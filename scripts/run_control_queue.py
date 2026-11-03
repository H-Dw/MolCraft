#!/usr/bin/env python3
"""One-shot, restartable DCU-gated experiment queue. Standard library only.

Deploy a copy outside the tracked source before scheduling: the running Agent
must keep its source checkout unchanged until the delayed gate opens.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request


TERMINAL = {"converged", "budget_complete"}


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path.is_relative_to(Path("/root/private_data").resolve()):
        raise ValueError("Queue paths must remain inside the non-private work root")
    return path


def file_sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def process_identity(pid, proc=Path("/proc")):
    try:
        fields = (proc / str(pid) / "stat").read_text().split(") ", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def alive(identity):
    return bool(identity and process_identity(identity["pid"]) == identity["start_time"])


def read_memory(pci_bus, sys_root=Path("/sys/class/drm"), smi="rocm-smi"):
    """Read only the allocated PCI device; never average host-wide cards."""
    for card in sys_root.glob("card[0-9]*"):
        if not card.name[4:].isdigit():
            continue
        device = card / "device"
        if device.resolve().name.lower() != pci_bus.lower():
            continue
        try:
            total = int((device / "mem_info_vram_total").read_text())
            used = int((device / "mem_info_vram_used").read_text())
            if not 0 <= used <= total or total <= 0:
                raise ValueError("Invalid device memory values")
            return {"pci_bus": pci_bus, "used_bytes": used, "total_bytes": total,
                    "fraction": used / total, "source": str(device)}
        except OSError:
            break
    result = subprocess.run([smi, "--showbus", "--showmeminfo", "vram", "--json"],
                            capture_output=True, text=True, timeout=20, check=True)
    payload = json.loads(result.stdout[result.stdout.index("{"):])
    for row in payload.values():
        if row.get("PCI Bus", "").lower() != pci_bus.lower():
            continue
        total = int(row["vram Total Memory (MiB)"]) * 2**20
        used = int(row["vram Total Used Memory (MiB)"]) * 2**20
        if not 0 <= used <= total or total <= 0:
            raise ValueError("Invalid device memory values")
        return {"pci_bus": pci_bus, "used_bytes": used, "total_bytes": total,
                "fraction": used / total, "source": "rocm-smi"}
    raise ValueError("Allocated device not found")


def gate(manifest, now, blockers_running, memory):
    if now < manifest["not_before_epoch"]:
        return False, "waiting_until_due"
    if blockers_running:
        return False, "waiting_existing_agent"
    if memory is None:
        return False, "waiting_memory_read_error"
    if memory["fraction"] >= manifest["memory_threshold"]:
        return False, "waiting_memory"
    return True, "ready"


def active_optimizations(proc=Path("/proc")):
    """Protect a live checkout even if a legacy process was restarted with a new PID."""
    found = []
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            command = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if "-m molcraft.cli optimize " in command and process_identity(int(p.name), proc):
                found.append(int(p.name))
        except (OSError, UnicodeError):
            continue
    return found


def sync_code(root, required_commit, log):
    request = urllib.request.Request(
        "https://api.github.com/repos/H-Dw/MolCraft/commits/main",
        headers={"User-Agent": "MolCraft-control-queue"})
    with urllib.request.urlopen(request, timeout=25) as response:
        latest = json.load(response)["sha"]
    if len(latest) != 40 or any(c not in "0123456789abcdef" for c in latest):
        raise ValueError("Invalid official main commit")
    subprocess.run(["bash", str(root / "scripts/sync_optagent.sh"), latest],
                   cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)
    subprocess.run(["git", "merge-base", "--is-ancestor", required_commit, latest],
                   cwd=root, check=True, timeout=15)
    # All native CPU tests must pass on the actual downloaded checkout, without probing DCU.
    cpu = ("source /root/opt/mosaic/activate_dtk.sh && "
           "export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' "
           "ROCR_VISIBLE_DEVICES='' && export PYTHONPATH=" + shlex.quote(str(root / "src")) +
           " && python -m pytest -q -m 'not model_integration' " + shlex.quote(str(root / "tests")))
    subprocess.run(["bash", "-lc", cpu], cwd=root, stdout=log, stderr=subprocess.STDOUT,
                   check=True, timeout=600)
    return latest


def prepare_jobs(root, manifest):
    import yaml
    from molcraft.io.examples import prepare_example
    from molcraft.io.config import load_config

    for job in manifest["jobs"]:
        if job.get("existing"):
            continue
        config_path = inside(root, job["config"])
        if config_path.exists():
            config = load_config(config_path)
            if config.optimization.strategy != job["strategy"] or config.msa_mode != job["msa_mode"]:
                raise ValueError("Existing control input has a different strategy or MSA mode")
            continue
        if job["case"] == "gba-2v3d":
            source = inside(root, manifest["gba_source_config"])
            data = yaml.safe_load(source.read_text(encoding="utf-8"))
            for protein in data["proteins"]:
                for field in ("msa_path", "template_path"):
                    if protein.get(field):
                        protein[field] = str((source.parent / protein[field]).resolve())
        else:
            prepare_example(job["case"], config_path.parent, manifest["model_dir"],
                            work_root=root, msa_mode="single_sequence")
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data.setdefault("run", {})["msa_mode"] = job["msa_mode"]
        data.setdefault("optimization", {})["strategy"] = job["strategy"]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        config = load_config(config_path)
        if (config.evaluation.candidate_budget != 8 or config.optimization.max_rounds != 25
                or config.seed != 0):
            raise ValueError("Control budget differs from the locked protocol")
        for protein in config.complex.proteins:
            if protein.msa_path:
                expected = read_json(Path(protein.msa_path).with_name("msa-provenance.json"))
                if file_sha(protein.msa_path) != expected["msa_sha256"]:
                    raise ValueError("GBA input MSA hash changed")


def summarize(root, manifest, state, destination):
    records = []
    for job in manifest["jobs"]:
        output = inside(root, job["output"])
        checkpoint = read_json(output / "checkpoint.json", {})
        rounds = [r for r in checkpoint.get("rounds", []) if "selected_id" in r]
        runtime = read_json(output / "runtime.json", {})
        record = {"job": job["id"], "strategy": job["strategy"], "msa_mode": job["msa_mode"],
                  "queue_status": state["jobs"][job["id"]]["status"],
                  "trajectory_status": checkpoint.get("status"), "completed_rounds": len(rounds),
                  "accepted_edits": sum(r["status"] == "improved" for r in rounds),
                  "candidate_evaluations": sum(r.get("evaluated_candidates", 0) for r in rounds),
                  "commit": runtime.get("commit"), "input_hashes": checkpoint.get("input_hashes"),
                  "final_graph": checkpoint.get("graph"), "seconds": sum(r["seconds"] for r in rounds),
                  "llm_requests": len(list(output.glob("round-*/requests/*.json"))),
                  "error": state["jobs"][job["id"]].get("error") or checkpoint.get("error")}
        if rounds:
            evaluation = read_json(output / f"round-{rounds[-1]['round']:03d}/evaluation.json", {})
            selected = next((c for c in evaluation.get("candidates", [])
                             if c["candidate_id"] == evaluation.get("selected_id")), evaluation.get("baseline", {}))
            record["final_loss"] = rounds[-1]["loss"]
            record["final_metrics"] = selected.get("metrics", {})
            record["final_artifacts"] = selected.get("artifacts", {})
        records.append(record)
    write_json(destination / "comparison.json", {"single_seed": True, "targets_use_different_msa_modes": True,
               "affinity_is_predicted_not_experimental": True, "trajectories": records})
    fields = ["job", "strategy", "msa_mode", "queue_status", "trajectory_status", "completed_rounds",
              "accepted_edits", "candidate_evaluations", "final_loss", "seconds", "llm_requests", "commit", "error"]
    with (destination / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def run(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    root = Path(manifest["work_root"]).resolve()
    destination = inside(root, manifest["queue_dir"])
    if not 0 < manifest["memory_threshold"] <= 0.1 or manifest["poll_seconds"] != 1800:
        raise ValueError("Queue requires threshold <=10% and a 30 minute poll interval")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".scheduler.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = destination / "queue_state.json"
        manifest_hash = file_sha(manifest_path)
        state = read_json(state_path, {"manifest_sha256": manifest_hash, "status": "created",
                          "jobs": {j["id"]: {"status": "pending"} for j in manifest["jobs"]}})
        if state["manifest_sha256"] != manifest_hash:
            raise ValueError("Queue manifest changed after scheduling")
        state["scheduler"] = {"pid": os.getpid(), "start_time": process_identity(os.getpid())}

        def persist(status, **extra):
            state.update(status=status, updated_epoch=time.time(), **extra)
            write_json(state_path, state)
            summarize(root, manifest, state, destination)
            print(json.dumps({"epoch": time.time(), "status": status, **extra}), flush=True)

        persist("scheduled", not_before_epoch=manifest["not_before_epoch"])
        while True:
            for job in manifest["jobs"]:
                entry = state["jobs"][job["id"]]
                checkpoint = read_json(inside(root, job["output"]) / "checkpoint.json", {})
                if entry["status"] != "failed" and checkpoint.get("status") in TERMINAL:
                    entry["status"] = "completed"
            pending = [j for j in manifest["jobs"] if state["jobs"][j["id"]]["status"] not in {"completed", "failed"}]
            if not pending:
                persist("completed" if all(v["status"]=="completed" for v in state["jobs"].values()) else "finished_with_failures")
                return
            # A child of a prior scheduler can survive its SSH/session loss. Do not duplicate it.
            surviving = [state["jobs"][j["id"]].get("process") for j in pending
                         if alive(state["jobs"][j["id"]].get("process"))]
            if surviving:
                persist("waiting_surviving_child")
                time.sleep(30)
                continue
            memory, error = None, None
            if time.time() >= manifest["not_before_epoch"]:
                try:
                    memory = read_memory(manifest["pci_bus"])
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            ready, reason = gate(manifest, time.time(), bool(active_optimizations()), memory)
            if not ready:
                wait = max(1, min(manifest["poll_seconds"], manifest["not_before_epoch"]-time.time())) if reason=="waiting_until_due" else manifest["poll_seconds"]
                persist(reason, memory=memory, probe_error=error, next_check_epoch=time.time()+wait)
                time.sleep(wait)
                continue
            if not state.get("synced_commit"):
                try:
                    persist("syncing_code", memory=memory)
                    with (destination / "sync.log").open("a") as log:
                        commit = sync_code(root, manifest["required_commit"], log)
                    state["synced_commit"] = commit
                except Exception as exc:
                    persist("waiting_code_sync", error=f"{type(exc).__name__}: {exc}",
                            next_check_epoch=time.time()+manifest["poll_seconds"])
                    time.sleep(manifest["poll_seconds"])
                    continue
            try:
                # Use the model environment to prepare YAML, without initializing a GPU.
                prepare_command = "source /root/opt/mosaic/activate_dtk.sh && export PYTHONPATH=" + shlex.quote(str(root / "src")) + " && python " + shlex.quote(str(Path(__file__).resolve())) + " --prepare " + shlex.quote(str(manifest_path))
                with (destination / "prepare.log").open("a") as log:
                    subprocess.run(["bash", "-lc", prepare_command], cwd=root, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=120)
            except Exception as exc:
                persist("input_preparation_failed", error=f"{type(exc).__name__}: {exc}")
                return
            job = pending[0]
            entry = state["jobs"][job["id"]]
            try:
                latest_memory = read_memory(manifest["pci_bus"])
                ready, _ = gate(manifest, time.time(), bool(active_optimizations()), latest_memory)
                if not ready:
                    persist("waiting_memory_before_launch", memory=latest_memory,
                            next_check_epoch=time.time()+manifest["poll_seconds"])
                    time.sleep(manifest["poll_seconds"])
                    continue
                current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,text=True).strip()
                if current != state["synced_commit"]:
                    raise ValueError("Checkout changed since queue synchronization")
                args = ["bash", "scripts/run_optagent.sh", "optimize", "--config", job["config"], "--output", job["output"]]
                if (inside(root, job["output"]) / "checkpoint.json").exists():
                    args.append("--resume")
                command = "cd " + shlex.quote(str(root)) + " && exec " + shlex.join(args)
                with (destination / f"{job['id']}.log").open("a") as log:
                    process = subprocess.Popen(["bash", "--noprofile", "-ic", command], cwd=root,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    entry.update(status="running", started_epoch=time.time(),
                                 process={"pid":process.pid,"start_time":process_identity(process.pid)},
                                 commit=current)
                    persist("running", current_job=job["id"], memory=latest_memory)
                    while process.poll() is None:
                        summarize(root, manifest, state, destination)
                        time.sleep(30)
                checkpoint = read_json(inside(root, job["output"]) / "checkpoint.json", {})
                entry.update(status="completed" if process.returncode==0 and checkpoint.get("status") in TERMINAL else "failed",
                             exit_code=process.returncode, finished_epoch=time.time())
                if entry["status"]=="failed":
                    entry["error"] = checkpoint.get("error", "Child exited without a complete trajectory")
            except Exception as exc:
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            persist("job_finished", current_job=job["id"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--prepare")
    args = parser.parse_args()
    if args.prepare:
        data = read_json(args.prepare)
        prepare_jobs(Path(data["work_root"]), data)
    elif args.manifest:
        run(args.manifest)
    else:
        parser.error("--manifest or --prepare is required")
