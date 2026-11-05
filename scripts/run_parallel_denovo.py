#!/usr/bin/env python3
"""Supervise two independently GPU-bound de novo trajectories; Linux only."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def run(root, output, configs, gpus):
    from molcraft.io.config import load_config
    from molcraft.io.examples import task_path, validate_msa_input
    from molcraft.chemistry.inventory import EvidenceIndex

    root = Path(root).resolve()
    output = task_path(root, output)
    if len(set(gpus)) != 2 or any(g < 0 for g in gpus):
        raise ValueError("GBA and VDR require two distinct nonnegative GPU indices")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("The launcher requires a configured DEEPSEEK_API_KEY")
    jobs = []
    for target, path, gpu, rounds in zip(("GBA", "VDR"), configs, gpus, (17, 16)):
        path = task_path(root, path)
        config = load_config(path)
        if (config.optimization.strategy != "denovo" or config.msa_mode != "provided"
                or config.optimization.min_rounds != rounds or config.optimization.max_rounds != rounds
                or config.evaluation.candidate_budget != 8 or config.denovo.max_api_retries != 3):
            raise ValueError("Both jobs require denovo, verified MSA, 17/16 rounds, 8 candidates and 3 API retries")
        for protein in config.complex.proteins:
            validate_msa_input(task_path(root, protein.msa_path), protein.sequence)
        index = EvidenceIndex.from_policy(config.denovo)
        index.close()
        jobs.append({"target": target, "config": str(path), "gpu_index": gpu,
                     "output": str(output / target.lower() / "trajectory"), "rounds": rounds})
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / "parallel_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": jobs}
        if [{k: j[k] for k in jobs[0]} for j in state["jobs"]] != jobs:
            raise ValueError("Existing supervisor inputs differ")
        state["supervisor"] = {"pid": os.getpid(), "start_time": identity(os.getpid())}
        children = {}
        for job in state["jobs"]:
            if job.get("status") in {"completed", "failed"}:
                continue
            if job.get("pid") and identity(job["pid"]) == job.get("start_time"):
                continue  # A child can survive its original supervisor.
            destination = Path(job["output"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            arguments = ["python", "-u", "-m", "molcraft.cli", "optimize", "--config", job["config"],
                         "--output", job["output"], "--work-root", str(root)]
            if (destination / "checkpoint.json").exists():
                arguments.append("--resume")
            command = ("source /root/opt/mosaic/activate_dtk.sh >/dev/null; "
                       "unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES; "
                       f"export HIP_VISIBLE_DEVICES={job['gpu_index']} MOLCRAFT_EXPECT_SINGLE_GPU=1 "
                       "JAX_PLATFORMS=gpu XLA_PYTHON_CLIENT_PREALLOCATE=false; "
                       "exec " + shlex.join(arguments))
            with (destination.parent / "run.log").open("a") as log:
                process = subprocess.Popen(["bash", "--noprofile", "--norc", "-c", command], cwd=root,
                    env={**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "8"},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children[job["target"]] = process
            job.update(pid=process.pid, start_time=identity(process.pid), status="running", started_epoch=time.time())
            state["status"] = "running"
            save(state_path, state)
        while any(j.get("status") == "running" for j in state["jobs"]):
            for job in state["jobs"]:
                if job.get("status") != "running":
                    continue
                process = children.get(job["target"])
                code = process.poll() if process else None
                if identity(job["pid"]) == job["start_time"] and code is None:
                    continue
                checkpoint_path = Path(job["output"]) / "checkpoint.json"
                checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
                completed = checkpoint.get("status") in {"converged", "budget_complete"}
                job.update(status="completed" if completed and code in (None, 0) else "failed",
                           exit_code=code, finished_epoch=time.time(), trajectory_status=checkpoint.get("status"),
                           error=checkpoint.get("error"))
            state["updated_epoch"] = time.time()
            save(state_path, state)
            if any(j.get("status") == "running" for j in state["jobs"]):
                time.sleep(15)
        state["status"] = "completed" if all(j["status"] == "completed" for j in state["jobs"]) else "finished_with_failures"
        save(state_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/workspace/MolCraft")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gba-config", required=True)
    parser.add_argument("--vdr-config", required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[0, 1])
    args = parser.parse_args()
    run(args.root, args.output, [args.gba_config, args.vdr_config], args.gpus)
