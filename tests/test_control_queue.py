import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


MODULE = Path(__file__).parents[1]/"scripts/run_control_queue.py"
spec = importlib.util.spec_from_file_location("control_queue", MODULE)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.mark.parametrize("fraction,ready",[(.0999,True),(.1,False),(.77,False)])
def test_memory_gate_strict_threshold(fraction, ready):
    manifest={"not_before_epoch":100,"memory_threshold":.1}
    assert queue.gate(manifest,100,False,{"fraction":fraction})[0] is ready
    assert not queue.gate(manifest,99,False,{"fraction":0})[0]
    assert not queue.gate(manifest,100,True,{"fraction":0})[0]
    assert queue.gate(manifest,100,False,None)==(False,"waiting_memory_read_error")


def test_memory_reads_allocated_pci_not_another_idle_card(tmp_path):
    allocated=tmp_path/"0000:0f:00.0"
    allocated.mkdir()
    (allocated/"mem_info_vram_total").write_text("1000")
    (allocated/"mem_info_vram_used").write_text("770")
    cards=tmp_path/"drm"
    (cards/"card3").mkdir(parents=True)
    (cards/"card3/device").symlink_to(allocated)
    result=queue.read_memory("0000:0f:00.0",sys_root=cards)
    assert result["fraction"]==.77 and result["used_bytes"]==770


def test_process_identity_and_path_boundaries(tmp_path):
    assert queue.alive({"pid":os.getpid(),"start_time":queue.process_identity(os.getpid())})
    assert not queue.alive({"pid":os.getpid(),"start_time":"wrong-start-time"})
    with pytest.raises(ValueError):
        queue.inside(tmp_path,"../outside")
    (tmp_path/"escape").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):
        queue.inside(tmp_path,"escape/outside")


def test_cpu_interpreter_keeps_virtualenv_symlink_and_rejects_directory_escape(tmp_path):
    binary=tmp_path/"cpu-env/bin/python"
    binary.parent.mkdir(parents=True)
    binary.symlink_to(sys.executable)
    selected=queue.cpu_interpreter(tmp_path,"cpu-env/bin/python")
    assert selected==binary and selected.is_symlink()
    assert selected.resolve()==Path(sys.executable).resolve()
    with pytest.raises(ValueError):
        queue.inside(tmp_path,"cpu-env/bin/python")
    (tmp_path/"escaped-env").symlink_to(binary.parent.parent.parent.parent)
    with pytest.raises(ValueError):
        queue.cpu_interpreter(tmp_path,"escaped-env/bin/python")
    with pytest.raises(ValueError):
        queue.cpu_interpreter(tmp_path,"cpu-env/bin/missing")


def test_denovo_queue_checks_evidence_before_launch(tmp_path):
    from molcraft.errors import ConfigurationError
    from molcraft.io.examples import prepare_example
    prepare_example("vdr-3a2i", tmp_path/"vdr", tmp_path, work_root=tmp_path,
                    strategy="denovo", msa_mode="single_sequence")
    manifest = {"jobs": [{"existing": True, "strategy": "denovo", "case": "vdr-3a2i",
                          "config": "vdr/mosaic-like.yaml"}]}
    with pytest.raises(ConfigurationError, match="not ready"):
        queue.prepare_jobs(tmp_path, manifest)


def test_denovo_queue_accepts_template_only_evidence(tmp_path):
    from molcraft.chemistry.inventory import build_index
    from molcraft.io.examples import prepare_example
    path = tmp_path / "templates.json"
    path.write_text(json.dumps(["[#6:1]-[#8:2]>>[#6:1].[#8:2]"]))
    build_index(None, tmp_path / "templates.sqlite", release="SYNTHETIC_TEST_ONLY",
                source_url="https://example.invalid", templates=path, template_source_url="https://example.invalid")
    prepare_example("vdr-3a2i", tmp_path/"vdr", tmp_path, work_root=tmp_path,
                    strategy="denovo", msa_mode="single_sequence",
                    evidence_manifest=tmp_path / "templates.sqlite.manifest.json")
    manifest = {"jobs": [{"existing": True, "strategy": "denovo", "case": "vdr-3a2i",
                          "config": "vdr/mosaic-like.yaml"}]}
    queue.prepare_jobs(tmp_path, manifest)


@pytest.mark.parametrize("due,memory,reason,wait",[
    (1000,None,"waiting_until_due",900),
    (0,{"fraction":.1},"waiting_memory",1800),
    (0,None,"waiting_memory_read_error",1800),
])
def test_real_queue_defers_without_syncing_or_launching(tmp_path,monkeypatch,due,memory,reason,wait):
    manifest={"work_root":str(tmp_path),"queue_dir":"queue","not_before_epoch":due,
              "memory_threshold":.1,"poll_seconds":1800,"pci_bus":"0000:0f:00.0",
              "jobs":[{"id":"vdr","output":"vdr","strategy":"argmax","msa_mode":"single_sequence"}]}
    path=tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(queue.time,"time",lambda:100)
    monkeypatch.setattr(queue,"active_optimizations",lambda:[])

    def read(*args):
        if memory is None:
            raise ValueError("probe unavailable")
        return memory

    monkeypatch.setattr(queue,"read_memory",read)
    class Deferred(Exception):
        pass

    def sleep(seconds):
        assert seconds==wait
        raise Deferred

    monkeypatch.setattr(queue.time,"sleep",sleep)
    monkeypatch.setattr(queue,"sync_code",lambda *a:pytest.fail("must not sync before gate"))
    with pytest.raises(Deferred):
        queue.run(path)
    state=json.loads((tmp_path/"queue/queue_state.json").read_text())
    assert state["status"]==reason and state["next_check_epoch"]==100+wait
    # Restart keeps its absolute due date, rather than starting a new six-hour delay.
    with pytest.raises(Deferred):
        queue.run(path)
    assert json.loads((tmp_path/"manifest.json").read_text())["not_before_epoch"]==due


def test_completed_queue_does_not_restart_existing_trajectory(tmp_path):
    manifest={"work_root":str(tmp_path),"queue_dir":"queue","not_before_epoch":0,
              "memory_threshold":.1,"poll_seconds":1800,
              "jobs":[{"id":"gba","output":"gba","strategy":"optagent","msa_mode":"provided"}]}
    (tmp_path/"gba").mkdir()
    (tmp_path/"gba/checkpoint.json").write_text(json.dumps({"status":"converged","rounds":[]}))
    path=tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    queue.run(path)
    state=json.loads((tmp_path/"queue/queue_state.json").read_text())
    assert state["status"]=="completed" and state["jobs"]["gba"]["status"]=="completed"


def test_queue_continues_after_failed_agent_and_runs_argmax(tmp_path,monkeypatch):
    commit="a"*40
    jobs=[{"id":name,"output":name,"config":name+".yaml","strategy":strategy,
           "msa_mode":"single_sequence"} for name,strategy in [("agent","optagent"),("control","argmax")]]
    manifest={"work_root":str(tmp_path),"queue_dir":"queue","not_before_epoch":0,
              "memory_threshold":.1,"poll_seconds":1800,"pci_bus":"0000:0f:00.0",
              "required_commit":commit,"cpu_python":"cpu-env/bin/python","jobs":jobs}
    path=tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    binary=tmp_path/"cpu-env/bin/python"
    binary.parent.mkdir(parents=True)
    binary.symlink_to(sys.executable)
    monkeypatch.setattr(queue,"read_memory",lambda *a:{"fraction":0.01})
    monkeypatch.setattr(queue,"active_optimizations",lambda:[])

    def sync(root,required,log,cpu_python):
        assert cpu_python==binary and cpu_python.is_symlink()
        return commit

    monkeypatch.setattr(queue,"sync_code",sync)
    monkeypatch.setattr(queue.subprocess,"run",lambda *a,**k:None)
    monkeypatch.setattr(queue.subprocess,"check_output",lambda *a,**k:commit)
    launched=[]

    class Process:
        pid=99999999

        def __init__(self,args,**kwargs):
            name="agent" if "agent.yaml" in args[-1] else "control"
            launched.append(name)
            self.returncode=2 if name=="agent" else 0
            queue.write_json(tmp_path/name/"checkpoint.json",
                             {"status":"provider_interrupted" if name=="agent" else "budget_complete",
                              "rounds":[],"error":"test provider failure" if name=="agent" else None})

        def poll(self):
            return self.returncode

    monkeypatch.setattr(queue.subprocess,"Popen",Process)
    queue.run(path)
    state=json.loads((tmp_path/"queue/queue_state.json").read_text())
    assert launched==["agent","control"]
    assert state["status"]=="finished_with_failures"
    assert state["jobs"]["agent"]["status"]=="failed"
    assert state["jobs"]["control"]["status"]=="completed"


def test_another_scheduler_holding_lock_prevents_duplicate(tmp_path):
    import fcntl
    (tmp_path/"queue").mkdir()
    manifest={"work_root":str(tmp_path),"queue_dir":"queue","memory_threshold":.1,
              "poll_seconds":1800}
    path=tmp_path/"manifest.json"
    path.write_text(json.dumps(manifest))
    with (tmp_path/"queue/.scheduler.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            queue.run(path)


def test_sync_prefers_official_git_without_rate_limited_api(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import io
    sha="a"*40
    calls=[]

    def command(args,**kwargs):
        calls.append(args)
        if "ls-remote" in args:
            assert "credential.helper=" in args and kwargs["env"]["GIT_TERMINAL_PROMPT"]=="0"
            return SimpleNamespace(stdout=sha+"\trefs/heads/main\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(queue.subprocess,"run",command)
    monkeypatch.setattr(queue.urllib.request,"urlopen",lambda *a,**k:pytest.fail("API must not be required"))
    assert queue.sync_code(tmp_path,sha,io.StringIO(),tmp_path/"cpu-env/bin/python")==sha
    assert any("--is-ancestor" in c for c in calls)
    assert any("env -u LD_LIBRARY_PATH" in str(c) and "cpu-env/bin/python" in str(c)
               and "JAX_ENABLE_X64=1" in str(c) for c in calls)
