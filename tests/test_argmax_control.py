from dataclasses import asdict, replace
import json
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from molcraft.agent.argmax import ArgmaxAgent
from molcraft.chemistry.graph import MolecularGraph
from molcraft.chemistry.mapping import AtomLayout
from molcraft.errors import ConfigurationError, StaleProposalError
from molcraft.guidance.proposals import ProposalPolicy, build_proposals
from molcraft.io.config import EvaluationSpec, OptimizationSpec, RunConfig, load_config
from molcraft.io.examples import case_data, prepare_example
from molcraft.losses.spec import ObjectiveSpec
from molcraft.optimization import _digest, _input_identity, optimize
from molcraft.sampling.core import Noise
from molcraft.types import ComplexSpec, GradientEvidence, PreparedComplex, ProteinSpec, ResidueRef


def config():
    return RunConfig(ComplexSpec((ProteinSpec("A", "ACD"),), "CCC", (ResidueRef("A", 1),)),
                     ObjectiveSpec(), ProposalPolicy(), evaluation=EvaluationSpec(candidate_budget=8),
                     optimization=OptimizationSpec(max_rounds=2, min_rounds=2, patience=1, strategy="argmax"),
                     msa_mode="single_sequence")


def test_argmax_factory_never_imports_or_contacts_provider():
    code = """
import sys, urllib.request
def forbidden(*args, **kwargs):
    raise AssertionError('network access')
urllib.request.urlopen = forbidden
from molcraft.agent.argmax import create_agent
agent, metadata = create_agent('argmax')
assert metadata == {'api_models_http': None, 'model': None, 'llm_calls': 0}
assert 'molcraft.agent.deepseek' not in sys.modules
assert 'molcraft.agent.optagent' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_argmax_top_k_excludes_noop_and_breaks_ties(graph, evidence):
    packet = build_proposals(graph, evidence)
    packet.candidates = packet.candidates[:5]
    packet.candidates[0].probability = 1
    a, b, c, d = packet.candidates[1:]
    a.probability, a.predicted_change, a.candidate_id = .8, -1, "z"
    b.probability, b.predicted_change, b.candidate_id = .8, -2, "b"
    c.probability, c.predicted_change, c.candidate_id = .8, -2, "a"
    d.probability, d.predicted_change, d.candidate_id = .1, -100, "d"
    agent = ArgmaxAgent()
    assert agent.select(packet, 8) == ["a", "b", "z", "d"]
    assert agent.select(packet, 2) == ["a", "b"]
    with pytest.raises(ConfigurationError):
        agent.select(packet, 0)


def test_argmax_exclusion_is_before_candidate_cap_and_normalization(graph, evidence):
    agent, policy = ArgmaxAgent(), ProposalPolicy(max_candidates=8)
    kwargs = dict(samples=10, round_index=1, context_id="context")
    first = agent.propose(graph, evidence, policy, **kwargs)
    excluded = {c.graph.smiles for c in first.candidates[1:3]}
    second = agent.propose(graph, evidence, policy, excluded=excluded, **kwargs)
    assert len(second.candidates) == 9
    assert not excluded.intersection(c.graph.smiles for c in second.candidates)
    assert sum(c.probability for c in second.candidates) == pytest.approx(1)
    assert second.diagnostics["llm_calls"] == 0
    assert second.diagnostics["context_id"] == "context"
    assert second.to_dict() == agent.propose(graph, evidence, policy, excluded=excluded, **kwargs).to_dict()


def test_argmax_no_legal_edits_returns_empty_selection(graph, evidence):
    packet = ArgmaxAgent().propose(graph, evidence, ProposalPolicy(frozen_atom_ids=graph.atom_ids),
                                 samples=10, round_index=1, context_id="fixed")
    assert ArgmaxAgent().select(packet, 8) == []
    assert packet.status == "no_legal_edit"


def test_vdr_single_sequence_preparation_and_msa_conflict(tmp_path):
    folder = tmp_path / "vdr"
    result = prepare_example("vdr-3a2i", folder, tmp_path, work_root=tmp_path,
                             msa_mode="single_sequence")
    cfg = load_config(folder / "mosaic-like.yaml")
    assert result["status"] == "ready" and result["input_depth"] == 1
    assert result["msa_service_called"] is False
    assert cfg.complex.proteins[0].sequence == case_data("vdr-3a2i")["sequence"]
    assert len(cfg.complex.proteins[0].sequence) == 259
    assert cfg.complex.ligand_smiles == "C"*31
    assert cfg.msa_mode == "single_sequence" and cfg.complex.proteins[0].msa_path is None
    assert cfg.complex.proteins[0].template_path is None
    assert not list(folder.glob("*.a3m"))
    assert cfg.evaluation.candidate_budget == 8
    with pytest.raises(ConfigurationError, match="cannot import"):
        prepare_example("vdr-3a2i", folder, tmp_path, work_root=tmp_path,
                        msa_source="unused.a3m", msa_mode="single_sequence")
    with pytest.raises(ConfigurationError, match="cannot also"):
        replace(cfg, complex=replace(cfg.complex, proteins=(
            replace(cfg.complex.proteins[0], msa_path="x.a3m"),)))


def test_legacy_identity_is_preserved_and_new_modes_are_distinct():
    legacy = replace(config(), optimization=OptimizationSpec(), msa_mode="provided")
    data = asdict(legacy)
    data.pop("synthesis")
    data["complex"].pop("cache_dir")
    data.pop("msa_mode")
    data["optimization"].pop("strategy")
    expected = _digest({"version": 1, "config": data, "inputs": {}})
    assert _input_identity(legacy)[0] == expected
    assert _input_identity(replace(legacy, msa_mode="single_sequence"))[0] != expected
    assert _input_identity(replace(legacy, optimization=OptimizationSpec(strategy="argmax")))[0] != expected
    with pytest.raises(ConfigurationError):
        OptimizationSpec(strategy="unknown")


@pytest.mark.parametrize("failed_candidate", [False, True])
def test_argmax_trajectory_resume_and_cross_mode_rejection(tmp_path, failed_candidate):
    cfg = config()

    def prepare(spec):
        graph = MolecularGraph.from_smiles(spec.ligand_smiles)
        layout = AtomLayout(graph.atom_ids, (1, 2, 3), (1, 2, 3), (0,), (0,), ((0,),), 4, 4)
        return PreparedComplex(spec, graph, {"msa": np.zeros((1,1,4,33))}, layout)

    class Backend:
        def draw_noise(self, prepared, seed):
            return Noise(jnp.zeros((4,3)), jnp.zeros((1,4,3)), jnp.eye(3)[None], jnp.zeros((1,3)))

        def value_and_grad(self, prepared, *args, **kwargs):
            g = prepared.graph
            return GradientEvidence(g.fingerprint, g.atom_ids, np.zeros((3,9)),
                                    np.zeros((3,5)), np.zeros(3))

    selected = []

    def score(prepared, packet, config, *, selected_ids, **kwargs):
        assert len(selected_ids) == 8 and "noop" not in selected_ids
        selected.extend(packet.candidate(i).graph.smiles for i in selected_ids)
        return {"selected_id":"noop", "selected_graph":prepared.graph.to_dict(),
                "status":"source_retained", "selected_loss":1., "baseline":{"loss":1.},
                "candidates":[{"candidate_id":i} for i in (selected_ids[:-1] if failed_candidate else selected_ids)],
                "unscored_candidates":[{"candidate_id":selected_ids[-1],"error":"no conformer"}] if failed_candidate else []}

    backend = Backend()
    args = dict(backend=backend, evaluation_backend=backend, prepare=prepare, score=score)
    first = optimize(cfg, ArgmaxAgent(), tmp_path, stop_after=1, **args)
    assert first["status"] == "window_complete"
    assert first["rounds"][0]["attempted_candidates"] == 8
    assert first["rounds"][0]["rejected_candidates"] == int(failed_candidate)
    last = optimize(cfg, ArgmaxAgent(), tmp_path, resume=True, **args)
    assert last["status"] == "converged" and last["next_round"] == 3
    assert len(set(selected)) == 16
    assert not list(tmp_path.glob("round-*/requests/*"))
    assert json.loads((tmp_path/"feature_profile.json").read_text())["msa_shape"] == [1,1,4,33]
    for other in (replace(cfg, msa_mode="provided"),
                  replace(cfg, optimization=replace(cfg.optimization, strategy="optagent"))):
        with pytest.raises(StaleProposalError):
            optimize(other, ArgmaxAgent(), tmp_path, resume=True, **args)


def test_cli_single_sequence_argmax_bypasses_msa_and_provider(tmp_path, monkeypatch):
    import jax
    from types import SimpleNamespace
    from molcraft.cli import main
    import molcraft.optimization as driver

    folder = tmp_path/"input"
    prepare_example("vdr-3a2i", folder, tmp_path, msa_mode="single_sequence", work_root=tmp_path)
    import yaml
    path = folder/"mosaic-like.yaml"
    data = yaml.safe_load(path.read_text())
    data["optimization"]["strategy"] = "argmax"
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(jax, "devices", lambda:[SimpleNamespace(platform="gpu",device_kind="fake-test-device")])

    def fake_optimize(cfg, agent, output, **kwargs):
        assert isinstance(agent, ArgmaxAgent) and cfg.msa_mode=="single_sequence"
        return {"status":"budget_complete", "next_round":26}

    monkeypatch.setattr(driver, "optimize", fake_optimize)
    result = main(["optimize","--config",str(path),"--output",str(tmp_path/"run"),
                   "--work-root",str(tmp_path)])
    assert result == 0
    runtime = json.loads((tmp_path/"run/runtime.json").read_text())
    assert runtime["llm_calls"]==0 and runtime["model"] is None

