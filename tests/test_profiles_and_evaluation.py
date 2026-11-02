"""Budget/depth/acceptance contracts; never load real model assets."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import pytest
import yaml

from molcraft.chemistry.graph import MolecularGraph
from molcraft.errors import ConfigurationError, GradientError, StaleProposalError
from molcraft.evaluation import evaluate_packet
from molcraft.guidance.proposals import build_proposals
from molcraft.io.config import EvaluationSpec, load_config

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("name,mode,geometry,budget,steps,passes,guard", [
    ("test", "through_structure", 1., 2, 25, 1, False),
    ("mosaic-like", "through_structure", 1., 8, 200, 4, True),
    ("bdmol-like", "trunk_only", 0., 8, 200, 4, True),
])
def test_profiles_load_with_explicit_distinct_design_and_evaluation_depths(
        name, mode, geometry, budget, steps, passes, guard):
    config = load_config(ROOT / "configs" / f"{name}.yaml")
    assert (config.gradient_mode, config.objective.geometry_weight) == (mode, geometry)
    assert (config.sampling_steps, config.trunk_passes) == (25, 1)
    assert (config.evaluation_sampling_steps, config.evaluation_trunk_passes) == (steps, passes)
    assert config.evaluation.candidate_budget == budget
    assert config.evaluation.require_affinity_nonworsening == guard
    assert config.policy.max_candidates == 32 and config.policy.max_bundle_size == 2


def test_legacy_configs_keep_old_evaluation_defaults():
    config = load_config(ROOT / "configs" / "boltz2.yaml")
    assert config.evaluation_sampling_steps == config.sampling_steps == 25
    assert config.evaluation_trunk_passes == config.trunk_passes == 1
    assert config.evaluation.candidate_budget == 2


@pytest.mark.parametrize("kwargs", [
    {"candidate_budget": 0}, {"candidate_budget": True}, {"sampling_steps": 0},
    {"trunk_passes": 1.5}, {"require_affinity_nonworsening": "false"},
])
def test_bad_evaluation_settings_rejected(kwargs):
    with pytest.raises(ConfigurationError):
        EvaluationSpec(**kwargs)


def test_trunk_cutoff_requires_zero_geometry_and_guard_requires_affinity(tmp_path):
    data = yaml.safe_load((ROOT / "configs" / "bdmol-like.yaml").read_text())
    path = tmp_path / "invalid.yaml"
    data["objective"]["geometry_weight"] = 1
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigurationError, match="trunk_only"):
        load_config(path)
    data["objective"].update(geometry_weight=0, affinity_weight=0)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigurationError, match="affinity"):
        load_config(path)


class FakeBackend:
    sampling_steps = 200
    trunk_passes = 4

    def __init__(self):
        self.noise = object()
        self.calls = 0

    def draw_noise(self, prepared, seed):
        return self.noise

    def evaluate(self, prepared, objective, seed, noise):
        assert noise is self.noise
        self.calls += 1
        return {"loss": 10., "metrics": {"affinity_loss": 2.}}


@pytest.mark.parametrize("guard,expected_rank", [(False, 0), (True, 1)])
def test_real_scores_not_proposal_scores_choose_graph_with_optional_affinity_guard(
        graph, evidence, monkeypatch, tmp_path, guard, expected_rank):
    config = load_config(ROOT / "configs" / "mosaic-like.yaml")
    config = replace(config, evaluation=replace(config.evaluation, candidate_budget=2,
                                                require_affinity_nonworsening=guard))
    packet = build_proposals(graph, evidence)
    ordered = sorted((c for c in packet.candidates if c.operations),
                     key=lambda c: (-c.probability, c.predicted_change, c.candidate_id))[:2]
    assert len(ordered) == 2
    backend = FakeBackend()
    seen = []

    def rescore(prepared, candidate, **kwargs):
        assert kwargs["backend"] is backend and kwargs["noise"] is backend.noise
        seen.append(candidate.candidate_id)
        # First has best joint loss but worse affinity; second meets both gates.
        loss, affinity = (6., 3.) if candidate.candidate_id == ordered[0].candidate_id else (7., 1.)
        return {"loss": loss, "metrics": {"affinity_loss": affinity}}

    monkeypatch.setattr("molcraft.api.evaluate_candidate", rescore)
    prepared = SimpleNamespace(graph=graph)
    result = evaluate_packet(prepared, packet, config, backend=backend, export_dir=tmp_path)
    assert backend.calls == 1  # fresh baseline at 200 / 4; do not use gradient evidence loss
    assert seen == [c.candidate_id for c in ordered]
    assert result["selected_id"] == ordered[expected_rank].candidate_id
    assert MolecularGraph.from_dict(result["selected_graph"]).fingerprint == ordered[expected_rank].graph.fingerprint
    assert prepared.graph.fingerprint == graph.fingerprint  # source never mutated
    assert result["evaluation"]["evaluated_candidates"] == 2


def test_no_improvement_retains_source_and_rejects_stale_or_wrong_depth(graph, evidence, monkeypatch):
    config = load_config(ROOT / "configs" / "mosaic-like.yaml")
    packet = build_proposals(graph, evidence)
    backend = FakeBackend()
    monkeypatch.setattr("molcraft.api.evaluate_candidate", lambda *a, **k:
                        {"loss": 11., "metrics": {"affinity_loss": 1.}})
    result = evaluate_packet(SimpleNamespace(graph=graph), packet, config, backend=backend)
    assert result["status"] == "source_retained" and result["selected_id"] == "noop"
    with pytest.raises(StaleProposalError):
        evaluate_packet(SimpleNamespace(graph=MolecularGraph.from_smiles("CCO")), packet, config, backend=backend)
    backend.sampling_steps = 25
    with pytest.raises(ConfigurationError, match="depth"):
        evaluate_packet(SimpleNamespace(graph=graph), packet, config, backend=backend)


def test_nonfinite_candidate_fails_instead_of_selecting(graph, evidence, monkeypatch):
    config = load_config(ROOT / "configs" / "mosaic-like.yaml")
    packet = build_proposals(graph, evidence)
    monkeypatch.setattr("molcraft.api.evaluate_candidate", lambda *a, **k:
                        {"loss": float("nan"), "metrics": {"affinity_loss": 1.}})
    with pytest.raises(GradientError):
        evaluate_packet(SimpleNamespace(graph=graph), packet, config, backend=FakeBackend())


def test_evaluate_set_cli_exposes_budgeted_path_without_loading_models():
    help_text = subprocess.check_output([sys.executable, "-m", "molcraft.cli", "evaluate-set", "--help"], text=True)
    assert "--config" in help_text and "--packet" in help_text
    report = subprocess.check_output([sys.executable, "-m", "molcraft.cli", "doctor"], text=True)
    assert json.loads(report)["model_weights_loaded"] is False


def test_no_legal_edit_only_scores_baseline_and_missing_guard_metric_fails(graph, evidence):
    from molcraft.guidance.proposals import ProposalPolicy

    config = load_config(ROOT / "configs" / "mosaic-like.yaml")
    packet = build_proposals(graph, evidence, ProposalPolicy(frozen_atom_ids=graph.atom_ids))
    backend = FakeBackend()
    result = evaluate_packet(SimpleNamespace(graph=graph), packet, config, backend=backend)
    assert result["selected_id"] == "noop" and result["evaluation"]["evaluated_candidates"] == 0
    backend.evaluate = lambda *a, **k: {"loss": 10., "metrics": {}}
    with pytest.raises(GradientError, match="affinity_loss"):
        evaluate_packet(SimpleNamespace(graph=graph), packet, config, backend=backend)


def test_evaluate_set_cli_consumes_profile_and_writes_next_graph(graph, evidence, tmp_path, monkeypatch):
    import molcraft
    from molcraft.cli import main
    from molcraft.types import PreparedComplex
    from molcraft.io.serialization import save_packet
    from molcraft.backends.boltz2_jax.backend import Boltz2Backend

    packet = build_proposals(graph, evidence)
    packet_path = tmp_path / "packet.json"
    save_packet(packet, packet_path)
    backend = FakeBackend()
    calls = []

    def load(cls, paths, sampling_steps, trunk_passes):
        calls.append((sampling_steps, trunk_passes))
        return backend

    monkeypatch.setattr(Boltz2Backend, "from_paths", classmethod(load))
    monkeypatch.setattr(molcraft, "prepare_complex", lambda spec: PreparedComplex(spec, graph, {}, None))
    monkeypatch.setattr("molcraft.api.evaluate_candidate", lambda *a, **k:
                        {"loss": 7., "metrics": {"affinity_loss": 1.}})
    output = tmp_path / "evaluation"
    assert main(["evaluate-set", "--config", str(ROOT / "configs" / "mosaic-like.yaml"),
                 "--packet", str(packet_path), "--output", str(output)]) == 0
    result = json.loads((output / "evaluation.json").read_text())
    assert calls == [(200, 4)]
    assert result["evaluation"]["evaluated_candidates"] == min(8, len(packet.candidates) - 1)
    assert result["selected_id"] != "noop" and result["status"] == "improved"
    next_graph = MolecularGraph.from_dict(json.loads((output / "next_graph.json").read_text()))
    assert next_graph.fingerprint == packet.candidate(result["selected_id"]).graph.fingerprint
