"""Candidate-local 3D failure isolation, bounded budgets and crash-safe replay."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from molcraft.backends.boltz2_jax.features import _reference_conformer, _single_native_record
from molcraft.chemistry.graph import MolecularGraph
from molcraft.errors import ConfigurationError, LigandPreparationError, MappingError, StaleProposalError
from molcraft.evaluation import evaluate_packet
from molcraft.guidance.proposals import build_proposals
from molcraft.io.config import load_config


def test_empty_native_manifest_only_recovers_explicit_failure_for_the_same_ligand():
    graph = MolecularGraph.from_smiles("CCP#C")
    manifest = SimpleNamespace(records=[])
    message = f"Failed to compute 3D conformer for {graph.mapped_smiles}"
    with pytest.raises(LigandPreparationError, match="3D conformer") as err:
        _single_native_record(manifest, graph, f"Failed to process input. Skipping. Error: {message}.")
    assert err.value.stage == "native_conformer"
    for wrong_log in ("", "Missing MSA file", "Failed to compute 3D conformer for CCC"):
        with pytest.raises(MappingError):
            _single_native_record(manifest, graph, wrong_log)
    record = object()
    assert _single_native_record(SimpleNamespace(records=[record]), graph, "") is record
    with pytest.raises(MappingError):
        _single_native_record(SimpleNamespace(records=[record, record]), graph, message)


@pytest.mark.parametrize("raises", [False, True])
def test_reference_embedding_failure_is_a_ligand_error_with_deterministic_fallback(monkeypatch, raises):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    attempts = []

    def embed(mol, options):
        attempts.append((options.randomSeed, options.useRandomCoords))
        if raises:
            raise RuntimeError("impossible distance bounds")
        return -1

    monkeypatch.setattr(AllChem, "EmbedMolecule", embed)
    with pytest.raises(LigandPreparationError) as err:
        _reference_conformer(Chem.MolFromSmiles("CCC"), 0)
    assert err.value.stage == "reference_conformer"
    assert attempts == [(0, False), (0, True)]


def setup_case(graph, evidence):
    config = load_config(Path(__file__).parents[1] / "configs/mosaic-like.yaml")
    config = replace(config, evaluation=replace(config.evaluation, candidate_budget=2))
    packet = build_proposals(graph, evidence)
    packet.diagnostics["context_id"] = "same-context"
    options = sorted((c for c in packet.candidates if c.operations),
                     key=lambda c: (-c.probability, c.predicted_change, c.candidate_id))[:2]
    assert len(options) == 2
    prepared = SimpleNamespace(graph=graph, native_writer=None)
    return config, packet, options, prepared


class Backend:
    sampling_steps = 200
    trunk_passes = 4

    def __init__(self):
        self.calls = 0
        self.noise = SimpleNamespace(init=np.zeros((4,3)), churn=np.zeros((2,4,3)),
                                     rotations=np.eye(3)[None], translations=np.zeros((1,3)))

    def draw_noise(self, *args):
        return self.noise

    def evaluate(self, *args):
        self.calls += 1
        return {"loss": 10., "metrics": {"affinity_loss": 2.}}


def test_bad_candidate_is_counted_without_refill_and_next_candidate_can_win(graph, evidence, monkeypatch, tmp_path):
    config, packet, options, prepared = setup_case(graph, evidence)
    backend, calls = Backend(), []

    def rescore(prepared, candidate, **kwargs):
        calls.append(candidate.candidate_id)
        if candidate is options[0]:
            raise LigandPreparationError("no native conformer", stage="native_conformer")
        return {"loss": 7., "metrics": {"affinity_loss": 1.}}

    monkeypatch.setattr("molcraft.api.evaluate_candidate", rescore)
    kwargs = dict(backend=backend, selected_ids=[c.candidate_id for c in options],
                  export_dir=tmp_path, context_id="same-context")
    result = evaluate_packet(prepared, packet, config, **kwargs)
    assert calls == [c.candidate_id for c in options]
    assert result["selected_id"] == options[1].candidate_id
    assert result["evaluation"]["attempted_candidates"] == 2
    assert result["evaluation"]["evaluated_candidates"] == result["evaluation"]["rejected_candidates"] == 1
    assert result["unscored_candidates"][0]["stage"] == "native_conformer"
    assert result["unscored_candidates"][0]["smiles"] == options[0].graph.smiles
    assert "loss" not in result["unscored_candidates"][0]
    calls.clear()
    resumed = evaluate_packet(prepared, packet, config, **kwargs)
    assert calls == [] and backend.calls == 1
    assert resumed["selected_id"] == result["selected_id"]
    assert resumed["evaluation"]["reused_candidates"] == 2
    with pytest.raises(StaleProposalError):
        evaluate_packet(prepared, packet, config, **{**kwargs, "selected_ids":list(reversed(kwargs["selected_ids"]))})
    backend.noise.init[0,0] = 1.
    with pytest.raises(StaleProposalError, match="noise"):
        evaluate_packet(prepared, packet, config, **kwargs)


def test_all_candidates_fail_retains_source_and_true_mapping_fault_still_stops(graph, evidence, monkeypatch, tmp_path):
    config, packet, options, prepared = setup_case(graph, evidence)

    def failed(*args, **kwargs):
        raise LigandPreparationError("ETKDG failed", stage="reference_conformer")

    monkeypatch.setattr("molcraft.api.evaluate_candidate", failed)
    result = evaluate_packet(prepared, packet, config, backend=Backend(),
        selected_ids=[c.candidate_id for c in options], context_id="same-context", export_dir=tmp_path)
    assert result["selected_id"] == "noop" and result["selected_loss"] == result["baseline"]["loss"]
    assert result["evaluation"]["attempted_candidates"] == result["evaluation"]["rejected_candidates"] == 2
    assert result["evaluation"]["evaluated_candidates"] == 0
    for error in (MappingError("lost atom identity"), ConfigurationError("bad input")):
        def fatal(*args, **kwargs):
            raise error
        monkeypatch.setattr("molcraft.api.evaluate_candidate", fatal)
        with pytest.raises(type(error)):
            evaluate_packet(prepared, packet, config, backend=Backend(),
                selected_ids=[c.candidate_id for c in options], context_id="same-context")


def test_progress_survives_unexpected_failure_without_rescoring_completed_candidate(graph, evidence, monkeypatch, tmp_path):
    config, packet, options, prepared = setup_case(graph, evidence)
    backend = Backend()
    calls = []

    def initial(prepared, candidate, **kwargs):
        calls.append(candidate.candidate_id)
        if candidate is options[1]:
            raise MappingError("mapping fault remains fatal")
        return {"loss": 8., "metrics": {"affinity_loss": 1.}}

    monkeypatch.setattr("molcraft.api.evaluate_candidate", initial)
    kwargs = dict(backend=backend, selected_ids=[c.candidate_id for c in options],
                  context_id="same-context", export_dir=tmp_path)
    with pytest.raises(MappingError):
        evaluate_packet(prepared, packet, config, **kwargs)
    progress = json.loads((tmp_path/"evaluation_progress.json").read_text())
    assert [c["candidate_id"] for c in progress["candidates"]] == [options[0].candidate_id]
    calls.clear()

    def continued(prepared, candidate, **kwargs):
        calls.append(candidate.candidate_id)
        return {"loss": 7., "metrics": {"affinity_loss": 1.}}

    monkeypatch.setattr("molcraft.api.evaluate_candidate", continued)
    result = evaluate_packet(prepared, packet, config, **kwargs)
    assert calls == [options[1].candidate_id] and backend.calls == 1
    assert result["selected_id"] == options[1].candidate_id


def test_baseline_failure_remains_fatal(graph, evidence, monkeypatch):
    config, packet, options, prepared = setup_case(graph, evidence)
    backend = Backend()

    def bad_baseline(*args):
        raise LigandPreparationError("source has no conformer", stage="reference_conformer")

    backend.evaluate = bad_baseline
    with pytest.raises(LigandPreparationError):
        evaluate_packet(prepared, packet, config, backend=backend,
            selected_ids=[c.candidate_id for c in options], context_id="same-context")
