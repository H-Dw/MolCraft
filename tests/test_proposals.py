import json
import numpy as np
import pytest

from molcraft.chemistry.graph import MolecularGraph
from molcraft.chemistry.edits import apply_edits
from molcraft.chemistry.vocabulary import ELEMENTS
from molcraft.guidance.proposals import ProposalPacket, ProposalPolicy, build_proposals
from molcraft.types import GradientEvidence
from molcraft.errors import ConfigurationError, GradientError, StaleProposalError


def test_packet_has_distinct_local_and_joint_distributions(graph, evidence):
    packet = build_proposals(graph, evidence)
    assert not packet.atoms.allowed[1, ELEMENTS.index("F")]
    assert packet.atoms.probabilities[1, ELEMENTS.index("F")] == 0
    assert packet.atoms.probabilities[1, ELEMENTS.index("N")] > packet.atoms.probabilities[1, ELEMENTS.index("O")]
    for table in (packet.atoms, packet.bonds, packet.charges):
        np.testing.assert_allclose(table.probabilities.sum(axis=1), 1)
    assert sum(c.probability for c in packet.candidates) == pytest.approx(1)
    assert sum(not c.operations for c in packet.candidates) == 1
    assert all(apply_edits(graph, c.operations).fingerprint == c.graph.fingerprint for c in packet.candidates)


def test_joint_generation_keeps_illegal_primitives_for_valid_bundles():
    graph = MolecularGraph.from_smiles("CC(C)(C)C")
    g = np.zeros((5, 9))
    g[1, ELEMENTS.index("N")] = -10
    evidence = GradientEvidence(graph.fingerprint, graph.atom_ids, g, np.zeros((10, 5)), np.zeros(5))
    packet = build_proposals(graph, evidence, ProposalPolicy(max_proposals=1000))
    assert not packet.atoms.allowed[1, ELEMENTS.index("N")]
    assert any(c.graph.smiles == "C[N+](C)(C)C" for c in packet.candidates)


def test_frozen_graph_has_only_noop_and_explicit_status(graph, evidence):
    packet = build_proposals(graph, evidence, ProposalPolicy(frozen_atom_ids=graph.atom_ids))
    assert packet.status == "no_legal_edit"
    assert len(packet.candidates) == 1
    assert packet.candidates[0].probability == 1


def test_packet_json_roundtrip_checks_identity_and_operations(graph, evidence):
    packet = build_proposals(graph, evidence)
    restored = ProposalPacket.from_dict(json.loads(json.dumps(packet.to_dict())))
    assert restored.to_dict() == packet.to_dict()
    data = packet.to_dict()
    data["source_fingerprint"] = "stale"
    with pytest.raises(StaleProposalError):
        ProposalPacket.from_dict(data)


def test_nonfinite_or_wrong_shape_evidence_is_not_silently_repaired(graph, evidence):
    evidence.atom_grad[0, 0] = np.nan
    with pytest.raises(GradientError):
        build_proposals(graph, evidence)
    evidence.atom_grad = np.zeros((1, 9))
    with pytest.raises(GradientError):
        build_proposals(graph, evidence)


def test_stale_evidence_is_rejected(graph, evidence):
    changed = MolecularGraph.from_smiles("CCO")
    with pytest.raises(StaleProposalError):
        build_proposals(changed, evidence)


def test_budget_is_finite_and_deterministic(graph, evidence):
    policy = ProposalPolicy(max_proposals=12, max_candidates=3)
    a = build_proposals(graph, evidence, policy)
    b = build_proposals(graph, evidence, policy)
    assert a.to_dict() == b.to_dict()
    assert len(a.candidates) <= 4
    assert a.diagnostics["checked_bundles"] <= 12


def test_invalid_policy_rejected():
    with pytest.raises(ConfigurationError):
        ProposalPolicy(max_bundle_size=3)


def test_joint_distribution_remains_finite_at_extreme_temperature(graph, evidence):
    packet = build_proposals(graph, evidence, ProposalPolicy(temperature=1e-320))
    probabilities = np.array([c.probability for c in packet.candidates])
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1)
