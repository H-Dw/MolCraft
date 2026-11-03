import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from molcraft.agent.deepseek import ProviderError, RequestJournal
from molcraft.agent.optagent import OptAgent, _regions, compile_draft
from molcraft.chemistry.graph import MolecularGraph
from molcraft.chemistry.mapping import AtomLayout
from molcraft.errors import ChemistryError, ConfigurationError, StaleProposalError
from molcraft.guidance.proposals import ProposalPolicy, ProposalPacket
from molcraft.io.config import EvaluationSpec, OptimizationSpec, RunConfig
from molcraft.io.examples import case_data, normalized_a3m, prepare_example
from molcraft.losses.spec import ObjectiveSpec
from molcraft.optimization import optimize
from molcraft.sampling.core import Noise
from molcraft.types import ComplexSpec, GradientEvidence, PreparedComplex, ProteinSpec, ResidueRef


def evidence(graph):
    n = len(graph.atom_ids)
    return GradientEvidence(graph.fingerprint, graph.atom_ids, np.zeros((n, 9)),
                            np.zeros((len(graph.pairs), 5)), np.zeros(n))


def test_examples_and_exact_a3m(tmp_path):
    for case_id, size, count in (("gba-2v3d", 497, 9), ("vdr-3a2i", 259, 7)):
        case = case_data(case_id)
        assert len(case["sequence"]) == size
        assert [row[0] for row in case["mapping"]] == list(range(1, size + 1))
        assert len(case["pocket"]) == count
    vdr = case_data("vdr-3a2i")
    result = prepare_example("vdr-3a2i", tmp_path / "vdr", tmp_path, work_root=tmp_path)
    request = json.loads((tmp_path / "vdr/af3_protein_request.json").read_text())
    assert request[0]["sequences"][0]["proteinChain"]["sequence"] == vdr["sequence"]
    assert result["status"] == "awaiting_af3_msa"
    assert not (tmp_path / "vdr/test.yaml").exists()
    data = b">query\nACD\n>dup\nAaaCD\n>homolog\nAaa-D\n>duplicate\nA-D\n"
    normalized, raw, used = normalized_a3m(data, "ACD")
    assert (raw, used) == (4, 2)
    assert b"Aaa-D" in normalized
    with pytest.raises(ConfigurationError, match="query"):
        normalized_a3m(data, "ACC")
    with pytest.raises(ConfigurationError, match="width"):
        normalized_a3m(data + b">wrong\nAC\n", "ACD")


def test_cycle_uses_actual_bond_gradient_and_respects_freeze():
    graph = MolecularGraph.from_smiles("CCCCCCCC")
    grad = evidence(graph)
    grad.bond_grad[graph.pairs.index((1, 6)), 1] = -2
    region, _ = _regions(graph, set())[4]
    draft = {"operations": [{"kind": "bond", "atom_ids": [6, 1], "value": "SINGLE"}],
             "rationale": "Close a six-membered ring.", "case_ids": []}
    _, updated, change, _ = compile_draft(graph, grad, draft, region, set(), set())
    assert change == -2
    molecule = updated.to_rdkit()
    assert 6 in [len(r) for r in molecule.GetRingInfo().AtomRings()]
    assert updated.atom_ids == graph.atom_ids
    with pytest.raises(ChemistryError, match="frozen"):
        compile_draft(graph, grad, draft, region, {6}, set())


class Cases:
    def retrieve(self, _graph):
        return [{"id": "case-1"}]


class Client:
    def __init__(self):
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        context = json.loads(messages[1]["content"])
        if isinstance(context, list):
            return {"candidate_ids": [context[0]["id"]]}, {"total_tokens": 5}
        return {"operations": [{"kind": "atom", "atom_ids": [2], "value": "N"}],
                "rationale": "Replace the central atom.", "case_ids": ["case-1"]}, {}


def test_agent_packet_and_id_replay(tmp_path):
    graph = MolecularGraph.from_smiles("CCC")
    client = Client()
    agent = OptAgent(client, Cases())
    kwargs = dict(samples=2, round_index=1, context_id="fixed", journal_dir=tmp_path)
    packet = agent.propose(graph, evidence(graph), ProposalPolicy(), **kwargs)
    assert len(packet.candidates) == 2
    assert sum(c.probability for c in packet.candidates) == pytest.approx(1)
    assert ProposalPacket.from_dict(packet.to_dict()).candidate("agent-01").graph.smiles == "CNC"
    assert agent.select(packet, 2, journal_dir=tmp_path) == ["agent-01"]
    count = client.calls
    assert agent.propose(graph, evidence(graph), ProposalPolicy(), **kwargs).to_dict() == packet.to_dict()
    assert agent.select(packet, 2, journal_dir=tmp_path) == ["agent-01"]
    assert client.calls == count


def test_provider_journal_does_not_reissue_reserved_slots(tmp_path):
    class Unavailable:
        def __init__(self):
            self.calls = 0

        def complete(self, _):
            self.calls += 1
            raise ProviderError("timeout", retryable=True)

    (tmp_path / "draft-1.json").write_text('{"status":"reserved"}', encoding="utf-8")
    service = Unavailable()
    for _ in range(2):
        with pytest.raises(ProviderError, match="exhausted"):
            RequestJournal(tmp_path).request("draft", service, [], lambda result: result)
    assert service.calls == 1


def test_resume_and_cross_round_noise(tmp_path):
    config = RunConfig(ComplexSpec((ProteinSpec("A", "ACD"),), "CCC", (ResidueRef("A", 1),)),
        ObjectiveSpec(), ProposalPolicy(), evaluation=EvaluationSpec(candidate_budget=1),
        optimization=OptimizationSpec(max_rounds=2, min_rounds=2, patience=1, drafts=1))
    mapped_noise = []

    def prepare(spec):
        graph = MolecularGraph.from_smiles(spec.ligand_smiles)
        indices = (3, 1, 2) if graph.smiles == "CNC" else (1, 2, 3)
        layout = AtomLayout(graph.atom_ids, indices, (1, 2, 3), (0,), (0,), ((0,),), 4, 4)
        return PreparedComplex(spec, graph, {}, layout)

    class Backend:
        def draw_noise(self, prepared, seed):
            return Noise(jnp.arange(12).reshape(4, 3), jnp.zeros((1, 4, 3)),
                         jnp.eye(3)[None], jnp.zeros((1, 3)))

        def value_and_grad(self, prepared, *args, noise, **kwargs):
            mapped_noise.append(np.asarray(noise.init)[list(prepared.layout.ligand_atom_indices)])
            return evidence(prepared.graph)

    def score(prepared, packet, _config, *, selected_ids, **kwargs):
        cid = selected_ids[0] if selected_ids else "noop"
        return {"selected_id": cid, "selected_graph": packet.candidate(cid).graph.to_dict(),
                "status": "improved" if selected_ids else "source_retained",
                "selected_loss": 1., "baseline": {"loss": 2.},
                "candidates": [{"candidate_id": i} for i in selected_ids]}

    model = Backend()
    kwargs = dict(backend=model, evaluation_backend=model, prepare=prepare, score=score)
    agent = OptAgent(Client(), Cases())
    first = optimize(config, agent, tmp_path, stop_after=1, **kwargs)
    assert first["status"] == "window_complete"
    assert first["rounds"][0]["evaluated_candidates"] == 1
    final = optimize(config, agent, tmp_path, resume=True, **kwargs)
    assert final["status"] == "converged"
    assert len(mapped_noise) == 2
    assert np.array_equal(*mapped_noise)
    with pytest.raises(StaleProposalError):
        optimize(RunConfig(config.complex, config.objective, config.policy, seed=42),
                 agent, Path(tmp_path), resume=True, **kwargs)
