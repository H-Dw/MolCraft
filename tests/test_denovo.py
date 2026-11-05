"""De novo chemistry contracts; tiny synthetic inventory fixtures are NOT vendor data."""

from contextlib import contextmanager
from dataclasses import replace
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from rdkit import Chem

from molcraft.agent.denovo import DeNovoAgent
from molcraft.agent.denovo_policy import DeNovoPolicy
from molcraft.agent.history import DraftHistory
from molcraft.agent.optagent import compile_draft
from molcraft.api import PreparedCandidatePool
from molcraft.chemistry.edits import Edit, apply_edits
from molcraft.chemistry.graph import MolecularGraph
from molcraft.chemistry.inventory import InventoryIndex, build_index, chemical_context
from molcraft.chemistry.mapping import AtomLayout
from molcraft.errors import ChemistryError, ConfigurationError, LigandPreparationError, StaleProposalError
from molcraft.guidance.proposals import ProposalPolicy, direction_tables
from molcraft.io.config import load_config
from molcraft.io.examples import prepare_example
from molcraft.losses.functions import evaluate_objective, confidence_loss
from molcraft.losses.spec import ObjectiveSpec
from molcraft.types import GradientEvidence


def gradients(graph):
    return GradientEvidence(graph.fingerprint, graph.atom_ids, np.zeros((len(graph.atom_ids), 9)),
                            np.zeros((len(graph.pairs), 5)), np.zeros(len(graph.atom_ids)))


@pytest.fixture
def inventory(tmp_path):
    stock = tmp_path / "synthetic_test_only.smi"
    stock.write_text("CCO TEST-1\nOCC TEST-DUP\nCC(=O)N TEST-2\nc1ccncc1 TEST-3\ninvalid TEST-BAD\n")
    templates = tmp_path / "templates.json"
    templates.write_text(json.dumps(["[C:1][O:2]>>[C:1].[O:2]", "bad"]))
    path = tmp_path / "index.sqlite"
    build_index(stock, path, release="SYNTHETIC_TEST_FIXTURE_NOT_ENAMINE_STOCK",
                source_url="https://example.invalid/test-fixture", templates=templates,
                template_source_url="https://example.invalid/template-fixture")
    index = InventoryIndex(path.with_suffix(".sqlite.manifest.json"))
    yield index
    index.close()


def evidence_ids(index):
    return [r[0] for r in index.db.execute("SELECT id FROM motifs")]


def draft(*ops):
    return {"operations": [op.to_dict() for op in ops], "rationale": "Synthetic test", "evidence_ids": []}


class Client:
    def __init__(self, response):
        self.response, self.messages = response, []

    def complete(self, messages):
        self.messages.append(messages)
        return self.response, {"prompt_tokens": 100, "completion_tokens": 20}


def propose(agent, graph, evidence=None, **kwargs):
    defaults = dict(samples=4, round_index=1, context_id="test", candidate_budget=8,
                    prepare_candidate=lambda graph: None)
    return agent.propose(graph, evidence or gradients(graph), ProposalPolicy(), **{**defaults, **kwargs})


def test_inventory_counts_traceability_and_tamper_detection(inventory, tmp_path):
    assert inventory.manifest["counts"] == dict(raw=5, valid=4, unique=3, invalid=1, duplicate=1,
        raw_templates=2, templates=1, invalid_templates=1, small_templates=0, unique_templates=1)
    assert inventory.db.execute("SELECT count(*) FROM stock_aliases").fetchone()[0] == 4
    assert inventory.manifest["route_validation"] == "not_performed"
    assert inventory.manifest["template_frequency_filter"] == "not_verified"
    card = next(inventory.get(i) for i in evidence_ids(inventory) if i.startswith("inventory"))
    source = card["example"]
    row = inventory.db.execute("SELECT vendor_id,smiles FROM stocks WHERE record_id=?", (source["stock_record"],)).fetchone()
    assert tuple(row) == (source["vendor_id"], source["stock_smiles"])
    manifest = tmp_path / "index.sqlite.manifest.json"
    data = json.loads(manifest.read_text())
    data["release"] = "FALSE RELEASE"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError, match="embedded provenance"):
        InventoryIndex(manifest)
    data["index_sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError, match="hash mismatch"):
        InventoryIndex(manifest)


def test_missing_inventory_and_case_json_fail_explicitly(tmp_path):
    with pytest.raises(ConfigurationError, match="not ready"):
        InventoryIndex(tmp_path / "missing.json")
    cases = tmp_path / "reasoning.json"
    cases.write_text('[{"source":"CCO"}]')
    with pytest.raises(ConfigurationError, match="never case JSON"):
        build_index(cases, tmp_path / "bad.sqlite", release="case", source_url="https://example.invalid")
    assert not (tmp_path / "bad.sqlite").exists()


@pytest.mark.parametrize("smiles,expected", [("CCO", True), ("CC[O-]", False), ("COC", False), ("CC=O", False)])
def test_inventory_match_checks_charge_hydrogen_degree_and_bond_environment(inventory, smiles, expected):
    source, product = MolecularGraph.from_smiles("CCC"), MolecularGraph.from_smiles(smiles)
    result = inventory.assess(source, product, [Edit("atom", (3,), "O")], evidence_ids(inventory))
    assert result["catalog_preferred"] is expected
    assert all(hit["route_validated"] is False for hit in result["matches"])


def test_template_alone_is_not_stock_or_route_evidence(inventory):
    graph = MolecularGraph.from_smiles("CCO")
    ids = [i for i in evidence_ids(inventory) if i.startswith("reaction")]
    result = inventory.assess(graph, graph, [Edit("atom", (3,), "O")], ids)
    assert result["matches"] and not result["catalog_preferred"]
    assert result["matches"][0]["source"]["stock_reactants_verified"] is False


@pytest.mark.parametrize("smiles", ["C[C@H](O)Cl", "CC[C@H](O)CCC"])
def test_aromaticity_and_stereochemical_attachment_environment_are_not_interchangeable(inventory, tmp_path, smiles):
    aromatic = MolecularGraph.from_smiles("c1ccncc1")
    ids = evidence_ids(inventory)
    edit = [Edit("atom", (4,), "N")]
    assert inventory.assess(aromatic, aromatic, edit, ids)["catalog_preferred"]
    saturated = MolecularGraph.from_smiles("C1CCNCC1")
    assert not inventory.assess(aromatic, saturated, edit, ids)["catalog_preferred"]
    stock = tmp_path / "stereo.smi"
    stock.write_text(smiles + " TEST-STEREO\n")
    build_index(stock, tmp_path / "stereo.sqlite", release="SYNTHETIC_TEST_ONLY", source_url="https://example.invalid")
    index = InventoryIndex(tmp_path / "stereo.sqlite.manifest.json")
    try:
        intended = MolecularGraph.from_smiles(smiles)
        inverse = MolecularGraph.from_smiles(smiles.replace("@", "@@"))
        oxygen = next(i for i, e, _ in intended.atoms if e == "O")
        edit = [Edit("atom", (oxygen,), "O")]
        assert index.assess(intended, intended, edit, evidence_ids(index))["catalog_preferred"]
        assert not index.assess(intended, inverse, edit, evidence_ids(index))["catalog_preferred"]
    finally:
        index.close()


def test_dynamic_window_closes_large_ring_and_conjugation():
    mol = Chem.MolFromSmiles("C1CCCCCCCCCCCC1")
    core, context = chemical_context(mol, {0})
    assert len(core) == len(context) == 13
    conjugated = Chem.MolFromSmiles("CC=CC=CC=O")
    core, _ = chemical_context(conjugated, {6})
    assert {1, 2, 3, 4, 5, 6} <= core


def test_joint_invalid_singles_remain_visible_and_compile_atomically():
    graph = MolecularGraph.from_smiles("CC(C)(C)C")
    evidence = gradients(graph)
    ops = (Edit("atom", (2,), "N"), Edit("charge", (2,), 1))
    for op in ops:
        with pytest.raises(ChemistryError):
            apply_edits(graph, [op])
    tables, _, _ = direction_tables(graph, evidence, ProposalPolicy(), joint=True)
    assert tables["atom"].allowed[1, 1] and tables["atom"].probabilities[1, 1] > 0
    payload = {"operations": [op.to_dict() for op in ops], "rationale": "", "case_ids": []}
    compiled = compile_draft(graph, evidence, payload, set(graph.atom_ids), set(), set(),
                             max_operations=None, max_touched=None)
    assert compiled[1].smiles == "C[N+](C)(C)C"
    assert graph.smiles == "CC(C)(C)C"


def test_no_fixed_operation_or_atom_count_chemical_gate():
    graph = MolecularGraph.from_smiles("CCCCCCCCC")
    payload = {"operations": [Edit("atom", (i,), "N").to_dict() for i in graph.atom_ids],
               "rationale": "", "case_ids": []}
    updated = compile_draft(graph, gradients(graph), payload, set(graph.atom_ids), set(), set(),
                            max_operations=None, max_touched=None)[1]
    assert updated.smiles == "NNNNNNNNN"


def test_empty_responses_and_pool_shortfall_are_bounded(inventory, tmp_path):
    client = Client({})
    agent = DeNovoAgent(client, DeNovoPolicy(max_batches=2, fallback_attempts=1), inventory)
    graph = MolecularGraph.from_smiles("CCC")
    packet = propose(agent, graph, journal_dir=tmp_path / "round" / "requests")
    assert len(client.messages) == 4
    assert packet.diagnostics["pool"]["shortfall"] > 0
    assert packet.diagnostics["funnel"]["fallback_attempts"] == 1
    assert len(agent.select(packet, 8)) <= 8
    assert (tmp_path / "round" / "draft_feedback.md").exists()
    replay = propose(agent, graph, journal_dir=tmp_path / "round" / "requests")
    assert len(client.messages) == 4 and replay.to_dict() == packet.to_dict()


@pytest.mark.parametrize("response", [None, [], {"usage": None, "choices": None},
    {"usage": {"prompt_tokens": 3, "completion_tokens_details": None},
     "choices": [{"finish_reason": "length", "message": {"content": None}}]}])
def test_provider_null_fields_are_counted_errors_instead_of_attribute_errors(monkeypatch, response):
    from molcraft.agent.deepseek import DeepSeekClient, ProviderError
    client = DeepSeekClient()
    monkeypatch.setattr(client, "_request", lambda *a, **k: response)
    with pytest.raises(ProviderError) as error:
        client.complete([])
    assert error.value.retryable
    if isinstance(response, dict) and response.get("usage"):
        assert error.value.usage["prompt_tokens"] == 3


def test_database_preference_is_not_a_gradient_rank_or_exclusion_gate(inventory):
    graph = MolecularGraph.from_smiles("CCC")
    evidence = gradients(graph)
    evidence.atom_grad[1, 1] = -4  # central C -> N is strongest but unsupported
    evidence.atom_grad[2, 2] = -1  # terminal alcohol has inventory evidence
    client = Client({"drafts": [draft(Edit("atom", (2,), "N")), draft(Edit("atom", (3,), "O"))]})
    packet = propose(DeNovoAgent(client, DeNovoPolicy(max_batches=1, fallback_attempts=1), inventory), graph, evidence)
    ids = DeNovoAgent.select(packet, 8)
    assert packet.candidate(ids[0]).graph.smiles == "CCO"
    central = next(c for c in packet.candidates if c.graph.smiles == "CNC")
    assert central.candidate_id in ids and central.probability > packet.candidate(ids[0]).probability


def test_valid_unmatched_candidates_native_failure_duplicates_and_feedback(inventory, tmp_path):
    graph = MolecularGraph.from_smiles("CCC")
    client = Client({"drafts": [draft(Edit("atom", (2,), "N")), draft(Edit("atom", (3,), "O")),
                                draft(Edit("atom", (2,), "N"))]})
    native_calls = []

    def native(product):
        native_calls.append(product.smiles)
        if product.smiles == "CCO":
            raise LigandPreparationError("Synthetic ETKDG failure", stage="native_conformer")

    agent = DeNovoAgent(client, DeNovoPolicy(max_batches=2, fallback_attempts=1), inventory)
    packet = propose(agent, graph, prepare_candidate=native, journal_dir=tmp_path / "requests")
    options = [c for c in packet.candidates if c.graph.smiles == "CNC"]
    assert len(options) == 1
    support = packet.diagnostics["agent_explanations"][options[0].candidate_id]["database_support"]
    assert not support["catalog_preferred"]
    assert "CCO" not in [c.graph.smiles for c in packet.candidates]
    assert native_calls.count("CCO") == 1
    assert len(client.messages) == 2  # Repeated drafts do not consume extra API retries.
    second = json.loads(client.messages[1][-1]["content"])
    assert second["same_round_feedback"] and "design" not in str(second["same_round_feedback"])
    assert "case_ids" not in str(second["evidence_cards"])
    assert packet.diagnostics["funnel"]["duplicate_combinations"] >= 3


def test_fallback_builds_pool_and_selection_is_separate_from_gradient_probability(inventory):
    graph = MolecularGraph.from_smiles("CCCCCC")
    packet = propose(DeNovoAgent(None, inventory=inventory), graph)
    assert packet.diagnostics["pool"]["native_ready"] == 16
    assert len(DeNovoAgent.select(packet, 8)) == 8
    assert sum(c.probability for c in packet.candidates) == pytest.approx(1)
    assert len({c.graph.smiles for c in packet.candidates}) == len(packet.candidates)
    assert all(c.predicted_change == 0 for c in packet.candidates)
    assert len({round(c.probability, 7) for c in packet.candidates}) == 1  # No rank-softmax.
    assert all(r["origin"] == "program_fallback" for r in packet.diagnostics["agent_explanations"].values())


def test_measured_history_is_not_pending_and_unmeasured_can_carry(inventory, tmp_path):
    graph = MolecularGraph.from_smiles("CCCCCC")
    agent = DeNovoAgent(None, inventory=inventory)
    packet = propose(agent, graph)
    selected = agent.select(packet, 8)
    history = DraftHistory(tmp_path, "same-protocol")
    history.record_round(1, graph, packet, {"selected_id": "noop", "candidates": [
        {"candidate_id": c, "loss": 1., "eligible": False} for c in selected]})
    old = history.before(2)
    measured_smiles = {packet.candidate(cid).graph.smiles for cid in selected}
    assert not any(r.get("outcome") == "pending" and r.get("smiles") in measured_smiles for r in old)
    second = propose(agent, graph, round_index=2, history=old, excluded=measured_smiles)
    reasons = second.diagnostics["agent_explanations"].values()
    assert any(r["origin"] == "carried_unmeasured" for r in reasons)
    assert not measured_smiles.intersection(c.graph.smiles for c in second.candidates)
    with pytest.raises(StaleProposalError):
        DraftHistory(tmp_path, "different-protocol")


def test_native_preparation_is_reused_and_closed_with_source_identity(graph):
    calls, closed = [], []
    from molcraft.types import ComplexSpec, ProteinSpec, ResidueRef
    spec = ComplexSpec((ProteinSpec("A", "ACD"),), graph.smiles, (ResidueRef("A", 1),))
    source = SimpleNamespace(graph=graph, spec=spec)

    @contextmanager
    def prepare(spec):
        calls.append(spec.ligand_smiles)
        try:
            yield SimpleNamespace(graph=MolecularGraph.from_smiles(spec.ligand_smiles))
        finally:
            closed.append(spec.ligand_smiles)

    product = apply_edits(graph, [Edit("atom", (3,), "O")])
    with PreparedCandidatePool(source, prepare) as pool:
        first = pool.prepare_graph(product)
        assert pool.prepare_graph(product) is first and pool.get(source, product) is first
        assert first.graph is product and len(calls) == 1 and not closed
        with pytest.raises(StaleProposalError):
            pool.get(SimpleNamespace(graph=product, spec=spec), product)
    assert len(closed) == 1 and not pool.entries


@pytest.mark.parametrize("case,rounds", [("gba-2v3d", 17), ("vdr-3a2i", 16)])
def test_profiles_objective_version_and_exact_budget(tmp_path, case, rounds):
    folder = tmp_path / case
    prepare_example(case, folder, tmp_path, work_root=tmp_path, msa_mode="single_sequence", strategy="denovo")
    cfg = load_config(folder / "mosaic-like.yaml")
    assert cfg.objective == ObjectiveSpec.denovo()
    assert cfg.optimization.min_rounds == cfg.optimization.max_rounds == rounds
    assert cfg.evaluation.candidate_budget == 8 and cfg.evaluation.coverage_mode == "off"
    assert not cfg.evaluation.require_affinity_nonworsening
    assert cfg.denovo.target_pool == 16 and cfg.denovo.batch_size * cfg.denovo.max_batches == 32
    with pytest.raises(ConfigurationError):
        replace(cfg, objective=ObjectiveSpec())
    with pytest.raises(ConfigurationError):
        replace(cfg, evaluation=replace(cfg.evaluation, coverage_mode="strict"))


def test_dbmol_formula_gradient_sum_finite_difference_and_real_geometry_at_zero_weight():
    layout = AtomLayout((1, 2), (1, 3), (1, 3), (0, 2), (0,), ((0, 2),), 4, 4, (1.7, 1.7))
    inputs = SimpleNamespace(atoms=jnp.zeros((2, 9)).at[:, 0].set(1))
    confidence = dict(ligand_iptm=.8, complex_iplddt=.7, complex_ipde=2., complex_pde=3.,
                      confidence_score=.6, ptm=.5)
    assert float(confidence_loss(confidence)) == pytest.approx(-.8-.21+.6+.3-.18-.05)

    def objective(x, weights):
        output = SimpleNamespace(affinity={"affinity_pred_value": x, "affinity_logits_binary": 2*x},
            confidence={k: v*x for k, v in confidence.items()}, distogram_bins=jnp.array([4., 12.]),
            distogram_logits=jnp.broadcast_to(jnp.array([x, 0.]), (4, 4, 2)),
            coordinates=jnp.array([[0.,0.,0.], [8.,0.,0.], [0.,1.,0.], [9.,0.,0.]]) * x)
        return evaluate_objective(output, inputs, layout, weights)

    spec = ObjectiveSpec.denovo()
    value, metrics = objective(1., spec)
    assert float(metrics["geometry_loss"]) > 0 and float(metrics["weighted_geometry"]) == 0
    assert float(value) == pytest.approx(sum(float(metrics["weighted_"+n]) for n in ("affinity", "contact", "confidence")))
    total_grad = jax.grad(lambda x: objective(x, spec)[0])(1.)
    names = ("affinity", "contact", "confidence", "geometry")
    parts = [jax.grad(lambda x: objective(x, replace(spec, **{n+"_weight": float(n == name) for n in names}))[0])(1.)
             for name in names]
    np.testing.assert_allclose(total_grad, parts[0] + 2.5*parts[1] + 2.5*parts[2], rtol=1e-5)
    finite = (objective(1.001, spec)[0] - objective(.999, spec)[0]) / .002
    np.testing.assert_allclose(total_grad, finite, rtol=1e-3, atol=1e-3)


def test_two_round_resume_reuses_native_inputs_carries_pool_and_separately_archives_success(inventory, tmp_path, monkeypatch):
    from molcraft.io.config import RunConfig, EvaluationSpec, OptimizationSpec
    from molcraft.optimization import optimize, _input_identity
    from molcraft.sampling.core import Noise
    from molcraft.types import ComplexSpec, ProteinSpec, ResidueRef, PreparedComplex

    spec = ComplexSpec((ProteinSpec("A", "ACD"),), "CCCCCC", (ResidueRef("A", 1),))
    policy = DeNovoPolicy(inventory_manifest=str(tmp_path / "index.sqlite.manifest.json"))
    cfg = RunConfig(spec, ObjectiveSpec.denovo(), ProposalPolicy(), msa_mode="single_sequence", denovo=policy,
        evaluation=EvaluationSpec(candidate_budget=8),
        optimization=OptimizationSpec(max_rounds=2, min_rounds=2, patience=5, strategy="denovo"))
    attempts, preparations = [], []

    def prepare(spec):
        graph = MolecularGraph.from_smiles(spec.ligand_smiles)
        preparations.append(graph.smiles)
        layout = AtomLayout(graph.atom_ids, tuple(range(1, 7)), tuple(range(1, 7)), (0,), (0,), ((0,),), 7, 7)
        return PreparedComplex(spec, graph, {"msa": np.zeros((1, 1, 7, 33))}, layout)

    class Backend:
        sampling_steps = 25
        trunk_passes = 1

        def draw_noise(self, *args):
            return Noise(jnp.zeros((7, 3)), jnp.zeros((1, 7, 3)), jnp.eye(3)[None], jnp.zeros((1, 3)))

        def value_and_grad(self, prepared, *args, **kwargs):
            return gradients(prepared.graph)

        def evaluate(self, *args):
            return {"loss": 1., "metrics": {"affinity_loss": 1., "bcov": .3, "affinity_pred_value": 1.}}

    def rescore(source, candidate, preparation_cache, **kwargs):
        prepared = preparation_cache.get(source, candidate.graph)
        assert prepared.graph is candidate.graph
        assert preparations.count(candidate.graph.smiles) == (2 if candidate.graph.smiles in carried else 1)
        attempts.append(candidate.graph.smiles)
        if len(attempts) == 1:
            from molcraft.errors import InvalidPredictionError
            raise InvalidPredictionError("Synthetic nonfinite prediction")
        return {"loss": 2., "metrics": {"affinity_loss": 1., "bcov": .9, "affinity_pred_value": -.2}}

    monkeypatch.setattr("molcraft.api.evaluate_candidate", rescore)
    agent, backend, carried = DeNovoAgent(None, policy, inventory), Backend(), set()
    output = tmp_path / "run"
    args = dict(backend=backend, evaluation_backend=backend, prepare=prepare)
    first = optimize(cfg, agent, output, stop_after=1, **args)
    assert first["status"] == "window_complete" and first["rounds"][0]["attempted_candidates"] == 8
    assert first["rounds"][0]["rejected_candidates"] == 1
    assert first["best_successful"]["metrics"]["bcov"] == .9  # archive survives coverage_mode=off
    records = DraftHistory(output, first["context_id"]).before(2)
    carried = {r["smiles"] for r in records if r.get("outcome") == "not_rescored"}
    final = optimize(cfg, agent, output, resume=True, **args)
    assert final["status"] == "budget_complete" and len(attempts) == len(set(attempts)) == 16
    assert len(final["rounds"]) == 2
    next_packet = json.loads((output / "round-002/proposals.json").read_text())
    assert any(r["origin"] == "carried_unmeasured" for r in next_packet["diagnostics"]["agent_explanations"].values())
    changed = replace(cfg, policy=replace(cfg.policy, temperature=.5))
    assert _input_identity(cfg)[0] != _input_identity(changed)[0]
    with pytest.raises(StaleProposalError):
        optimize(changed, agent, output, resume=True, **args)
