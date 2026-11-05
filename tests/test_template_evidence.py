"""Template-only inference and optional stock enrichment, without vendor fixtures."""

from dataclasses import replace
import hashlib
import json

import pytest

from molcraft.agent.denovo import DeNovoAgent
from molcraft.agent.denovo_policy import DeNovoPolicy
from molcraft.chemistry.edits import Edit, apply_edits
from molcraft.chemistry.graph import MolecularGraph
from molcraft.chemistry.inventory import EvidenceIndex, InventoryIndex, build_index
from molcraft.cli import main
from molcraft.errors import ConfigurationError, LigandPreparationError
from molcraft.guidance.proposals import ProposalPolicy, direction_tables
from molcraft.io.config import RunConfig, EvaluationSpec, OptimizationSpec, load_config
from molcraft.io.examples import prepare_example
from molcraft.losses.spec import ObjectiveSpec
from molcraft.optimization import _input_identity
from molcraft.types import ComplexSpec, ProteinSpec, ResidueRef

from test_denovo import Client, draft, gradients, propose, evidence_ids


@pytest.fixture
def template_index(tmp_path):
    path = tmp_path / "templates.json"
    path.write_text(json.dumps([
        "[#6:1]-[#8;H1;+0:2]>>[#6:1].[#8:2]",
        "[#6:1](=[#8:2])-[#7:3]>>[#6:1](=[#8:2]).[#7:3]",
        "bad",
    ]))
    build_index(None, tmp_path / "templates.sqlite", release="SYNTHETIC_TEMPLATE_TEST_ONLY",
                source_url="https://example.invalid/templates", templates=path,
                template_source_url="https://example.invalid/templates")
    index = EvidenceIndex(tmp_path / "templates.sqlite.manifest.json")
    yield index
    index.close()


def policy_for(tmp_path, **kwargs):
    return DeNovoPolicy(evidence_manifest=str(tmp_path / "templates.sqlite.manifest.json"), **kwargs)


def test_template_only_index_and_prompt_never_load_enamine(template_index, tmp_path, monkeypatch):
    import molcraft.chemistry.inventory as module

    def forbidden(*args, **kwargs):
        pytest.fail("Template-only inference must not load or require stock")

    monkeypatch.setattr(module, "InventoryIndex", forbidden)
    monkeypatch.setattr(module, "_stock_records", forbidden)
    graph = MolecularGraph.from_smiles("CCC")
    client = Client({"drafts": [draft(Edit("atom", (3,), "O"))]})
    agent = DeNovoAgent(client, policy_for(tmp_path, max_batches=1, fallback_attempts=1))
    try:
        packet = propose(agent, graph)
        metadata = packet.diagnostics["chemical_evidence"]
        assert metadata["primary"]["provider"] == "USPTO/SimpRetro"
        assert metadata["primary"]["resource_mode"] == "templates_only"
        assert metadata["primary"]["stock_sha256"] is None
        assert metadata["primary"]["counts"]["raw"] == 0
        assert not metadata["stock_loaded"] and metadata["inventory_enhancement"] is None
        context = json.loads(client.messages[0][-1]["content"])
        assert context["evidence_cards"]
        assert all(c["kind"] == "reaction_template_support" for c in context["evidence_cards"])
        assert all(c["example"]["stock_reactants_verified"] is False for c in context["evidence_cards"])
        selected = packet.candidate(agent.select(packet, 1)[0])
        support = packet.diagnostics["agent_explanations"][selected.candidate_id]["database_support"]
        assert selected.graph.smiles == "CCO"
        assert support["template_preferred"] and not support["catalog_preferred"]
        assert support["preference"] == "reaction_product_pattern"
    finally:
        agent.evidence.close()


def test_new_product_pattern_softly_precedes_unmatched_but_keeps_other_legal_edits(template_index, tmp_path):
    graph = MolecularGraph.from_smiles("CCC")
    evidence = gradients(graph)
    evidence.atom_grad[1, 1] = -4
    evidence.atom_grad[2, 2] = -1
    client = Client({"drafts": [draft(Edit("atom", (2,), "N")), draft(Edit("atom", (3,), "O"))]})
    agent = DeNovoAgent(client, policy_for(tmp_path, max_batches=1, fallback_attempts=1), template_index)
    packet = propose(agent, graph, evidence)
    selected = agent.select(packet, 8)
    assert packet.candidate(selected[0]).graph.smiles == "CCO"
    unmatched = next(c for c in packet.candidates if c.graph.smiles == "CNC")
    assert unmatched.candidate_id in selected
    assert unmatched.probability > packet.candidate(selected[0]).probability


@pytest.mark.parametrize("smiles,preferred", [("CCO", True), ("CC[O-]", False), ("COC", False), ("CC=O", False)])
def test_product_pattern_constraints_are_checked_in_product(template_index, smiles, preferred):
    source = MolecularGraph.from_smiles("CCC")
    product = MolecularGraph.from_smiles(smiles)
    assessment = template_index.assess(source, product, [Edit("atom", (3,), "O")], evidence_ids(template_index))
    assert assessment["template_preferred"] is preferred
    assert not assessment["catalog_preferred"]
    assert assessment["route_validation"] == "not_performed"


def test_atomic_bundle_must_form_one_complete_template_pattern(template_index):
    source = MolecularGraph.from_smiles("CCC")
    operations = [Edit("atom", (1,), "O"), Edit("atom", (3,), "N"), Edit("bond", (1, 2), "DOUBLE")]
    product = apply_edits(source, operations)
    assessment = template_index.assess(source, product, operations, evidence_ids(template_index))
    assert product.smiles == "NC=O" and assessment["template_preferred"]
    # A remote edit outside the matching motif prevents promoting the whole bundle.
    source = MolecularGraph.from_smiles("CCCCCC")
    operations = [Edit("atom", (1,), "N"), Edit("atom", (6,), "O")]
    assessment = template_index.assess(source, apply_edits(source, operations), operations, evidence_ids(template_index))
    assert assessment["matches"] and not assessment["template_preferred"]


@pytest.mark.parametrize("pattern,smiles,operations", [
    ("[#6:1]-[*:2]>>[#6:1].[*:2]", "CCC", [Edit("atom", (3,), "O")]),
    ("[#6:1]-[#8:2]>>[#6:1].[#8:2]", "CCC", [Edit("atom", (3,), "O"), Edit("charge", (3,), -1)]),
    ("[#6:1]~[#8:2]>>[#6:1].[#8:2]", "CCO", [Edit("bond", (2, 3), "DOUBLE")]),
])
def test_permissive_templates_do_not_assert_element_charge_or_bond_support(tmp_path, pattern, smiles, operations):
    path = tmp_path / "generic.json"
    path.write_text(json.dumps([pattern]))
    build_index(None, tmp_path / "generic.sqlite", release="SYNTHETIC_TEST_ONLY",
                source_url="https://example.invalid", templates=path, template_source_url="https://example.invalid")
    index = EvidenceIndex(tmp_path / "generic.sqlite.manifest.json")
    try:
        graph = MolecularGraph.from_smiles(smiles)
        assessment = index.assess(graph, apply_edits(graph, operations), operations, evidence_ids(index))
        assert assessment["matches"] and not assessment["template_preferred"]
    finally:
        index.close()


def test_template_match_cannot_bypass_graph_validation_or_native_input_preparation(template_index, tmp_path):
    client = Client({"drafts": [draft(Edit("atom", (2,), "F")), draft(Edit("atom", (3,), "O")),
                                draft(Edit("atom", (2,), "N"))]})
    agent = DeNovoAgent(client, policy_for(tmp_path, max_batches=1, fallback_attempts=1), template_index)
    native = []

    def prepare(product):
        native.append(product.smiles)
        if product.smiles == "CCO":
            raise LigandPreparationError("Synthetic native input failure", stage="native_conformer")

    packet = propose(agent, MolecularGraph.from_smiles("CCC"), prepare_candidate=prepare)
    assert "CCO" in native and "CNC" in native
    assert not any(c.graph.smiles == "CCO" for c in packet.candidates)
    funnel = packet.diagnostics["funnel"]
    assert funnel["invalid_graph"] >= 1 and funnel["preparation_failed"] >= 1
    assert any(c.graph.smiles == "CNC" for c in packet.candidates)


def test_explicit_stock_is_optional_validated_and_included_in_cache_identity(template_index, tmp_path):
    stock = tmp_path / "stock.smi"
    stock.write_text("CCO SYNTHETIC_TEST_ONLY\n")
    build_index(stock, tmp_path / "stock.sqlite", release="SYNTHETIC_TEST_ONLY", source_url="https://example.invalid")
    stock_manifest = str(tmp_path / "stock.sqlite.manifest.json")
    config = RunConfig(ComplexSpec((ProteinSpec("A", "ACD"),), "CCC", (ResidueRef("A", 1),)),
        ObjectiveSpec.denovo(), ProposalPolicy(), denovo=policy_for(tmp_path),
        evaluation=EvaluationSpec(), optimization=OptimizationSpec(strategy="denovo"))
    enriched = replace(config, denovo=replace(config.denovo, inventory_manifest=stock_manifest))
    assert _input_identity(config)[0] != _input_identity(enriched)[0]
    index = EvidenceIndex.from_policy(enriched.denovo)
    try:
        metadata = index.resource_metadata()
        assert metadata["stock_loaded"] and metadata["inventory_enhancement"]["provider"] == "Enamine"
        graph = MolecularGraph.from_smiles("CCC")
        tables, _, _ = direction_tables(graph, gradients(graph), ProposalPolicy(), joint=True)
        cards = index.retrieve(graph, tables)
        assert {c["kind"] for c in cards} == {"reaction_template_support", "inventory_fragment"}
        product = apply_edits(graph, [Edit("atom", (3,), "O")])
        support = index.assess(graph, product, [Edit("atom", (3,), "O")], [c["id"] for c in cards])
        assert support["catalog_preferred"] and support["template_preferred"]
    finally:
        index.close()
    with pytest.raises(ConfigurationError, match="not ready"):
        EvidenceIndex.from_policy(replace(config.denovo, inventory_manifest=str(tmp_path / "absent.json")))
    with pytest.raises(ConfigurationError, match="real Enamine"):
        InventoryIndex(config.denovo.evidence_manifest)


def test_cli_builds_and_verifies_template_only_index(template_index, tmp_path, capsys):
    source = tmp_path / "templates.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "cli.sqlite"
    assert main(["build-template-index", "--templates", str(source), "--output", str(output),
                 "--release", "SYNTHETIC_TEST_ONLY", "--source-url", "https://example.invalid",
                 "--expected-sha256", digest]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["resource_mode"] == "templates_only" and built["stock_sha256"] is None
    assert main(["check-evidence", "--manifest", str(output)+".manifest.json"]) == 0
    assert not json.loads(capsys.readouterr().out)["stock_loaded"]
    with pytest.raises(ConfigurationError, match="Template SHA-256"):
        build_index(None, tmp_path / "bad.sqlite", release="test", source_url="https://example.invalid",
                    templates=source, template_source_url="https://example.invalid", expected_template_sha256="0"*64)


def test_generated_profile_uses_template_manifest_without_inventory(tmp_path):
    output = tmp_path / "vdr"
    prepare_example("vdr-3a2i", output, tmp_path, work_root=tmp_path, msa_mode="single_sequence", strategy="denovo")
    config = load_config(output / "mosaic-like.yaml")
    assert config.denovo.inventory_manifest is None
    assert config.denovo.evidence_manifest == str(tmp_path / "data/inventory/uspto_simpretro.sqlite.manifest.json")
    assert config.denovo.protocol == "denovo-motif-v2"
