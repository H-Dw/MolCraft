from dataclasses import replace
import json

import numpy as np
import pytest

from molcraft.agent.syncraft import SynCraftAgent, compile_synthesis_draft
from molcraft.agent.optagent import SynCraftCases
from molcraft.chemistry.edits import Edit, apply_edits
from molcraft.chemistry.graph import MolecularGraph
from molcraft.chemistry.synthesis import (
    SynthesisPolicy, assess_synthesis, check_plan, complete_regions, groups, resonance_equivalent, sa_score,
)
from molcraft.errors import ChemistryError, ConfigurationError, StaleProposalError
from molcraft.guidance.proposals import ProposalPolicy, ProposalPacket
from molcraft.io.config import OptimizationSpec, RunConfig, load_config
from molcraft.io.examples import prepare_example
from molcraft.losses.spec import ObjectiveSpec
from molcraft.optimization import _input_identity, optimize
from molcraft.types import ComplexSpec, GradientEvidence, PreparedComplex, ProteinSpec, ResidueRef


def evidence(graph):
    return GradientEvidence(graph.fingerprint, graph.atom_ids,
        np.zeros((len(graph.atom_ids),9)), np.zeros((len(graph.pairs),5)), np.zeros(len(graph.atom_ids)))


def draft():
    return {"operations":[{"kind":"atom","atom_ids":[2],"value":"N"}],
            "rationale":"Introduce an amine analog.","case_ids":[],
            "synthesis":{"liability":"The alkane is accessible; avoid introducing hard motifs.",
                         "transformation":"Introduce a secondary amine functional group.",
                         "target_motifs":[{"motif":"amine","atom_ids":[2]}],
                         "replaced_source_groups":[],
                         "feasibility_basis":"Use an amine building-block analogy; route not validated."}}


def compile_example(content=None, focus=(2,), policy=None):
    graph=MolecularGraph.from_smiles("CCC")
    return compile_synthesis_draft(graph,evidence(graph),content or draft(),set(graph.atom_ids),
                                   set(),set(),focus,policy or SynthesisPolicy())


def test_real_amine_group_sa_and_conformer_preflight():
    _,updated,_,reason=compile_example()
    assert updated.smiles=="CNC"
    assessment=reason["synthesis_assessment"]
    assert 1 <= assessment["sa_after"] <= 6
    assert assessment["conformer_preflight"]=="passed"
    assert assessment["retrosynthesis"]=="not_run"
    assert assessment["experimentally_synthesizable"]=="not_established"


def test_atomic_rationale_without_synthesis_plan_is_rejected():
    content=draft()
    content.pop("synthesis")
    with pytest.raises(ChemistryError,match="requires"):
        compile_example(content)


@pytest.mark.parametrize("motif,ids",[("amide",[2]),("amine",[1]),("invented",[2])])
def test_unmatched_motif_claims_are_flagged_without_rejecting_legal_chemistry(motif,ids):
    content=draft()
    content["synthesis"]["target_motifs"]=[{"motif":motif,"atom_ids":ids}]
    _, updated, _, reason = compile_example(content)
    assert updated.smiles == "CNC"
    assessment = reason["synthesis_assessment"]
    assert assessment["catalog_preferred"] and assessment["catalog_warnings"]
    assert not assessment["matched_motifs"]
    assert any(m["motif"]=="amine" for m in assessment["recognized_product_motifs"])


def test_malformed_motif_atom_ids_are_still_rejected():
    content=draft()
    content["synthesis"]["target_motifs"]=[{"motif":"amine","atom_ids":[2,2]}]
    with pytest.raises(ChemistryError, match="Malformed"):
        compile_example(content)


def test_changed_atoms_need_group_coverage_and_high_preference_focus():
    with pytest.raises(ChemistryError,match="focus"):
        compile_example(focus=(1,))
    source=MolecularGraph.from_smiles("CCCCCC")
    product=apply_edits(source,[Edit("atom",(2,),"N"),Edit("atom",(5,),"O")])
    assert check_plan(draft()["synthesis"],source,product,{2,5},{2})["catalog_warnings"]


def test_whole_source_amide_replacement_must_be_explicit():
    source=MolecularGraph.from_smiles("CC(=O)N")
    product=apply_edits(source,[Edit("atom",(4,),"C")])
    plan=draft()["synthesis"]
    plan["target_motifs"]=[{"motif":"ketone","atom_ids":[1,2,3,4]}]
    assert check_plan(plan,source,product,{4},{4})["catalog_warnings"]
    plan["replaced_source_groups"]=["amide:2,3,4"]
    assert check_plan(plan,source,product,{4},{4})["catalog_preferred"]


def test_regions_complete_groups_and_do_not_truncate_large_rings():
    source=MolecularGraph.from_smiles("CCCC(=O)NCCCCC")
    units=[g for g in groups(source) if g["motif"]!="alkyl_segment"]
    for region,_ in complete_regions(source,set()).values():
        assert all(not set(g["atom_ids"]) & region or set(g["atom_ids"]) <= region for g in units)
    macrocycle=MolecularGraph.from_smiles("C1CCCCCCCCCCC1")
    assert not complete_regions(macrocycle,set(),limit=8)


@pytest.mark.parametrize("length",[15,31])
def test_actual_carbon_chain_initializations_have_local_complete_regions(length):
    source=MolecularGraph.from_smiles("C"*length)
    regions=complete_regions(source,set())
    assert len(regions)==length
    assert max(len(region) for region,_ in regions.values())==7
    assert complete_regions(source,set(source.atom_ids))=={}


@pytest.mark.parametrize("source,product",[
    ("C"*15,"CCCCCCCCC#[PH]CCCCC"),
    ("C"*31,"CCCCCCCCCCCCCCCCCCCCC#[SH+]CCCCCCCCC"),
    ("CCCCC","CC[Cl+]CC"),
    ("CCCCC","CCOOC"),
])
def test_regression_unusual_valence_and_peroxide_are_candidate_rejections(source,product):
    with pytest.raises(ChemistryError,match="New synthesis liability"):
        assess_synthesis(MolecularGraph.from_smiles(source),MolecularGraph.from_smiles(product),SynthesisPolicy())


def test_gba_resonance_change_is_not_a_new_chemical_entity():
    source=MolecularGraph.from_smiles("NC=NNCC")
    product=apply_edits(source,[Edit("bond",(1,2),"DOUBLE"),Edit("bond",(2,3),"SINGLE"),
                               Edit("charge",(1,),1),Edit("charge",(3,),-1)])
    assert resonance_equivalent(source,product)
    with pytest.raises(ChemistryError,match="Resonance-only"):
        assess_synthesis(source,product,SynthesisPolicy())


def test_sa_budget_and_failed_embedding_are_enforced(monkeypatch):
    with pytest.raises(ChemistryError,match="SA proxy"):
        compile_example(policy=SynthesisPolicy(max_sa_score=1))
    import molcraft.chemistry.synthesis as checks
    monkeypatch.setattr(checks.AllChem,"EmbedMolecule",lambda *a:-1)
    with pytest.raises(ChemistryError,match="conformer preflight"):
        compile_example()


class Cases:
    def retrieve(self,graph):
        return [{"id":"case-1","evidence_level":"analogy_only"}]


class Client:
    def __init__(self):
        self.messages=[]

    def complete(self,messages):
        self.messages.append(messages)
        assert "complete_source_groups" in json.loads(messages[1]["content"])
        assert "synthesis" in messages[0]["content"]
        return draft(),{"total_tokens":10}


def test_agent_uses_group_guidance_and_deterministic_selection_with_journal_replay(tmp_path):
    source=MolecularGraph.from_smiles("CCC")
    grad=evidence(source)
    grad.atom_grad[1,1]=-2  # N preference focuses atom 2.
    client=Client()
    agent=SynCraftAgent(client,Cases(),SynthesisPolicy(check_conformer=False,focus_atoms=1))
    kwargs=dict(samples=2,round_index=1,context_id="fixed",journal_dir=tmp_path)
    packet=agent.propose(source,grad,ProposalPolicy(),**kwargs)
    assert len(packet.candidates)==2 and packet.candidates[1].graph.smiles=="CNC"
    assert packet.diagnostics["focus_atom_ids"]==[2]
    assert packet.diagnostics["retrosynthesis"]=="not_run"
    assert ProposalPacket.from_dict(packet.to_dict()).candidate("synthesis-01").graph.smiles=="CNC"
    n=len(client.messages)
    assert agent.select(packet,8,journal_dir=tmp_path)==["synthesis-01"]
    assert len(client.messages)==n and not list(tmp_path.glob("select-*.json"))
    assert agent.propose(source,grad,ProposalPolicy(),**kwargs).to_dict()==packet.to_dict()
    assert len(client.messages)==n
    assert agent.propose(source,grad,ProposalPolicy(),excluded={"CNC"},samples=1,
                         round_index=2,context_id="other",journal_dir=tmp_path/"other").status=="no_legal_edit"


def test_retrieval_includes_actual_case_edits_and_atom_namespace():
    cases=SynCraftCases(synthesis_context=True).retrieve(MolecularGraph.from_smiles("CCC"))
    assert len(cases)==5
    assert all(c["edit_sequence"] and c["atom_id_namespace"]=="case_only_do_not_copy_to_query" for c in cases)


def test_sa_adjusted_selection_can_differ_from_gradient_only_ranking():
    source=MolecularGraph.from_smiles("CCCCCC")
    changes=[Edit("atom",(2,),"N"),Edit("atom",(3,),"O")]
    products=[apply_edits(source,[op]) for op in changes]
    scores=[sa_score(g.smiles) for g in products]
    assert abs(scores[0]-scores[1])>.01
    lower=int(np.argmin(scores))
    grad=evidence(source)
    for i,op in enumerate(changes):
        value=-.1 if i==lower else -.1-abs(scores[0]-scores[1])/2
        grad.atom_grad[op.atom_ids[0]-1,{"N":1,"O":2}[op.value]]=value

    class TwoDrafts:
        def complete(self,messages):
            i=json.loads(messages[1]["content"])["sample"]-1
            content=draft()
            content["operations"]=[changes[i].to_dict()]
            content["synthesis"]["target_motifs"]=[
                {"motif":"amine","atom_ids":[2]} if i==0 else {"motif":"ether","atom_ids":[2,3,4]}]
            return content,{}

    agent=SynCraftAgent(TwoDrafts(),Cases(),SynthesisPolicy(sa_weight=1,max_sa_score=10,
                            max_sa_increase=10,check_conformer=False))
    packet=agent.propose(source,grad,ProposalPolicy(),samples=2,round_index=1,context_id="ranking")
    assert len(packet.candidates)==3
    picked=packet.candidate(agent.select(packet,1)[0])
    assert picked.graph.smiles==products[lower].smiles
    assert picked.candidate_id!=min(packet.candidates[1:],key=lambda c:c.predicted_change).candidate_id


def test_factory_and_cli_use_the_synthesis_agent_without_extra_selection_calls(tmp_path,monkeypatch):
    import molcraft
    import molcraft.agent.argmax as factory
    import molcraft.agent.deepseek as provider
    import molcraft.evaluation as evaluation
    from molcraft.backends.boltz2_jax.backend import Boltz2Backend
    from molcraft.cli import main
    import yaml

    monkeypatch.setattr(provider.DeepSeekClient,"available",lambda self:True)
    real_agent,metadata=factory.create_agent("syncraft")
    assert isinstance(real_agent,SynCraftAgent) and metadata["agent_protocol"]=="syncraft-group-v2"
    model_paths={}
    for name in ("structure_checkpoint","affinity_checkpoint","ccd_path"):
        target=tmp_path/name
        target.write_bytes(b"mock-model-not-loaded")
        model_paths[name]=str(target)
    model_paths["molecule_dir"]=str(tmp_path)
    config_path=tmp_path/"config.yaml"
    config_path.write_text(yaml.safe_dump({"proteins":[{"chain_id":"A","sequence":"ACD"}],
        "ligand":{"smiles":"CCC"},"pocket":[{"chain_id":"A","position":1}],
        "model_paths":model_paths,
        "optimization":{"strategy":"syncraft","drafts":1}}))
    config=load_config(config_path)
    source=MolecularGraph.from_smiles("CCC")
    monkeypatch.setattr(molcraft,"prepare_complex",lambda spec:PreparedComplex(spec,source,{},None))
    monkeypatch.setattr(molcraft,"value_and_grad",lambda *a,**k:evidence(source))
    monkeypatch.setattr(Boltz2Backend,"from_paths",lambda *a,**k:object())
    client=Client()
    agent=SynCraftAgent(client,Cases())
    monkeypatch.setattr(factory,"create_agent",lambda *a,**k:(agent,{}))
    assert main(["propose","--config",str(config_path),"--output",str(tmp_path/"proposal")])==0
    proposal=tmp_path/"proposal/proposals.json"
    packet=ProposalPacket.from_dict(json.loads(proposal.read_text()))
    assert packet.diagnostics["strategy"]=="syncraft"
    calls=len(client.messages)

    def rescore(prepared,loaded,cfg,**kwargs):
        assert kwargs["context_id"]==packet.diagnostics["context_id"]
        assert _input_identity(cfg)==_input_identity(config)
        assert kwargs["selected_ids"]==["synthesis-01"]
        return {"status":"improved","selected_id":"synthesis-01","selected_loss":1.,
                "selected_graph":loaded.candidate("synthesis-01").graph.to_dict()}

    monkeypatch.setattr(evaluation,"evaluate_packet",rescore)
    assert main(["evaluate-set","--config",str(config_path),"--packet",str(proposal),
                 "--output",str(tmp_path/"evaluation")])==0
    assert len(client.messages)==calls


def test_new_example_mode_is_explicit_and_checkpoint_identity_is_separate(tmp_path):
    prepare_example("vdr-3a2i",tmp_path/"input",tmp_path,work_root=tmp_path,msa_mode="single_sequence")
    config=load_config(tmp_path/"input/mosaic-like.yaml")
    assert config.optimization.strategy=="syncraft"
    base=RunConfig(ComplexSpec((ProteinSpec("A","ACD"),),"CCC",(ResidueRef("A",1),)),
                   ObjectiveSpec(),ProposalPolicy())
    new=replace(base,optimization=OptimizationSpec(strategy="syncraft"))
    assert _input_identity(base)[0]!=_input_identity(new)[0]
    assert _input_identity(new)[0]!=_input_identity(replace(new,synthesis=replace(new.synthesis,sa_weight=.5)))[0]
    out=tmp_path/"run"
    out.mkdir()
    (out/"checkpoint.json").write_text(json.dumps({"context_id":_input_identity(base)[0],"status":"converged"}))
    with pytest.raises(StaleProposalError,match="strategy"):
        optimize(new,None,out,resume=True)


@pytest.mark.parametrize("kwargs",[{"max_sa_score":11},{"focus_atoms":0},{"max_sa_increase":-1},{"check_conformer":"yes"}])
def test_invalid_synthesis_policy_is_not_silently_ignored(kwargs):
    with pytest.raises(ConfigurationError):
        SynthesisPolicy(**kwargs)


def test_synthesis_settings_cannot_be_silently_applied_to_a_different_control_mode():
    base=RunConfig(ComplexSpec((ProteinSpec("A","ACD"),),"CCC",(ResidueRef("A",1),)),
                   ObjectiveSpec(),ProposalPolicy())
    with pytest.raises(ConfigurationError,match="require the syncraft strategy"):
        replace(base,synthesis=SynthesisPolicy(sa_weight=1))
