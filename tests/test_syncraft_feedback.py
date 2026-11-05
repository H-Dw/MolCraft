"""Candidate diversity, catalogue fallback, compact hints and durable feedback."""

from dataclasses import replace
import json

import numpy as np
import pytest

from molcraft.agent.context import operation_key, top_half_directions
from molcraft.agent.deepseek import DeepSeekClient, ProviderError, RequestJournal
from molcraft.agent.history import DraftHistory
from molcraft.agent.syncraft import SynCraftAgent, compile_synthesis_draft
from molcraft.chemistry.graph import MolecularGraph
from molcraft.chemistry.synthesis import SynthesisPolicy
from molcraft.errors import ConfigurationError, StaleProposalError
from molcraft.guidance.proposals import ProposalPolicy, ProbabilityTable
from molcraft.io.config import EvaluationSpec
from molcraft.types import GradientEvidence


def evidence(graph):
    return GradientEvidence(graph.fingerprint, graph.atom_ids, np.zeros((len(graph.atom_ids),9)),
                            np.zeros((len(graph.pairs),5)), np.zeros(len(graph.atom_ids)))


class Cases:
    def retrieve(self, graph):
        return [{"id":"case-1"}]


def design(atom=2, element="N", motifs=None):
    return {"operations":[{"kind":"atom","atom_ids":[atom],"value":element}],
            "rationale":"Introduce a local analog.", "case_ids":[],
            "synthesis":{"liability":"Accessible source.", "transformation":"Local group change.",
                         "target_motifs":motifs if motifs is not None else [{"motif":"amine","atom_ids":[atom]}],
                         "replaced_source_groups":[], "feasibility_basis":"Building-block analogy; route not run."}}


def test_adaptive_pool_contains_sixteen_unique_candidates_before_selecting_eight(tmp_path):
    graph=MolecularGraph.from_smiles("C"*20)

    class Client:
        def __init__(self):
            self.contexts=[]
        def complete(self,messages):
            c=json.loads(messages[1]["content"])
            self.contexts.append(c)
            i=(c["sample"]-1)%8+1
            element="N" if c["sample"]<=8 else "O"
            motifs=([{"motif":"amine","atom_ids":[i]}] if element=="N" else
                    [{"motif":"alcohol","atom_ids":[1,2]}] if i==1 else
                    [{"motif":"ether","atom_ids":[i-1,i,i+1]}])
            return design(i,element,motifs),{"total_tokens":10}

    client=Client()
    agent=SynCraftAgent(client,Cases(),SynthesisPolicy(check_conformer=False))
    kwargs=dict(samples=10,round_index=1,context_id="fixed",candidate_budget=8,journal_dir=tmp_path)
    packet=agent.propose(graph,evidence(graph),ProposalPolicy(),**kwargs)
    assert len(packet.candidates)==17 and len(agent.select(packet,8))==8
    assert packet.diagnostics["pool"]["target"]==16
    assert packet.diagnostics["pool"]["shortfall"]==0
    assert len(client.contexts)==16
    assert any(c["draft_history"] for c in client.contexts[2:])
    before=len(client.contexts)
    assert agent.propose(graph,evidence(graph),ProposalPolicy(),**kwargs).to_dict()==packet.to_dict()
    assert len(client.contexts)==before


def test_duplicate_storm_is_bounded_and_reports_pool_shortfall(tmp_path):
    graph=MolecularGraph.from_smiles("CCC")
    class Repeats:
        calls=0
        def complete(self,messages):
            self.calls+=1
            return design(),{}
    client=Repeats()
    agent=SynCraftAgent(client,Cases(),SynthesisPolicy(check_conformer=False,max_drafts=4,draft_batch_size=1))
    packet=agent.propose(graph,evidence(graph),ProposalPolicy(),samples=1,round_index=1,
                         context_id="same",candidate_budget=2,journal_dir=tmp_path)
    assert len(packet.candidates)==2
    assert packet.diagnostics["attempted_samples"]==4
    assert packet.diagnostics["pool"]["shortfall"]==3
    assert client.calls<=16
    assert all(json.loads(p.read_text()).get("content") for p in tmp_path.glob("draft-*-*.json"))
    before=client.calls
    replay=agent.propose(graph,evidence(graph),ProposalPolicy(),samples=1,round_index=1,
                          context_id="same",candidate_budget=2,journal_dir=tmp_path)
    assert replay.to_dict()==packet.to_dict() and client.calls==before


def test_catalogue_preferred_first_with_legal_uncatalogued_fallback():
    graph=MolecularGraph.from_smiles("C"*8)
    grad=evidence(graph)
    grad.bond_grad[graph.pairs.index((3,4)),3]=-100
    class Client:
        def complete(self,messages):
            sample=json.loads(messages[1]["content"])["sample"]
            content=design(2) if sample==1 else design(4,motifs=[])
            if sample!=1:
                content["operations"]=[{"kind":"bond","atom_ids":[3,4],"value":"TRIPLE"}]
            return content,{}
    agent=SynCraftAgent(Client(),Cases(),SynthesisPolicy(check_conformer=False))
    packet=agent.propose(graph,grad,ProposalPolicy(),samples=2,round_index=1,context_id="x")
    ids=agent.select(packet,2)
    assert len(ids)==2
    assert packet.candidate(ids[0]).operations[0].atom_ids==(2,)
    assert packet.candidate(ids[1]).predicted_change==-100
    assert packet.candidate(ids[0]).probability>packet.candidate(ids[1]).probability
    assert not packet.diagnostics["agent_explanations"][ids[1]]["synthesis_assessment"]["catalog_preferred"]


def test_ordinary_halide_matches_and_uncatalogued_legal_chemistry_is_not_gated():
    graph=MolecularGraph.from_smiles("CCCC")
    content=design(4,"Cl",[{"motif":"alkyl_halide","atom_ids":[3,4]}])
    args=(graph,evidence(graph),content,set(graph.atom_ids),set(),set(),{4},SynthesisPolicy(check_conformer=False))
    _,product,_,reason=compile_synthesis_draft(*args)
    assert product.smiles=="CCCCl" and reason["synthesis_assessment"]["catalog_preferred"]
    content["synthesis"]["target_motifs"]=[]
    _,product,_,reason=compile_synthesis_draft(*args)
    assert product.smiles=="CCCCl" and reason["synthesis_assessment"]["catalog_preferred"]
    content["operations"]=[{"kind":"bond","atom_ids":[2,3],"value":"TRIPLE"}]
    _,product,_,reason=compile_synthesis_draft(graph,evidence(graph),content,set(graph.atom_ids),
                                             set(),set(),{2},SynthesisPolicy(check_conformer=False))
    assert product.smiles=="CC#CC" and not reason["synthesis_assessment"]["catalog_preferred"]


def test_probability_top_half_ranks_each_row_and_excludes_noop_zero_and_ties():
    graph=MolecularGraph.from_smiles("CCC")
    labels=("C","N","O","F","S")
    probabilities=np.array([[.1,.4,.3,.2,0],[.5,.2,.2,.1,0],[1,0,0,0,0]])
    t=ProbabilityTable(tuple((i,) for i in graph.atom_ids),labels,np.zeros((3,5)),
                       probabilities>0,probabilities,[[""]*5 for _ in range(3)])
    hints=top_half_directions(graph,{"atom":t},set(graph.atom_ids))["atom"]
    assert [r["ids"] for r in hints]==[[1],[2]]
    assert [[d[0] for d in row["directions"]] for row in hints]==[["N","O"],["N","O"]]
    assert not top_half_directions(graph,{"atom":t},{3})["atom"]


def test_history_updates_actual_metrics_once_and_prevents_cross_round_repeats(tmp_path):
    graph=MolecularGraph.from_smiles("CCC")
    class Client:
        def complete(self,messages):
            return design(),{}
    agent=SynCraftAgent(Client(),Cases(),SynthesisPolicy(check_conformer=False))
    kwargs=dict(samples=1,round_index=1,context_id="c",journal_dir=tmp_path/"round-001/requests")
    packet=agent.propose(graph,evidence(graph),ProposalPolicy(),**kwargs)
    history=DraftHistory(tmp_path,"run")
    history.record_round(1,graph,packet,requests=kwargs["journal_dir"])
    assert history.before(1)==[]
    result={"selected_id":"synthesis-01","baseline":{"loss":2},"candidates":[
        {"candidate_id":"synthesis-01","loss":1,"actual_change":-1,"eligible":True,
         "metrics":{"affinity_pred_value":-.4,"bcov":.75,"affinity_probability_binary":.8}}]}
    history.record_round(1,graph,packet,result,requests=kwargs["journal_dir"])
    history=DraftHistory(tmp_path,"run")
    assert len(history.data["rounds"])==1
    assert history.data["rounds"][0]["candidates"][0]["metrics"]["bcov"]==.75
    assert "0.75" in (tmp_path/"draft_history.md").read_text()
    second=agent.propose(graph,evidence(graph),ProposalPolicy(),samples=1,round_index=2,
                         context_id="c",history=history.before(2),journal_dir=tmp_path/"round-002/requests")
    assert len(second.candidates)==1
    operations=design()["operations"]+[{"kind":"charge","atom_ids":[2],"value":1}]
    assert operation_key(operations)==operation_key(list(reversed(operations)))
    with pytest.raises(StaleProposalError):
        DraftHistory(tmp_path,"different")


@pytest.mark.parametrize("interrupt",[False,True])
def test_optimization_writes_round_metrics_and_feeds_them_after_resume(tmp_path,interrupt):
    import jax.numpy as jnp
    from molcraft.chemistry.mapping import AtomLayout
    from molcraft.io.config import RunConfig, OptimizationSpec
    from molcraft.losses.spec import ObjectiveSpec
    from molcraft.optimization import optimize
    from molcraft.sampling.core import Noise
    from molcraft.types import ComplexSpec, PreparedComplex, ProteinSpec, ResidueRef

    class Client:
        def __init__(self):
            self.contexts=[]
        def complete(self,messages):
            self.contexts.append(json.loads(messages[1]["content"]))
            return design(),{}
    def prepare(spec):
        graph=MolecularGraph.from_smiles(spec.ligand_smiles)
        layout=AtomLayout(graph.atom_ids,(1,2,3),(1,2,3),(0,),(0,),((0,),),4,4)
        return PreparedComplex(spec,graph,{},layout)
    class Backend:
        def draw_noise(self,*args):
            return Noise(jnp.zeros((4,3)),jnp.zeros((1,4,3)),jnp.eye(3)[None],jnp.zeros((1,3)))
        def value_and_grad(self,prepared,*args,**kwargs):
            return evidence(prepared.graph)
    interrupted=[]
    def score(prepared,packet,cfg,*,selected_ids,**kwargs):
        if interrupt and not interrupted:
            interrupted.append(True)
            raise RuntimeError("simulated evaluation interruption")
        cid=selected_ids[0] if selected_ids else "noop"
        rows=[{"candidate_id":i,"loss":1.,"actual_change":-1.,"eligible":True,
               "metrics":{"affinity_pred_value":-.3,"bcov":.75}} for i in selected_ids]
        return {"selected_id":cid,"selected_graph":packet.candidate(cid).graph.to_dict(),
                "status":"improved" if selected_ids else "source_retained",
                "selected_loss":1.,"baseline":{"loss":2.},"candidates":rows,
                "best_successful":{"candidate_id":cid,"graph":packet.candidate(cid).graph.to_dict(),
                                   "loss":1.,"metrics":{"affinity_pred_value":-.3,"bcov":.75}}}
    config=RunConfig(ComplexSpec((ProteinSpec("A","ACD"),),"CCC",(ResidueRef("A",1),)),
                     ObjectiveSpec(),ProposalPolicy(),evaluation=EvaluationSpec(candidate_budget=1),
                     optimization=OptimizationSpec(strategy="syncraft",drafts=1,max_rounds=2,min_rounds=2,patience=1),
                     synthesis=SynthesisPolicy(check_conformer=False,max_drafts=2,draft_batch_size=1))
    client=Client()
    agent=SynCraftAgent(client,Cases(),config.synthesis)
    kwargs=dict(backend=Backend(),evaluation_backend=Backend(),prepare=prepare,score=score)
    if interrupt:
        with pytest.raises(RuntimeError,match="simulated"):
            optimize(config,agent,tmp_path,stop_after=1,**kwargs)
        calls=len(client.contexts)
        first=optimize(config,agent,tmp_path,stop_after=1,resume=True,**kwargs)
        assert len(client.contexts)==calls
    else:
        first=optimize(config,agent,tmp_path,stop_after=1,**kwargs)
    assert first["status"]=="window_complete"
    history=json.loads((tmp_path/"draft_history.json").read_text())
    assert history["rounds"][0]["candidates"][0]["metrics"]["bcov"]==.75
    assert (tmp_path/"best_successful.json").exists()
    client.contexts.clear()
    state=optimize(config,agent,tmp_path,resume=True,**kwargs)
    assert len(state["rounds"])==2
    assert any(r.get("metrics",{}).get("bcov")==.75 for c in client.contexts for r in c["draft_history"])
    assert len(json.loads((tmp_path/"draft_history.json").read_text())["rounds"])==2
    with pytest.raises(StaleProposalError):
        optimize(replace(config,synthesis=replace(config.synthesis,max_drafts=3)),agent,tmp_path,resume=True,**kwargs)


def test_operation_history_from_another_source_does_not_ban_a_new_analog():
    graph=MolecularGraph.from_smiles("CCC")
    other=MolecularGraph.from_smiles("CCO")
    class Client:
        def complete(self,messages):
            return design(),{}
    agent=SynCraftAgent(Client(),Cases(),SynthesisPolicy(check_conformer=False))
    packet=agent.propose(graph,evidence(graph),ProposalPolicy(),samples=1,round_index=2,context_id="x",
                         history=[{"round":1,"source_fingerprint":other.fingerprint,
                                   "operations":design()["operations"],"outcome":"rejected"}])
    assert len(packet.candidates)==2


def test_failed_json_preserves_usage_and_finish_reason(tmp_path,monkeypatch):
    client=DeepSeekClient()
    client.retry_backoff_seconds=0
    monkeypatch.setattr(client,"_request",lambda *args:{
        "choices":[{"finish_reason":"length","message":{"content":"{"}}],
        "usage":{"prompt_tokens":100,"completion_tokens":200,"total_tokens":300,
                 "prompt_cache_hit_tokens":50,"completion_tokens_details":{"reasoning_tokens":180}}})
    with pytest.raises(ProviderError):
        RequestJournal(tmp_path,bind_messages=True).request("draft-00",client,[],lambda c:c)
    receipt=json.loads((tmp_path/"draft-00-4.json").read_text())
    assert receipt["usage"]["total_tokens"]==300
    assert receipt["usage"]["reasoning_tokens"]==180 and receipt["finish_reason"]=="length"
    with pytest.raises(StaleProposalError):
        RequestJournal(tmp_path,bind_messages=True).request("draft-00",client,[{"role":"user","content":"changed"}],lambda c:c)


@pytest.mark.parametrize("kwargs",[{"coverage_mode":"unknown"},{"min_coverage":1},{"min_coverage":-1},
                                    {"min_coverage":float("nan")}])
def test_invalid_coverage_contract(kwargs):
    with pytest.raises(ConfigurationError):
        EvaluationSpec(**kwargs)
