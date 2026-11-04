"""Coverage is a design-task constraint, with exploration before first success."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from molcraft.evaluation import evaluate_packet
from molcraft.errors import GradientError, StaleProposalError
from molcraft.guidance.proposals import build_proposals
from molcraft.io.config import load_config


ROOT=Path(__file__).parents[1]


@pytest.mark.parametrize("mode,baseline_cov,first_cov,expected",[
    ("preserve",.8,.4,1), ("preserve",.3,.4,0), ("strict",.3,.4,1),
    ("off",.8,.4,0), ("preserve",.5,.4,0), ("preserve",.8,.5,1),
])
def test_coverage_guards_keep_a_success_floor_without_blocking_initial_search(
        graph,evidence,monkeypatch,mode,baseline_cov,first_cov,expected):
    cfg=load_config(ROOT/"configs/mosaic-like.yaml")
    cfg=replace(cfg,evaluation=replace(cfg.evaluation,candidate_budget=2,coverage_mode=mode))
    packet=build_proposals(graph,evidence)
    options=sorted([c for c in packet.candidates if c.operations],key=lambda c:-c.probability)[:2]
    class Backend:
        sampling_steps=200
        trunk_passes=4
        def draw_noise(self,*args):
            return None
        def evaluate(self,*args):
            return {"loss":10.,"metrics":{"affinity_loss":2.,"bcov":baseline_cov,"affinity_pred_value":.2}}
    def rescore(prepared,candidate,**kwargs):
        first=candidate is options[0]
        return {"loss":6. if first else 7.,"metrics":{"affinity_loss":1.,
                "bcov":first_cov if first else .6,"affinity_pred_value":-.2}}
    monkeypatch.setattr("molcraft.api.evaluate_candidate",rescore)
    result=evaluate_packet(SimpleNamespace(graph=graph),packet,cfg,backend=Backend())
    assert result["selected_id"]==options[expected].candidate_id
    if mode!="off":
        assert result["best_successful"]["metrics"]["bcov"]>.5
    if expected==1:
        assert "coverage_guard" in result["candidates"][0]["rejection_reasons"]


def test_active_coverage_guard_does_not_silently_accept_missing_metrics(graph,evidence):
    cfg=load_config(ROOT/"configs/mosaic-like.yaml")
    cfg=replace(cfg,evaluation=replace(cfg.evaluation,coverage_mode="preserve"))
    class Backend:
        sampling_steps=200
        trunk_passes=4
        def draw_noise(self,*args):
            return None
        def evaluate(self,*args):
            return {"loss":10.,"metrics":{"affinity_loss":2.}}
    with pytest.raises(GradientError,match="bcov"):
        evaluate_packet(SimpleNamespace(graph=graph),build_proposals(graph,evidence),cfg,backend=Backend())


def test_success_archive_can_preserve_a_successful_nonwinning_candidate(graph,evidence,monkeypatch):
    cfg=load_config(ROOT/"configs/mosaic-like.yaml")
    cfg=replace(cfg,evaluation=replace(cfg.evaluation,candidate_budget=2,coverage_mode="preserve"))
    packet=build_proposals(graph,evidence)
    options=sorted([c for c in packet.candidates if c.operations],key=lambda c:-c.probability)[:2]
    class Backend:
        sampling_steps=200
        trunk_passes=4
        def draw_noise(self,*args):
            return None
        def evaluate(self,*args):
            return {"loss":10.,"metrics":{"affinity_loss":2.,"bcov":.8,"affinity_pred_value":.3}}
    def rescore(prepared,candidate,**kwargs):
        first=candidate is options[0]
        return {"loss":6. if first else 7.,"metrics":{"affinity_loss":1.,"bcov":.7,
                                                       "affinity_pred_value":.2 if first else -.2}}
    monkeypatch.setattr("molcraft.api.evaluate_candidate",rescore)
    result=evaluate_packet(SimpleNamespace(graph=graph),packet,cfg,backend=Backend())
    assert result["selected_id"]==options[0].candidate_id
    assert result["best_successful"]["candidate_id"]==options[1].candidate_id


def test_changed_coverage_contract_cannot_reuse_cached_evaluation(graph,evidence,monkeypatch,tmp_path):
    import numpy as np
    cfg=load_config(ROOT/"configs/mosaic-like.yaml")
    cfg=replace(cfg,evaluation=replace(cfg.evaluation,candidate_budget=1,coverage_mode="preserve"))
    packet=build_proposals(graph,evidence)
    packet.diagnostics["context_id"]="same"
    candidate=next(c for c in packet.candidates if c.operations)
    noise=SimpleNamespace(init=np.zeros((1,3)),churn=np.zeros((1,1,3)),
                          rotations=np.eye(3)[None],translations=np.zeros((1,3)))
    class Backend:
        sampling_steps=200
        trunk_passes=4
        def evaluate(self,*args):
            return {"loss":10.,"metrics":{"affinity_loss":2.,"bcov":.8}}
    monkeypatch.setattr("molcraft.api.evaluate_candidate",lambda *a,**k:
                        {"loss":9.,"metrics":{"affinity_loss":1.,"bcov":.7}})
    kwargs=dict(backend=Backend(),noise=noise,selected_ids=[candidate.candidate_id],
                export_dir=tmp_path,context_id="same")
    prepared=SimpleNamespace(graph=graph,native_writer=None)
    evaluate_packet(prepared,packet,cfg,**kwargs)
    changed=replace(cfg,evaluation=replace(cfg.evaluation,min_coverage=.6))
    with pytest.raises(StaleProposalError):
        evaluate_packet(prepared,packet,changed,**kwargs)
