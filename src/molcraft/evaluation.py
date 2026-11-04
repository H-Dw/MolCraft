"""Bounded single-round rescoring; recommendation does not mutate the source graph."""

from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import math
import time

import numpy as np

from molcraft.errors import ChemistryError, ConfigurationError, GradientError, StaleProposalError
from molcraft.io.serialization import save_json


EVALUATION_PROTOCOL = "candidate-failure-isolation-v1"


def _progress_identity(prepared, candidates, config, context_id, noise):
    # Optimization supplies a context including file hashes and an explicit Noise
    # replay. Generic callers without that contract do not reuse cached scores.
    fields = ("init", "churn", "rotations", "translations")
    if context_id is None or not all(hasattr(noise, name) for name in fields):
        return None
    digest = hashlib.sha256()
    for name in fields:
        array = np.ascontiguousarray(np.asarray(getattr(noise, name)))
        digest.update(str((name, array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return {"protocol": EVALUATION_PROTOCOL, "context_id": context_id,
            "source_fingerprint": prepared.graph.fingerprint,
            "candidate_fingerprints": [{"candidate_id": c.candidate_id, "fingerprint": c.graph.fingerprint}
                                       for c in candidates],
            "seed": config.seed, "objective": asdict(config.objective),
            "sampling_steps": config.evaluation_sampling_steps,
            "trunk_passes": config.evaluation_trunk_passes,
            "require_affinity_nonworsening": config.evaluation.require_affinity_nonworsening,
            "noise_sha256": digest.hexdigest()}


def _scores(result):
    loss = float(result["loss"])
    metrics = {name: float(value) for name, value in result["metrics"].items()}
    if not math.isfinite(loss) or not all(math.isfinite(v) for v in metrics.values()):
        raise GradientError("Rescoring returned nonfinite objective metrics")
    return {"loss": loss, "metrics": metrics}


def evaluate_packet(prepared, packet, config, *, backend=None, export_dir=None,
                    selected_ids=None, noise=None, context_id=None):
    """Rescore at one shared depth/noise and recommend at most one legal next graph.

    The baseline is recomputed here; a 25-step gradient loss is never compared
    against a 200-step candidate loss. The caller explicitly adopts selected_graph
    if desired. This function does not run a multi-round search or invoke an LLM.
    """
    if packet.source_graph.fingerprint != prepared.graph.fingerprint:
        raise StaleProposalError("Packet belongs to another source molecule")
    if packet.diagnostics.get("context_id") != context_id:
        raise StaleProposalError("Packet belongs to a different optimization context")
    if selected_ids is not None:
        if len(selected_ids) > config.evaluation.candidate_budget or len(set(selected_ids)) != len(selected_ids):
            raise ConfigurationError("Candidate IDs must be unique and within the evaluation budget")
        candidates = [packet.candidate(candidate_id) for candidate_id in selected_ids]
        if any(not c.operations for c in candidates):
            raise ConfigurationError("The unchanged source is already included as the baseline")
    else:
        candidates = sorted((c for c in packet.candidates if c.operations),
                            key=lambda c: (-c.probability, c.predicted_change, c.candidate_id))
        candidates = candidates[:config.evaluation.candidate_budget]
    config.objective.validate(config.gradient_mode)
    if config.evaluation.require_affinity_nonworsening and config.objective.affinity_weight == 0:
        raise ConfigurationError("The affinity acceptance guard requires an active affinity loss")
    steps, passes = config.evaluation_sampling_steps, config.evaluation_trunk_passes
    if backend is None:
        from molcraft.backends.boltz2_jax.backend import Boltz2Backend

        backend = Boltz2Backend.from_paths(config.complex.model_paths, steps, passes)
    if backend.sampling_steps != steps or backend.trunk_passes != passes:
        raise ConfigurationError("Evaluation backend depth does not match the configuration")
    from molcraft.api import evaluate_candidate

    if noise is None:
        noise = backend.draw_noise(prepared, config.seed)
    identity = _progress_identity(prepared, candidates, config, context_id, noise)
    progress_path = Path(export_dir) / "evaluation_progress.json" if export_dir is not None and identity else None
    progress = {}
    if progress_path is not None and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["identity"] != identity:
            raise StaleProposalError("Saved evaluation progress has different inputs, candidates or noise")
    cached = {r["candidate_id"]: r for r in [*progress.get("candidates", []),
                                               *progress.get("unscored_candidates", [])]}
    if len(cached) != len(progress.get("candidates", [])) + len(progress.get("unscored_candidates", [])):
        raise StaleProposalError("Saved evaluation progress contains duplicate candidates")
    if not set(cached) <= {c.candidate_id for c in candidates}:
        raise StaleProposalError("Saved evaluation progress contains unknown candidates")
    if progress:
        baseline = progress["baseline"]
        _scores(baseline)
    else:
        started = time.monotonic()
        baseline_result = backend.evaluate(prepared, config.objective, config.seed, noise)
        baseline = _scores(baseline_result)
        baseline["seconds"] = time.monotonic() - started
        if export_dir is not None and selected_ids is not None and prepared.native_writer is not None:
            from molcraft.api import export_prediction

            baseline["artifacts"] = export_prediction(prepared, prepared.graph, baseline_result,
                                                     Path(export_dir) / "baseline")
    guard = config.evaluation.require_affinity_nonworsening
    if guard and "affinity_loss" not in baseline["metrics"]:
        raise GradientError("Baseline is missing affinity_loss required by the acceptance guard")
    measured, rejected = [], []

    def persist():
        if progress_path is not None:
            completed = {**cached, **{r["candidate_id"]: r for r in [*measured, *rejected]}}
            ordered = [completed[c.candidate_id] for c in candidates if c.candidate_id in completed]
            save_json({"identity": identity, "baseline": baseline,
                       "candidates": [r for r in ordered if "error" not in r],
                       "unscored_candidates": [r for r in ordered if "error" in r]}, progress_path)

    persist()
    selected_graph, selected_id, best_loss = prepared.graph, "noop", baseline["loss"]
    for index, candidate in enumerate(candidates, 1):
        # Packet IDs are data, not filesystem paths. Use a local numeric directory.
        destination = Path(export_dir) / f"evaluation-{index:04d}" if export_dir is not None else None
        started = time.monotonic()
        saved = cached.get(candidate.candidate_id)
        if saved is not None and "error" in saved:
            rejected.append(saved)
            persist()
            continue
        try:
            if saved is not None:
                result = saved
            else:
                result = evaluate_candidate(prepared, candidate, noise=noise, objective=config.objective,
                    noise_seed=config.seed, backend=backend, export_dir=destination)
        except ChemistryError as exc:
            if selected_ids is None:
                raise
            rejected.append({"candidate_id": candidate.candidate_id, "error": str(exc),
                             "error_type": type(exc).__name__, "stage": getattr(exc, "stage", "chemistry"),
                             "smiles": candidate.graph.smiles, "seconds": time.monotonic() - started})
            persist()
            continue
        scores = _scores(result)
        if guard and "affinity_loss" not in scores["metrics"]:
            raise GradientError("Candidate is missing affinity_loss required by the acceptance guard")
        affinity_ok = not guard or scores["metrics"]["affinity_loss"] <= baseline["metrics"]["affinity_loss"]
        improves = scores["loss"] < baseline["loss"]
        eligible = improves and affinity_ok
        measured.append({"candidate_id": candidate.candidate_id, **scores,
            "predicted_change": candidate.predicted_change,
            "actual_change": scores["loss"] - baseline["loss"],
            "eligible": eligible, "affinity_guard_passed": affinity_ok,
            "seconds": saved["seconds"] if saved is not None else time.monotonic() - started,
            "operations": [op.to_dict() for op in candidate.operations],
            "artifacts": result.get("artifacts", {})})
        persist()
        if eligible and scores["loss"] < best_loss:
            selected_graph, selected_id, best_loss = candidate.graph, candidate.candidate_id, scores["loss"]
    return {"status": "improved" if selected_id != "noop" else "source_retained",
        "source_fingerprint": prepared.graph.fingerprint, "baseline": baseline,
        "evaluation": {"candidate_budget": config.evaluation.candidate_budget,
            "protocol": EVALUATION_PROTOCOL, "requested_candidates": len(candidates),
            "attempted_candidates": len(measured) + len(rejected), "rejected_candidates": len(rejected),
            "reused_candidates": len(cached),
            "evaluated_candidates": len(measured), "sampling_steps": steps, "trunk_passes": passes,
            "seed": config.seed, "require_affinity_nonworsening": guard},
        "candidates": measured, "unscored_candidates": rejected,
        "selected_id": selected_id, "selected_graph": selected_graph.to_dict(),
        "selected_loss": best_loss}
