"""Bounded single-round rescoring; recommendation does not mutate the source graph."""

from pathlib import Path
import math

from molcraft.errors import ConfigurationError, GradientError, StaleProposalError


def _scores(result):
    loss = float(result["loss"])
    metrics = {name: float(value) for name, value in result["metrics"].items()}
    if not math.isfinite(loss) or not all(math.isfinite(v) for v in metrics.values()):
        raise GradientError("Rescoring returned nonfinite objective metrics")
    return {"loss": loss, "metrics": metrics}


def evaluate_packet(prepared, packet, config, *, backend=None, export_dir=None):
    """Rescore at one shared depth/noise and recommend at most one legal next graph.

    The baseline is recomputed here; a 25-step gradient loss is never compared
    against a 200-step candidate loss. The caller explicitly adopts selected_graph
    if desired. This function does not run a multi-round search or invoke an LLM.
    """
    if packet.source_graph.fingerprint != prepared.graph.fingerprint:
        raise StaleProposalError("Packet belongs to another source molecule")
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

    noise = backend.draw_noise(prepared, config.seed)
    baseline = _scores(backend.evaluate(prepared, config.objective, config.seed, noise))
    guard = config.evaluation.require_affinity_nonworsening
    if guard and "affinity_loss" not in baseline["metrics"]:
        raise GradientError("Baseline is missing affinity_loss required by the acceptance guard")
    candidates = sorted((c for c in packet.candidates if c.operations),
                        key=lambda c: (-c.probability, c.predicted_change, c.candidate_id))
    candidates = candidates[:config.evaluation.candidate_budget]
    measured = []
    selected_graph, selected_id, best_loss = prepared.graph, "noop", baseline["loss"]
    for index, candidate in enumerate(candidates, 1):
        # Packet IDs are data, not filesystem paths. Use a local numeric directory.
        destination = Path(export_dir) / f"evaluation-{index:04d}" if export_dir is not None else None
        result = evaluate_candidate(prepared, candidate, noise=noise, objective=config.objective,
            noise_seed=config.seed, backend=backend, export_dir=destination)
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
            "operations": [op.to_dict() for op in candidate.operations],
            "artifacts": result.get("artifacts", {})})
        if eligible and scores["loss"] < best_loss:
            selected_graph, selected_id, best_loss = candidate.graph, candidate.candidate_id, scores["loss"]
    return {"status": "improved" if selected_id != "noop" else "source_retained",
        "source_fingerprint": prepared.graph.fingerprint, "baseline": baseline,
        "evaluation": {"candidate_budget": config.evaluation.candidate_budget,
            "evaluated_candidates": len(measured), "sampling_steps": steps, "trunk_passes": passes,
            "seed": config.seed, "require_affinity_nonworsening": guard},
        "candidates": measured, "selected_id": selected_id, "selected_graph": selected_graph.to_dict(),
        "selected_loss": best_loss}
