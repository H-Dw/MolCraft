"""Restartable multi-round search with stable reference noise at both depths."""

from dataclasses import asdict, replace
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from molcraft.api import prepare_complex, value_and_grad
from molcraft.agent.deepseek import ProviderError
from molcraft.backends.boltz2_jax.backend import Boltz2Backend, remap_noise
from molcraft.chemistry.graph import MolecularGraph
from molcraft.errors import ConfigurationError, StaleProposalError
from molcraft.evaluation import evaluate_packet
from molcraft.io.examples import task_path
from molcraft.io.serialization import save_json, save_packet
from molcraft.types import GradientEvidence


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _input_identity(config):
    spec = config.complex
    inputs = {f"{p.chain_id}:{name}": _file_hash(path)
              for p in spec.proteins
              for name, path in (("msa", p.msa_path), ("template", p.template_path))
              if path}
    if config.optimization.strategy == "syncraft":
        from importlib.resources import files

        for name in ("data/syncraft_synthesis_guidance.md", "data/syncraft_reasoning.json",
                     "agent/syncraft.py", "agent/context.py", "agent/history.py", "agent/deepseek.py",
                     "chemistry/synthesis.py"):
            resource = files("molcraft").joinpath(*name.split("/"))
            inputs[name] = hashlib.sha256(resource.read_bytes()).hexdigest()
    if spec.model_paths:
        paths = spec.model_paths
        for name in ("structure_checkpoint", "affinity_checkpoint", "ccd_path"):
            inputs[name] = _file_hash(getattr(paths, name))
        inputs["molecule_dir"] = str(paths.molecule_dir.resolve())
    cfg = asdict(config)
    cfg["complex"].pop("cache_dir", None)
    # A new, versioned protocol; old OptAgent/argmax identities remain intact.
    if config.optimization.strategy != "syncraft":
        cfg.pop("synthesis")
        if config.evaluation.coverage_mode == "off" and config.evaluation.min_coverage == 0.5:
            cfg["evaluation"].pop("coverage_mode")
            cfg["evaluation"].pop("min_coverage")
    # Preserve the identity of published, default OptAgent checkpoints.
    if cfg["optimization"]["strategy"] == "optagent":
        cfg["optimization"].pop("strategy")
    if cfg["msa_mode"] == "provided":
        cfg.pop("msa_mode")
    return _digest({"version": 1, "config": cfg, "inputs": inputs}), inputs


def _restore_gradient(path, context, graph):
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.pop("context_id") != context:
        raise StaleProposalError("Saved gradient has a different context")
    data["atom_ids"] = tuple(data["atom_ids"])
    data["pocket_distances"] = tuple(data["pocket_distances"])
    for name in ("atom_grad", "bond_grad", "charge_grad"):
        data[name] = np.asarray(data[name], dtype=np.float32)
    data["bond_grad"] = data["bond_grad"].reshape((-1, 5))
    evidence = GradientEvidence(**data)
    evidence.validate(graph)
    return evidence


def optimize(config, agent, output, *, resume=False, stop_after=None, backend=None,
             evaluation_backend=None, prepare=prepare_complex, score=evaluate_packet,
             on_event=None):
    output = Path(output).resolve()
    budget = config.optimization
    if stop_after is not None and not 1 <= stop_after <= budget.max_rounds:
        raise ConfigurationError("stop_after must be within the configured round budget")
    output.mkdir(parents=True, exist_ok=True)
    with task_path(output, output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ConfigurationError("The trajectory already has an active process") from None
        return _run(config, agent, output, resume, stop_after, backend, evaluation_backend,
                    prepare, score, on_event or (lambda event: None))


def _run(config, agent, output, resume, stop_after, backend, evaluation_backend, prepare,
         score, event):
    key, inputs = _input_identity(config)
    path = task_path(output, output / "checkpoint.json")
    graph0 = MolecularGraph.from_smiles(config.complex.ligand_smiles)
    if path.exists():
        if not resume:
            raise ConfigurationError("Run already exists; specify --resume")
        state = json.loads(path.read_text(encoding="utf-8"))
        if (state.get("strategy", "optagent") != config.optimization.strategy
                or state.get("msa_mode", "provided") != config.msa_mode):
            raise StaleProposalError("Checkpoint strategy or MSA mode differs from current inputs")
        if state["context_id"] != key:
            raise StaleProposalError("Checkpoint context differs from current inputs")
        if state["status"] in {"converged", "budget_complete"}:
            return state
    elif resume:
        raise ConfigurationError("No checkpoint exists to resume")
    else:
        state = {"context_id": key, "input_hashes": inputs, "next_round": 1, "rounds": [],
                 "strategy": config.optimization.strategy, "msa_mode": config.msa_mode,
                 "no_accept": 0, "graph": graph0.to_dict(), "status": "ready",
                 "budget": asdict(config.optimization), "external_metrics": {
                     "af3": "not_evaluated", "vina": "not_evaluated"}}
        save_json(state, path)
    draft_history = None
    if config.optimization.strategy == "syncraft":
        from molcraft.agent.history import DraftHistory

        draft_history = DraftHistory(output, key)
    if backend is None:
        event({"phase": "weights"})
        backend = Boltz2Backend.from_paths(config.complex.model_paths, config.sampling_steps,
                                           config.trunk_passes)
    if evaluation_backend is None:
        evaluation_backend = Boltz2Backend(backend.bundle, config.evaluation_sampling_steps,
                                          config.evaluation_trunk_passes)
    event({"phase": "reference_features"})
    with prepare(config.complex) as initial:
        shape = np.shape(initial.features.get("msa"))
        if config.msa_mode == "single_sequence" and (len(shape) < 3 or shape[1] != 1):
            raise ConfigurationError("single_sequence mode requires exactly one query MSA row")
        reference_layout = initial.layout
        gradient_noise = backend.draw_noise(initial, config.seed)
        evaluation_noise = evaluation_backend.draw_noise(initial, config.seed)
    graph = MolecularGraph.from_dict(state["graph"])
    last_evidence = None
    stop = stop_after or config.optimization.max_rounds
    for index in range(state["next_round"], stop + 1):
        round_dir = task_path(output, output / f"round-{index:03d}")
        requests = task_path(output, round_dir / "requests")
        source = graph.fingerprint
        context = _digest((key, source))
        spec = replace(config.complex, ligand_smiles=graph.mapped_smiles)
        started = time.monotonic()
        source_graph, packet, result = graph, None, None
        state.pop("error", None)
        try:
            event({"phase": "prepare", "round": index, "smiles": graph.smiles})
            with prepare(spec) as prepared:
                if prepared.graph.mapped_smiles != graph.mapped_smiles:
                    raise ConfigurationError("Native preparation changed the current graph")
                prepared.graph = graph
                evidence_path = round_dir / "gradient.json"
                evidence = _restore_gradient(evidence_path, context, graph)
                if evidence is None and last_evidence and last_evidence.source_fingerprint == source:
                    evidence = last_evidence
                if evidence is None:
                    event({"phase": "gradient", "round": index,
                           "msa_shape": list(np.shape(prepared.features.get("msa")))})
                    noise = remap_noise(gradient_noise, reference_layout, prepared.layout)
                    evidence = value_and_grad(prepared, config.objective, config.gradient_mode,
                                              noise, config.seed, backend)
                save_json({**evidence.to_dict(), "context_id": context}, evidence_path)
                if not (output / "feature_profile.json").exists():
                    save_json({"strategy": config.optimization.strategy, "msa_mode": config.msa_mode,
                               "msa_shape": list(np.shape(prepared.features.get("msa"))),
                               "sequence_lengths": [len(p.sequence) for p in spec.proteins],
                               "homolog_msa_supplied": any(p.msa_path for p in spec.proteins)},
                              output / "feature_profile.json")
                last_evidence = evidence
                excluded = {smiles for row in state["rounds"] if row.get("source") == source
                            for smiles in row.get("attempted_smiles", row.get("evaluated_smiles", []))}
                event({"phase": "propose", "round": index})
                proposal_context = ({"candidate_budget": config.evaluation.candidate_budget,
                                     "history": draft_history.before(index)} if draft_history else {})
                packet = agent.propose(graph, evidence, config.policy,
                    samples=config.optimization.drafts, round_index=index, context_id=context,
                    journal_dir=requests, excluded=excluded, **proposal_context)
                save_packet(packet, round_dir / "proposals.json")
                if draft_history:
                    draft_history.record_round(index, source_graph, packet, requests=requests)
                selected = agent.select(packet, config.evaluation.candidate_budget,
                                        journal_dir=requests)
                event({"phase": "rescore", "round": index, "selected_ids": selected})
                noise = remap_noise(evaluation_noise, reference_layout, prepared.layout)
                result = score(prepared, packet, config, backend=evaluation_backend, noise=noise,
                               selected_ids=selected, context_id=context, export_dir=round_dir)
            graph = MolecularGraph.from_dict(result["selected_graph"])
            save_json(result, round_dir / "evaluation.json")
            if draft_history:
                draft_history.record_round(index, source_graph, packet, result, requests=requests)
                successful = result.get("best_successful")
                if successful and (not state.get("best_successful")
                                   or successful["loss"] < state["best_successful"]["loss"]):
                    state["best_successful"] = {**successful, "round": index}
                    save_json(state["best_successful"], output / "best_successful.json")
        except ProviderError as exc:
            if draft_history:
                draft_history.record_round(index, source_graph, packet, result,
                                           requests=requests, error=str(exc))
            state.update(status="provider_interrupted", error=str(exc))
            if exc.exhausted:
                state["rounds"].append({"round": index, "status": "provider_exhausted",
                                        "source": source, "seconds": time.monotonic() - started})
                state["next_round"] = index + 1
            save_json(state, path)
            event({"phase": "provider_interrupted", "round": index, "error": str(exc)})
            return state
        except Exception as exc:
            if draft_history:
                progress = round_dir / "evaluation_progress.json"
                partial = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else result
                draft_history.record_round(index, source_graph, packet, partial,
                                           requests=requests, error=f"{type(exc).__name__}: {exc}")
            state.update(status="evaluation_interrupted", error=f"{type(exc).__name__}: {exc}")
            save_json(state, path)
            raise
        improved = result["selected_id"] != "noop"
        state["no_accept"] = 0 if improved else state["no_accept"] + 1
        row = {"round": index, "source": source, "status": result["status"],
               "strategy": config.optimization.strategy, "msa_mode": config.msa_mode,
               "selected_id": result["selected_id"], "loss": result["selected_loss"],
               "baseline_loss": result["baseline"]["loss"],
               "evaluated_smiles": [packet.candidate(c["candidate_id"]).graph.smiles
                                    for c in result["candidates"]],
               "attempted_smiles": [packet.candidate(c["candidate_id"]).graph.smiles
                                    for c in [*result["candidates"], *result.get("unscored_candidates", [])]],
               "attempted_candidates": len(result["candidates"]) + len(result.get("unscored_candidates", [])),
               "rejected_candidates": len(result.get("unscored_candidates", [])),
               "evaluation_protocol": result.get("evaluation", {}).get("protocol"),
               "evaluated_candidates": len(result["candidates"]),
               "legal_candidates": len(packet.candidates) - 1,
               "seconds": time.monotonic() - started}
        if draft_history:
            row["proposal_pool"] = packet.diagnostics["pool"]
        state["rounds"].append(row)
        state.update(graph=graph.to_dict(), next_round=index + 1, status="running")
        if index >= config.optimization.min_rounds and state["no_accept"] >= config.optimization.patience:
            state["status"] = "converged"
        elif index == config.optimization.max_rounds:
            state["status"] = "budget_complete"
        elif index == stop:
            state["status"] = "window_complete"
        save_json(state, path)
        if index % 5 == 0:
            save_json(state, output / f"snapshot-{index:03d}.json")
        event({"phase": "round_complete", **row, "run_status": state["status"]})
        if state["status"] != "running":
            return state
    if state["next_round"] > config.optimization.max_rounds:
        state["status"] = "budget_complete"
        save_json(state, path)
    return state
