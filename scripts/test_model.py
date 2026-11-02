#!/usr/bin/env python3
"""Opt-in real-device acceptance. Never substitutes synthetic gradients or weights."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-budget", type=int, default=2)
    parser.add_argument("--skip-components", action="store_true")
    args = parser.parse_args()
    if args.candidate_budget < 1:
        parser.error("--candidate-budget must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "config": str(args.config.resolve()), "stages": [],
              "components_requested": not args.skip_components}

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8", newline="\n")

    def stage(name):
        report["stages"].append({"name": name, "elapsed_seconds": time.monotonic() - started})
        persist()
        print(name, flush=True)

    try:
        import jax
        import numpy as np
        from molcraft import prepare_complex, value_and_grad, build_proposals, evaluate_candidate
        from molcraft.backends.boltz2_jax.backend import Boltz2Backend
        from molcraft.io.config import load_config
        from molcraft.io.serialization import save_json, save_packet, load_packet, write_ligand_sdf

        report["versions"] = {name: importlib.metadata.version(name) for name in
                              ("molcraft-local", "jax", "jaxlib", "equinox", "boltz", "joltz", "rdkit")}
        report["devices"] = [{"platform": d.platform, "kind": d.device_kind} for d in jax.devices()]
        if not any(d.platform in ("gpu", "rocm", "cuda") for d in jax.devices()):
            raise RuntimeError("This acceptance command requires an actual GPU/DCU device")
        root = Path(__file__).resolve().parents[1]
        report["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        report["working_tree_clean"] = not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True).strip()
        config = load_config(args.config)
        report["sampling_steps"] = config.sampling_steps
        report["gradient_mode"] = config.gradient_mode
        stage("device_verified")

        with prepare_complex(config.complex) as prepared:
            report["source_graph"] = prepared.graph.to_dict()
            report["layout"] = {"atom_ids": prepared.graph.atom_ids,
                                "total_atoms": prepared.layout.total_atoms,
                                "total_tokens": prepared.layout.total_tokens,
                                "ligand_atom_indices": prepared.layout.ligand_atom_indices}
            stage("native_features_prepared_offline")
            backend = Boltz2Backend.from_paths(config.complex.model_paths,
                                               config.sampling_steps, config.trunk_passes)
            stage("real_structure_and_affinity_weights_loaded")
            noise = backend.draw_noise(prepared, config.seed)
            evidence = value_and_grad(prepared, config.objective, config.gradient_mode,
                                      noise=noise, noise_seed=config.seed, backend=backend)
            evidence.validate(prepared.graph)
            norms = evidence.metadata["gradient_norms"]
            if not all(np.isfinite(x) and x > 0 for x in norms.values()):
                raise RuntimeError(f"Combined objective has an inactive chemical channel: {norms}")
            save_json(evidence.to_dict(), output / "gradient_evidence.json")
            report["combined"] = {"metrics": evidence.metrics, "gradient_norms": norms}
            stage("three_chemical_gradient_channels_passed")

            if not args.skip_components:
                components = {}
                for name in ("affinity", "contact", "confidence", "geometry"):
                    if name == "geometry" and config.gradient_mode == "trunk_only":
                        components[name] = {"status": "disabled_by_trunk_only_contract"}
                        continue
                    objective = replace(config.objective, **{
                        f"{k}_weight": float(k == name) for k in
                        ("affinity", "contact", "confidence", "geometry")})
                    component = value_and_grad(prepared, objective, config.gradient_mode,
                                               noise=noise, noise_seed=config.seed, backend=backend)
                    component_norms = component.metadata["gradient_norms"]
                    if sum(component_norms.values()) <= 0:
                        raise RuntimeError(f"{name} has no input gradient")
                    components[name] = {"metrics": component.metrics,
                                        "gradient_norms": component_norms}
                    save_json(component.to_dict(), output / f"gradient_{name}.json")
                    report["components"] = components
                    stage(f"{name}_input_gradient_passed")

            baseline = backend.evaluate(prepared, config.objective, config.seed, noise)
            if not np.isfinite(baseline["coordinates"]).all():
                raise RuntimeError("Nonfinite source coordinates")
            np.testing.assert_allclose(baseline["loss"], evidence.metrics["loss"], rtol=1e-4, atol=1e-4)
            np.savez_compressed(output / "source_coordinates.npz", coordinates=baseline["coordinates"])
            write_ligand_sdf(prepared.graph, baseline["ligand_coordinates"], output / "source.sdf")
            prepared.native_writer(baseline["coordinates"], output / "source.cif")
            report["baseline"] = {"loss": baseline["loss"], "metrics": baseline["metrics"]}
            stage("source_forward_and_native_export_passed")

            packet = build_proposals(prepared.graph, evidence, config.policy)
            save_packet(packet, output / "proposals.json")
            packet = load_packet(output / "proposals.json")
            candidates = sorted((c for c in packet.candidates if c.operations),
                                key=lambda c: (-c.probability, c.candidate_id))[:args.candidate_budget]
            if not candidates:
                raise RuntimeError("No legal edited candidate was produced")
            report["candidates"] = []
            for candidate in candidates:
                result = evaluate_candidate(prepared, candidate, noise=noise,
                    objective=config.objective, noise_seed=config.seed, backend=backend,
                    export_dir=output / candidate.candidate_id)
                if not np.isfinite(result["coordinates"]).all():
                    raise RuntimeError("Nonfinite candidate coordinates")
                np.savez_compressed(output / candidate.candidate_id / "coordinates.npz",
                                    coordinates=result["coordinates"])
                measured = {k: v for k, v in result.items() if k not in
                            ("coordinates", "ligand_coordinates")}
                measured["operations"] = [op.to_dict() for op in candidate.operations]
                measured["predicted_change"] = candidate.predicted_change
                measured["actual_change"] = result["loss"] - baseline["loss"]
                save_json(measured, output / candidate.candidate_id / "evaluation.json")
                report["candidates"].append(measured)
                stage(f"{candidate.candidate_id}_rebuilt_rescored_exported")
            report["best_actual_change"] = min(c["actual_change"] for c in report["candidates"])
            report["interpretation"] = (
                "Runtime acceptance only; this toy target is not a biological efficacy benchmark. "
                "Directional derivatives do not guarantee a lower discrete-candidate loss.")
        report["status"] = "passed"
        stage("acceptance_passed")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        persist()
        raise


if __name__ == "__main__":
    main()
