#!/usr/bin/env python3
"""Prepare new fixed-budget inputs from prior GBA/VDR configs; never launch a job."""

import argparse
from pathlib import Path
import yaml


def prepare_configs(gba, vdr, output, inventory_manifest=None, *, evidence_manifest=None, vdr_msa=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, source, rounds in (("GBA", gba, 17), ("VDR", vdr, 16)):
        destination = output / f"{name}.yaml"
        if destination.exists():
            raise ValueError("Use a fresh configuration directory; historical inputs are immutable")
        source = Path(source).resolve()
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        for protein in data["proteins"]:
            for field in ("msa_path", "template_path"):
                if protein.get(field):
                    protein[field] = str((source.parent / protein[field]).resolve())
        data["model_paths"] = {k: str((source.parent / p).resolve()) for k, p in data["model_paths"].items()}
        data.pop("synthesis", None)
        data["objective"] = {"protocol": "dbmol-denovo-v1", "affinity_weight": 1.,
                             "contact_weight": 2.5, "confidence_weight": 2.5, "geometry_weight": 0.}
        data.setdefault("run", {}).update(gradient_mode="through_structure", sampling_steps=25, trunk_passes=1)
        if name == "VDR" and vdr_msa:
            from molcraft.io.examples import validate_msa_input
            msa = Path(vdr_msa).resolve()
            if len(data["proteins"]) != 1:
                raise ValueError("VDR MSA override requires the established single-protein construct")
            validate_msa_input(msa, data["proteins"][0]["sequence"])
            data["proteins"][0]["msa_path"] = str(msa)
            data["run"]["msa_mode"] = "provided"
        data["evaluation"] = {"candidate_budget": 8, "sampling_steps": 200, "trunk_passes": 4,
                              "require_affinity_nonworsening": False, "coverage_mode": "off", "min_coverage": .5}
        data["optimization"] = {"strategy": "denovo", "max_rounds": rounds, "min_rounds": rounds,
                                "patience": 5, "drafts": 4}
        manifest = Path(evidence_manifest).resolve() if evidence_manifest else (
            Path.cwd() / "data/inventory/uspto_simpretro.sqlite.manifest.json")
        data["denovo"] = {"protocol": "denovo-motif-v2", "evidence_manifest": str(manifest),
                          "target_pool": 16, "batch_size": 4, "max_batches": 8,
                          "fallback_attempts": 64, "evidence_cards": 6, "gradient_diagnostics": False}
        data["denovo"]["max_api_retries"] = 3
        if inventory_manifest:
            data["denovo"]["inventory_manifest"] = str(Path(inventory_manifest).resolve())
        with destination.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
        paths.append(destination)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gba-config", required=True)
    parser.add_argument("--vdr-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evidence-manifest", help="Defaults to data/inventory/uspto_simpretro.sqlite.manifest.json")
    parser.add_argument("--inventory-manifest", help="Optional Enamine enhancement")
    parser.add_argument("--vdr-msa", help="Verified latest VDR A3M; overrides old single-sequence input")
    args = parser.parse_args()
    for path in prepare_configs(args.gba_config, args.vdr_config, args.output, args.inventory_manifest,
                                evidence_manifest=args.evidence_manifest, vdr_msa=args.vdr_msa):
        print(path)
