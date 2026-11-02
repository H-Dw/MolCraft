"""CLI with a weight-free CPU demo and explicit real-backend commands."""

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

from molcraft.errors import MolCraftError


def _doctor(_args):
    packages = {}
    for name in ("numpy", "rdkit", "jax", "equinox", "torch", "boltz", "joltz"):
        installed = importlib.util.find_spec(name) is not None
        try:
            version = importlib.metadata.version(name) if installed else None
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        packages[name] = {"installed": installed, "version": version}
    print(json.dumps({"python": sys.version.split()[0], "packages": packages,
                      "model_weights_loaded": False, "gpu_probed": False,
                      "msa_service_called": False}, indent=2))


def _demo(args):
    import numpy as np
    from molcraft.chemistry.graph import MolecularGraph
    from molcraft.types import GradientEvidence
    from molcraft.guidance.proposals import ProposalPolicy, build_proposals
    from molcraft.agent.selectors import GreedySelector
    from molcraft.io.serialization import save_json, save_packet

    graph = MolecularGraph.from_smiles("CCC")
    atom_grad = np.zeros((3, 9))
    # Fictional local derivatives for the central carbon, not Boltz predictions.
    atom_grad[1, :4] = [0.2, -0.4, -0.1, -0.8]
    evidence = GradientEvidence(graph.fingerprint, graph.atom_ids, atom_grad,
        np.zeros((3, 5)), np.zeros(3), gradient_mode="synthetic",
        metadata={"description": "Synthetic teaching example; no model was run"})
    # Keep this teaching example to single edits. The library's default joint
    # policy is exercised independently, including atom+charge cooperation.
    packet = build_proposals(graph, evidence, ProposalPolicy(temperature=0.3, max_bundle_size=1))
    output = Path(args.output)
    save_packet(packet, output / "proposals.json")
    save_json(evidence.to_dict(), output / "gradient_evidence.json")
    selected = GreedySelector().select(packet)[0]
    summary = {"status": packet.status, "gradient_source": "synthetic_not_boltz",
               "policy": "single_edit_demonstration",
               "candidate_count": len(packet.candidates), "selected_id": selected,
               "selected_smiles": packet.candidate(selected).graph.smiles,
               "source_fingerprint": graph.fingerprint}
    save_json(summary, output / "summary.json")
    print(json.dumps(summary, indent=2))


def _propose(args):
    import numpy as np
    from molcraft import prepare_complex, value_and_grad, build_proposals
    from molcraft.backends.boltz2_jax.backend import Boltz2Backend
    from molcraft.io.config import load_config
    from molcraft.io.serialization import save_json, save_packet

    config = load_config(args.config)
    with prepare_complex(config.complex) as prepared:
        backend = Boltz2Backend.from_paths(config.complex.model_paths,
                                           config.sampling_steps, config.trunk_passes)
        evidence = value_and_grad(prepared, config.objective, config.gradient_mode,
                                  noise_seed=config.seed, backend=backend)
        packet = build_proposals(prepared.graph, evidence, config.policy)
        output = Path(args.output)
        save_packet(packet, output / "proposals.json")
        save_json(evidence.to_dict(), output / "gradient_evidence.json")
        np.savez_compressed(output / "gradients.npz", atoms=evidence.atom_grad,
                            bonds=evidence.bond_grad, charges=evidence.charge_grad)
        print(json.dumps({"status": packet.status, "packet": str(output / "proposals.json")}, indent=2))


def _evaluate(args):
    import numpy as np
    from molcraft import prepare_complex, evaluate_candidate
    from molcraft.backends.boltz2_jax.backend import Boltz2Backend
    from molcraft.io.config import load_config
    from molcraft.io.serialization import load_packet, save_json

    config = load_config(args.config)
    packet = load_packet(args.packet)
    with prepare_complex(config.complex) as prepared:
        backend = Boltz2Backend.from_paths(config.complex.model_paths,
                                           config.sampling_steps, config.trunk_passes)
        result = evaluate_candidate(prepared, packet.candidate(args.candidate),
            objective=config.objective, noise_seed=config.seed, backend=backend,
            export_dir=Path(args.output))
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "coordinates.npz", coordinates=result.pop("coordinates"),
                            ligand_coordinates=result.pop("ligand_coordinates"))
        save_json(result, output / "evaluation.json")
        print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="molcraft")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect packages without probing GPU or loading weights")
    doctor.set_defaults(func=_doctor)
    demo = commands.add_parser("demo", help="Synthetic-gradient CPU example; not a model prediction")
    demo.add_argument("--output", default="outputs/demo")
    demo.set_defaults(func=_demo)
    propose = commands.add_parser("propose", help="Run the real backend with explicit local assets")
    propose.add_argument("--config", required=True)
    propose.add_argument("--output", required=True)
    propose.set_defaults(func=_propose)
    evaluate = commands.add_parser("evaluate", help="Rebuild and rescore a legal discrete candidate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--packet", required=True)
    evaluate.add_argument("--candidate", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(func=_evaluate)
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except (MolCraftError, OSError, ImportError) as exc:
        print(f"molcraft: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
