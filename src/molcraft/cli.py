"""CLI with a weight-free CPU demo and explicit real-backend commands."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

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
        output = Path(args.output)
        if config.optimization.strategy == "syncraft":
            from molcraft.agent.argmax import create_agent
            from molcraft.optimization import _input_identity, _digest

            agent, _ = create_agent("syncraft", synthesis_policy=config.synthesis)
            key, _ = _input_identity(config)
            packet = agent.propose(prepared.graph, evidence, config.policy,
                samples=config.optimization.drafts, round_index=1,
                context_id=_digest((key, prepared.graph.fingerprint)), journal_dir=output/"requests")
        else:
            packet = build_proposals(prepared.graph, evidence, config.policy)
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
                                           config.evaluation_sampling_steps, config.evaluation_trunk_passes)
        result = evaluate_candidate(prepared, packet.candidate(args.candidate),
            objective=config.objective, noise_seed=config.seed, backend=backend,
            export_dir=Path(args.output))
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "coordinates.npz", coordinates=result.pop("coordinates"),
                            ligand_coordinates=result.pop("ligand_coordinates"))
        save_json(result, output / "evaluation.json")
        print(json.dumps(result, indent=2))


def _evaluate_set(args):
    from molcraft import prepare_complex
    from molcraft.evaluation import evaluate_packet
    from molcraft.io.config import load_config
    from molcraft.io.serialization import load_packet, save_json

    config = load_config(args.config)
    packet = load_packet(args.packet)
    with prepare_complex(config.complex) as prepared:
        kwargs = {}
        if config.optimization.strategy == "syncraft":
            from molcraft.agent.syncraft import SynCraftAgent
            from molcraft.optimization import _input_identity, _digest

            key, _ = _input_identity(config)
            kwargs = {"selected_ids": SynCraftAgent(None, policy=config.synthesis).select(
                        packet, config.evaluation.candidate_budget),
                      "context_id": _digest((key, prepared.graph.fingerprint))}
        result = evaluate_packet(prepared, packet, config, export_dir=Path(args.output), **kwargs)
        save_json(result, Path(args.output) / "evaluation.json")
        save_json(result["selected_graph"], Path(args.output) / "next_graph.json")
        print(json.dumps({"status": result["status"], "selected_id": result["selected_id"],
                          "selected_loss": result["selected_loss"]}, indent=2))


def _prepare_example(args):
    from molcraft.io.examples import prepare_example

    result = prepare_example(args.case, args.output, args.model_dir, msa_source=args.msa,
                             work_root=args.work_root, msa_mode=args.msa_mode, strategy=args.strategy)
    print(json.dumps(result, indent=2))


def _optimize(args):
    from dataclasses import replace
    import hashlib

    from molcraft.errors import ConfigurationError
    from molcraft.io.config import load_config
    from molcraft.io.examples import a3m_records, task_path
    from molcraft.io.serialization import save_json

    output = task_path(args.work_root, args.output)
    cache = task_path(args.work_root, output / "cache")
    temp = task_path(args.work_root, output / "tmp")
    for directory in (cache, temp):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[name] = str(temp)
    for name in ("XDG_CACHE_HOME", "JAX_COMPILATION_CACHE_DIR", "HF_HOME", "PIP_CACHE_DIR"):
        os.environ[name] = str(cache)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    config = load_config(task_path(args.work_root, args.config))
    config = replace(config, complex=replace(config.complex, cache_dir=temp))
    for protein in config.complex.proteins:
        if config.msa_mode == "single_sequence":
            continue
        if protein.msa_path is None:
            raise ConfigurationError("Optimization requires a verified local AF3 A3M")
        msa = task_path(args.work_root, protein.msa_path)
        raw = msa.read_bytes()
        records = a3m_records(raw, protein.sequence)
        if not 2 <= len(records) <= 1024:
            raise ConfigurationError("Optimization requires 2 to 1024 aligned AF3 MSA rows")
        provenance = task_path(args.work_root, msa.with_name("msa-provenance.json"))
        metadata = json.loads(provenance.read_text(encoding="utf-8"))
        if (metadata.get("msa_sha256") != hashlib.sha256(raw).hexdigest()
                or metadata.get("input_depth") != len(records)
                or metadata.get("sequence_sha256") != hashlib.sha256(protein.sequence.encode()).hexdigest()):
            raise ConfigurationError("AF3 MSA differs from the verified example import")

    import jax
    from molcraft.agent.argmax import create_agent
    from molcraft.agent.deepseek import ProviderError
    from molcraft.optimization import optimize

    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise ConfigurationError("The optimization CLI requires an actual GPU/DCU device")
    agent, provider_metadata = create_agent(config.optimization.strategy, synthesis_policy=config.synthesis)
    import subprocess
    git = subprocess.run(["git", "-C", str(Path(args.work_root).resolve()), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    save_json({"device": str(devices[0]), "kind": devices[0].device_kind,
               "jax": jax.__version__, "commit": git.stdout.strip() or None,
               "strategy": config.optimization.strategy, "msa_mode": config.msa_mode,
               **provider_metadata}, output / "runtime.json")

    def event(row):
        print(json.dumps({"unix_time": time.time(), **row}), flush=True)

    state = optimize(config, agent, output, resume=args.resume,
                     stop_after=args.stop_after, on_event=event)
    print(json.dumps({"status": state["status"], "next_round": state["next_round"],
                      "checkpoint": str(output / "checkpoint.json")}), flush=True)
    if state["status"] == "provider_interrupted":
        raise ProviderError(state["error"])


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
    evaluate_set = commands.add_parser("evaluate-set", help="Rescore a bounded candidate set and recommend one graph")
    evaluate_set.add_argument("--config", required=True)
    evaluate_set.add_argument("--packet", required=True)
    evaluate_set.add_argument("--output", required=True)
    evaluate_set.set_defaults(func=_evaluate_set)
    example = commands.add_parser("prepare-example", help="Prepare a DBMol example with local AF3 MSA")
    example.add_argument("--case", default="vdr-3a2i", choices=("vdr-3a2i", "gba-2v3d"))
    example.add_argument("--msa")
    example.add_argument("--msa-mode", choices=("provided", "single_sequence"), default="provided")
    example.add_argument("--strategy", choices=("syncraft", "optagent", "argmax"), default="syncraft")
    example.add_argument("--output", required=True)
    example.add_argument("--model-dir", required=True)
    example.add_argument("--work-root", default=".")
    example.set_defaults(func=_prepare_example)
    search = commands.add_parser("optimize", help="Run a restartable OptAgent trajectory on GPU/DCU")
    search.add_argument("--config", required=True)
    search.add_argument("--output", required=True)
    search.add_argument("--work-root", default=".")
    search.add_argument("--stop-after", type=int)
    search.add_argument("--resume", action="store_true")
    search.set_defaults(func=_optimize)
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except (MolCraftError, OSError, ImportError) as exc:
        print(f"molcraft: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
