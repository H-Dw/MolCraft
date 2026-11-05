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


def _evidence_preflight(config):
    if config.optimization.strategy == "denovo":
        from molcraft.chemistry.inventory import EvidenceIndex
        index = EvidenceIndex.from_policy(config.denovo)
        index.close()


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
    from contextlib import ExitStack
    import numpy as np
    from molcraft import prepare_complex, value_and_grad, build_proposals
    from molcraft.backends.boltz2_jax.backend import Boltz2Backend
    from molcraft.io.config import load_config
    from molcraft.io.serialization import save_json, save_packet

    config = load_config(args.config)
    _evidence_preflight(config)
    with prepare_complex(config.complex) as prepared, ExitStack() as stack:
        backend = Boltz2Backend.from_paths(config.complex.model_paths,
                                           config.sampling_steps, config.trunk_passes)
        evidence = value_and_grad(prepared, config.objective, config.gradient_mode,
                                  noise_seed=config.seed, backend=backend)
        output = Path(args.output)
        if config.optimization.strategy in {"syncraft", "denovo"}:
            from molcraft.agent.argmax import create_agent
            from molcraft.optimization import _input_identity, _digest

            agent, _ = create_agent(config.optimization.strategy, synthesis_policy=config.synthesis,
                                    denovo_policy=config.denovo)
            key, _ = _input_identity(config)
            from molcraft.agent.history import DraftHistory

            history = DraftHistory(output, key)
            extra = {}
            if config.optimization.strategy == "denovo":
                from molcraft.api import PreparedCandidatePool
                pool = stack.enter_context(PreparedCandidatePool(prepared))
                extra["prepare_candidate"] = pool.prepare_graph
            packet = agent.propose(prepared.graph, evidence, config.policy,
                samples=config.optimization.drafts, round_index=1,
                context_id=_digest((key, prepared.graph.fingerprint)), journal_dir=output/"requests",
                candidate_budget=config.evaluation.candidate_budget, history=history.before(1), **extra)
            history.record_round(1, prepared.graph, packet, requests=output/"requests")
        else:
            packet = build_proposals(prepared.graph, evidence, config.policy)
        save_packet(packet, output / "proposals.json")
        save_json(evidence.to_dict(), output / "gradient_evidence.json")
        np.savez_compressed(output / "gradients.npz", atoms=evidence.atom_grad,
                            bonds=evidence.bond_grad, charges=evidence.charge_grad)
        print(json.dumps({"status": packet.status, "packet": str(output / "proposals.json"),
                          **({"pool": packet.diagnostics["pool"]} if "pool" in packet.diagnostics else {})}, indent=2))


def _evaluate(args):
    import numpy as np
    from molcraft import prepare_complex, evaluate_candidate
    from molcraft.backends.boltz2_jax.backend import Boltz2Backend
    from molcraft.io.config import load_config
    from molcraft.io.serialization import load_packet, save_json

    config = load_config(args.config)
    _evidence_preflight(config)
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
    from contextlib import ExitStack
    from molcraft import prepare_complex
    from molcraft.evaluation import evaluate_packet
    from molcraft.io.config import load_config
    from molcraft.io.serialization import load_packet, save_json

    config = load_config(args.config)
    _evidence_preflight(config)
    packet = load_packet(args.packet)
    with prepare_complex(config.complex) as prepared, ExitStack() as stack:
        kwargs = {}
        if config.optimization.strategy in {"syncraft", "denovo"}:
            from molcraft.agent.syncraft import SynCraftAgent
            from molcraft.optimization import _input_identity, _digest

            key, _ = _input_identity(config)
            kwargs = {"context_id": _digest((key, prepared.graph.fingerprint))}
            if config.optimization.strategy == "syncraft":
                kwargs["selected_ids"] = SynCraftAgent(None, policy=config.synthesis).select(
                    packet, config.evaluation.candidate_budget)
            if config.optimization.strategy == "denovo":
                from molcraft.agent.denovo import DeNovoAgent
                from molcraft.api import PreparedCandidatePool
                from molcraft.errors import ChemistryError
                pool = stack.enter_context(PreparedCandidatePool(prepared))
                selected = []
                for cid in DeNovoAgent.select(packet, len(packet.candidates)):
                    try:
                        pool.prepare_graph(packet.candidate(cid).graph)
                        selected.append(cid)
                    except ChemistryError:
                        continue
                    if len(selected) == config.evaluation.candidate_budget:
                        break
                kwargs.update(selected_ids=selected, preparation_cache=pool)
        result = evaluate_packet(prepared, packet, config, export_dir=Path(args.output), **kwargs)
        if config.optimization.strategy in {"syncraft", "denovo"}:
            from molcraft.agent.history import DraftHistory

            proposal_root = Path(args.packet).parent
            DraftHistory(proposal_root, key).record_round(1, prepared.graph, packet, result,
                                                        requests=proposal_root/"requests")
        save_json(result, Path(args.output) / "evaluation.json")
        save_json(result["selected_graph"], Path(args.output) / "next_graph.json")
        print(json.dumps({"status": result["status"], "selected_id": result["selected_id"],
                          "selected_loss": result["selected_loss"]}, indent=2))


def _prepare_example(args):
    from molcraft.io.examples import prepare_example

    result = prepare_example(args.case, args.output, args.model_dir, msa_source=args.msa,
                             work_root=args.work_root, msa_mode=args.msa_mode, strategy=args.strategy,
                             inventory_manifest=args.inventory_manifest, evidence_manifest=args.evidence_manifest)
    print(json.dumps(result, indent=2))


def _optimize(args):
    from dataclasses import replace

    from molcraft.errors import ConfigurationError
    from molcraft.io.config import load_config
    from molcraft.io.examples import validate_msa_input, task_path
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
    _evidence_preflight(config)
    config = replace(config, complex=replace(config.complex, cache_dir=temp))
    for protein in config.complex.proteins:
        if config.msa_mode == "single_sequence":
            continue
        if protein.msa_path is None:
            raise ConfigurationError("Optimization requires a verified local AF3 A3M")
        msa = task_path(args.work_root, protein.msa_path)
        task_path(args.work_root, msa.with_name("msa-provenance.json"))
        validate_msa_input(msa, protein.sequence)

    import jax
    from molcraft.agent.argmax import create_agent
    from molcraft.agent.deepseek import ProviderError
    from molcraft.optimization import optimize

    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise ConfigurationError("The optimization CLI requires an actual GPU/DCU device")
    agent, provider_metadata = create_agent(config.optimization.strategy, synthesis_policy=config.synthesis,
                                           denovo_policy=config.denovo)
    from molcraft.losses.provenance import weight_audit
    save_json(weight_audit(config), output / "weight_audit.json")
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


def _build_inventory(args):
    from molcraft.chemistry.inventory import build_index
    result = build_index(args.stock, args.output, release=args.release, source_url=args.source_url,
                         templates=args.templates, template_source_url=args.template_source_url,
                         expected_sha256=args.expected_sha256, expected_raw_count=args.expected_raw_count)
    print(json.dumps(result, indent=2))


def _check_inventory(args):
    from molcraft.chemistry.inventory import InventoryIndex
    index = InventoryIndex(args.manifest)
    print(json.dumps(index.manifest, indent=2))
    index.close()


def _build_template_index(args):
    from molcraft.chemistry.inventory import build_index
    result = build_index(None, args.output, release=args.release, source_url=args.source_url,
                         templates=args.templates, template_source_url=args.source_url,
                         expected_template_sha256=args.expected_sha256)
    print(json.dumps(result, indent=2))


def _check_evidence(args):
    from molcraft.chemistry.inventory import EvidenceIndex
    index = EvidenceIndex(args.manifest, inventory_manifest=args.inventory_manifest)
    print(json.dumps(index.resource_metadata(), indent=2))
    index.close()


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
    example.add_argument("--strategy", choices=("syncraft", "optagent", "argmax", "denovo"), default="syncraft")
    example.add_argument("--evidence-manifest", help="Product-pattern index manifest for denovo")
    example.add_argument("--inventory-manifest", help="Optional real Enamine stock enhancement")
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
    inventory = commands.add_parser("build-inventory", help="Index a real Enamine stock file; never cases")
    inventory.add_argument("--stock", required=True)
    inventory.add_argument("--output", required=True)
    inventory.add_argument("--release", required=True)
    inventory.add_argument("--source-url", required=True)
    inventory.add_argument("--expected-sha256")
    inventory.add_argument("--expected-raw-count", type=int)
    inventory.add_argument("--templates")
    inventory.add_argument("--template-source-url")
    inventory.set_defaults(func=_build_inventory)
    check = commands.add_parser("check-inventory", help="Verify inventory provenance and content hashes")
    check.add_argument("--manifest", required=True)
    check.set_defaults(func=_check_inventory)
    templates = commands.add_parser("build-template-index", help="Index USPTO/SimpRetro product patterns; no stock required")
    templates.add_argument("--templates", required=True)
    templates.add_argument("--output", required=True)
    templates.add_argument("--release", required=True)
    templates.add_argument("--source-url", required=True)
    templates.add_argument("--expected-sha256")
    templates.set_defaults(func=_build_template_index)
    evidence = commands.add_parser("check-evidence", help="Verify product-pattern evidence and optional Enamine enhancement")
    evidence.add_argument("--manifest", required=True)
    evidence.add_argument("--inventory-manifest")
    evidence.set_defaults(func=_check_evidence)
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except (MolCraftError, OSError, ImportError) as exc:
        print(f"molcraft: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
