"""High-level entry points; model construction is explicit or cached per prepared complex."""

from dataclasses import replace

from molcraft.chemistry.edits import apply_edits
from molcraft.errors import ConfigurationError, StaleProposalError
from molcraft.losses.spec import ObjectiveSpec
from molcraft.guidance.proposals import build_proposals


def prepare_complex(spec):
    from molcraft.backends.boltz2_jax.features import prepare_complex as prepare

    return prepare(spec)


def _backend(prepared, backend):
    if backend is not None:
        return backend
    if not hasattr(prepared, "_backend"):
        from molcraft.backends.boltz2_jax.backend import Boltz2Backend

        prepared._backend = Boltz2Backend.from_paths(prepared.spec.model_paths)
    return prepared._backend


def value_and_grad(prepared, objective=None, gradient_mode="through_structure", noise=None,
                   noise_seed=0, backend=None):
    objective = objective or ObjectiveSpec()
    objective.validate(gradient_mode)
    return _backend(prepared, backend).value_and_grad(prepared, objective, gradient_mode,
                                                      noise_seed=noise_seed, noise=noise)


def evaluate_candidate(prepared, candidate, noise=None, objective=None, noise_seed=0, backend=None,
                       export_dir=None):
    if candidate.source_fingerprint != prepared.graph.fingerprint:
        raise StaleProposalError("Candidate belongs to a different source molecule")
    actual = apply_edits(prepared.graph, candidate.operations)
    if actual.fingerprint != candidate.graph.fingerprint:
        raise ConfigurationError("Candidate graph disagrees with its declared operations")
    model = _backend(prepared, backend)
    if noise is None:
        noise = model.draw_noise(prepared, noise_seed)
    # mapped SMILES is accompanied by original hydrogen policy when validating;
    # native preparation needs the actual chemistry, not an optimization target.
    spec = replace(prepared.spec, ligand_smiles=actual.mapped_smiles)
    from molcraft.backends.boltz2_jax.backend import remap_noise

    with prepare_complex(spec) as updated:
        if updated.graph.mapped_smiles != actual.mapped_smiles:
            raise ConfigurationError("Native candidate preparation changed chemistry")
        updated.graph = actual  # preserve original explicit/implicit hydrogen editing policy
        replay = remap_noise(noise, prepared.layout, updated.layout)
        result = model.evaluate(updated, objective or ObjectiveSpec(), noise_seed, replay)
        result["source_fingerprint"] = prepared.graph.fingerprint
        result["candidate_fingerprint"] = actual.fingerprint
        result["candidate_id"] = candidate.candidate_id
        if export_dir is not None:
            result["artifacts"] = export_prediction(updated, actual, result, export_dir)
        return result


def export_prediction(prepared, graph, result, export_dir):
    from pathlib import Path
    from molcraft.io.serialization import write_ligand_sdf

    output = Path(export_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_ligand_sdf(graph, result["ligand_coordinates"], output / "ligand.sdf")
    prepared.native_writer(result["coordinates"], output / "complex.cif")
    return {"ligand": str(output / "ligand.sdf"), "complex": str(output / "complex.cif")}


__all__ = ["prepare_complex", "value_and_grad", "build_proposals", "evaluate_candidate"]
