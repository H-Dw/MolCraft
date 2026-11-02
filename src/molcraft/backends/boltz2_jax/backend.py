"""Actual JAX evaluation and input differentiation, with immutable chemistry state."""

from dataclasses import dataclass
from functools import lru_cache
import numpy as np

from molcraft.chemistry.encoding import encode_graph, ChemistryInputs
from molcraft.errors import ConfigurationError, GradientError, MappingError
from molcraft.types import GradientEvidence


def _model_objective(inputs, bundle, features, layout, vocabulary, sampler, noise,
                     objective, mode, passes, key):
    from molcraft.losses.functions import evaluate_objective
    from .injection import inject_chemistry
    from .forward import forward

    injected = inject_chemistry(features, inputs, layout, vocabulary)
    output = forward(bundle, injected, layout, sampler, noise, mode, passes, key)
    loss, metrics = evaluate_objective(output, inputs, layout, objective)
    return loss, (metrics, output)


@lru_cache(maxsize=1)
def _compiled_functions():
    import equinox as eqx

    # Model weights are explicit PyTree arguments, not giant constants captured
    # in a newly compiled closure for every molecule or candidate.
    return (eqx.filter_jit(eqx.filter_value_and_grad(_model_objective, has_aux=True)),
            eqx.filter_jit(_model_objective))


@dataclass
class Boltz2Backend:
    bundle: object
    sampling_steps: int = 25
    trunk_passes: int = 1

    def __post_init__(self):
        if self.sampling_steps < 1 or self.trunk_passes < 1:
            raise ConfigurationError("Sampling steps and trunk passes must be positive")
        if not getattr(self.bundle.structure, "bond_type_feature", False):
            raise ConfigurationError("A bond-type-enabled model is required for all three chemical channels")

    @classmethod
    def from_paths(cls, paths, sampling_steps=25, trunk_passes=1):
        from .weights import load_weights

        return cls(load_weights(paths), sampling_steps, trunk_passes)

    def draw_noise(self, prepared, seed):
        import jax
        from molcraft.sampling.core import Noise

        return Noise.draw(jax.random.key(seed), num_steps=self.sampling_steps,
                          num_atoms=prepared.layout.total_atoms)

    def _arguments(self, prepared, objective, mode, noise, seed):
        import jax
        import jax.numpy as jnp
        from molcraft.sampling.boltz2 import boltz2_sampler

        objective.validate(mode)
        features = jax.tree.map(jnp.asarray, prepared.features)
        vocabulary = prepared.native_data["vocabulary"]
        vocabulary.validate()
        sampler = boltz2_sampler(self.bundle.structure.structure_module, self.sampling_steps)

        return (self.bundle, features, prepared.layout, vocabulary, sampler, noise,
                objective, mode, self.trunk_passes, jax.random.key(seed))

    def value_and_grad(self, prepared, objective, gradient_mode="through_structure", noise_seed=0, noise=None):
        import jax
        import jax.numpy as jnp

        noise = self.draw_noise(prepared, noise_seed) if noise is None else noise
        inputs = ChemistryInputs(*(jnp.asarray(x) for x in encode_graph(prepared.graph)))
        arguments = self._arguments(prepared, objective, gradient_mode, noise, noise_seed)
        compute_grad, _ = _compiled_functions()
        (value, (metrics, _output)), gradient = compute_grad(inputs, *arguments)
        if not np.isfinite(np.asarray(value)).all():
            raise GradientError("Model returned a non-finite objective")
        scalar_metrics = {k: float(np.asarray(v)) for k, v in metrics.items()}
        result = GradientEvidence(prepared.graph.fingerprint, prepared.graph.atom_ids,
            np.asarray(gradient.atoms), np.asarray(gradient.bonds), np.asarray(gradient.charges),
            scalar_metrics, gradient_mode, noise_seed,
            {"sampling_steps": self.sampling_steps, "trunk_passes": self.trunk_passes,
             "coordinate_confidence_bins": "native_hard_bins",
             "affinity_mw_correction": False,
             "gradient_norms": {name: float(np.linalg.norm(np.asarray(g))) for name, g in
                                zip(("atom", "bond", "charge"), gradient)}})
        result.validate(prepared.graph)
        jax.block_until_ready(value)
        return result

    def evaluate(self, prepared, objective, noise_seed=0, noise=None):
        import jax.numpy as jnp

        noise = self.draw_noise(prepared, noise_seed) if noise is None else noise
        inputs = ChemistryInputs(*(jnp.asarray(x) for x in encode_graph(prepared.graph)))
        arguments = self._arguments(prepared, objective, "through_structure", noise, noise_seed)
        _, compute = _compiled_functions()
        value, (metrics, output) = compute(inputs, *arguments)
        if not np.isfinite(np.asarray(value)).all():
            raise GradientError("Candidate model evaluation returned a non-finite objective")
        return {"loss": float(value), "metrics": {k: float(v) for k, v in metrics.items()},
                "coordinates": np.asarray(output.coordinates),
                "ligand_coordinates": prepared.layout.ligand_coordinates(output.coordinates)[0]}


def remap_noise(noise, source_layout, target_layout):
    """Common random numbers follow stable atom IDs after native atom reordering."""
    import equinox as eqx
    import jax.numpy as jnp

    if (source_layout.atom_ids != target_layout.atom_ids
            or source_layout.total_atoms != target_layout.total_atoms
            or source_layout.protein_atom_indices != target_layout.protein_atom_indices):
        raise MappingError("Noise replay requires the same fixed context and atom identities")
    permutation = np.arange(source_layout.total_atoms)
    for old, new in zip(source_layout.ligand_atom_indices, target_layout.ligand_atom_indices):
        permutation[new] = old
    if len(set(permutation.tolist())) != len(permutation):
        raise MappingError("Noise remapping is not a bijection")
    idx = jnp.asarray(permutation)
    return eqx.tree_at(lambda n: (n.init, n.churn), noise,
                       (noise.init[idx], noise.churn[:, idx]))
