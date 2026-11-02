import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from molcraft.sampling.core import Noise, Sampler, integrate, no_alignment, no_centering
from molcraft.losses.functions import contact_loss, aggregate_affinity, geometry_loss
from molcraft.losses.spec import ObjectiveSpec
from molcraft.chemistry.mapping import AtomLayout
from molcraft.errors import ConfigurationError


class ToyDenoiser(eqx.Module):
    chemistry: jax.Array

    def __call__(self, x, sigma):
        return x / (1 + sigma) + self.chemistry[0] * jnp.array([1.0, 0.2, -0.1])


def test_diffusion_coordinate_gradient_and_fixed_noise_finite_difference():
    sampler = Sampler(jnp.array([2.0, 1.0, 0.0]), gamma_0=0, step_scale=1,
                      augment=False, center=no_centering, align=no_alignment)
    noise = Noise.draw(jax.random.key(4), num_steps=2, num_atoms=3)

    def loss(chemistry):
        xyz = integrate(ToyDenoiser(chemistry), jnp.ones(3), sampler, noise)
        return jnp.sum(xyz * xyz)

    x = jnp.array([0.3])
    gradient = jax.grad(loss)(x)
    finite = (loss(x + 1e-3) - loss(x - 1e-3)) / 2e-3
    assert np.isfinite(gradient).all() and abs(float(gradient[0])) > 1e-4
    np.testing.assert_allclose(gradient[0], finite, rtol=1e-3, atol=1e-3)
    assert jax.devices()[0].platform == "cpu"


def test_contact_uses_8_angstrom_probability_mass():
    logits = jnp.log(jnp.array([[[0.1, 0.2, 0.7]]]))
    actual = contact_loss(logits, jnp.array([3.0, 7.0, 12.0]), 8.0)
    np.testing.assert_allclose(actual, -np.log(0.3), rtol=1e-6)
    assert np.linalg.norm(np.asarray(jax.grad(lambda x: contact_loss(x, jnp.array([3., 7., 12.])))(logits))) > 0


def test_affinity_ensemble_averages_probabilities_not_logits():
    heads = [{"affinity_pred_value": jnp.array(1.), "affinity_logits_binary": jnp.array(-1.)},
             {"affinity_pred_value": jnp.array(3.), "affinity_logits_binary": jnp.array(3.)}]
    result = aggregate_affinity(heads)
    assert float(result["affinity_pred_value"]) == 2
    expected = (float(jax.nn.sigmoid(-1.)) + float(jax.nn.sigmoid(3.))) / 2
    np.testing.assert_allclose(result["affinity_probability_binary"], expected, rtol=1e-6)
    assert not np.isclose(float(result["affinity_logits_binary"]), 1)


def test_raw_ligand_geometry_provides_coordinate_gradient():
    layout = AtomLayout((1, 2), (1, 3), (1, 3), (0, 2), (0,), ((0, 2),), 4, 4, (1.7, 1.7))
    xyz = jnp.array([[0., 0., 0.], [8., 0., 0.], [0., 1., 0.], [9., 0., 0.]])
    weights = jnp.zeros((2, 9)).at[:, 0].set(1)
    gradient = jax.grad(lambda x: geometry_loss(x, weights, layout, ObjectiveSpec())[0])(xyz)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0


def test_stop_gradient_and_hard_binning_are_not_claimed_as_smooth_paths():
    coordinates = jnp.array([2.2, 3.2])
    def hard(x):
        indices = jnp.sum(x[:, None] > jnp.array([2., 3.]), axis=-1)
        return jnp.sum(jnp.array([0., 1., 4.])[indices])
    np.testing.assert_array_equal(jax.grad(hard)(coordinates), 0)
    np.testing.assert_array_equal(jax.grad(lambda x: jnp.sum(jax.lax.stop_gradient(x) ** 2))(coordinates), 0)
    with pytest.raises(ConfigurationError):
        ObjectiveSpec().validate("trunk_only")
    ObjectiveSpec(geometry_weight=0).validate("trunk_only")
