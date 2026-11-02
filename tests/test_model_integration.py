"""Explicitly unexecuted in local acceptance: requires real model assets."""

import os
import pytest

pytestmark = pytest.mark.model_integration


def test_real_boltz_three_chemical_gradient_channels():
    if os.environ.get("MOLCRAFT_RUN_MODEL_TESTS") != "1":
        pytest.skip("Real-model execution is disabled; CPU acceptance never loads weights")
    from molcraft import prepare_complex, value_and_grad
    from molcraft.backends.boltz2_jax.backend import Boltz2Backend
    from molcraft.io.config import load_config

    config = load_config(os.environ["MOLCRAFT_MODEL_CONFIG"])
    with prepare_complex(config.complex) as prepared:
        backend = Boltz2Backend.from_paths(config.complex.model_paths,
                                           config.sampling_steps, config.trunk_passes)
        result = value_and_grad(prepared, config.objective, config.gradient_mode,
                                noise_seed=config.seed, backend=backend)
        result.validate(prepared.graph)
