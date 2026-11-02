import numpy as np
import pytest

from molcraft.errors import ConfigurationError, GradientError
from molcraft.guidance.probability import directional_scores, masked_softmax


def test_directional_scores_are_feasible_category_differences():
    g = np.array([[0.2, -0.4, -0.1, -0.8]])
    np.testing.assert_allclose(directional_scores(g, [0]), [[0, -0.6, -0.3, -1.0]])
    np.testing.assert_allclose(directional_scores(g + 10, [0]), directional_scores(g, [0]))


def test_mask_removes_favored_illegal_option_and_normalizes():
    q = masked_softmax([[0, -0.6, -0.3, -1.0]], [[1, 1, 1, 0]], 0.3)
    np.testing.assert_allclose(q[0], [0.090030573, 0.665240956, 0.244728471, 0], atol=1e-8)


def test_temperature_and_prior_have_explicit_effects():
    low = masked_softmax([[0, -1]], [[1, 1]], 0.1)
    high = masked_softmax([[0, -1]], [[1, 1]], 10)
    assert low[0, 1] > high[0, 1]
    np.testing.assert_allclose(masked_softmax([[0, 0]], [[1, 1]], prior=[[1, 3]]), [[0.25, 0.75]])


def test_large_scores_remain_stable():
    q = masked_softmax([[100000, -100000]], [[1, 1]])
    np.testing.assert_array_equal(q, [[0, 1]])


def test_very_low_temperature_does_not_create_nan():
    q = masked_softmax([[-1e300, 1e300]], [[1, 1]], temperature=1e-300)
    np.testing.assert_array_equal(q, [[1, 0]])


def test_overflowing_directional_differences_fail_explicitly():
    with pytest.raises(GradientError):
        directional_scores([[-1e308, 1e308]], [0])


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_fails(temperature):
    with pytest.raises(ConfigurationError):
        masked_softmax([[0, 1]], [[1, 1]], temperature)


def test_empty_allowed_set_fails_instead_of_inventing_a_distribution():
    with pytest.raises(ConfigurationError):
        masked_softmax([[0, 1]], [[0, 0]])


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_gradients_fail(value):
    with pytest.raises(GradientError):
        directional_scores([[value, 1]], [0])


def test_zero_prior_is_explicitly_rejected():
    with pytest.raises(ConfigurationError):
        masked_softmax([[0, 1]], [[1, 1]], prior=[[1, 0]])
