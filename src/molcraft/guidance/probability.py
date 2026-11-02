"""First-order preferences, not calibrated molecular posterior probabilities."""

import numpy as np

from molcraft.errors import ConfigurationError, GradientError


def directional_scores(gradient, current_indices):
    g = np.asarray(gradient, dtype=np.float64)
    current = np.asarray(current_indices, dtype=int)
    if g.ndim != 2 or current.shape != (g.shape[0],) or not np.isfinite(g).all():
        raise GradientError("Finite [positions, categories] gradients are required")
    if np.any(current < 0) or np.any(current >= g.shape[1]):
        raise ConfigurationError("Current category index out of range")
    with np.errstate(over="ignore", invalid="ignore"):
        result = g - g[np.arange(len(g)), current, None]
    if not np.isfinite(result).all():
        raise GradientError("Directional differences overflowed; rescale the objective explicitly")
    return result


def masked_softmax(scores, allowed, temperature=1.0, prior=None):
    scores = np.asarray(scores, dtype=np.float64)
    try:
        prior = (np.ones_like(scores) if prior is None
                 else np.broadcast_to(np.asarray(prior, dtype=float), scores.shape))
    except (ValueError, TypeError) as exc:
        raise ConfigurationError("Prior cannot be broadcast to the category table") from exc
    if not np.isfinite(prior).all() or np.any(prior <= 0):
        raise ConfigurationError("Every category prior must be finite and strictly positive")
    return masked_log_prior_softmax(scores, allowed, temperature, np.log(prior))


def masked_log_prior_softmax(scores, allowed, temperature, log_prior):
    """Internal log-prior form also supports products of extreme bundle priors."""
    scores = np.asarray(scores, dtype=np.float64)
    allowed = np.asarray(allowed, dtype=bool)
    if scores.ndim != 2 or scores.shape != allowed.shape or not np.isfinite(scores).all():
        raise GradientError("Scores and masks must have identical finite 2D shape")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ConfigurationError("temperature must be finite and positive")
    log_prior = np.broadcast_to(np.asarray(log_prior, dtype=float), scores.shape)
    if not np.isfinite(log_prior).all():
        raise ConfigurationError("Log priors must be finite")
    if np.any(~allowed.any(axis=1)):
        raise ConfigurationError("Each row must contain an allowed option, including NO_OP")
    if scores.shape[0] == 0:
        return np.empty_like(scores)
    # Subtract the best allowed score before division, avoiding +inf - +inf
    # when a user deliberately chooses a very low temperature.
    minimum = np.min(np.where(allowed, scores, np.inf), axis=1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logits = np.where(allowed, -(scores - minimum) / temperature + log_prior, -np.inf)
    logits -= np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)
