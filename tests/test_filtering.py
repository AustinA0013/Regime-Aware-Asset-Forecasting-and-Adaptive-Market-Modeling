"""Causality and numerical validation for the explicit forward recursion."""

from __future__ import annotations

import numpy as np
from hmmlearn.hmm import GaussianHMM

from market_regimes.models.filtering import forward_filter


def _parameters() -> tuple[np.ndarray, ...]:
    start = np.array([0.7, 0.3])
    transition = np.array([[0.92, 0.08], [0.12, 0.88]])
    means = np.array([[-1.5], [1.5]])
    covariances = np.array([[[0.6]], [[0.8]]])
    return start, transition, means, covariances


def test_filter_probabilities_are_normalized_and_finite() -> None:
    start, transition, means, covariances = _parameters()
    observations = np.array([[-1.0], [-0.8], [0.2], [1.8], [1.4]])
    result = forward_filter(observations, start, transition, means, covariances)
    assert np.all(np.isfinite(result.probabilities))
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_filter_is_strictly_causal() -> None:
    start, transition, means, covariances = _parameters()
    prefix = np.array([[-1.0], [-0.8], [0.2]])
    future_a = np.array([[1.8], [1.4]])
    future_b = np.array([[-8.0], [-8.0]])
    probabilities_a = forward_filter(
        np.vstack([prefix, future_a]), start, transition, means, covariances
    ).probabilities
    probabilities_b = forward_filter(
        np.vstack([prefix, future_b]), start, transition, means, covariances
    ).probabilities
    assert np.array_equal(probabilities_a[: len(prefix)], probabilities_b[: len(prefix)])


def test_forward_log_likelihood_matches_hmmlearn() -> None:
    start, transition, means, covariances = _parameters()
    observations = np.array([[-1.0], [-0.8], [0.2], [1.8], [1.4]])
    reference = GaussianHMM(n_components=2, covariance_type="full", init_params="")
    reference.startprob_ = start
    reference.transmat_ = transition
    reference.means_ = means
    reference.covars_ = covariances
    explicit = forward_filter(observations, start, transition, means, covariances)
    assert np.isclose(explicit.log_likelihood, reference.score(observations), atol=1e-10)

