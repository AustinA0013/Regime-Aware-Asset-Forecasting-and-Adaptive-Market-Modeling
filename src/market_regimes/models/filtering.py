"""Numerically stable, forward-only filtering for a Gaussian HMM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class FilterResult:
    """Causal posterior probabilities and their predictive normalizers."""

    probabilities: FloatArray
    log_normalizers: FloatArray

    @property
    def log_likelihood(self) -> float:
        """Sequence log likelihood accumulated by the forward recursion."""
        return float(self.log_normalizers.sum())


def gaussian_log_densities(
    observations: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
    *,
    jitter: float = 1e-10,
) -> FloatArray:
    """Return ``log p(x_t | z_t=k)`` for every time and state.

    Cholesky solves avoid explicit matrix inversion. A tiny deterministic jitter
    is added only when a supplied covariance is numerically semidefinite.
    """
    x = np.asarray(observations, dtype=np.float64)
    mu = np.asarray(means, dtype=np.float64)
    sigma = np.asarray(covariances, dtype=np.float64)
    if x.ndim != 2 or mu.ndim != 2:
        raise ValueError("observations and means must be two-dimensional")
    n_samples, n_features = x.shape
    n_states = mu.shape[0]
    if mu.shape[1] != n_features or sigma.shape != (n_states, n_features, n_features):
        raise ValueError("mean/covariance dimensions do not match observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("observations must contain only finite values")

    result = np.empty((n_samples, n_states), dtype=np.float64)
    constant = n_features * np.log(2.0 * np.pi)
    identity = np.eye(n_features)
    for state in range(n_states):
        covariance = sigma[state]
        try:
            cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            cholesky = np.linalg.cholesky(covariance + jitter * identity)
        centered = (x - mu[state]).T
        solved = np.linalg.solve(cholesky, centered)
        mahalanobis = np.einsum("ij,ij->j", solved, solved)
        log_determinant = 2.0 * np.log(np.diag(cholesky)).sum()
        result[:, state] = -0.5 * (constant + log_determinant + mahalanobis)
    return result


def _validate_probabilities(start: FloatArray, transition: FloatArray) -> None:
    if start.ndim != 1:
        raise ValueError("start_probabilities must be one-dimensional")
    if transition.shape != (start.size, start.size):
        raise ValueError("transition_matrix has an invalid shape")
    if np.any(start < 0) or not np.isclose(start.sum(), 1.0):
        raise ValueError("start_probabilities must be nonnegative and sum to one")
    if np.any(transition < 0) or not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("transition_matrix rows must be nonnegative and sum to one")


def forward_filter(
    observations: FloatArray,
    start_probabilities: FloatArray,
    transition_matrix: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
) -> FilterResult:
    r"""Calculate filtered state probabilities using only data through time ``t``.

    In log space the implemented recursion is

    ``log alpha_t(k) ∝ log p(x_t|z_t=k) + logsumexp_j(log alpha_(t-1)(j)+log A_jk)``.

    Each row is normalized immediately. Consequently row ``t`` is exactly
    :math:`P(z_t=k\mid x_{0:t})`; no backward/smoothing pass occurs.
    """
    x = np.asarray(observations, dtype=np.float64)
    start = np.asarray(start_probabilities, dtype=np.float64)
    transition = np.asarray(transition_matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("observations must be a nonempty two-dimensional array")
    _validate_probabilities(start, transition)

    log_emissions = gaussian_log_densities(x, means, covariances)
    with np.errstate(divide="ignore"):
        log_start = np.log(start)
        log_transition = np.log(transition)

    n_samples, n_states = log_emissions.shape
    log_alpha = np.empty((n_samples, n_states), dtype=np.float64)
    log_normalizers = np.empty(n_samples, dtype=np.float64)

    unnormalized = log_start + log_emissions[0]
    log_normalizers[0] = logsumexp(unnormalized)
    log_alpha[0] = unnormalized - log_normalizers[0]
    for index in range(1, n_samples):
        log_prediction = logsumexp(
            log_alpha[index - 1, :, np.newaxis] + log_transition,
            axis=0,
        )
        unnormalized = log_prediction + log_emissions[index]
        log_normalizers[index] = logsumexp(unnormalized)
        log_alpha[index] = unnormalized - log_normalizers[index]

    return FilterResult(
        probabilities=np.exp(log_alpha),
        log_normalizers=log_normalizers,
    )

