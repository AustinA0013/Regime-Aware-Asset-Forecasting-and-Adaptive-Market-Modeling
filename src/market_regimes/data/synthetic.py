"""Synthetic observations from a known multivariate Gaussian HMM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from market_regimes.config import SyntheticConfig

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SyntheticHMMData:
    """A sampled series together with its unobserved ground-truth states."""

    observations: FloatArray
    states: IntArray


def generate_synthetic_hmm(config: SyntheticConfig, seed: int) -> SyntheticHMMData:
    """Sample a first-order Gaussian HMM using NumPy's local random generator.

    The generator follows ``z_0 ~ pi``, ``z_t ~ A[z_(t-1), :]``, and
    ``x_t | z_t=k ~ N(mu_k, Sigma_k)``. It never modifies NumPy's global RNG.
    """
    config.validate()
    rng = np.random.default_rng(seed)
    n_states = config.start_probabilities.size
    n_features = config.means.shape[1]
    states = np.empty(config.n_samples, dtype=np.int64)
    observations = np.empty((config.n_samples, n_features), dtype=np.float64)

    states[0] = rng.choice(n_states, p=config.start_probabilities)
    observations[0] = rng.multivariate_normal(
        config.means[states[0]], config.covariances[states[0]], check_valid="raise"
    )
    for index in range(1, config.n_samples):
        states[index] = rng.choice(
            n_states, p=config.transition_matrix[states[index - 1]]
        )
        observations[index] = rng.multivariate_normal(
            config.means[states[index]],
            config.covariances[states[index]],
            check_valid="raise",
        )
    return SyntheticHMMData(observations=observations, states=states)

