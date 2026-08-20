"""Tests for the known-data generator."""

from __future__ import annotations

import numpy as np

from market_regimes.config import ExperimentConfig
from market_regimes.data.synthetic import generate_synthetic_hmm


def test_generator_is_reproducible(baseline_config: ExperimentConfig) -> None:
    first = generate_synthetic_hmm(baseline_config.synthetic, seed=19)
    second = generate_synthetic_hmm(baseline_config.synthetic, seed=19)
    assert np.array_equal(first.states, second.states)
    assert np.array_equal(first.observations, second.observations)


def test_generator_shapes_and_state_support(baseline_config: ExperimentConfig) -> None:
    data = generate_synthetic_hmm(baseline_config.synthetic, seed=7)
    assert data.observations.shape == (
        baseline_config.synthetic.n_samples,
        baseline_config.synthetic.means.shape[1],
    )
    assert data.states.shape == (baseline_config.synthetic.n_samples,)
    assert set(np.unique(data.states)) == {0, 1, 2}

