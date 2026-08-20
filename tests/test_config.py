"""Configuration validation tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from market_regimes.config import ExperimentConfig


def test_baseline_configuration_is_dimensionally_consistent(
    baseline_config: ExperimentConfig,
) -> None:
    baseline_config.validate()
    assert baseline_config.synthetic.means.shape == (3, 2)
    assert baseline_config.model.n_states == 3


def test_invalid_transition_rows_are_rejected(baseline_config: ExperimentConfig) -> None:
    invalid_matrix = np.array(baseline_config.synthetic.transition_matrix, copy=True)
    invalid_matrix[0, 0] = 0.5
    invalid_synthetic = replace(baseline_config.synthetic, transition_matrix=invalid_matrix)
    with pytest.raises(ValueError, match="sum to one"):
        invalid_synthetic.validate()

