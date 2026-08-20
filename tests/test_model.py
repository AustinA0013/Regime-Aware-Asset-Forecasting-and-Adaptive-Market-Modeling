"""GaussianRegimeModel API and persistence tests."""

from __future__ import annotations

import numpy as np
import pytest

from market_regimes.data.synthetic import generate_synthetic_hmm
from market_regimes.models.gaussian_hmm import GaussianRegimeModel


@pytest.mark.parametrize("covariance_type", ["diag", "full"])
def test_wrapper_supports_required_covariance_types(
    baseline_config, covariance_type: str
) -> None:
    data = generate_synthetic_hmm(baseline_config.synthetic, seed=41)
    observations = data.observations[:600]
    model = GaussianRegimeModel(
        3, covariance_type=covariance_type, n_iter=100, seeds=(3, 5)
    ).fit(observations)
    probabilities = model.predict_filtered_probabilities(observations)
    assert probabilities.shape == (600, 3)
    assert model.predict_filtered_probabilities(observations[:1]).shape == (1, 3)
    assert model.covariance_matrices().shape == (3, 2, 2)
    assert np.all(model.expected_state_duration() >= 1.0)
    assert len(model.state_statistics(observations)) == 3


def test_model_round_trip(tmp_path, baseline_config) -> None:
    data = generate_synthetic_hmm(baseline_config.synthetic, seed=43)
    observations = data.observations[:500]
    model = GaussianRegimeModel(3, n_iter=100, seeds=(2,)).fit(observations)
    before = model.predict_filtered_probabilities(observations)
    path = tmp_path / "model.joblib"
    model.save(path)
    restored = GaussianRegimeModel.load(path)
    assert np.array_equal(before, restored.predict_filtered_probabilities(observations))
