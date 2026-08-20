"""End-to-end statistical acceptance test for Milestone 1."""

from __future__ import annotations

import pytest

from market_regimes.data.synthetic import generate_synthetic_hmm
from market_regimes.models.gaussian_hmm import GaussianRegimeModel
from market_regimes.validation.recovery import evaluate_recovery


@pytest.mark.recovery
@pytest.mark.parametrize("data_seed", [20260814, 20260815, 20260816])
def test_known_hmm_is_recovered_with_causal_filtering(baseline_config, data_seed: int) -> None:
    data = generate_synthetic_hmm(baseline_config.synthetic, data_seed)
    model_config = baseline_config.model
    model = GaussianRegimeModel(
        model_config.n_states,
        covariance_type=model_config.covariance_type,
        n_iter=model_config.n_iter,
        tolerance=model_config.tolerance,
        min_covar=model_config.min_covar,
        seeds=model_config.seeds,
    ).fit(data.observations)
    report, _ = evaluate_recovery(
        baseline_config.synthetic,
        data.states,
        data.observations,
        model,
        baseline_config.validation,
    )
    assert report.passed, report
