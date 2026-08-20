"""Causality, target timing, and feedback tests for adaptive forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regimes.forecast_config import OnlineForecastConfig
from market_regimes.forecasting.online import evaluate_forecast_ledger, walk_forward_forecast


def _config() -> OnlineForecastConfig:
    return OnlineForecastConfig(
        horizons=(1, 5, 21),
        initial_epochs=4,
        random_seed=17,
        learning_rate=0.002,
        regularization=0.001,
        target_multiplier=100.0,
        rolling_metric_window=30,
    )


def _inputs(n_rows: int = 280) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-01", periods=n_rows)
    regime = (np.arange(n_rows) // 35) % 3
    rng = np.random.default_rng(9)
    state_return = np.array([0.0015, 0.0002, -0.0018])
    returns = state_return[regime] + rng.normal(0, 0.006, n_rows)
    features = pd.DataFrame(
        {
            "log_return_1d": returns,
            "momentum": pd.Series(returns, index=dates).rolling(10, min_periods=1).sum(),
            "stress": regime + rng.normal(0, 0.1, n_rows),
        },
        index=dates,
    )
    probabilities = np.full((n_rows, 3), 0.025)
    probabilities[np.arange(n_rows), regime] = 0.95
    return features, pd.DataFrame(
        probabilities,
        index=dates,
        columns=["state_0", "state_1", "state_2"],
    )


def test_forecast_target_starts_after_origin_date() -> None:
    features, probabilities = _inputs()
    result = walk_forward_forecast(
        features,
        probabilities,
        train_end=features.index[159],
        test_start=features.index[160],
        config=_config(),
    )
    first_origin = features.index[160]
    for horizon in (1, 5, 21):
        row = result.ledger.loc[
            (result.ledger["origin_date"] == first_origin)
            & (result.ledger["horizon_trading_days"] == horizon)
        ].iloc[0]
        expected = features["log_return_1d"].iloc[161 : 161 + horizon].sum()
        assert np.isclose(row["actual_cumulative_log_return"], expected)
        assert row["forecast_for_date"] == features.index[160 + horizon]


def test_future_perturbation_cannot_change_earlier_forecasts() -> None:
    features, probabilities = _inputs()
    base = walk_forward_forecast(
        features,
        probabilities,
        train_end=features.index[159],
        test_start=features.index[160],
        config=_config(),
    ).ledger
    changed_features = features.copy()
    changed_features.loc[features.index[230] :, "log_return_1d"] += 0.25
    changed_features.loc[features.index[230] :, "stress"] += 20
    changed = walk_forward_forecast(
        changed_features,
        probabilities,
        train_end=features.index[159],
        test_start=features.index[160],
        config=_config(),
    ).ledger
    prediction_columns = [
        "predicted_cumulative_log_return",
        "predicted_up_probability",
        "predicted_regime",
        "predicted_regime_probability_0",
        "predicted_regime_probability_1",
        "predicted_regime_probability_2",
    ]
    base_prefix = base.loc[base["origin_date"] < features.index[230], prediction_columns]
    changed_prefix = changed.loc[
        changed["origin_date"] < features.index[230], prediction_columns
    ]
    pd.testing.assert_frame_equal(
        base_prefix.reset_index(drop=True),
        changed_prefix.reset_index(drop=True),
    )


def test_forecasts_are_scored_and_models_receive_matured_feedback() -> None:
    features, probabilities = _inputs()
    result = walk_forward_forecast(
        features,
        probabilities,
        train_end=features.index[159],
        test_start=features.index[160],
        config=_config(),
    )
    metrics = evaluate_forecast_ledger(result.ledger, n_states=3)
    for horizon in (1, 5, 21):
        assert result.models[horizon].online_updates > 0
        assert metrics[str(horizon)]["resolved_forecasts"] == 120 - horizon
        assert metrics[str(horizon)]["pending_forecasts"] == horizon
        assert np.isfinite(metrics[str(horizon)]["return_mae"])
        assert 0 <= metrics[str(horizon)]["directional_accuracy"] <= 1
        assert 0 <= metrics[str(horizon)]["regime_accuracy"] <= 1
