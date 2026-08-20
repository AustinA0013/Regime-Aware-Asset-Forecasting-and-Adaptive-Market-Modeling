"""Delayed feedback, deterministic replay, and causal forecast tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regimes.asset_config import AnchoredFoldConfig, AssetOnlineConfig
from market_regimes.forecasting.asset_online import walk_forward_asset_fold


def _config() -> AssetOnlineConfig:
    return AssetOnlineConfig(
        horizons=(1, 5, 21),
        initial_epochs=3,
        random_seed=41,
        learning_rate=0.002,
        regularization=0.001,
        target_multiplier=100.0,
        min_initial_samples=100,
        rolling_metric_window=30,
    )


def _inputs():
    n_rows = 430
    dates = pd.bdate_range("2020-01-01", periods=n_rows)
    rng = np.random.default_rng(12)
    states = (np.arange(n_rows) // 45) % 3
    market_returns = np.array([0.0008, 0.0001, -0.0007])[states] + rng.normal(
        0, 0.006, n_rows
    )
    asset_returns = 1.15 * market_returns + np.array([0.0004, 0, -0.0002])[states]
    asset_returns += rng.normal(0, 0.004, n_rows)
    prices = pd.DataFrame(
        {
            "SPY": 100 * np.exp(np.cumsum(market_returns)),
            "AAA": 60 * np.exp(np.cumsum(asset_returns)),
        },
        index=dates,
    )
    market = pd.DataFrame(
        rng.normal(size=(n_rows, 10)),
        index=dates,
        columns=[f"m{i}" for i in range(10)],
    )
    probabilities = np.full((n_rows, 3), 0.025)
    probabilities[np.arange(n_rows), states] = 0.95
    probability_frame = pd.DataFrame(
        probabilities, index=dates, columns=[f"state_{i}" for i in range(3)]
    )
    local_columns = [f"asset_feature_{i}" for i in range(14)] + [
        "asset_realized_volatility_20d",
        "asset_drawdown_252d",
        "asset_relative_strength_60d",
    ]
    local = pd.DataFrame(
        rng.normal(size=(n_rows, 17)), index=dates, columns=local_columns
    )
    local["asset_realized_volatility_20d"] = 0.2
    local["asset_drawdown_252d"] = -0.1
    local["asset_relative_strength_60d"] = 0.03
    fold = AnchoredFoldConfig(
        "test",
        train_end=dates[259].date(),
        test_start=dates[260].date(),
        test_end=dates[409].date(),
    )
    return dates, market, probability_frame, local, prices, fold


def _run(market, probabilities, local, prices, fold):
    return walk_forward_asset_fold(
        ticker="AAA",
        benchmark="SPY",
        market_features=market,
        probabilities=probabilities,
        asset_features=local,
        prices=prices,
        fold=fold,
        config=_config(),
    )


def test_asset_walk_forward_releases_targets_only_at_maturity() -> None:
    dates, market, probabilities, local, prices, fold = _inputs()
    result = _run(market, probabilities, local, prices, fold)
    first_origin = dates[260]
    for horizon in (1, 5, 21):
        row = result.ledger.loc[
            (result.ledger["origin_date"] == first_origin)
            & (result.ledger["horizon_trading_days"] == horizon)
        ].iloc[0]
        expected = np.log(prices.loc[dates[260 + horizon], "AAA"] / prices.loc[first_origin, "AAA"])
        assert np.isclose(row["actual_cumulative_log_return"], expected)
        assert row["forecast_for_date"] == dates[260 + horizon]
        assert result.models[horizon].online_updates > 0


def test_deterministic_replay_and_future_perturbation() -> None:
    dates, market, probabilities, local, prices, fold = _inputs()
    base = _run(market, probabilities, local, prices, fold).ledger
    replay = _run(market, probabilities, local, prices, fold).ledger
    prediction_columns = [
        "predicted_cumulative_log_return",
        "predicted_up_probability",
        "predicted_excess_log_return",
        "predicted_outperform_probability",
    ]
    pd.testing.assert_frame_equal(base[prediction_columns], replay[prediction_columns])

    changed_prices = prices.copy()
    changed_local = local.copy()
    change_date = dates[360]
    changed_prices.loc[change_date:, "AAA"] *= 3
    changed_local.loc[change_date:, :] += 50
    changed = _run(market, probabilities, changed_local, changed_prices, fold).ledger
    base_prefix = base.loc[base["origin_date"] < change_date, prediction_columns]
    changed_prefix = changed.loc[changed["origin_date"] < change_date, prediction_columns]
    pd.testing.assert_frame_equal(
        base_prefix.reset_index(drop=True), changed_prefix.reset_index(drop=True)
    )
