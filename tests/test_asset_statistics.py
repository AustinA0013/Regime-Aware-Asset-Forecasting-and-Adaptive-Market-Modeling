"""Regime-conditioned sample-size and probability-weighting tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regimes.asset_config import AssetStatisticsConfig
from market_regimes.assets.statistics import calculate_asset_state_statistics


def test_asset_state_statistics_report_hard_and_weighted_sample_sizes() -> None:
    dates = pd.bdate_range("2020-01-01", periods=180)
    states = np.repeat([0, 1, 2], 60)
    probabilities = np.full((180, 3), 0.05)
    probabilities[np.arange(180), states] = 0.90
    feature = pd.DataFrame(
        {
            "timestamp": dates,
            "ticker": "AAA",
            "asset_log_return_1d": np.array([0.002, 0.0, -0.002])[states],
            "asset_drawdown_252d": np.array([-0.02, -0.08, -0.20])[states],
        }
    )
    result = calculate_asset_state_statistics(
        feature,
        pd.DataFrame(probabilities, index=dates),
        {0: "Risk-On-like", 1: "Transitional-like", 2: "Risk-Off-like"},
        drawdown_column="asset_drawdown_252d",
        config=AssetStatisticsConfig(40, 50, 10),
        random_seed=3,
    )
    assert set(result["membership_method"]) == {"hard_argmax", "probability_weighted"}
    hard = result.loc[
        (result["membership_method"] == "hard_argmax") & (result["state"] == 0)
    ].iloc[0]
    assert hard["sample_count"] == 60
    assert bool(hard["sufficient_sample"])
    assert hard["mean_daily_return"] > 0
