"""Asset target horizon and benchmark-excess tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regimes.assets.targets import future_asset_targets


def test_asset_target_is_close_to_future_close_after_origin() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.DataFrame(
        {
            "SPY": 100 * np.exp(np.arange(8) * 0.01),
            "AAA": 50 * np.exp(np.arange(8) * 0.02),
        },
        index=dates,
    )
    result = future_asset_targets(prices, benchmark="SPY", horizon=3)
    assert np.isclose(result.cumulative_return.loc[dates[0], "AAA"], 0.06)
    assert np.isclose(result.cumulative_excess_return.loc[dates[0], "AAA"], 0.03)
    assert result.target_dates.loc[dates[0]] == dates[3]
    assert result.cumulative_return.iloc[-3:].isna().all().all()
