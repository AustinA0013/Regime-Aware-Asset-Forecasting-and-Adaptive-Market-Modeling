"""Formula and no-future-dependence tests for asset features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regimes.asset_config import AssetFeatureConfig
from market_regimes.assets.features import asset_feature_columns, build_asset_features


def _config() -> AssetFeatureConfig:
    return AssetFeatureConfig(
        volatility_windows=(5, 20, 60),
        momentum_windows=(5, 20, 60, 120),
        drawdown_window=252,
        relative_strength_windows=(20, 60, 120),
        beta_window=60,
        correlation_window=60,
        volume_window=20,
        volume_change_window=5,
        annualization_factor=252,
    )


def _frame(ticker: str, dates: pd.DatetimeIndex, prices: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": dates,
            "ticker": ticker,
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "adjusted_close": prices,
            "volume": 1_000_000 + np.arange(len(dates)) * 1000,
            "price_for_returns": prices,
        }
    )


def _inputs() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-01", periods=340)
    market_returns = 0.0003 + 0.005 * np.sin(np.arange(len(dates)) / 13)
    market_price = 100 * np.exp(np.cumsum(market_returns))
    asset_price = 80 * np.exp(np.cumsum(1.4 * market_returns + 0.0001))
    return {
        "SPY": _frame("SPY", dates, market_price),
        "AAA": _frame("AAA", dates, asset_price),
    }


def test_asset_feature_contract_and_relative_strength() -> None:
    result = build_asset_features(_inputs(), benchmark="SPY", config=_config())
    assert len(asset_feature_columns(_config())) == 17
    aaa = result.features.loc[result.features["ticker"] == "AAA"].set_index("timestamp")
    final = aaa.dropna().iloc[-1]
    prices = result.prices
    expected_relative_strength = np.log(
        prices["AAA"].iloc[-1] / prices["AAA"].iloc[-61]
    ) - np.log(prices["SPY"].iloc[-1] / prices["SPY"].iloc[-61])
    assert np.isclose(final["asset_relative_strength_60d"], expected_relative_strength)
    assert 1.2 < final["asset_beta_60d"] < 1.6
    assert final["asset_market_correlation_60d"] > 0.99


def test_future_price_change_cannot_change_earlier_asset_features() -> None:
    frames = _inputs()
    base = build_asset_features(frames, benchmark="SPY", config=_config()).features
    changed = {ticker: frame.copy() for ticker, frame in frames.items()}
    cutoff = pd.Timestamp("2021-03-01")
    mask = changed["AAA"]["timestamp"] > cutoff
    for column in ("open", "high", "low", "close", "adjusted_close", "price_for_returns"):
        changed["AAA"].loc[mask, column] *= 4.0
    later = build_asset_features(changed, benchmark="SPY", config=_config()).features
    columns = asset_feature_columns(_config())
    base_prefix = base.loc[(base["ticker"] == "AAA") & (base["timestamp"] <= cutoff), columns]
    later_prefix = later.loc[
        (later["ticker"] == "AAA") & (later["timestamp"] <= cutoff), columns
    ]
    pd.testing.assert_frame_equal(
        base_prefix.reset_index(drop=True), later_prefix.reset_index(drop=True)
    )
