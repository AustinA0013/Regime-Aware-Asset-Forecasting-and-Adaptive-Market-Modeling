"""Causal, economically interpretable features for individual assets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_regimes.asset_config import AssetFeatureConfig


@dataclass(frozen=True)
class AssetFeatureResult:
    """Long asset-feature table and aligned adjusted-return prices."""

    features: pd.DataFrame
    prices: pd.DataFrame
    raw_aligned: pd.DataFrame


def asset_feature_columns(config: AssetFeatureConfig) -> list[str]:
    columns = ["asset_log_return_1d"]
    columns.extend(f"asset_realized_volatility_{window}d" for window in config.volatility_windows)
    columns.extend(f"asset_momentum_{window}d" for window in config.momentum_windows)
    columns.append(f"asset_drawdown_{config.drawdown_window}d")
    columns.extend(
        f"asset_relative_strength_{window}d" for window in config.relative_strength_windows
    )
    columns.extend(
        [
            f"asset_beta_{config.beta_window}d",
            f"asset_market_correlation_{config.correlation_window}d",
            f"asset_volume_relative_{config.volume_window}d",
            f"asset_volume_zscore_{config.volume_window}d",
            f"asset_volume_change_{config.volume_change_window}d",
        ]
    )
    return columns


def _ticker_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    required = {
        "timestamp",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "price_for_returns",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"{ticker} asset frame is missing columns: {sorted(required-set(frame))}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"]).dt.normalize()
    result = result.loc[result["ticker"] == ticker].copy()
    result = result.drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
    if result.empty:
        raise ValueError(f"{ticker} has no observations")
    return result


def build_asset_features(
    frames: Mapping[str, pd.DataFrame],
    *,
    benchmark: str,
    config: AssetFeatureConfig,
) -> AssetFeatureResult:
    """Build per-asset features using only current and earlier daily observations.

    Relative strength is the difference between asset and benchmark cumulative
    log returns. This is stable when returns are near zero and directly represents
    cumulative outperformance in log-return units.
    """
    config.validate()
    if benchmark not in frames:
        raise ValueError("benchmark frame is required")
    normalized = {ticker: _ticker_frame(frame, ticker) for ticker, frame in frames.items()}
    benchmark_frame = normalized[benchmark]
    calendar = pd.DatetimeIndex(benchmark_frame.index)
    benchmark_price = benchmark_frame["price_for_returns"].reindex(calendar)
    benchmark_return = np.log(benchmark_price / benchmark_price.shift(1))
    expected = asset_feature_columns(config)
    feature_parts: list[pd.DataFrame] = []
    raw_parts: list[pd.DataFrame] = []
    price_table = pd.DataFrame(index=calendar)
    price_table.index.name = "timestamp"

    for ticker, source in normalized.items():
        aligned = source.reindex(calendar)
        price = aligned["price_for_returns"].where(aligned["price_for_returns"] > 0)
        price_table[ticker] = price
        returns = np.log(price / price.shift(1))
        features = pd.DataFrame(index=calendar)
        features["asset_log_return_1d"] = returns
        for window in config.volatility_windows:
            features[f"asset_realized_volatility_{window}d"] = (
                returns.rolling(window, min_periods=window).std(ddof=0)
                * np.sqrt(config.annualization_factor)
            )
        for window in config.momentum_windows:
            features[f"asset_momentum_{window}d"] = np.log(price / price.shift(window))
        rolling_peak = price.rolling(
            config.drawdown_window, min_periods=config.drawdown_window
        ).max()
        features[f"asset_drawdown_{config.drawdown_window}d"] = price / rolling_peak - 1.0
        for window in config.relative_strength_windows:
            asset_momentum = np.log(price / price.shift(window))
            benchmark_momentum = np.log(benchmark_price / benchmark_price.shift(window))
            features[f"asset_relative_strength_{window}d"] = (
                asset_momentum - benchmark_momentum
            )
        covariance = returns.rolling(
            config.beta_window, min_periods=config.beta_window
        ).cov(benchmark_return)
        market_variance = benchmark_return.rolling(
            config.beta_window, min_periods=config.beta_window
        ).var(ddof=1)
        features[f"asset_beta_{config.beta_window}d"] = covariance / market_variance
        features[f"asset_market_correlation_{config.correlation_window}d"] = (
            returns.rolling(
                config.correlation_window, min_periods=config.correlation_window
            ).corr(benchmark_return)
        )
        volume = aligned["volume"].where(aligned["volume"] > 0)
        volume_mean = volume.rolling(
            config.volume_window, min_periods=config.volume_window
        ).mean()
        volume_std = volume.rolling(
            config.volume_window, min_periods=config.volume_window
        ).std(ddof=0)
        features[f"asset_volume_relative_{config.volume_window}d"] = (
            volume / volume_mean - 1.0
        )
        features[f"asset_volume_zscore_{config.volume_window}d"] = (
            volume - volume_mean
        ) / volume_std.replace(0.0, np.nan)
        features[f"asset_volume_change_{config.volume_change_window}d"] = np.log(
            volume / volume.shift(config.volume_change_window)
        )
        if list(features.columns) != expected:
            raise RuntimeError("asset feature column order does not match contract")
        features.insert(0, "ticker", ticker)
        features.insert(0, "timestamp", calendar)
        feature_parts.append(features.reset_index(drop=True))

        raw = aligned[["open", "high", "low", "close", "adjusted_close", "volume"]].copy()
        raw.insert(0, "price_for_returns", price)
        raw.insert(0, "ticker", ticker)
        raw.insert(0, "timestamp", calendar)
        raw_parts.append(raw.reset_index(drop=True))

    feature_table = pd.concat(feature_parts, ignore_index=True)
    feature_table = feature_table.sort_values(["timestamp", "ticker"], kind="stable")
    raw_table = pd.concat(raw_parts, ignore_index=True)
    raw_table = raw_table.sort_values(["timestamp", "ticker"], kind="stable")
    return AssetFeatureResult(
        features=feature_table.reset_index(drop=True),
        prices=price_table,
        raw_aligned=raw_table.reset_index(drop=True),
    )
