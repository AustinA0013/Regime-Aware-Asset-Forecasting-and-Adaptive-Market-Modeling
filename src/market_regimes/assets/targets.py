"""Close-to-future-close asset and benchmark-relative targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssetTargetResult:
    cumulative_return: pd.DataFrame
    cumulative_excess_return: pd.DataFrame
    target_dates: pd.Series


def future_asset_targets(
    prices: pd.DataFrame,
    *,
    benchmark: str,
    horizon: int,
) -> AssetTargetResult:
    """Create targets from t to t+h without including the origin-day return."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if benchmark not in prices:
        raise ValueError("benchmark must be present in price table")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("price index must be strictly increasing")
    cumulative = np.log(prices.shift(-horizon) / prices)
    benchmark_return = cumulative[benchmark]
    excess = cumulative.subtract(benchmark_return, axis=0)
    target_dates = pd.Series(pd.NaT, index=prices.index, dtype="datetime64[ns]")
    if len(prices) > horizon:
        target_dates.iloc[:-horizon] = prices.index[horizon:].to_numpy()
    return AssetTargetResult(
        cumulative_return=cumulative,
        cumulative_excess_return=excess,
        target_dates=target_dates,
    )
