"""Causal and interpretable financial feature construction."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_regimes.real_config import FeatureConfig

LOGGER = logging.getLogger(__name__)
FEATURE_COLUMNS = [
    "log_return_1d",
    "realized_volatility_5d",
    "realized_volatility_20d",
    "momentum_20d",
    "momentum_60d",
    "drawdown_252d",
    "vix_level",
    "vix_change_1d",
    "yield_curve_10y_2y",
    "credit_spread",
]


def feature_columns(config: FeatureConfig) -> list[str]:
    """Return ordered feature names implied by a configurable window definition."""
    volatility_short, volatility_long = config.realized_volatility_windows
    momentum_short, momentum_long = config.momentum_windows
    return [
        "log_return_1d",
        f"realized_volatility_{volatility_short}d",
        f"realized_volatility_{volatility_long}d",
        f"momentum_{momentum_short}d",
        f"momentum_{momentum_long}d",
        f"drawdown_{config.drawdown_window}d",
        "vix_level",
        "vix_change_1d",
        "yield_curve_10y_2y",
        "credit_spread",
    ]


@dataclass(frozen=True)
class FillAudit:
    """Observed consequences of one explicit as-of alignment rule."""

    source_id: str
    max_staleness_days: int
    exact_matches: int
    carried_forward: int
    missing: int
    maximum_observed_age_days: float


@dataclass(frozen=True)
class MarketFeatureResult:
    """Causal feature table, aligned levels, and fill audit records."""

    features: pd.DataFrame
    aligned_levels: pd.DataFrame
    fill_audit: tuple[FillAudit, ...]


def _normalized_series(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "available_at", "source_id", "raw_value"}
    if not required.issubset(frame.columns):
        raise ValueError(f"series is missing columns: {sorted(required - set(frame.columns))}")
    result = frame.loc[:, list(required)].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"]).dt.normalize()
    result["available_at"] = pd.to_datetime(result["available_at"]).dt.normalize()
    result = result.sort_values("available_at", kind="stable")
    if result["available_at"].duplicated().any():
        result = result.drop_duplicates("available_at", keep="last")
    return result


def align_asof(
    calendar: pd.DatetimeIndex,
    frame: pd.DataFrame,
    *,
    max_staleness_days: int,
) -> tuple[pd.Series, FillAudit]:
    """Align only values already available at each calendar time.

    This is an explicitly logged backward as-of join, conventionally called a
    forward fill in time-series work. ``max_staleness_days`` prevents old values
    from being carried indefinitely.
    """
    if max_staleness_days < 0:
        raise ValueError("max_staleness_days cannot be negative")
    source = _normalized_series(frame).rename(
        columns={"timestamp": "source_timestamp", "raw_value": "value"}
    )
    left = pd.DataFrame({"timestamp": pd.DatetimeIndex(calendar).sort_values()})
    joined = pd.merge_asof(
        left,
        source[["available_at", "source_timestamp", "value"]],
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
        tolerance=pd.Timedelta(days=max_staleness_days),
        allow_exact_matches=True,
    )
    if (joined["available_at"] > joined["timestamp"]).fillna(False).any():
        raise RuntimeError("as-of alignment used a value before it was available")
    valid = joined["value"].notna()
    age = (joined["timestamp"] - joined["available_at"]).dt.total_seconds() / 86400.0
    exact = int((valid & age.eq(0)).sum())
    carried = int((valid & age.gt(0)).sum())
    missing = int((~valid).sum())
    maximum_age = float(age[valid].max()) if valid.any() else float("nan")
    source_id = str(frame["source_id"].iloc[0])
    audit = FillAudit(
        source_id=source_id,
        max_staleness_days=max_staleness_days,
        exact_matches=exact,
        carried_forward=carried,
        missing=missing,
        maximum_observed_age_days=maximum_age,
    )
    LOGGER.info(
        "Aligned %s exact=%d carried=%d missing=%d max_age=%.1f days",
        source_id,
        exact,
        carried,
        missing,
        maximum_age,
    )
    return pd.Series(joined["value"].to_numpy(), index=left["timestamp"], name=source_id), audit


def build_market_features(
    series_by_role: Mapping[str, pd.DataFrame],
    feature_config: FeatureConfig,
    max_staleness_by_source: Mapping[str, int],
) -> MarketFeatureResult:
    """Construct the ten-feature baseline using no observation after time ``t``."""
    feature_config.validate()
    required_roles = {
        "price",
        "vix",
        "treasury_10y",
        "treasury_2y",
        "credit_spread",
    }
    if set(series_by_role) != required_roles:
        raise ValueError(f"series roles must be exactly {sorted(required_roles)}")

    price_frame = _normalized_series(series_by_role["price"])
    price_frame = price_frame.sort_values("timestamp", kind="stable")
    price_frame = price_frame.drop_duplicates("timestamp", keep="last")
    calendar = pd.DatetimeIndex(price_frame["timestamp"])
    levels = pd.DataFrame(index=calendar)
    levels.index.name = "timestamp"
    levels["price"] = price_frame.set_index("timestamp")["raw_value"].reindex(calendar)
    audits: list[FillAudit] = []
    for role in ("vix", "treasury_10y", "treasury_2y", "credit_spread"):
        frame = series_by_role[role]
        source_id = str(frame["source_id"].iloc[0])
        if source_id not in max_staleness_by_source:
            raise ValueError(f"missing max-staleness policy for {source_id}")
        aligned, audit = align_asof(
            calendar,
            frame,
            max_staleness_days=max_staleness_by_source[source_id],
        )
        levels[role] = aligned.reindex(calendar).to_numpy()
        audits.append(audit)

    volatility_short, volatility_long = feature_config.realized_volatility_windows
    momentum_short, momentum_long = feature_config.momentum_windows
    returns = np.log(levels["price"] / levels["price"].shift(1))
    features = pd.DataFrame(index=calendar)
    features["log_return_1d"] = returns
    features[f"realized_volatility_{volatility_short}d"] = (
        returns.rolling(volatility_short, min_periods=volatility_short).std(ddof=0)
        * np.sqrt(feature_config.annualization_factor)
    )
    features[f"realized_volatility_{volatility_long}d"] = (
        returns.rolling(volatility_long, min_periods=volatility_long).std(ddof=0)
        * np.sqrt(feature_config.annualization_factor)
    )
    features[f"momentum_{momentum_short}d"] = np.log(
        levels["price"] / levels["price"].shift(momentum_short)
    )
    features[f"momentum_{momentum_long}d"] = np.log(
        levels["price"] / levels["price"].shift(momentum_long)
    )
    rolling_peak = levels["price"].rolling(
        feature_config.drawdown_window,
        min_periods=feature_config.drawdown_window,
    ).max()
    features[f"drawdown_{feature_config.drawdown_window}d"] = levels["price"] / rolling_peak - 1.0
    features["vix_level"] = levels["vix"]
    features["vix_change_1d"] = levels["vix"].diff()
    features["yield_curve_10y_2y"] = levels["treasury_10y"] - levels["treasury_2y"]
    features["credit_spread"] = levels["credit_spread"]
    if list(features.columns) != feature_columns(feature_config):
        raise RuntimeError(f"unexpected feature columns: {list(features.columns)}")
    return MarketFeatureResult(
        features=features,
        aligned_levels=levels,
        fill_audit=tuple(audits),
    )
