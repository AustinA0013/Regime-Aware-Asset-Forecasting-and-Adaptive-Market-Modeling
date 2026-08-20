"""Causality, formula, and no-silent-fill tests for market features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regimes.features.market import align_asof, build_market_features
from market_regimes.real_config import FeatureConfig


def _frame(source_id: str, dates, values, lag_days: int = 0) -> pd.DataFrame:
    timestamp = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "available_at": timestamp + pd.to_timedelta(lag_days, unit="D"),
            "source_id": source_id,
            "raw_value": values,
        }
    )


def test_asof_alignment_obeys_availability_and_staleness() -> None:
    calendar = pd.date_range("2024-01-01", periods=5, freq="D")
    delayed = _frame("RATE", ["2024-01-01", "2024-01-04"], [1.0, 2.0], lag_days=1)
    aligned, audit = align_asof(calendar, delayed, max_staleness_days=2)
    assert np.isnan(aligned.iloc[0])
    assert aligned.iloc[1:4].tolist() == [1.0, 1.0, 1.0]
    assert aligned.iloc[4] == 2.0
    assert audit.carried_forward == 2
    assert audit.missing == 1


def test_feature_engineering_is_causal_under_future_perturbation() -> None:
    dates = pd.bdate_range("2020-01-01", periods=320)
    price = 100.0 * np.exp(np.linspace(0, 0.2, len(dates)))
    base = {
        "price": _frame("PRICE", dates, price),
        "vix": _frame("VIX", dates, 20 + np.sin(np.arange(len(dates)) / 10)),
        "treasury_10y": _frame("DGS10", dates, np.full(len(dates), 3.0)),
        "treasury_2y": _frame("DGS2", dates, np.full(len(dates), 2.0)),
        "credit_spread": _frame("CREDIT", dates, np.full(len(dates), 4.0)),
    }
    config = FeatureConfig((5, 20), (20, 60), 252, 252)
    policies = {"VIX": 4, "DGS10": 7, "DGS2": 7, "CREDIT": 7}
    original = build_market_features(base, config, policies).features
    changed = {key: value.copy() for key, value in base.items()}
    changed["price"].loc[changed["price"].index >= 300, "raw_value"] *= 10
    changed["vix"].loc[changed["vix"].index >= 300, "raw_value"] = 999
    perturbed = build_market_features(changed, config, policies).features
    pd.testing.assert_frame_equal(original.iloc[:300], perturbed.iloc[:300])


def test_return_momentum_and_yield_curve_formulas() -> None:
    dates = pd.bdate_range("2020-01-01", periods=320)
    price = 100 * np.power(1.01, np.arange(len(dates)))
    inputs = {
        "price": _frame("PRICE", dates, price),
        "vix": _frame("VIX", dates, np.arange(len(dates), dtype=float)),
        "treasury_10y": _frame("DGS10", dates, np.full(len(dates), 4.0)),
        "treasury_2y": _frame("DGS2", dates, np.full(len(dates), 1.5)),
        "credit_spread": _frame("CREDIT", dates, np.full(len(dates), 3.5)),
    }
    result = build_market_features(
        inputs,
        FeatureConfig((5, 20), (20, 60), 252, 252),
        {"VIX": 4, "DGS10": 7, "DGS2": 7, "CREDIT": 7},
    ).features
    row = result.iloc[-1]
    assert np.isclose(row["log_return_1d"], np.log(1.01))
    assert np.isclose(row["momentum_20d"], 20 * np.log(1.01))
    assert np.isclose(row["momentum_60d"], 60 * np.log(1.01))
    assert np.isclose(row["yield_curve_10y_2y"], 2.5)
