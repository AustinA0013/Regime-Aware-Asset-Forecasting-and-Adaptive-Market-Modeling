"""Matured-only individual asset metric tests."""

from __future__ import annotations

import pandas as pd

from market_regimes.forecasting.asset_metrics import evaluate_asset_ledger


def test_asset_metrics_ignore_pending_forecasts_and_compare_baselines() -> None:
    rows = []
    for index, actual in enumerate([0.01, -0.02, 0.015, 0.004]):
        rows.append(
            {
                "fold_id": "f1",
                "origin_date": pd.Timestamp("2024-01-01") + pd.offsets.BDay(index),
                "ticker": "AAA",
                "horizon_trading_days": 5,
                "resolved": True,
                "actual_cumulative_log_return": actual,
                "actual_excess_log_return": actual - 0.002,
                "predicted_cumulative_log_return": actual * 0.8,
                "predicted_excess_log_return": (actual - 0.002) * 0.8,
                "predicted_up_probability": 0.8 if actual > 0 else 0.2,
                "predicted_outperform_probability": 0.8 if actual > 0.002 else 0.2,
                "expanding_baseline_return": 0.0,
                "regime_baseline_return": 0.0,
                "expanding_baseline_up_probability": 0.5,
                "regime_baseline_up_probability": 0.5,
                "expanding_baseline_excess_return": 0.0,
                "regime_baseline_excess_return": 0.0,
                "expanding_baseline_outperform_probability": 0.5,
                "regime_baseline_outperform_probability": 0.5,
            }
        )
    pending = rows[-1].copy()
    pending["origin_date"] += pd.offsets.BDay(10)
    pending["resolved"] = False
    pending["actual_cumulative_log_return"] = None
    pending["actual_excess_log_return"] = None
    metrics = evaluate_asset_ledger(pd.DataFrame([*rows, pending]))
    aggregate = metrics.loc[metrics["scope"] == "all_folds"].iloc[0]
    assert aggregate["resolved_forecasts"] == 4
    assert aggregate["return_mae"] < aggregate["expanding_return_mae"]
    assert aggregate["direction_brier_score"] < aggregate["expanding_direction_brier_score"]
