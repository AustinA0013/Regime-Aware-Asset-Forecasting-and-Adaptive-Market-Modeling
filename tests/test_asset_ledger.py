"""Append-only asset ledger immutability tests."""

from __future__ import annotations

import pandas as pd

from market_regimes.forecasting.asset_ledger import merge_asset_forecast_ledgers


def _row(origin: str, prediction: float, resolved: bool, actual: float | None = None):
    return {
        "fold_id": "live",
        "origin_date": pd.Timestamp(origin),
        "forecast_for_date": pd.Timestamp(origin) + pd.offsets.BDay(5),
        "forecast_date_is_estimate": not resolved,
        "ticker": "AAA",
        "horizon_trading_days": 5,
        "resolved": resolved,
        "predicted_cumulative_log_return": prediction,
        "actual_cumulative_log_return": actual,
        "actual_benchmark_log_return": 0.001 if actual is not None else None,
        "actual_excess_log_return": actual - 0.001 if actual is not None else None,
        "actual_direction_up": int(actual > 0) if actual is not None else None,
        "actual_outperformed": int(actual > 0.001) if actual is not None else None,
    }


def test_asset_ledger_preserves_prediction_and_only_settles_outcome() -> None:
    existing = pd.DataFrame([_row("2026-08-10", 0.123, False)])
    replay = pd.DataFrame([_row("2026-08-10", 9.999, True, -0.02)])
    merged, audit = merge_asset_forecast_ledgers(existing, replay)
    assert merged.iloc[0]["predicted_cumulative_log_return"] == 0.123
    assert merged.iloc[0]["actual_cumulative_log_return"] == -0.02
    assert audit.newly_resolved == 1


def test_duplicate_asset_forecast_keys_are_rejected() -> None:
    duplicate = pd.DataFrame([_row("2026-08-10", 0.1, False)] * 2)
    try:
        merge_asset_forecast_ledgers(None, duplicate)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate asset ledger keys should be rejected")


def test_fold_id_type_changes_do_not_append_duplicate_forecasts() -> None:
    existing = pd.DataFrame([_row("2026-08-10", 0.123, False)])
    existing["fold_id"] = 20222023
    replay = pd.DataFrame([_row("2026-08-10", 9.999, True, -0.02)])
    replay["fold_id"] = "20222023"

    merged, audit = merge_asset_forecast_ledgers(existing, replay)

    assert len(merged) == 1
    assert merged.iloc[0]["fold_id"] == "20222023"
    assert merged.iloc[0]["predicted_cumulative_log_return"] == 0.123
    assert merged.iloc[0]["actual_cumulative_log_return"] == -0.02
    assert audit.new_forecasts == 0
    assert audit.newly_resolved == 1
