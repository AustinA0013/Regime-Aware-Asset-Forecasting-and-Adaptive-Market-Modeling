"""Append-only reconciliation and dynamic-date tests for scheduled daily runs."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from market_regimes.forecasting.ledger import merge_forecast_ledgers
from market_regimes.real_config import resolve_data_end


def _row(
    origin: str,
    horizon: int,
    *,
    prediction: float,
    resolved: bool,
    actual: float | None = None,
) -> dict[str, object]:
    return {
        "origin_date": pd.Timestamp(origin),
        "forecast_for_date": pd.Timestamp(origin) + pd.offsets.BDay(horizon),
        "forecast_date_is_estimate": not resolved,
        "horizon_trading_days": horizon,
        "resolved": resolved,
        "predicted_cumulative_log_return": prediction,
        "actual_cumulative_log_return": actual,
        "actual_direction_up": int(actual > 0) if actual is not None else None,
        "actual_regime": 0 if actual is not None else None,
        "return_error": actual - prediction if actual is not None else None,
        "absolute_return_error": abs(actual - prediction) if actual is not None else None,
    }


def test_latest_end_uses_new_york_calendar_date() -> None:
    before_midnight_new_york = datetime(2026, 8, 19, 3, 30, tzinfo=UTC)
    assert resolve_data_end("latest", now=before_midnight_new_york) == date(2026, 8, 18)
    assert resolve_data_end("2025-01-02", now=before_midnight_new_york) == date(2025, 1, 2)


def test_live_merge_preserves_prediction_settles_and_appends() -> None:
    existing = pd.DataFrame(
        [_row("2026-08-14", 1, prediction=0.123, resolved=False)]
    )
    replay = pd.DataFrame(
        [
            _row("2026-08-14", 1, prediction=9.999, resolved=True, actual=-0.01),
            _row("2026-08-17", 1, prediction=0.002, resolved=False),
        ]
    )
    merged, audit = merge_forecast_ledgers(existing, replay)
    old = merged.loc[merged["origin_date"] == pd.Timestamp("2026-08-14")].iloc[0]
    assert old["predicted_cumulative_log_return"] == 0.123
    assert bool(old["resolved"])
    assert old["actual_cumulative_log_return"] == -0.01
    assert audit.new_forecasts == 1
    assert audit.newly_resolved == 1


def test_identical_daily_replay_is_a_no_op() -> None:
    ledger = pd.DataFrame([_row("2026-08-17", 5, prediction=0.002, resolved=False)])
    merged, audit = merge_forecast_ledgers(ledger, ledger.copy())
    pd.testing.assert_frame_equal(merged, ledger)
    assert audit.new_forecasts == 0
    assert audit.newly_resolved == 0
