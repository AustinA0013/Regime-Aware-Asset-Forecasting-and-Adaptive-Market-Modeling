"""Immutable asset prediction reconciliation with outcome-only settlement."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

KEY_COLUMNS = ["fold_id", "origin_date", "ticker", "horizon_trading_days"]
OUTCOME_COLUMNS = [
    "resolved",
    "forecast_for_date",
    "forecast_date_is_estimate",
    "actual_cumulative_log_return",
    "actual_benchmark_log_return",
    "actual_excess_log_return",
    "actual_direction_up",
    "actual_outperformed",
]


@dataclass(frozen=True)
class AssetLedgerMergeAudit:
    existing_rows: int
    new_forecasts: int
    newly_resolved: int
    final_rows: int


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    missing = set(KEY_COLUMNS + OUTCOME_COLUMNS) - set(result.columns)
    if missing:
        raise ValueError(f"asset ledger is missing columns: {sorted(missing)}")
    result["fold_id"] = result["fold_id"].astype(str)
    result["ticker"] = result["ticker"].astype(str)
    result["origin_date"] = pd.to_datetime(result["origin_date"])
    result["forecast_for_date"] = pd.to_datetime(result["forecast_for_date"])
    result["horizon_trading_days"] = result["horizon_trading_days"].astype(int)
    result["resolved"] = result["resolved"].astype(bool)
    if result.duplicated(KEY_COLUMNS).any():
        raise ValueError("asset ledger contains duplicate fold/origin/ticker/horizon keys")
    return result


def merge_asset_forecast_ledgers(
    existing: pd.DataFrame | None,
    replay: pd.DataFrame,
) -> tuple[pd.DataFrame, AssetLedgerMergeAudit]:
    """Preserve original predictions, settle outcomes, and append new origins."""
    replay_normalized = _normalize(replay)
    if existing is None or existing.empty:
        merged = replay_normalized.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
        return merged, AssetLedgerMergeAudit(0, len(merged), 0, len(merged))
    existing_normalized = _normalize(existing)
    replay_by_key = replay_normalized.set_index(KEY_COLUMNS, drop=False)
    merged = existing_normalized.copy()
    newly_resolved = 0
    for row_index, row in merged.iterrows():
        key = (
            row["fold_id"],
            row["origin_date"],
            row["ticker"],
            int(row["horizon_trading_days"]),
        )
        if bool(row["resolved"]) or key not in replay_by_key.index:
            continue
        replay_row = replay_by_key.loc[key]
        if isinstance(replay_row, pd.DataFrame):
            raise ValueError("replay asset ledger contains duplicate keys")
        if not bool(replay_row["resolved"]):
            continue
        for column in OUTCOME_COLUMNS:
            merged.at[row_index, column] = replay_row[column]
        newly_resolved += 1
    existing_keys = pd.MultiIndex.from_frame(existing_normalized[KEY_COLUMNS])
    replay_keys = pd.MultiIndex.from_frame(replay_normalized[KEY_COLUMNS])
    new_rows = replay_normalized.loc[~replay_keys.isin(existing_keys)]
    merged = pd.concat([merged, new_rows], ignore_index=True)
    merged = merged.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    return merged, AssetLedgerMergeAudit(
        existing_rows=len(existing_normalized),
        new_forecasts=len(new_rows),
        newly_resolved=newly_resolved,
        final_rows=len(merged),
    )
