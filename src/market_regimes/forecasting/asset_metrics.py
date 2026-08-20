"""Matured-only individual asset forecast metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    return (
        float(np.corrcoef(actual, predicted)[0, 1])
        if len(actual) > 1 and np.std(actual) > 0 and np.std(predicted) > 0
        else float("nan")
    )


def _log_loss(actual: np.ndarray, probability: np.ndarray) -> float:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped))
    )


def _metrics_row(group: pd.DataFrame, *, fold_id: str, scope: str) -> dict[str, object]:
    resolved = group.loc[group["resolved"]].dropna(
        subset=["actual_cumulative_log_return", "actual_excess_log_return"]
    )
    if resolved.empty:
        raise ValueError("asset metric group has no matured forecasts")
    actual_return = resolved["actual_cumulative_log_return"].to_numpy(dtype=float)
    actual_excess = resolved["actual_excess_log_return"].to_numpy(dtype=float)
    actual_direction = (actual_return > 0).astype(float)
    actual_outperform = (actual_excess > 0).astype(float)
    predicted_return = resolved["predicted_cumulative_log_return"].to_numpy(dtype=float)
    predicted_excess = resolved["predicted_excess_log_return"].to_numpy(dtype=float)
    probability_up = resolved["predicted_up_probability"].to_numpy(dtype=float)
    probability_outperform = resolved["predicted_outperform_probability"].to_numpy(dtype=float)
    row: dict[str, object] = {
        "scope": scope,
        "fold_id": fold_id,
        "ticker": str(resolved["ticker"].iloc[0]),
        "horizon_trading_days": int(resolved["horizon_trading_days"].iloc[0]),
        "resolved_forecasts": len(resolved),
        "mean_prediction": float(predicted_return.mean()),
        "mean_realized_return": float(actual_return.mean()),
        "return_mae": float(np.mean(np.abs(actual_return - predicted_return))),
        "return_rmse": float(np.sqrt(np.mean((actual_return - predicted_return) ** 2))),
        "return_correlation": _correlation(actual_return, predicted_return),
        "directional_accuracy": float(np.mean((probability_up >= 0.5) == actual_direction)),
        "direction_brier_score": float(np.mean((probability_up - actual_direction) ** 2)),
        "direction_log_loss": _log_loss(actual_direction, probability_up),
        "mean_predicted_excess": float(predicted_excess.mean()),
        "mean_realized_excess": float(actual_excess.mean()),
        "excess_mae": float(np.mean(np.abs(actual_excess - predicted_excess))),
        "excess_rmse": float(np.sqrt(np.mean((actual_excess - predicted_excess) ** 2))),
        "excess_correlation": _correlation(actual_excess, predicted_excess),
        "outperform_accuracy": float(
            np.mean((probability_outperform >= 0.5) == actual_outperform)
        ),
        "outperform_brier_score": float(
            np.mean((probability_outperform - actual_outperform) ** 2)
        ),
        "outperform_log_loss": _log_loss(actual_outperform, probability_outperform),
    }
    baseline_columns = {
        "expanding": (
            "expanding_baseline_return",
            "expanding_baseline_up_probability",
            "expanding_baseline_excess_return",
            "expanding_baseline_outperform_probability",
        ),
        "regime": (
            "regime_baseline_return",
            "regime_baseline_up_probability",
            "regime_baseline_excess_return",
            "regime_baseline_outperform_probability",
        ),
    }
    for name, (return_col, up_col, excess_col, outperform_col) in baseline_columns.items():
        baseline_return = resolved[return_col].to_numpy(dtype=float)
        baseline_up = resolved[up_col].to_numpy(dtype=float)
        baseline_excess = resolved[excess_col].to_numpy(dtype=float)
        baseline_outperform = resolved[outperform_col].to_numpy(dtype=float)
        row[f"{name}_return_mae"] = float(
            np.mean(np.abs(actual_return - baseline_return))
        )
        row[f"{name}_return_rmse"] = float(
            np.sqrt(np.mean((actual_return - baseline_return) ** 2))
        )
        row[f"{name}_directional_accuracy"] = float(
            np.mean((baseline_up >= 0.5) == actual_direction)
        )
        row[f"{name}_direction_brier_score"] = float(
            np.mean((baseline_up - actual_direction) ** 2)
        )
        row[f"{name}_direction_log_loss"] = _log_loss(actual_direction, baseline_up)
        row[f"{name}_excess_mae"] = float(
            np.mean(np.abs(actual_excess - baseline_excess))
        )
        row[f"{name}_excess_rmse"] = float(
            np.sqrt(np.mean((actual_excess - baseline_excess) ** 2))
        )
        row[f"{name}_outperform_accuracy"] = float(
            np.mean((baseline_outperform >= 0.5) == actual_outperform)
        )
        row[f"{name}_outperform_brier_score"] = float(
            np.mean((baseline_outperform - actual_outperform) ** 2)
        )
        row[f"{name}_outperform_log_loss"] = _log_loss(
            actual_outperform, baseline_outperform
        )
    row["beats_both_return_mae_baselines"] = bool(
        row["return_mae"] < row["expanding_return_mae"]
        and row["return_mae"] < row["regime_return_mae"]
    )
    row["beats_both_direction_brier_baselines"] = bool(
        row["direction_brier_score"] < row["expanding_direction_brier_score"]
        and row["direction_brier_score"] < row["regime_direction_brier_score"]
    )
    row["beats_both_excess_mae_baselines"] = bool(
        row["excess_mae"] < row["expanding_excess_mae"]
        and row["excess_mae"] < row["regime_excess_mae"]
    )
    return row


def evaluate_asset_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return fold-level and all-fold metrics for every ticker/horizon."""
    records: list[dict[str, object]] = []
    for (fold_id, _ticker, _horizon), group in ledger.groupby(
        ["fold_id", "ticker", "horizon_trading_days"], sort=True
    ):
        if group["resolved"].any():
            records.append(_metrics_row(group, fold_id=str(fold_id), scope="fold"))
    for (_ticker, _horizon), group in ledger.groupby(
        ["ticker", "horizon_trading_days"], sort=True
    ):
        if group["resolved"].any():
            records.append(_metrics_row(group, fold_id="ALL", scope="all_folds"))
    return pd.DataFrame.from_records(records)
