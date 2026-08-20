"""Cross-sectional ranking diagnostics for asset selection research."""

from __future__ import annotations

import pandas as pd


def _spearman(predicted: pd.Series, actual: pd.Series) -> float:
    predicted_rank = predicted.rank(method="average")
    actual_rank = actual.rank(method="average")
    if predicted_rank.nunique() < 2 or actual_rank.nunique() < 2:
        return float("nan")
    return float(predicted_rank.corr(actual_rank))


def cross_sectional_daily_metrics(
    ledger: pd.DataFrame,
    *,
    top_k: int,
    min_assets: int,
) -> pd.DataFrame:
    """Score predicted excess-return ranks at each origin and horizon."""
    records: list[dict[str, object]] = []
    resolved = ledger.loc[ledger["resolved"]].dropna(
        subset=["predicted_excess_log_return", "actual_excess_log_return"]
    )
    for (fold_id, origin, horizon), group in resolved.groupby(
        ["fold_id", "origin_date", "horizon_trading_days"], sort=True
    ):
        group = group.drop_duplicates("ticker", keep="last")
        if len(group) < min_assets:
            continue
        ranked = group.sort_values("predicted_excess_log_return", ascending=False)
        selected = ranked.head(min(top_k, len(ranked)))
        records.append(
            {
                "fold_id": fold_id,
                "origin_date": origin,
                "horizon_trading_days": int(horizon),
                "eligible_assets": len(group),
                "information_coefficient": _spearman(
                    group["predicted_excess_log_return"],
                    group["actual_excess_log_return"],
                ),
                "top_k": min(top_k, len(ranked)),
                "top_k_realized_return": float(
                    selected["actual_cumulative_log_return"].mean()
                ),
                "top_k_realized_excess": float(selected["actual_excess_log_return"].mean()),
                "universe_realized_return": float(
                    group["actual_cumulative_log_return"].mean()
                ),
                "universe_realized_excess": float(group["actual_excess_log_return"].mean()),
                "benchmark_realized_return": float(
                    group["actual_benchmark_log_return"].mean()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_cross_sectional(daily: pd.DataFrame) -> dict[str, object]:
    """Aggregate IC and top-k outcomes by horizon and fold."""
    result: dict[str, object] = {"by_horizon": {}, "by_fold_and_horizon": {}}

    def summary(group: pd.DataFrame) -> dict[str, float | int]:
        ic = group["information_coefficient"].dropna()
        return {
            "periods": int(len(group)),
            "ic_periods": int(len(ic)),
            "mean_ic": float(ic.mean()) if len(ic) else float("nan"),
            "median_ic": float(ic.median()) if len(ic) else float("nan"),
            "ic_standard_deviation": float(ic.std(ddof=0)) if len(ic) else float("nan"),
            "fraction_positive_ic": float((ic > 0).mean()) if len(ic) else float("nan"),
            "mean_top_k_realized_return": float(group["top_k_realized_return"].mean()),
            "mean_top_k_realized_excess": float(group["top_k_realized_excess"].mean()),
            "mean_universe_realized_return": float(group["universe_realized_return"].mean()),
            "mean_benchmark_realized_return": float(group["benchmark_realized_return"].mean()),
            "top_k_minus_universe": float(
                (group["top_k_realized_return"] - group["universe_realized_return"]).mean()
            ),
            "top_k_minus_benchmark": float(
                (group["top_k_realized_return"] - group["benchmark_realized_return"]).mean()
            ),
        }

    for horizon, group in daily.groupby("horizon_trading_days", sort=True):
        result["by_horizon"][str(int(horizon))] = summary(group)
    for (fold, horizon), group in daily.groupby(
        ["fold_id", "horizon_trading_days"], sort=True
    ):
        result["by_fold_and_horizon"][f"{fold}:{int(horizon)}"] = summary(group)
    return result


def realized_return_by_predicted_rank(ledger: pd.DataFrame) -> pd.DataFrame:
    """Assign within-origin predicted-excess quintiles and summarize outcomes."""
    resolved = ledger.loc[ledger["resolved"]].dropna(
        subset=["predicted_excess_log_return", "actual_excess_log_return"]
    ).copy()
    pieces: list[pd.DataFrame] = []
    for _, group in resolved.groupby(
        ["fold_id", "origin_date", "horizon_trading_days"], sort=False
    ):
        if len(group) < 5:
            continue
        ranked = group.copy()
        ranked["predicted_rank_bucket"] = pd.qcut(
            ranked["predicted_excess_log_return"].rank(method="first"),
            q=min(5, len(ranked)),
            labels=False,
            duplicates="drop",
        )
        pieces.append(ranked)
    if not pieces:
        return pd.DataFrame()
    combined = pd.concat(pieces, ignore_index=True)
    return (
        combined.groupby(
            ["horizon_trading_days", "predicted_rank_bucket"], as_index=False
        )
        .agg(
            observations=("actual_excess_log_return", "size"),
            mean_realized_return=("actual_cumulative_log_return", "mean"),
            mean_realized_excess=("actual_excess_log_return", "mean"),
        )
        .sort_values(["horizon_trading_days", "predicted_rank_bucket"])
    )
