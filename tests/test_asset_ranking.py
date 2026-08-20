"""Cross-sectional Information Coefficient and top-k tests."""

from __future__ import annotations

import pandas as pd

from market_regimes.forecasting.ranking import (
    cross_sectional_daily_metrics,
    summarize_cross_sectional,
)


def test_perfect_asset_ranking_has_unit_information_coefficient() -> None:
    rows = []
    for rank, ticker in enumerate(["A", "B", "C", "D", "E"]):
        value = 0.05 - rank * 0.01
        rows.append(
            {
                "fold_id": "f1",
                "origin_date": pd.Timestamp("2024-01-02"),
                "horizon_trading_days": 5,
                "ticker": ticker,
                "resolved": True,
                "predicted_excess_log_return": value,
                "actual_excess_log_return": value,
                "actual_cumulative_log_return": value + 0.002,
                "actual_benchmark_log_return": 0.002,
            }
        )
    daily = cross_sectional_daily_metrics(pd.DataFrame(rows), top_k=2, min_assets=5)
    assert abs(daily.iloc[0]["information_coefficient"] - 1.0) < 1e-12
    assert daily.iloc[0]["top_k_realized_return"] > daily.iloc[0]["universe_realized_return"]
    summary = summarize_cross_sectional(daily)
    assert summary["by_horizon"]["5"]["fraction_positive_ic"] == 1.0
