"""Tests for provider schema, dates, and explicit availability assumptions."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from market_regimes.data.providers import CSVDataProvider, FREDDataProvider, SeriesRequest


def test_csv_provider_emits_standard_schema_and_lag(tmp_path) -> None:
    path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "close": [100.0, ".", 102.0],
        }
    ).to_csv(path, index=False)
    provider = CSVDataProvider(path, timestamp_column="date", value_column="close")
    result = provider.fetch(
        SeriesRequest(
            "TEST",
            date(2024, 1, 2),
            date(2024, 1, 4),
            availability_lag_days=1,
        )
    )
    assert result["raw_value"].tolist() == [100.0, 102.0]
    assert result["source_id"].unique().tolist() == ["TEST"]
    assert (result["available_at"] - result["timestamp"]).dt.days.tolist() == [1, 1]


def test_csv_provider_rejects_missing_columns(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"wrong": [1]}).to_csv(path, index=False)
    provider = CSVDataProvider(path, timestamp_column="date", value_column="value")
    with pytest.raises(ValueError, match="missing required columns"):
        provider.fetch(SeriesRequest("TEST", date(2024, 1, 1), date(2024, 1, 2)))


def test_current_incomplete_latest_cache_requires_refresh() -> None:
    request = SeriesRequest(
        source_id="TEST",
        start=date(2026, 8, 1),
        end=date(2026, 8, 18),
    )
    incomplete = pd.DataFrame({"timestamp": pd.to_datetime(["2026-08-17"])})
    assert FREDDataProvider._latest_cache_needs_refresh(
        request,
        incomplete,
        today=date(2026, 8, 18),
    )
    assert not FREDDataProvider._latest_cache_needs_refresh(
        request,
        incomplete,
        today=date(2026, 8, 19),
    )
