"""Replaceable daily OHLCV providers with an adjusted-return price contract."""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
ASSET_COLUMNS = [
    "timestamp",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
    "price_for_returns",
    "adjustment_policy",
    "retrieved_at",
]


@dataclass(frozen=True)
class AssetRequest:
    ticker: str
    start: date
    end: date

    def validate(self) -> None:
        if not self.ticker or self.ticker != self.ticker.upper():
            raise ValueError("ticker must be a nonempty uppercase symbol")
        if self.start > self.end:
            raise ValueError("asset request start cannot be after end")


class AssetProvider(ABC):
    """Source-neutral interface for daily prices and volume."""

    @abstractmethod
    def fetch(self, request: AssetRequest) -> pd.DataFrame:
        """Return one ticker in the standard adjusted-price schema."""


class YahooChartAssetProvider(AssetProvider):
    """Fetch public Yahoo Finance chart data and cache the exact response table.

    Predictive returns use Yahoo's adjusted-close series when it is present. That
    field is intended to account for splits and cash distributions. Raw OHLC fields
    remain unadjusted and are retained for auditing; they are not used for targets.
    If adjusted close is absent, the provider falls back to close and marks the row.
    The interface is isolated because Yahoo's public chart endpoint is not a formal
    stable data-service contract and can later be replaced without changing features.
    """

    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"

    def __init__(self, cache_dir: str | Path, *, timeout_seconds: int = 30) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds

    def _cache_path(self, request: AssetRequest) -> Path:
        today_ny = datetime.now(ZoneInfo("America/New_York")).date()
        end_token = "latest" if request.end >= today_ny else request.end.isoformat()
        return self.cache_dir / (
            f"{request.ticker}_{request.start.isoformat()}_{end_token}_1d.csv"
        )

    @staticmethod
    def _unix_seconds(value: date) -> int:
        stamp = datetime.combine(value, time.min, tzinfo=ZoneInfo("America/New_York"))
        return int(stamp.timestamp())

    def _download_json(self, url: str) -> dict[str, object]:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "market-regimes-research/0.2"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def _fetch_remote(self, request: AssetRequest) -> pd.DataFrame:
        parameters = {
            "period1": str(self._unix_seconds(request.start)),
            "period2": str(self._unix_seconds(request.end + timedelta(days=1))),
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
        url = (
            f"{self.BASE_URL}/{urllib.parse.quote(request.ticker, safe='')}?"
            f"{urllib.parse.urlencode(parameters)}"
        )
        payload = self._download_json(url)
        chart = payload.get("chart")
        if not isinstance(chart, dict):
            raise ValueError(f"Yahoo returned an invalid chart payload for {request.ticker}")
        error = chart.get("error")
        if error:
            raise ValueError(f"Yahoo chart error for {request.ticker}: {error}")
        results = chart.get("result")
        if not isinstance(results, list) or not results:
            raise ValueError(f"Yahoo returned no result for {request.ticker}")
        result = results[0]
        if not isinstance(result, dict):
            raise ValueError(f"Yahoo returned an invalid result for {request.ticker}")
        timestamps = result.get("timestamp")
        indicators = result.get("indicators")
        if not isinstance(timestamps, list) or not isinstance(indicators, dict):
            raise ValueError(f"Yahoo result lacks timestamps/indicators for {request.ticker}")
        quotes = indicators.get("quote")
        adjusted = indicators.get("adjclose")
        if not isinstance(quotes, list) or not quotes or not isinstance(quotes[0], dict):
            raise ValueError(f"Yahoo result lacks OHLCV quote data for {request.ticker}")
        quote = quotes[0]
        adjusted_values: list[object] | None = None
        if isinstance(adjusted, list) and adjusted and isinstance(adjusted[0], dict):
            candidate = adjusted[0].get("adjclose")
            if isinstance(candidate, list):
                adjusted_values = candidate

        def values(name: str) -> list[object]:
            raw = quote.get(name)
            if not isinstance(raw, list) or len(raw) != len(timestamps):
                return [None] * len(timestamps)
            return raw

        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": values("open"),
                "high": values("high"),
                "low": values("low"),
                "close": values("close"),
                "adjusted_close": (
                    adjusted_values
                    if adjusted_values is not None and len(adjusted_values) == len(timestamps)
                    else [None] * len(timestamps)
                ),
                "volume": values("volume"),
            }
        )
        frame["timestamp"] = (
            frame["timestamp"]
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )
        numeric = ["open", "high", "low", "close", "adjusted_close", "volume"]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.loc[
            (frame["timestamp"].dt.date >= request.start)
            & (frame["timestamp"].dt.date <= request.end)
        ].copy()
        frame = frame.dropna(subset=["timestamp", "close"]).copy()
        frame["ticker"] = request.ticker
        has_adjusted = frame["adjusted_close"].notna() & (frame["adjusted_close"] > 0)
        frame["price_for_returns"] = frame["adjusted_close"].where(
            has_adjusted, frame["close"]
        )
        frame["adjustment_policy"] = np.where(
            has_adjusted,
            "yahoo_adjusted_close_total_return_compatible",
            "raw_close_fallback_unadjusted",
        )
        frame["retrieved_at"] = datetime.now(UTC).isoformat()
        frame = frame.drop_duplicates("timestamp", keep="last")
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"Yahoo returned no usable observations for {request.ticker}")
        return frame[ASSET_COLUMNS]

    @staticmethod
    def _needs_refresh(request: AssetRequest, frame: pd.DataFrame) -> bool:
        today_ny = datetime.now(ZoneInfo("America/New_York")).date()
        latest = pd.to_datetime(frame["timestamp"]).max().date()
        return request.end >= today_ny and latest < request.end

    def fetch(self, request: AssetRequest) -> pd.DataFrame:
        request.validate()
        cache_path = self._cache_path(request)
        if cache_path.exists():
            frame = pd.read_csv(cache_path, parse_dates=["timestamp"])
            missing = set(ASSET_COLUMNS) - set(frame.columns)
            if missing:
                raise ValueError(f"asset cache is missing columns: {sorted(missing)}")
            frame = frame[ASSET_COLUMNS]
            if self._needs_refresh(request, frame):
                LOGGER.info("Refreshing incomplete Yahoo asset cache %s", cache_path)
                frame = self._fetch_remote(request)
                frame.to_csv(cache_path, index=False)
                mode = "yahoo-chart-refresh"
            else:
                mode = "yahoo-chart-cache"
        else:
            frame = self._fetch_remote(request)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache_path, index=False)
            mode = "yahoo-chart"
        frame.attrs["source_path"] = str(cache_path.resolve())
        frame.attrs["provider_mode"] = mode
        return frame
