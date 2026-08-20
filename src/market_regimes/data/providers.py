"""Auditable source-independent time-series provider interfaces."""

from __future__ import annotations

import io
import json
import logging
import os
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
STANDARD_COLUMNS = [
    "timestamp",
    "available_at",
    "source_id",
    "raw_value",
    "transformed_value",
    "vintage_date",
    "retrieved_at",
]


@dataclass(frozen=True)
class SeriesRequest:
    """A bounded series request with an explicit availability assumption."""

    source_id: str
    start: date
    end: date
    availability_lag_days: int = 0
    vintage_date: date | None = None

    def validate(self) -> None:
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        if self.start > self.end:
            raise ValueError("series request start cannot be after end")
        if self.availability_lag_days < 0:
            raise ValueError("availability_lag_days cannot be negative")


class DataProvider(ABC):
    """Source-neutral interface returning the standard longitudinal schema."""

    @abstractmethod
    def fetch(self, request: SeriesRequest) -> pd.DataFrame:
        """Return timestamped raw values and explicit availability timestamps."""


def _standardize(
    timestamps: pd.Series,
    values: pd.Series,
    request: SeriesRequest,
    *,
    retrieved_at: str,
    vintage_date: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, errors="coerce"),
            "raw_value": pd.to_numeric(values, errors="coerce"),
        }
    )
    before = len(frame)
    frame = frame.dropna(subset=["timestamp", "raw_value"]).copy()
    dropped = before - len(frame)
    if dropped:
        LOGGER.info("Dropped %d missing/non-numeric %s observations", dropped, request.source_id)
    frame = frame.loc[
        (frame["timestamp"].dt.date >= request.start)
        & (frame["timestamp"].dt.date <= request.end)
    ].copy()
    frame["timestamp"] = frame["timestamp"].dt.normalize()
    frame["available_at"] = frame["timestamp"] + pd.to_timedelta(
        request.availability_lag_days, unit="D"
    )
    frame["source_id"] = request.source_id
    frame["transformed_value"] = frame["raw_value"]
    frame["vintage_date"] = vintage_date
    frame["retrieved_at"] = retrieved_at
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no usable observations returned for {request.source_id}")
    return frame[STANDARD_COLUMNS]


class CSVDataProvider(DataProvider):
    """Read a local CSV through the same schema used by remote providers."""

    def __init__(
        self,
        path: str | Path,
        *,
        timestamp_column: str,
        value_column: str,
    ) -> None:
        self.path = Path(path)
        self.timestamp_column = timestamp_column
        self.value_column = value_column

    def fetch(self, request: SeriesRequest) -> pd.DataFrame:
        request.validate()
        raw = pd.read_csv(self.path)
        missing = {self.timestamp_column, self.value_column} - set(raw.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        retrieved_at = datetime.fromtimestamp(self.path.stat().st_mtime, tz=UTC).isoformat()
        result = _standardize(
            raw[self.timestamp_column],
            raw[self.value_column],
            request,
            retrieved_at=retrieved_at,
            vintage_date=(request.vintage_date or request.end).isoformat(),
        )
        result.attrs["source_path"] = str(self.path.resolve())
        result.attrs["provider_mode"] = "csv"
        return result


class FREDDataProvider(DataProvider):
    """Fetch FRED observations with an API-key or public latest-vintage mode.

    API-key mode uses the documented ``fred/series/observations`` endpoint and
    supports a requested vintage date. Public mode uses FRED's graph CSV export,
    which is appropriate for this reproducible diagnostic snapshot but is not an
    ALFRED point-in-time data source.
    """

    API_URL = "https://api.stlouisfed.org/fred/series/observations"
    PUBLIC_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        api_key: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        self.timeout_seconds = timeout_seconds

    def _cache_path(self, request: SeriesRequest) -> Path:
        vintage = request.vintage_date.isoformat() if request.vintage_date else "latest"
        filename = (
            f"{request.source_id}_{request.start.isoformat()}_"
            f"{request.end.isoformat()}_{vintage}_lag{request.availability_lag_days}d.csv"
        )
        return self.cache_dir / filename

    def _download_text(self, url: str) -> str:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "market-regimes-research/0.1"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read().decode("utf-8")

    @staticmethod
    def _latest_cache_needs_refresh(
        request: SeriesRequest,
        result: pd.DataFrame,
        *,
        today: date | None = None,
    ) -> bool:
        """Refresh an incomplete current-date cache while preserving old snapshots."""
        if request.vintage_date is not None:
            return False
        current_date = today or datetime.now(UTC).date()
        latest_observation = pd.to_datetime(result["timestamp"]).max().date()
        return request.end >= current_date and latest_observation < request.end

    def _fetch_api(self, request: SeriesRequest, retrieved_at: str) -> pd.DataFrame:
        parameters: dict[str, str] = {
            "series_id": request.source_id,
            "api_key": str(self.api_key),
            "file_type": "json",
            "observation_start": request.start.isoformat(),
            "observation_end": request.end.isoformat(),
        }
        if request.vintage_date:
            parameters["vintage_dates"] = request.vintage_date.isoformat()
        url = f"{self.API_URL}?{urllib.parse.urlencode(parameters)}"
        payload = json.loads(self._download_text(url))
        observations = payload.get("observations", [])
        raw = pd.DataFrame(observations)
        if raw.empty:
            raise ValueError(f"FRED API returned no observations for {request.source_id}")
        vintage = request.vintage_date.isoformat() if request.vintage_date else "latest"
        return _standardize(
            raw["date"],
            raw["value"],
            request,
            retrieved_at=retrieved_at,
            vintage_date=vintage,
        )

    def _fetch_public(self, request: SeriesRequest, retrieved_at: str) -> pd.DataFrame:
        if request.vintage_date is not None:
            raise ValueError("a FRED_API_KEY is required for vintage-date requests")
        parameters = {
            "id": request.source_id,
            "cosd": request.start.isoformat(),
            "coed": request.end.isoformat(),
        }
        raw = pd.read_csv(io.StringIO(self._download_text(
            f"{self.PUBLIC_CSV_URL}?{urllib.parse.urlencode(parameters)}"
        )))
        date_column = "observation_date" if "observation_date" in raw.columns else "DATE"
        if date_column not in raw.columns or request.source_id not in raw.columns:
            raise ValueError(
                f"unexpected FRED CSV columns for {request.source_id}: {list(raw.columns)}"
            )
        return _standardize(
            raw[date_column],
            raw[request.source_id],
            request,
            retrieved_at=retrieved_at,
            vintage_date="latest",
        )

    def fetch(self, request: SeriesRequest) -> pd.DataFrame:
        request.validate()
        cache_path = self._cache_path(request)
        if cache_path.exists():
            result = pd.read_csv(
                cache_path,
                parse_dates=["timestamp", "available_at"],
            )
            result = result[STANDARD_COLUMNS]
            if self._latest_cache_needs_refresh(request, result):
                LOGGER.info("Refreshing incomplete current-date FRED snapshot %s", cache_path)
                retrieved_at = datetime.now(UTC).isoformat()
                result = (
                    self._fetch_api(request, retrieved_at)
                    if self.api_key
                    else self._fetch_public(request, retrieved_at)
                )
                result.to_csv(cache_path, index=False)
                mode = "api-refresh" if self.api_key else "public-latest-refresh"
            else:
                LOGGER.info("Loaded cached FRED snapshot %s", cache_path)
                mode = "api-cache" if self.api_key else "public-cache"
        else:
            retrieved_at = datetime.now(UTC).isoformat()
            result = (
                self._fetch_api(request, retrieved_at)
                if self.api_key
                else self._fetch_public(request, retrieved_at)
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(cache_path, index=False)
            LOGGER.info(
                "Cached %d %s observations at %s",
                len(result),
                request.source_id,
                cache_path,
            )
            mode = "api" if self.api_key else "public-latest"
        result.attrs["source_path"] = str(cache_path.resolve())
        result.attrs["provider_mode"] = mode
        return result
