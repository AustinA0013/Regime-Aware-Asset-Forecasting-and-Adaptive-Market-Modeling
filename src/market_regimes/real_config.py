"""Configuration for the real-data regime diagnostic milestone."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


@dataclass(frozen=True)
class RealProjectConfig:
    """Reproducibility, output, and logging settings."""

    seed: int
    output_dir: Path
    log_level: str


@dataclass(frozen=True)
class RealDataConfig:
    """FRED series, interval, availability assumptions, and split date."""

    start: date
    end: date
    train_end: date
    cache_dir: Path
    allow_latest_vintage: bool
    series: dict[str, str]
    availability_lag_days: dict[str, int]
    max_staleness_days: dict[str, int]

    def validate(self) -> None:
        expected_roles = {
            "price",
            "vix",
            "treasury_10y",
            "treasury_2y",
            "credit_spread",
        }
        if set(self.series) != expected_roles:
            raise ValueError(f"data.series must define exactly {sorted(expected_roles)}")
        if not self.start < self.train_end < self.end:
            raise ValueError("require data.start < data.train_end < data.end")
        if not self.allow_latest_vintage:
            raise ValueError(
                "the public diagnostic requires explicit allow_latest_vintage=true; "
                "use an API-key vintage workflow for point-in-time research"
            )
        for series_id in self.series.values():
            if series_id not in self.availability_lag_days:
                raise ValueError(f"missing availability lag for {series_id}")
            if self.availability_lag_days[series_id] < 0:
                raise ValueError("availability lags cannot be negative")
        auxiliary = set(self.series.values()) - {self.series["price"]}
        if not auxiliary.issubset(self.max_staleness_days):
            raise ValueError("every auxiliary series requires a max-staleness rule")


@dataclass(frozen=True)
class FeatureConfig:
    """Causal rolling-feature definitions."""

    realized_volatility_windows: tuple[int, int]
    momentum_windows: tuple[int, int]
    drawdown_window: int
    annualization_factor: int

    def validate(self) -> None:
        windows = (*self.realized_volatility_windows, *self.momentum_windows)
        if any(window < 2 for window in windows) or self.drawdown_window < 2:
            raise ValueError("all rolling windows must be at least two observations")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")


@dataclass(frozen=True)
class RealModelConfig:
    """Frozen HMM specification for this diagnostic, not model selection."""

    n_states: int
    covariance_type: str
    n_iter: int
    tolerance: float
    min_covar: float
    seeds: tuple[int, ...]

    def validate(self) -> None:
        if self.n_states < 2:
            raise ValueError("model.n_states must be at least two")
        if self.covariance_type not in {"diag", "full"}:
            raise ValueError("covariance_type must be 'diag' or 'full'")
        if self.n_iter < 1 or self.tolerance <= 0 or self.min_covar <= 0:
            raise ValueError("model optimization settings must be positive")
        if not self.seeds:
            raise ValueError("at least one deterministic model seed is required")


@dataclass(frozen=True)
class RealExperimentConfig:
    """Complete real-data diagnostic configuration."""

    project: RealProjectConfig
    data: RealDataConfig
    features: FeatureConfig
    model: RealModelConfig

    def validate(self) -> None:
        self.data.validate()
        self.features.validate()
        self.model.validate()


def _date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def resolve_data_end(value: Any, *, now: datetime | None = None) -> date:
    """Resolve a fixed ISO date or the dynamic ``latest`` request boundary.

    ``latest`` deliberately requests the current New York calendar date rather than
    guessing whether FRED has published a same-day close. The provider safely returns
    the latest observation available on or before the requested date.
    """
    if str(value).strip().lower() != "latest":
        return _date(value)
    reference = now or datetime.now(ZoneInfo("America/New_York"))
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=ZoneInfo("America/New_York"))
    return reference.astimezone(ZoneInfo("America/New_York")).date()


def load_real_experiment_config(path: str | Path) -> RealExperimentConfig:
    """Load and validate a human-readable real-data experiment definition."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    expected = {"project", "data", "features", "model"}
    if set(raw) != expected:
        raise ValueError(f"configuration sections must be exactly {sorted(expected)}")
    project = raw["project"]
    data = raw["data"]
    features = raw["features"]
    model = raw["model"]
    config = RealExperimentConfig(
        project=RealProjectConfig(
            seed=int(project["seed"]),
            output_dir=Path(project["output_dir"]),
            log_level=str(project.get("log_level", "INFO")).upper(),
        ),
        data=RealDataConfig(
            start=_date(data["start"]),
            end=resolve_data_end(data["end"]),
            train_end=_date(data["train_end"]),
            cache_dir=Path(data["cache_dir"]),
            allow_latest_vintage=bool(data["allow_latest_vintage"]),
            series={str(key): str(value) for key, value in data["series"].items()},
            availability_lag_days={
                str(key): int(value) for key, value in data["availability_lag_days"].items()
            },
            max_staleness_days={
                str(key): int(value) for key, value in data["max_staleness_days"].items()
            },
        ),
        features=FeatureConfig(
            realized_volatility_windows=tuple(
                int(value) for value in features["realized_volatility_windows"]
            ),
            momentum_windows=tuple(int(value) for value in features["momentum_windows"]),
            drawdown_window=int(features["drawdown_window"]),
            annualization_factor=int(features["annualization_factor"]),
        ),
        model=RealModelConfig(
            n_states=int(model["n_states"]),
            covariance_type=str(model["covariance_type"]),
            n_iter=int(model["n_iter"]),
            tolerance=float(model["tolerance"]),
            min_covar=float(model["min_covar"]),
            seeds=tuple(int(value) for value in model["seeds"]),
        ),
    )
    config.validate()
    return config
