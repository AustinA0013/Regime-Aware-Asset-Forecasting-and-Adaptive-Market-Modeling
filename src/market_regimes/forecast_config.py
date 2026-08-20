"""Validated configuration for the adaptive walk-forward forecasting milestone."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ForecastProjectConfig:
    """Artifact and logging settings."""

    output_dir: Path
    log_level: str


@dataclass(frozen=True)
class ForecastSourceConfig:
    """Real-data experiment dependency and chronological forecast boundary."""

    real_config: Path
    test_start: date


@dataclass(frozen=True)
class OnlineForecastConfig:
    """Online learner horizons and deterministic SGD hyperparameters."""

    horizons: tuple[int, ...]
    initial_epochs: int
    random_seed: int
    learning_rate: float
    regularization: float
    target_multiplier: float
    rolling_metric_window: int

    def validate(self) -> None:
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("forecast horizons must be unique and strictly increasing")
        if any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("forecast horizons must be positive trading-day counts")
        if self.initial_epochs <= 0:
            raise ValueError("initial_epochs must be positive")
        if self.learning_rate <= 0 or self.regularization < 0:
            raise ValueError("learning_rate must be positive and regularization nonnegative")
        if self.target_multiplier <= 0:
            raise ValueError("target_multiplier must be positive")
        if self.rolling_metric_window < 20:
            raise ValueError("rolling_metric_window must be at least 20")


@dataclass(frozen=True)
class AdaptiveExperimentConfig:
    """Complete adaptive experiment configuration."""

    project: ForecastProjectConfig
    source: ForecastSourceConfig
    forecast: OnlineForecastConfig

    def validate(self) -> None:
        self.forecast.validate()
        if not self.project.log_level:
            raise ValueError("project.log_level cannot be empty")


def _require_mapping(raw: Any, context: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping")
    return raw


def _exact_keys(raw: dict[str, Any], expected: set[str], context: str) -> None:
    if set(raw) != expected:
        raise ValueError(
            f"{context} keys must be exactly {sorted(expected)}; received {sorted(raw)}"
        )


def load_adaptive_experiment_config(path: str | Path) -> AdaptiveExperimentConfig:
    """Load a strict adaptive-forecast YAML file."""
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as handle:
        root = _require_mapping(yaml.safe_load(handle), "configuration")
    _exact_keys(root, {"project", "source", "forecast"}, "configuration")

    project = _require_mapping(root["project"], "project")
    source = _require_mapping(root["source"], "source")
    forecast = _require_mapping(root["forecast"], "forecast")
    _exact_keys(project, {"output_dir", "log_level"}, "project")
    _exact_keys(source, {"real_config", "test_start"}, "source")
    _exact_keys(
        forecast,
        {
            "horizons",
            "initial_epochs",
            "random_seed",
            "learning_rate",
            "regularization",
            "target_multiplier",
            "rolling_metric_window",
        },
        "forecast",
    )

    config = AdaptiveExperimentConfig(
        project=ForecastProjectConfig(
            output_dir=Path(project["output_dir"]),
            log_level=str(project["log_level"]),
        ),
        source=ForecastSourceConfig(
            real_config=Path(source["real_config"]),
            test_start=date.fromisoformat(str(source["test_start"])),
        ),
        forecast=OnlineForecastConfig(
            horizons=tuple(int(value) for value in forecast["horizons"]),
            initial_epochs=int(forecast["initial_epochs"]),
            random_seed=int(forecast["random_seed"]),
            learning_rate=float(forecast["learning_rate"]),
            regularization=float(forecast["regularization"]),
            target_multiplier=float(forecast["target_multiplier"]),
            rolling_metric_window=int(forecast["rolling_metric_window"]),
        ),
    )
    config.validate()
    return config
