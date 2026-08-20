"""Strict configuration for causal individual-asset forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from market_regimes.real_config import resolve_data_end


@dataclass(frozen=True)
class AssetProjectConfig:
    output_dir: Path
    log_level: str


@dataclass(frozen=True)
class AssetSourceConfig:
    real_config: Path


@dataclass(frozen=True)
class AssetDataConfig:
    provider: str
    start: date
    end: date
    cache_dir: Path
    benchmark: str
    tickers: tuple[str, ...]
    timeout_seconds: int

    def validate(self) -> None:
        if self.provider != "yahoo_chart":
            raise ValueError("the initial asset provider must be yahoo_chart")
        if self.start > self.end:
            raise ValueError("asset data start cannot be after end")
        if self.timeout_seconds <= 0:
            raise ValueError("asset provider timeout_seconds must be positive")
        if not self.tickers or len(set(self.tickers)) != len(self.tickers):
            raise ValueError("asset tickers must be nonempty and unique")
        if self.benchmark not in self.tickers:
            raise ValueError("benchmark must be included in asset tickers")
        if any(not ticker or ticker != ticker.upper() for ticker in self.tickers):
            raise ValueError("asset tickers must be nonempty uppercase symbols")


@dataclass(frozen=True)
class AssetFeatureConfig:
    volatility_windows: tuple[int, ...]
    momentum_windows: tuple[int, ...]
    drawdown_window: int
    relative_strength_windows: tuple[int, ...]
    beta_window: int
    correlation_window: int
    volume_window: int
    volume_change_window: int
    annualization_factor: int

    def validate(self) -> None:
        windows = (
            self.volatility_windows
            + self.momentum_windows
            + self.relative_strength_windows
        )
        if any(value <= 1 for value in windows):
            raise ValueError("rolling feature windows must exceed one observation")
        if any(
            value <= 1
            for value in (
                self.drawdown_window,
                self.beta_window,
                self.correlation_window,
                self.volume_window,
                self.volume_change_window,
            )
        ):
            raise ValueError("asset feature windows must exceed one observation")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")


@dataclass(frozen=True)
class AssetOnlineConfig:
    horizons: tuple[int, ...]
    initial_epochs: int
    random_seed: int
    learning_rate: float
    regularization: float
    target_multiplier: float
    min_initial_samples: int
    rolling_metric_window: int

    def validate(self) -> None:
        if tuple(sorted(set(self.horizons))) != self.horizons or not self.horizons:
            raise ValueError("asset horizons must be nonempty, unique, and increasing")
        if any(value <= 0 for value in self.horizons):
            raise ValueError("asset horizons must be positive")
        if self.initial_epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("initial_epochs and learning_rate must be positive")
        if self.regularization < 0 or self.target_multiplier <= 0:
            raise ValueError("regularization must be nonnegative and multiplier positive")
        if self.min_initial_samples < 50:
            raise ValueError("min_initial_samples must be at least 50")
        if self.rolling_metric_window < 20:
            raise ValueError("rolling_metric_window must be at least 20")


@dataclass(frozen=True)
class AnchoredFoldConfig:
    fold_id: str
    train_end: date
    test_start: date
    test_end: date

    def validate(self) -> None:
        if not self.fold_id:
            raise ValueError("fold_id cannot be empty")
        if not self.train_end < self.test_start <= self.test_end:
            raise ValueError("fold dates must satisfy train_end < test_start <= test_end")


@dataclass(frozen=True)
class AssetEvaluationConfig:
    folds: tuple[AnchoredFoldConfig, ...]
    top_k: int
    min_cross_sectional_assets: int

    def validate(self) -> None:
        if not self.folds or len({fold.fold_id for fold in self.folds}) != len(self.folds):
            raise ValueError("anchored folds must be nonempty with unique IDs")
        for fold in self.folds:
            fold.validate()
        ordered = sorted(self.folds, key=lambda fold: fold.test_start)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous.test_end >= current.test_start:
                raise ValueError("anchored test intervals cannot overlap")
            if current.train_end < previous.test_end:
                raise ValueError("each later fold must train through the earlier test interval")
        if self.top_k <= 0 or self.min_cross_sectional_assets < 3:
            raise ValueError("top_k must be positive and min_cross_sectional_assets at least 3")


@dataclass(frozen=True)
class AssetStatisticsConfig:
    min_state_samples: int
    bootstrap_samples: int
    bootstrap_block_length: int

    def validate(self) -> None:
        if self.min_state_samples < 20:
            raise ValueError("min_state_samples must be at least 20")
        if self.bootstrap_samples < 50 or self.bootstrap_block_length < 2:
            raise ValueError("bootstrap requires at least 50 samples and block length at least 2")


@dataclass(frozen=True)
class AssetExperimentConfig:
    project: AssetProjectConfig
    source: AssetSourceConfig
    data: AssetDataConfig
    features: AssetFeatureConfig
    forecast: AssetOnlineConfig
    evaluation: AssetEvaluationConfig
    statistics: AssetStatisticsConfig

    def validate(self) -> None:
        if not self.project.log_level:
            raise ValueError("project log_level cannot be empty")
        self.data.validate()
        self.features.validate()
        self.forecast.validate()
        self.evaluation.validate()
        self.statistics.validate()
        if self.evaluation.top_k >= len(self.data.tickers):
            raise ValueError("top_k must be smaller than the configured universe")


def _mapping(raw: Any, context: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a mapping")
    return raw


def _keys(raw: dict[str, Any], expected: set[str], context: str) -> None:
    if set(raw) != expected:
        raise ValueError(
            f"{context} keys must be exactly {sorted(expected)}; received {sorted(raw)}"
        )


def _int_tuple(raw: Any, context: str) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{context} must be a list")
    return tuple(int(value) for value in raw)


def load_asset_experiment_config(path: str | Path) -> AssetExperimentConfig:
    """Load a strict YAML definition for the asset milestone."""
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as handle:
        root = _mapping(yaml.safe_load(handle), "configuration")
    _keys(
        root,
        {"project", "source", "data", "features", "forecast", "evaluation", "statistics"},
        "configuration",
    )
    project = _mapping(root["project"], "project")
    source = _mapping(root["source"], "source")
    data = _mapping(root["data"], "data")
    features = _mapping(root["features"], "features")
    forecast = _mapping(root["forecast"], "forecast")
    evaluation = _mapping(root["evaluation"], "evaluation")
    statistics = _mapping(root["statistics"], "statistics")
    _keys(project, {"output_dir", "log_level"}, "project")
    _keys(source, {"real_config"}, "source")
    _keys(
        data,
        {"provider", "start", "end", "cache_dir", "benchmark", "tickers", "timeout_seconds"},
        "data",
    )
    _keys(
        features,
        {
            "volatility_windows",
            "momentum_windows",
            "drawdown_window",
            "relative_strength_windows",
            "beta_window",
            "correlation_window",
            "volume_window",
            "volume_change_window",
            "annualization_factor",
        },
        "features",
    )
    _keys(
        forecast,
        {
            "horizons",
            "initial_epochs",
            "random_seed",
            "learning_rate",
            "regularization",
            "target_multiplier",
            "min_initial_samples",
            "rolling_metric_window",
        },
        "forecast",
    )
    _keys(evaluation, {"folds", "top_k", "min_cross_sectional_assets"}, "evaluation")
    _keys(
        statistics,
        {"min_state_samples", "bootstrap_samples", "bootstrap_block_length"},
        "statistics",
    )
    raw_tickers = data["tickers"]
    if not isinstance(raw_tickers, list):
        raise ValueError("data.tickers must be a list")
    raw_folds = evaluation["folds"]
    if not isinstance(raw_folds, list):
        raise ValueError("evaluation.folds must be a list")
    folds: list[AnchoredFoldConfig] = []
    for index, raw_fold in enumerate(raw_folds):
        fold = _mapping(raw_fold, f"evaluation.folds[{index}]")
        _keys(fold, {"id", "train_end", "test_start", "test_end"}, f"fold {index}")
        folds.append(
            AnchoredFoldConfig(
                fold_id=str(fold["id"]),
                train_end=date.fromisoformat(str(fold["train_end"])),
                test_start=date.fromisoformat(str(fold["test_start"])),
                test_end=resolve_data_end(fold["test_end"]),
            )
        )
    config = AssetExperimentConfig(
        project=AssetProjectConfig(Path(project["output_dir"]), str(project["log_level"])),
        source=AssetSourceConfig(Path(source["real_config"])),
        data=AssetDataConfig(
            provider=str(data["provider"]),
            start=date.fromisoformat(str(data["start"])),
            end=resolve_data_end(data["end"]),
            cache_dir=Path(data["cache_dir"]),
            benchmark=str(data["benchmark"]).upper(),
            tickers=tuple(str(value).upper() for value in raw_tickers),
            timeout_seconds=int(data["timeout_seconds"]),
        ),
        features=AssetFeatureConfig(
            volatility_windows=_int_tuple(features["volatility_windows"], "volatility_windows"),
            momentum_windows=_int_tuple(features["momentum_windows"], "momentum_windows"),
            drawdown_window=int(features["drawdown_window"]),
            relative_strength_windows=_int_tuple(
                features["relative_strength_windows"], "relative_strength_windows"
            ),
            beta_window=int(features["beta_window"]),
            correlation_window=int(features["correlation_window"]),
            volume_window=int(features["volume_window"]),
            volume_change_window=int(features["volume_change_window"]),
            annualization_factor=int(features["annualization_factor"]),
        ),
        forecast=AssetOnlineConfig(
            horizons=_int_tuple(forecast["horizons"], "horizons"),
            initial_epochs=int(forecast["initial_epochs"]),
            random_seed=int(forecast["random_seed"]),
            learning_rate=float(forecast["learning_rate"]),
            regularization=float(forecast["regularization"]),
            target_multiplier=float(forecast["target_multiplier"]),
            min_initial_samples=int(forecast["min_initial_samples"]),
            rolling_metric_window=int(forecast["rolling_metric_window"]),
        ),
        evaluation=AssetEvaluationConfig(
            folds=tuple(folds),
            top_k=int(evaluation["top_k"]),
            min_cross_sectional_assets=int(evaluation["min_cross_sectional_assets"]),
        ),
        statistics=AssetStatisticsConfig(
            min_state_samples=int(statistics["min_state_samples"]),
            bootstrap_samples=int(statistics["bootstrap_samples"]),
            bootstrap_block_length=int(statistics["bootstrap_block_length"]),
        ),
    )
    config.validate()
    return config
