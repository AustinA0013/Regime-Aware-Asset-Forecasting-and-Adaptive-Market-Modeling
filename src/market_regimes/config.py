"""Typed, validated configuration for the synthetic milestone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class ProjectConfig:
    """Project-wide reproducibility and output settings."""

    seed: int
    output_dir: Path
    log_level: str = "INFO"


@dataclass(frozen=True)
class SyntheticConfig:
    """Known Gaussian HMM used to generate a validation data set."""

    n_samples: int
    start_probabilities: FloatArray
    transition_matrix: FloatArray
    means: FloatArray
    covariances: FloatArray

    def validate(self) -> None:
        """Raise ``ValueError`` when dimensions or probabilities are invalid."""
        if self.n_samples < 2:
            raise ValueError("synthetic.n_samples must be at least 2")
        n_states = self.start_probabilities.shape[0]
        if self.start_probabilities.ndim != 1 or n_states < 2:
            raise ValueError("start_probabilities must be a one-dimensional vector")
        if self.transition_matrix.shape != (n_states, n_states):
            raise ValueError("transition_matrix must have shape (n_states, n_states)")
        if self.means.ndim != 2 or self.means.shape[0] != n_states:
            raise ValueError("means must have shape (n_states, n_features)")
        n_features = self.means.shape[1]
        if self.covariances.shape != (n_states, n_features, n_features):
            raise ValueError(
                "covariances must have shape (n_states, n_features, n_features)"
            )
        if np.any(self.start_probabilities < 0) or not np.isclose(
            self.start_probabilities.sum(), 1.0
        ):
            raise ValueError("start_probabilities must be nonnegative and sum to one")
        if np.any(self.transition_matrix < 0) or not np.allclose(
            self.transition_matrix.sum(axis=1), 1.0
        ):
            raise ValueError("each transition_matrix row must be nonnegative and sum to one")
        for state, covariance in enumerate(self.covariances):
            if not np.allclose(covariance, covariance.T):
                raise ValueError(f"covariance for state {state} must be symmetric")
            if np.min(np.linalg.eigvalsh(covariance)) <= 0:
                raise ValueError(f"covariance for state {state} must be positive definite")


@dataclass(frozen=True)
class ModelConfig:
    """Gaussian HMM fit settings, including deterministic EM restarts."""

    n_states: int
    covariance_type: str
    n_iter: int
    tolerance: float
    min_covar: float
    seeds: tuple[int, ...]

    def validate(self) -> None:
        if self.n_states < 2:
            raise ValueError("model.n_states must be at least 2")
        if self.covariance_type not in {"diag", "full"}:
            raise ValueError("model.covariance_type must be 'diag' or 'full'")
        if self.n_iter < 1 or self.tolerance <= 0 or self.min_covar <= 0:
            raise ValueError("iteration, tolerance, and covariance settings must be positive")
        if not self.seeds:
            raise ValueError("model.seeds must contain at least one deterministic seed")


@dataclass(frozen=True)
class ValidationConfig:
    """Explicit statistical recovery thresholds for the milestone gate."""

    transition_tolerance: int
    max_mean_rmse: float
    max_covariance_relative_error: float
    max_transition_mae: float
    min_filtered_accuracy: float
    min_boundary_f1: float

    def validate(self) -> None:
        if self.transition_tolerance < 0:
            raise ValueError("transition_tolerance cannot be negative")
        if min(
            self.max_mean_rmse,
            self.max_covariance_relative_error,
            self.max_transition_mae,
        ) <= 0:
            raise ValueError("maximum-error thresholds must be positive")
        if not 0 <= self.min_filtered_accuracy <= 1:
            raise ValueError("min_filtered_accuracy must be in [0, 1]")
        if not 0 <= self.min_boundary_f1 <= 1:
            raise ValueError("min_boundary_f1 must be in [0, 1]")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration for one synthetic validation run."""

    project: ProjectConfig
    synthetic: SyntheticConfig
    model: ModelConfig
    validation: ValidationConfig

    def validate(self) -> None:
        self.synthetic.validate()
        self.model.validate()
        self.validation.validate()
        if self.model.n_states != self.synthetic.start_probabilities.size:
            raise ValueError("model.n_states must match the synthetic generator")


def _array(value: Any) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


def _check_sections(raw: dict[str, Any]) -> None:
    expected = {"project", "synthetic", "model", "validation"}
    missing = expected - raw.keys()
    unknown = raw.keys() - expected
    if missing:
        raise ValueError(f"missing configuration sections: {sorted(missing)}")
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment specification without mutating global state."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    _check_sections(raw)

    project_raw = raw["project"]
    synthetic_raw = raw["synthetic"]
    model_raw = raw["model"]
    validation_raw = raw["validation"]
    config = ExperimentConfig(
        project=ProjectConfig(
            seed=int(project_raw["seed"]),
            output_dir=Path(project_raw["output_dir"]),
            log_level=str(project_raw.get("log_level", "INFO")).upper(),
        ),
        synthetic=SyntheticConfig(
            n_samples=int(synthetic_raw["n_samples"]),
            start_probabilities=_array(synthetic_raw["start_probabilities"]),
            transition_matrix=_array(synthetic_raw["transition_matrix"]),
            means=_array(synthetic_raw["means"]),
            covariances=_array(synthetic_raw["covariances"]),
        ),
        model=ModelConfig(
            n_states=int(model_raw["n_states"]),
            covariance_type=str(model_raw["covariance_type"]),
            n_iter=int(model_raw["n_iter"]),
            tolerance=float(model_raw["tolerance"]),
            min_covar=float(model_raw["min_covar"]),
            seeds=tuple(int(seed) for seed in model_raw["seeds"]),
        ),
        validation=ValidationConfig(
            transition_tolerance=int(validation_raw["transition_tolerance"]),
            max_mean_rmse=float(validation_raw["max_mean_rmse"]),
            max_covariance_relative_error=float(
                validation_raw["max_covariance_relative_error"]
            ),
            max_transition_mae=float(validation_raw["max_transition_mae"]),
            min_filtered_accuracy=float(validation_raw["min_filtered_accuracy"]),
            min_boundary_f1=float(validation_raw["min_boundary_f1"]),
        ),
    )
    config.validate()
    return config

