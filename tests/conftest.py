"""Shared deterministic fixtures for unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from market_regimes.config import ExperimentConfig, load_experiment_config


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def baseline_config(project_root: Path) -> ExperimentConfig:
    return load_experiment_config(project_root / "configs" / "synthetic_baseline.yaml")

