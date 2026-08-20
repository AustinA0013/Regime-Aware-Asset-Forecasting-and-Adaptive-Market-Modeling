"""Adaptive walk-forward forecasts driven by causal HMM state probabilities."""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from market_regimes.forecast_config import (
    AdaptiveExperimentConfig,
    load_adaptive_experiment_config,
)
from market_regimes.forecasting.ledger import merge_forecast_ledgers
from market_regimes.forecasting.online import evaluate_forecast_ledger, walk_forward_forecast
from market_regimes.logging_utils import configure_logging
from market_regimes.plotting.forecast import create_adaptive_forecast_plots
from market_regimes.real_experiment import run_real_experiment

LOGGER = logging.getLogger(__name__)
MAX_LOCK_AGE_SECONDS = 6 * 60 * 60


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _latest_forecast_table(
    ledger: pd.DataFrame,
    state_labels: dict[int, str],
) -> pd.DataFrame:
    latest_date = ledger["origin_date"].max()
    latest = ledger.loc[ledger["origin_date"] == latest_date].copy()
    latest["predicted_regime_label"] = latest["predicted_regime"].map(state_labels)
    columns = [
        "origin_date",
        "forecast_for_date",
        "forecast_date_is_estimate",
        "horizon_trading_days",
        "predicted_cumulative_log_return",
        "predicted_up_probability",
        "predicted_regime",
        "predicted_regime_label",
    ]
    columns.extend(
        f"predicted_regime_probability_{state}" for state in sorted(state_labels)
    )
    return latest.loc[:, columns].sort_values("horizon_trading_days")


@contextmanager
def _exclusive_run_lock(path: Path):
    """Prevent overlapping scheduled runs and recover an abandoned old lock."""
    if path.exists():
        age_seconds = datetime.now(UTC).timestamp() - path.stat().st_mtime
        if age_seconds <= MAX_LOCK_AGE_SECONDS:
            raise RuntimeError(f"another adaptive forecast run holds {path}")
        path.unlink()
        LOGGER.warning("Removed stale adaptive forecast lock %s", path)
    handle = path.open("x", encoding="utf-8")
    try:
        handle.write(datetime.now(UTC).isoformat())
        handle.flush()
        yield
    finally:
        handle.close()
        path.unlink(missing_ok=True)


def _append_run_history(path: Path, record: dict[str, Any]) -> None:
    history = pd.read_csv(path) if path.exists() else pd.DataFrame()
    history = pd.concat([history, pd.DataFrame([record])], ignore_index=True)
    history.to_csv(path, index=False)


def _run_adaptive_locked(
    config: AdaptiveExperimentConfig,
    project_root: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Run one daily refresh while the caller holds the project lock."""
    real_config_path = (
        config.source.real_config
        if config.source.real_config.is_absolute()
        else project_root / config.source.real_config
    )
    real_output_dir, real_metrics = run_real_experiment(real_config_path)

    features = pd.read_csv(
        real_output_dir / "causal_features.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )
    probabilities = pd.read_csv(
        real_output_dir / "filtered_probabilities.csv",
        parse_dates=["timestamp"],
        index_col="timestamp",
    )
    forecast_result = walk_forward_forecast(
        features,
        probabilities,
        train_end=real_metrics["train_end"],
        test_start=config.source.test_start,
        config=config.forecast,
    )
    replay_ledger = forecast_result.ledger
    ledger_path = output_dir / "forecast_ledger.csv"
    existing_ledger = (
        pd.read_csv(
            ledger_path,
            parse_dates=["origin_date", "forecast_for_date"],
        )
        if ledger_path.exists()
        else None
    )
    ledger, merge_audit = merge_forecast_ledgers(existing_ledger, replay_ledger)
    n_states = probabilities.shape[1]
    performance = evaluate_forecast_ledger(ledger, n_states)
    state_labels = {int(state): str(label) for state, label in real_metrics["state_labels"].items()}
    latest = _latest_forecast_table(ledger, state_labels)

    metrics: dict[str, Any] = {
        "status": "daily_adaptive_forecast_no_strategy",
        "generated_at": datetime.now(UTC).isoformat(),
        "feedback_contract": (
            "A forecast target is withheld until its configured trading-day horizon has elapsed; "
            "only then is one SGD update allowed."
        ),
        "real_data_artifact": real_output_dir,
        "train_end": real_metrics["train_end"],
        "test_start": config.source.test_start.isoformat(),
        "last_origin_date": pd.Timestamp(ledger["origin_date"].max()),
        "horizons_trading_days": config.forecast.horizons,
        "input_features": forecast_result.input_features,
        "initial_training_rows": forecast_result.initial_training_rows,
        "online_updates": {
            horizon: model.online_updates for horizon, model in forecast_result.models.items()
        },
        "state_labels": state_labels,
        "performance": performance,
        "daily_run": {
            "existing_ledger_rows": merge_audit.existing_rows,
            "new_forecasts": merge_audit.new_forecasts,
            "newly_resolved": merge_audit.newly_resolved,
            "final_ledger_rows": merge_audit.final_rows,
            "latest_source_observation": pd.Timestamp(features.index.max()),
        },
        "limitations": [
            "Future regime targets are later HMM classifications, not objective market labels.",
            "Latest-vintage FRED history can contain revisions.",
            "No news, earnings text, transaction costs, or trading strategy is included.",
            "Forecast skill must be judged against baselines; adaptation does not guarantee skill.",
        ],
    }

    ledger.to_csv(ledger_path, index=False)
    latest.to_csv(output_dir / "latest_forecasts.csv", index=False)
    joblib.dump(
        {
            "scaler": forecast_result.scaler,
            "models": forecast_result.models,
            "baselines": forecast_result.baselines,
            "input_features": forecast_result.input_features,
            "state_labels": state_labels,
            "config": config,
        },
        output_dir / "adaptive_forecasters.joblib",
    )
    with (output_dir / "adaptive_forecast_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=_json_default)
    create_adaptive_forecast_plots(
        ledger,
        performance,
        state_labels,
        rolling_window=config.forecast.rolling_metric_window,
        output_dir=output_dir / "plots",
    )
    _append_run_history(
        output_dir / "daily_run_history.csv",
        {
            "run_at_utc": metrics["generated_at"],
            "latest_source_observation": pd.Timestamp(features.index.max()).isoformat(),
            "new_forecasts": merge_audit.new_forecasts,
            "newly_resolved": merge_audit.newly_resolved,
            "ledger_rows": merge_audit.final_rows,
            "status": "success",
        },
    )
    LOGGER.info("Adaptive forecast outputs written to %s", output_dir)
    return output_dir, metrics


def run_adaptive_experiment(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Refresh latest data and safely reconcile one append-only daily forecast run."""
    source = Path(config_path).resolve()
    config = load_adaptive_experiment_config(source)
    configure_logging(config.project.log_level)
    project_root = source.parent.parent
    output_dir = (
        config.project.output_dir
        if config.project.output_dir.is_absolute()
        else project_root / config.project.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with _exclusive_run_lock(output_dir / ".daily_run.lock"):
        return _run_adaptive_locked(config, project_root, output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/adaptive_forecast.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the configured adaptive walk-forward forecast experiment."""
    arguments = _parse_args()
    output_dir, _ = run_adaptive_experiment(arguments.config)
    print(f"Adaptive forecast diagnostic complete: {output_dir}")


if __name__ == "__main__":
    main()
