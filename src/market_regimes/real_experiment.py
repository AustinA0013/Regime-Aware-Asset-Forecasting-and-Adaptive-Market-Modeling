"""Chronological real-financial-data regime diagnostic without a trading strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from market_regimes.data.providers import FREDDataProvider, SeriesRequest
from market_regimes.features.market import build_market_features, feature_columns
from market_regimes.logging_utils import configure_logging
from market_regimes.models.gaussian_hmm import GaussianRegimeModel
from market_regimes.plotting.real import create_real_diagnostic_plots
from market_regimes.real_config import RealExperimentConfig, load_real_experiment_config

LOGGER = logging.getLogger(__name__)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weighted_centroids(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    denominators = probabilities.sum(axis=0)
    if np.any(denominators <= 0):
        raise ValueError("a state has zero posterior mass and cannot be characterized")
    return probabilities.T @ values / denominators[:, np.newaxis]


def _state_labels(
    standardized_centroids: pd.DataFrame,
    config: RealExperimentConfig,
) -> tuple[dict[int, str], dict[int, float]]:
    """Assign descriptive labels from training statistics only.

    The transparent stress ranking is not a fitted model or selection criterion:
    high VIX, long-window volatility, and corporate credit spreads increase the score;
    momentum and drawdown increase stress when they are negative.
    """
    volatility_long = config.features.realized_volatility_windows[1]
    momentum_long = config.features.momentum_windows[1]
    drawdown_window = config.features.drawdown_window
    scores = (
        standardized_centroids["vix_level"]
        + standardized_centroids[f"realized_volatility_{volatility_long}d"]
        + standardized_centroids["credit_spread"]
        - standardized_centroids[f"momentum_{momentum_long}d"]
        - standardized_centroids[f"drawdown_{drawdown_window}d"]
    )
    ordered = scores.sort_values().index.tolist()
    descriptions = ["Risk-On-like", "Transitional-like", "Risk-Off-like"]
    if len(ordered) != len(descriptions):
        raise ValueError("the descriptive labeling rule currently requires exactly three states")
    labels = {int(state): label for state, label in zip(ordered, descriptions, strict=True)}
    return labels, {int(state): float(score) for state, score in scores.items()}


def _normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-15, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1) / np.log(probabilities.shape[1])


def _load_series(
    config: RealExperimentConfig,
    project_root: Path,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    cache_dir = (
        config.data.cache_dir
        if config.data.cache_dir.is_absolute()
        else project_root / config.data.cache_dir
    )
    provider = FREDDataProvider(cache_dir)
    series: dict[str, pd.DataFrame] = {}
    manifest: list[dict[str, Any]] = []
    for role, source_id in config.data.series.items():
        frame = provider.fetch(
            SeriesRequest(
                source_id=source_id,
                start=config.data.start,
                end=config.data.end,
                availability_lag_days=config.data.availability_lag_days[source_id],
            )
        )
        source_path = Path(frame.attrs["source_path"])
        manifest.append(
            {
                "role": role,
                "source_id": source_id,
                "provider_mode": frame.attrs["provider_mode"],
                "cache_path": str(source_path),
                "sha256": _sha256(source_path),
                "rows": len(frame),
                "first_observation": frame["timestamp"].min(),
                "last_observation": frame["timestamp"].max(),
                "vintage_date": str(frame["vintage_date"].iloc[0]),
                "retrieved_at": str(frame["retrieved_at"].iloc[0]),
                "availability_lag_days": config.data.availability_lag_days[source_id],
            }
        )
        series[role] = frame
    return series, manifest


def run_real_experiment(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Download/cache, transform, train, freeze, filter, characterize, and report."""
    source = Path(config_path).resolve()
    config = load_real_experiment_config(source)
    configure_logging(config.project.log_level)
    project_root = source.parent.parent
    output_dir = (
        config.project.output_dir
        if config.project.output_dir.is_absolute()
        else project_root / config.project.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    series, manifest = _load_series(config, project_root)
    built = build_market_features(
        series,
        config.features,
        config.data.max_staleness_days,
    )
    expected_features = feature_columns(config.features)
    complete = built.features.dropna(how="any").copy()
    complete = complete.loc[:, expected_features]
    levels = built.aligned_levels.reindex(complete.index)
    train_end = pd.Timestamp(config.data.train_end)
    train = complete.loc[complete.index <= train_end]
    test = complete.loc[complete.index > train_end]
    if len(train) < 500 or len(test) < 100:
        raise ValueError(
            f"insufficient complete observations after split: train={len(train)}, test={len(test)}"
        )
    LOGGER.info(
        "Complete feature rows=%d; train=%d (%s to %s); test=%d (%s to %s)",
        len(complete),
        len(train),
        train.index.min().date(),
        train.index.max().date(),
        len(test),
        test.index.min().date(),
        test.index.max().date(),
    )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    model_config = config.model
    model = GaussianRegimeModel(
        model_config.n_states,
        covariance_type=model_config.covariance_type,
        n_iter=model_config.n_iter,
        tolerance=model_config.tolerance,
        min_covar=model_config.min_covar,
        seeds=model_config.seeds,
    ).fit(train_scaled)

    all_scaled = np.vstack([train_scaled, test_scaled])
    filtered = model.filter(all_scaled)
    train_probabilities = filtered.probabilities[: len(train)]
    test_probabilities = filtered.probabilities[len(train) :]
    raw_centroids = _weighted_centroids(train.to_numpy(), train_probabilities)
    standardized_centroids_array = _weighted_centroids(train_scaled, train_probabilities)
    raw_centroids_frame = pd.DataFrame(
        raw_centroids,
        columns=expected_features,
        index=pd.Index(range(model_config.n_states), name="state"),
    )
    standardized_centroids = pd.DataFrame(
        standardized_centroids_array,
        columns=expected_features,
        index=pd.Index(range(model_config.n_states), name="state"),
    )
    labels, stress_scores = _state_labels(standardized_centroids, config)
    durations = model.expected_state_duration()
    train_occupancy = train_probabilities.mean(axis=0)
    test_occupancy = test_probabilities.mean(axis=0)
    characteristics = raw_centroids_frame.copy()
    characteristics.insert(0, "label", [labels[state] for state in characteristics.index])
    characteristics.insert(
        1,
        "stress_score",
        [stress_scores[state] for state in characteristics.index],
    )
    characteristics.insert(2, "expected_duration", durations)
    characteristics.insert(3, "train_occupancy", train_occupancy)
    characteristics.insert(4, "test_occupancy", test_occupancy)

    test_entropy = _normalized_entropy(test_probabilities)
    test_maximum_probability = test_probabilities.max(axis=1)
    test_states = np.argmax(test_probabilities, axis=1)
    train_log_likelihood = float(filtered.log_normalizers[: len(train)].sum())
    test_log_likelihood = float(filtered.log_normalizers[len(train) :].sum())
    metrics: dict[str, Any] = {
        "status": "diagnostic_only_no_strategy",
        "latest_vintage_acknowledged": config.data.allow_latest_vintage,
        "train_start": train.index.min(),
        "train_end": train.index.max(),
        "test_start": test.index.min(),
        "test_end": test.index.max(),
        "train_rows": len(train),
        "test_rows": len(test),
        "features": expected_features,
        "train_log_likelihood": train_log_likelihood,
        "train_log_likelihood_per_observation": train_log_likelihood / len(train),
        "conditional_test_log_likelihood": test_log_likelihood,
        "conditional_test_log_likelihood_per_observation": test_log_likelihood / len(test),
        "test_mean_maximum_probability": float(test_maximum_probability.mean()),
        "test_mean_normalized_entropy": float(test_entropy.mean()),
        "test_fraction_max_probability_below_0_60": float(
            np.mean(test_maximum_probability < 0.60)
        ),
        "test_fraction_max_probability_below_0_80": float(
            np.mean(test_maximum_probability < 0.80)
        ),
        "test_argmax_transitions": int(np.sum(test_states[1:] != test_states[:-1])),
        "transition_matrix": model.transition_matrix(),
        "expected_durations": durations,
        "state_labels": labels,
        "stress_scores": stress_scores,
        "train_occupancy": train_occupancy,
        "test_occupancy": test_occupancy,
        "fit_records": [asdict(record) for record in model.fit_records_],
        "fill_audit": [asdict(audit) for audit in built.fill_audit],
        "data_manifest": manifest,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }

    combined_probabilities = np.vstack([train_probabilities, test_probabilities])
    model.save(output_dir / "real_gaussian_regime_model.joblib")
    joblib.dump(scaler, output_dir / "training_standard_scaler.joblib")
    complete.to_csv(output_dir / "causal_features.csv", index_label="timestamp")
    characteristics.to_csv(output_dir / "state_characteristics.csv")
    pd.DataFrame(
        combined_probabilities,
        index=complete.index,
        columns=[f"state_{state}" for state in range(model_config.n_states)],
    ).to_csv(output_dir / "filtered_probabilities.csv", index_label="timestamp")
    with (output_dir / "real_data_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=_json_default)
    with (output_dir / "data_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=_json_default)
    create_real_diagnostic_plots(
        levels=levels,
        features=complete,
        probabilities=combined_probabilities,
        state_labels=labels,
        transition_matrix=model.transition_matrix(),
        standardized_centroids=standardized_centroids,
        train_end=train_end,
        output_dir=output_dir / "plots",
    )
    LOGGER.info("Real-data diagnostic outputs written to %s", output_dir)
    return output_dir, metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/real_data_diagnostic.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    """Run the configured diagnostic and print its artifact location."""
    arguments = _parse_args()
    output_dir, _ = run_real_experiment(arguments.config)
    print(f"Real-data diagnostic complete: {output_dir}")


if __name__ == "__main__":
    main()
