"""Command-line runner for the complete Milestone 1 synthetic experiment."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from market_regimes.config import load_experiment_config
from market_regimes.data.synthetic import generate_synthetic_hmm
from market_regimes.logging_utils import configure_logging
from market_regimes.models.gaussian_hmm import GaussianRegimeModel
from market_regimes.plotting.synthetic import create_synthetic_diagnostic_plots
from market_regimes.validation.recovery import evaluate_recovery

LOGGER = logging.getLogger(__name__)


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def run_experiment(config_path: str | Path) -> tuple[bool, Path]:
    """Run generation, fitting, causal filtering, alignment, validation, and plots."""
    source = Path(config_path).resolve()
    config = load_experiment_config(source)
    configure_logging(config.project.log_level)
    project_root = source.parent.parent
    output_dir = (
        config.project.output_dir
        if config.project.output_dir.is_absolute()
        else project_root / config.project.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Generating %d synthetic observations", config.synthetic.n_samples)
    data = generate_synthetic_hmm(config.synthetic, config.project.seed)

    model = GaussianRegimeModel(
        config.model.n_states,
        covariance_type=config.model.covariance_type,
        n_iter=config.model.n_iter,
        tolerance=config.model.tolerance,
        min_covar=config.model.min_covar,
        seeds=config.model.seeds,
    ).fit(data.observations)
    report, alignment = evaluate_recovery(
        config.synthetic,
        data.states,
        data.observations,
        model,
        config.validation,
    )

    fitted = model._require_model()
    fitted_covariances = model.covariance_matrices()
    aligned_means, aligned_covariances = alignment.align_parameters(
        np.asarray(fitted.means_), fitted_covariances
    )
    aligned_transition = alignment.align_transition_matrix(model.transition_matrix())
    aligned_probabilities = alignment.align_probabilities(
        model.predict_filtered_probabilities(data.observations)
    )

    model.save(output_dir / "gaussian_regime_model.joblib")
    np.savez_compressed(
        output_dir / "synthetic_data.npz",
        observations=data.observations,
        states=data.states,
    )
    np.savez_compressed(
        output_dir / "aligned_parameters.npz",
        means=aligned_means,
        covariances=aligned_covariances,
        transition_matrix=aligned_transition,
        filtered_probabilities=aligned_probabilities,
    )
    with (output_dir / "recovery_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(report), handle, indent=2, default=_json_default)
    with (output_dir / "fit_records.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(record) for record in model.fit_records_], handle, indent=2)
    create_synthetic_diagnostic_plots(
        data.observations,
        data.states,
        aligned_probabilities,
        config.synthetic.transition_matrix,
        aligned_transition,
        config.synthetic.means,
        aligned_means,
        output_dir / "plots",
    )
    LOGGER.info("Synthetic recovery gate passed=%s; outputs=%s", report.passed, output_dir)
    return report.passed, output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_baseline.yaml"),
        help="Path to the YAML experiment configuration",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point; a failed recovery gate returns a nonzero exit code."""
    arguments = _parse_args()
    passed, output_dir = run_experiment(arguments.config)
    print(f"Synthetic recovery {'PASSED' if passed else 'FAILED'}: {output_dir}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

