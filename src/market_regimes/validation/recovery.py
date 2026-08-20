"""Quantitative recovery metrics for a known synthetic Gaussian HMM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from market_regimes.config import SyntheticConfig, ValidationConfig
from market_regimes.models.gaussian_hmm import GaussianRegimeModel
from market_regimes.validation.alignment import StateAlignment, align_states

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SyntheticRecoveryReport:
    """Aligned parameter, state, persistence, and transition-timing metrics."""

    passed: bool
    convergence_passed: bool
    mean_rmse: float
    covariance_relative_error: float
    transition_mae: float
    duration_mae: float
    filtered_accuracy: float
    mean_true_state_probability: float
    boundary_precision: float
    boundary_recall: float
    boundary_f1: float
    true_occupancy: tuple[float, ...]
    filtered_occupancy: tuple[float, ...]
    candidate_to_reference: tuple[int, ...]


def _boundaries(states: IntArray) -> IntArray:
    return (np.flatnonzero(states[1:] != states[:-1]) + 1).astype(np.int64)


def transition_timing_scores(
    true_states: IntArray,
    predicted_states: IntArray,
    tolerance: int,
) -> tuple[float, float, float]:
    """Score predicted state boundaries with one-to-one matches inside a tolerance."""
    true_boundaries = _boundaries(np.asarray(true_states, dtype=np.int64))
    predicted_boundaries = _boundaries(np.asarray(predicted_states, dtype=np.int64))
    unmatched = set(int(value) for value in true_boundaries)
    matches = 0
    for prediction in predicted_boundaries:
        candidates = [value for value in unmatched if abs(value - int(prediction)) <= tolerance]
        if candidates:
            closest = min(candidates, key=lambda value: abs(value - int(prediction)))
            unmatched.remove(closest)
            matches += 1
    precision = matches / len(predicted_boundaries) if len(predicted_boundaries) else 0.0
    recall = matches / len(true_boundaries) if len(true_boundaries) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def evaluate_recovery(
    truth: SyntheticConfig,
    true_states: IntArray,
    observations: FloatArray,
    model: GaussianRegimeModel,
    thresholds: ValidationConfig,
) -> tuple[SyntheticRecoveryReport, StateAlignment]:
    """Align a fit to the generator and apply the configured milestone gate."""
    fitted = model._require_model()
    fitted_covariances = model.covariance_matrices()
    alignment = align_states(
        truth.means,
        truth.covariances,
        np.asarray(fitted.means_, dtype=np.float64),
        fitted_covariances,
    )
    aligned_means, aligned_covariances = alignment.align_parameters(
        np.asarray(fitted.means_, dtype=np.float64), fitted_covariances
    )
    aligned_transition = alignment.align_transition_matrix(model.transition_matrix())
    filtered = alignment.align_probabilities(model.predict_filtered_probabilities(observations))
    predicted_states = np.argmax(filtered, axis=1).astype(np.int64)

    mean_rmse = float(np.sqrt(np.mean((aligned_means - truth.means) ** 2)))
    covariance_relative_error = float(
        np.linalg.norm(aligned_covariances - truth.covariances)
        / np.linalg.norm(truth.covariances)
    )
    transition_mae = float(np.mean(np.abs(aligned_transition - truth.transition_matrix)))
    true_durations = 1.0 / (1.0 - np.diag(truth.transition_matrix))
    fitted_durations = 1.0 / (1.0 - np.diag(aligned_transition))
    duration_mae = float(np.mean(np.abs(fitted_durations - true_durations)))
    true_states_array = np.asarray(true_states, dtype=np.int64)
    filtered_accuracy = float(np.mean(predicted_states == true_states_array))
    mean_true_state_probability = float(
        np.mean(filtered[np.arange(true_states_array.size), true_states_array])
    )
    precision, recall, f1 = transition_timing_scores(
        true_states_array, predicted_states, thresholds.transition_tolerance
    )
    true_occupancy_array = np.bincount(
        true_states_array, minlength=truth.start_probabilities.size
    ) / true_states_array.size
    filtered_occupancy_array = filtered.mean(axis=0)
    selected_log_likelihood = model.log_likelihood(observations)
    convergence_passed = any(
        record.converged and np.isclose(record.log_likelihood, selected_log_likelihood)
        for record in model.fit_records_
    )
    passed = bool(
        convergence_passed
        and mean_rmse <= thresholds.max_mean_rmse
        and covariance_relative_error <= thresholds.max_covariance_relative_error
        and transition_mae <= thresholds.max_transition_mae
        and filtered_accuracy >= thresholds.min_filtered_accuracy
        and f1 >= thresholds.min_boundary_f1
    )
    report = SyntheticRecoveryReport(
        passed=passed,
        convergence_passed=convergence_passed,
        mean_rmse=mean_rmse,
        covariance_relative_error=covariance_relative_error,
        transition_mae=transition_mae,
        duration_mae=duration_mae,
        filtered_accuracy=filtered_accuracy,
        mean_true_state_probability=mean_true_state_probability,
        boundary_precision=precision,
        boundary_recall=recall,
        boundary_f1=f1,
        true_occupancy=tuple(float(value) for value in true_occupancy_array),
        filtered_occupancy=tuple(float(value) for value in filtered_occupancy_array),
        candidate_to_reference=tuple(int(value) for value in alignment.candidate_to_reference),
    )
    return report, alignment

