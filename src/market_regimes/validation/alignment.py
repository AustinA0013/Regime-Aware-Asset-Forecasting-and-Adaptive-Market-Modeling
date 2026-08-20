"""Distribution-based hidden-state label alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class StateAlignment:
    """A one-to-one mapping from candidate state IDs to reference IDs."""

    candidate_to_reference: IntArray
    cost_matrix: FloatArray
    assignment_cost: float

    @property
    def reference_to_candidate(self) -> IntArray:
        """Return the inverse permutation for parameter-array reordering."""
        return np.argsort(self.candidate_to_reference).astype(np.int64)

    def align_state_sequence(self, candidate_states: IntArray) -> IntArray:
        """Replace arbitrary candidate IDs with their reference IDs."""
        states = np.asarray(candidate_states, dtype=np.int64)
        if np.any((states < 0) | (states >= self.candidate_to_reference.size)):
            raise ValueError("candidate state sequence contains an unknown state")
        return self.candidate_to_reference[states]

    def align_probabilities(self, candidate_probabilities: FloatArray) -> FloatArray:
        """Reorder candidate probability columns into reference-state order."""
        probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.candidate_to_reference.size:
            raise ValueError("probability matrix has an invalid shape")
        return probabilities[:, self.reference_to_candidate]

    def align_transition_matrix(self, candidate_transition: FloatArray) -> FloatArray:
        """Reorder both axes of a candidate transition matrix."""
        transition = np.asarray(candidate_transition, dtype=np.float64)
        inverse = self.reference_to_candidate
        if transition.shape != (inverse.size, inverse.size):
            raise ValueError("transition matrix has an invalid shape")
        return transition[np.ix_(inverse, inverse)]

    def align_parameters(
        self, candidate_means: FloatArray, candidate_covariances: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Reorder candidate emission parameters into reference-state order."""
        inverse = self.reference_to_candidate
        return candidate_means[inverse], candidate_covariances[inverse]


def _alignment_costs(
    reference_means: FloatArray,
    reference_covariances: FloatArray,
    candidate_means: FloatArray,
    candidate_covariances: FloatArray,
    covariance_weight: float,
) -> FloatArray:
    """Build a unit-aware centroid-plus-covariance distance matrix."""
    pooled_variance = np.mean(
        np.diagonal(reference_covariances, axis1=1, axis2=2), axis=0
    )
    feature_scale = np.sqrt(np.maximum(pooled_variance, 1e-12))
    scale_matrix = np.outer(feature_scale, feature_scale)
    n_states = reference_means.shape[0]
    costs = np.empty((n_states, n_states), dtype=np.float64)
    for reference in range(n_states):
        for candidate in range(n_states):
            mean_distance = np.linalg.norm(
                (reference_means[reference] - candidate_means[candidate]) / feature_scale
            )
            covariance_distance = np.linalg.norm(
                (reference_covariances[reference] - candidate_covariances[candidate])
                / scale_matrix,
                ord="fro",
            )
            costs[reference, candidate] = mean_distance + covariance_weight * covariance_distance
    return costs


def align_states(
    reference_means: FloatArray,
    reference_covariances: FloatArray,
    candidate_means: FloatArray,
    candidate_covariances: FloatArray,
    *,
    covariance_weight: float = 0.25,
) -> StateAlignment:
    """Align arbitrary HMM labels by minimum-cost linear assignment.

    The cost is a feature-scale-normalized Euclidean distance between state means
    plus ``covariance_weight`` times the corresponding normalized covariance
    Frobenius distance. The Hungarian assignment supplies the global one-to-one
    minimum rather than independent nearest neighbors.
    """
    ref_mu = np.asarray(reference_means, dtype=np.float64)
    ref_sigma = np.asarray(reference_covariances, dtype=np.float64)
    cand_mu = np.asarray(candidate_means, dtype=np.float64)
    cand_sigma = np.asarray(candidate_covariances, dtype=np.float64)
    if ref_mu.shape != cand_mu.shape or ref_mu.ndim != 2:
        raise ValueError("reference and candidate means must have the same 2D shape")
    n_states, n_features = ref_mu.shape
    expected_covariance_shape = (n_states, n_features, n_features)
    if (
        ref_sigma.shape != expected_covariance_shape
        or cand_sigma.shape != expected_covariance_shape
    ):
        raise ValueError("reference and candidate covariance shapes must match their means")
    if covariance_weight < 0:
        raise ValueError("covariance_weight cannot be negative")

    costs = _alignment_costs(ref_mu, ref_sigma, cand_mu, cand_sigma, covariance_weight)
    reference_rows, candidate_columns = linear_sum_assignment(costs)
    candidate_to_reference = np.empty(n_states, dtype=np.int64)
    candidate_to_reference[candidate_columns] = reference_rows
    return StateAlignment(
        candidate_to_reference=candidate_to_reference,
        cost_matrix=costs,
        assignment_cost=float(costs[reference_rows, candidate_columns].sum()),
    )
