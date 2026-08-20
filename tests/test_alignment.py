"""Tests that arbitrary hidden-state labels are removed correctly."""

from __future__ import annotations

import numpy as np

from market_regimes.validation.alignment import align_states


def test_hungarian_alignment_recovers_known_permutation() -> None:
    means = np.array([[-3.0, 0.0], [0.0, 3.0], [3.0, -1.0]])
    covariances = np.array([np.eye(2), 2 * np.eye(2), 0.5 * np.eye(2)])
    permutation = np.array([2, 0, 1])  # candidate index -> reference index
    inverse = np.argsort(permutation)
    candidate_means = means[permutation]
    candidate_covariances = covariances[permutation]
    alignment = align_states(means, covariances, candidate_means, candidate_covariances)
    assert np.array_equal(alignment.candidate_to_reference, permutation)
    aligned_means, aligned_covariances = alignment.align_parameters(
        candidate_means, candidate_covariances
    )
    assert np.array_equal(aligned_means, means)
    assert np.array_equal(aligned_covariances, covariances)

    transition = np.array([[0.9, 0.08, 0.02], [0.1, 0.8, 0.1], [0.03, 0.07, 0.9]])
    candidate_transition = transition[np.ix_(permutation, permutation)]
    assert np.allclose(alignment.align_transition_matrix(candidate_transition), transition)
    candidate_probabilities = np.eye(3)[inverse]
    assert np.array_equal(alignment.align_probabilities(candidate_probabilities), np.eye(3))

