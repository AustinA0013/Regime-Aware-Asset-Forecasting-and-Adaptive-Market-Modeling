"""A reproducible hmmlearn wrapper with an independent causal filter."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
from hmmlearn.hmm import GaussianHMM

from market_regimes.models.filtering import FilterResult, forward_filter

LOGGER = logging.getLogger(__name__)
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HMMFitRecord:
    """Auditable result from one EM initialization."""

    seed: int
    log_likelihood: float
    converged: bool
    n_iterations: int
    final_improvement: float | None


class GaussianRegimeModel:
    """Fit a multivariate Gaussian HMM and expose causal inference methods.

    EM optimization is performed by :class:`hmmlearn.hmm.GaussianHMM`. Historical
    state probabilities are *not* delegated to hmmlearn: :meth:`filter` calls the
    explicit forward-only recursion in ``models.filtering``.
    """

    def __init__(
        self,
        n_states: int,
        *,
        covariance_type: str = "full",
        n_iter: int = 300,
        tolerance: float = 1e-4,
        min_covar: float = 1e-4,
        seeds: tuple[int, ...] = (0,),
    ) -> None:
        if n_states < 2:
            raise ValueError("n_states must be at least 2")
        if covariance_type not in {"diag", "full"}:
            raise ValueError("covariance_type must be 'diag' or 'full'")
        if not seeds:
            raise ValueError("at least one seed is required")
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tolerance = tolerance
        self.min_covar = min_covar
        self.seeds = tuple(seeds)
        self.model_: GaussianHMM | None = None
        self.fit_records_: list[HMMFitRecord] = []

    @staticmethod
    def _check_observations(
        observations: FloatArray, *, min_samples: int = 1
    ) -> FloatArray:
        x = np.asarray(observations, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] < min_samples:
            raise ValueError("observations must have shape (n_samples, n_features)")
        if not np.all(np.isfinite(x)):
            raise ValueError("observations must contain only finite values")
        return x

    def _require_model(self) -> GaussianHMM:
        if self.model_ is None:
            raise RuntimeError("fit or load the model before inference")
        return self.model_

    def fit(self, observations: FloatArray) -> GaussianRegimeModel:
        """Fit all configured EM restarts and retain the maximum-likelihood fit."""
        x = self._check_observations(observations, min_samples=2)
        if x.shape[0] <= self.n_states:
            raise ValueError("the number of samples must exceed the number of states")
        candidates: list[tuple[float, GaussianHMM]] = []
        records: list[HMMFitRecord] = []
        for seed in self.seeds:
            candidate = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                min_covar=self.min_covar,
                n_iter=self.n_iter,
                tol=self.tolerance,
                random_state=seed,
                implementation="log",
            )
            candidate.fit(x)
            score = float(candidate.score(x))
            history = tuple(float(value) for value in candidate.monitor_.history)
            final_improvement = history[-1] - history[-2] if len(history) >= 2 else None
            tolerance_reached = bool(
                final_improvement is not None
                and -1e-8 <= final_improvement < self.tolerance
            )
            record = HMMFitRecord(
                seed=seed,
                log_likelihood=score,
                converged=tolerance_reached,
                n_iterations=int(candidate.monitor_.iter),
                final_improvement=final_improvement,
            )
            records.append(record)
            if np.isfinite(score):
                candidates.append((score, candidate))
            LOGGER.info(
                "HMM restart seed=%d log_likelihood=%.3f converged=%s iterations=%d",
                seed,
                score,
                record.converged,
                record.n_iterations,
            )
        if not candidates:
            raise RuntimeError("all HMM initializations produced non-finite likelihoods")
        self.fit_records_ = records
        self.model_ = max(candidates, key=lambda item: item[0])[1]
        return self

    def covariance_matrices(self) -> FloatArray:
        """Return all covariance types as explicit ``(K, d, d)`` matrices."""
        model = self._require_model()
        covariances = np.asarray(model.covars_, dtype=np.float64)
        if covariances.ndim == 3:
            return covariances.copy()
        if covariances.ndim == 2 and covariances.shape[0] == self.n_states:
            return np.stack([np.diag(row) for row in covariances])
        raise RuntimeError(f"unexpected hmmlearn covariance shape: {covariances.shape}")

    def emission_parameters(self) -> tuple[FloatArray, FloatArray]:
        """Return defensive copies of state means and full covariance matrices."""
        means = np.asarray(self._require_model().means_, dtype=np.float64).copy()
        return means, self.covariance_matrices()

    def filter(self, observations: FloatArray) -> FilterResult:
        """Run the independent causal forward filter with frozen parameters."""
        x = self._check_observations(observations)
        model = self._require_model()
        return forward_filter(
            x,
            np.asarray(model.startprob_, dtype=np.float64),
            np.asarray(model.transmat_, dtype=np.float64),
            np.asarray(model.means_, dtype=np.float64),
            self.covariance_matrices(),
        )

    def predict_filtered_probabilities(self, observations: FloatArray) -> FloatArray:
        """Return ``P(z_t=k | x_0, ..., x_t)`` for every historical time."""
        return self.filter(observations).probabilities

    def log_likelihood(self, observations: FloatArray) -> float:
        """Evaluate the frozen HMM sequence log likelihood."""
        x = self._check_observations(observations)
        return float(self._require_model().score(x))

    def transition_matrix(self) -> FloatArray:
        """Return a defensive copy of the fitted row-stochastic matrix."""
        return np.asarray(self._require_model().transmat_, dtype=np.float64).copy()

    def expected_state_duration(self) -> FloatArray:
        r"""Return :math:`E[D_k]=1/(1-A_{kk})`, including infinity if absorbing."""
        diagonal = np.diag(self.transition_matrix())
        with np.errstate(divide="ignore"):
            return 1.0 / (1.0 - diagonal)

    def state_statistics(self, observations: FloatArray | None = None) -> list[dict[str, Any]]:
        """Summarize emissions, persistence, and optional filtered occupancy."""
        model = self._require_model()
        occupancy = None
        if observations is not None:
            occupancy = self.predict_filtered_probabilities(observations).mean(axis=0)
        durations = self.expected_state_duration()
        covariances = self.covariance_matrices()
        return [
            {
                "state": state,
                "mean": np.asarray(model.means_[state]).tolist(),
                "covariance": covariances[state].tolist(),
                "self_transition_probability": float(model.transmat_[state, state]),
                "expected_duration": float(durations[state]),
                "filtered_occupancy": (
                    None if occupancy is None else float(occupancy[state])
                ),
            }
            for state in range(self.n_states)
        ]

    def save(self, path: str | Path) -> None:
        """Persist the fitted wrapper. Only load model files from trusted sources."""
        self._require_model()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> GaussianRegimeModel:
        """Load a wrapper created by :meth:`save` from a trusted local file."""
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError("serialized object is not a GaussianRegimeModel")
        model._require_model()
        return model
