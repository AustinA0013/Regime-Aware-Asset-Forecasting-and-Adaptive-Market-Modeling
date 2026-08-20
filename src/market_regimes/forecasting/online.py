"""Online supervised forecasts whose labels are released only after maturity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

from market_regimes.forecast_config import OnlineForecastConfig

FloatArray = npt.NDArray[np.float64]


@dataclass
class ExpandingBaseline:
    """Causal historical-mean and positive-frequency return baseline."""

    count: int
    return_mean: float
    positive_count: int

    @classmethod
    def fit(cls, outcomes: FloatArray) -> ExpandingBaseline:
        values = np.asarray(outcomes, dtype=np.float64)
        return cls(
            count=len(values),
            return_mean=float(values.mean()),
            positive_count=int(np.sum(values > 0)),
        )

    @property
    def positive_probability(self) -> float:
        """Laplace-smoothed probability of a positive future return."""
        return (self.positive_count + 1.0) / (self.count + 2.0)

    def update(self, outcome: float) -> None:
        """Release one newly matured return to the expanding baseline."""
        self.count += 1
        self.return_mean += (outcome - self.return_mean) / self.count
        self.positive_count += int(outcome > 0)


class AdaptiveHorizonModel:
    """Three SGD learners updated when one horizon's outcome becomes observable."""

    def __init__(
        self,
        horizon: int,
        n_states: int,
        config: OnlineForecastConfig,
    ) -> None:
        seed = config.random_seed + horizon
        common: dict[str, Any] = {
            "penalty": "l2",
            "alpha": config.regularization,
            "learning_rate": "constant",
            "eta0": config.learning_rate,
            "random_state": seed,
        }
        self.horizon = horizon
        self.n_states = n_states
        self.target_multiplier = config.target_multiplier
        self.return_model = SGDRegressor(loss="huber", epsilon=1.35, **common)
        self.direction_model = SGDClassifier(loss="log_loss", **common)
        self.regime_model = SGDClassifier(loss="log_loss", **common)
        self.initial_epochs = config.initial_epochs
        self.online_updates = 0

    def initial_fit(
        self,
        features: FloatArray,
        future_returns: FloatArray,
        future_regimes: npt.NDArray[np.int64],
    ) -> None:
        """Fit only labels whose forecast horizons finish inside training."""
        directions = (future_returns > 0).astype(np.int64)
        for _ in range(self.initial_epochs):
            self.return_model.partial_fit(features, future_returns * self.target_multiplier)
            self.direction_model.partial_fit(features, directions, classes=np.array([0, 1]))
            self.regime_model.partial_fit(
                features,
                future_regimes,
                classes=np.arange(self.n_states),
            )

    def update(self, features: FloatArray, future_return: float, future_regime: int) -> None:
        """Apply one stochastic-gradient update from a newly matured forecast target."""
        row = np.asarray(features, dtype=np.float64).reshape(1, -1)
        self.return_model.partial_fit(row, np.array([future_return * self.target_multiplier]))
        self.direction_model.partial_fit(row, np.array([int(future_return > 0)]))
        self.regime_model.partial_fit(row, np.array([future_regime]))
        self.online_updates += 1

    def predict(self, features: FloatArray) -> tuple[float, float, FloatArray]:
        """Predict cumulative return, positive-return probability, and future regime."""
        row = np.asarray(features, dtype=np.float64).reshape(1, -1)
        predicted_return = float(self.return_model.predict(row)[0] / self.target_multiplier)
        up_probability = float(self.direction_model.predict_proba(row)[0, 1])
        regime_probability = np.zeros(self.n_states, dtype=np.float64)
        raw_regime_probability = self.regime_model.predict_proba(row)[0]
        regime_probability[self.regime_model.classes_.astype(int)] = raw_regime_probability
        return predicted_return, up_probability, regime_probability


@dataclass(frozen=True)
class AdaptiveForecastResult:
    """Forecast ledger and final state of every adaptive learner."""

    ledger: pd.DataFrame
    scaler: StandardScaler
    models: dict[int, AdaptiveHorizonModel]
    baselines: dict[int, ExpandingBaseline]
    input_features: tuple[str, ...]
    initial_training_rows: dict[int, int]


def _future_targets(
    returns: FloatArray,
    regimes: npt.NDArray[np.int64],
    horizon: int,
) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """Create close-to-future-close targets excluding the origin day's return."""
    n_rows = len(returns)
    future_returns = np.full(n_rows, np.nan, dtype=np.float64)
    future_regimes = np.full(n_rows, -1, dtype=np.int64)
    for origin in range(n_rows - horizon):
        future_returns[origin] = float(returns[origin + 1 : origin + horizon + 1].sum())
        future_regimes[origin] = int(regimes[origin + horizon])
    return future_returns, future_regimes


def _validate_inputs(features: pd.DataFrame, probabilities: pd.DataFrame) -> None:
    if features.empty or probabilities.empty:
        raise ValueError("features and probabilities cannot be empty")
    if not features.index.equals(probabilities.index):
        raise ValueError("features and probabilities must have identical time indices")
    if not features.index.is_monotonic_increasing or features.index.has_duplicates:
        raise ValueError("time index must be strictly increasing")
    if not np.all(np.isfinite(features.to_numpy(dtype=float))):
        raise ValueError("features must contain only finite values")
    probability_values = probabilities.to_numpy(dtype=float)
    if np.any(probability_values < 0) or not np.allclose(probability_values.sum(axis=1), 1.0):
        raise ValueError("state probabilities must be nonnegative and sum to one")
    if "log_return_1d" not in features:
        raise ValueError("features must contain log_return_1d")


def walk_forward_forecast(
    features: pd.DataFrame,
    probabilities: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    test_start: str | pd.Timestamp,
    config: OnlineForecastConfig,
) -> AdaptiveForecastResult:
    """Train, predict, settle, and update without exposing an outcome before maturity."""
    config.validate()
    _validate_inputs(features, probabilities)
    dates = pd.DatetimeIndex(features.index)
    train_end_timestamp = pd.Timestamp(train_end)
    test_start_timestamp = pd.Timestamp(test_start)
    train_mask = dates <= train_end_timestamp
    test_positions = np.flatnonzero(dates >= test_start_timestamp)
    if train_mask.sum() < 100 or len(test_positions) == 0:
        raise ValueError("walk-forward forecast requires at least 100 train rows and one test row")
    test_start_position = int(test_positions[0])

    probability_features = probabilities.copy()
    probability_features.columns = [
        f"filtered_state_probability_{i}" for i in range(len(probabilities.columns))
    ]
    model_inputs = pd.concat([features, probability_features], axis=1)
    input_features = tuple(str(column) for column in model_inputs.columns)
    scaler = StandardScaler().fit(model_inputs.loc[train_mask])
    scaled_inputs = scaler.transform(model_inputs)
    returns = features["log_return_1d"].to_numpy(dtype=np.float64)
    regimes = np.argmax(probabilities.to_numpy(dtype=np.float64), axis=1).astype(np.int64)
    n_states = probabilities.shape[1]

    models: dict[int, AdaptiveHorizonModel] = {}
    baselines: dict[int, ExpandingBaseline] = {}
    targets: dict[int, tuple[FloatArray, npt.NDArray[np.int64]]] = {}
    training_cutoffs: dict[int, int] = {}
    initial_training_rows: dict[int, int] = {}
    for horizon in config.horizons:
        future_returns, future_regimes = _future_targets(returns, regimes, horizon)
        targets[horizon] = (future_returns, future_regimes)
        outcome_positions = np.arange(len(dates)) + horizon
        initial_mask = (
            (outcome_positions < len(dates))
            & (dates <= train_end_timestamp)
            & (dates[np.minimum(outcome_positions, len(dates) - 1)] <= train_end_timestamp)
            & np.isfinite(future_returns)
        )
        origins = np.flatnonzero(initial_mask)
        if len(origins) < 100:
            raise ValueError(f"horizon {horizon} has fewer than 100 matured training examples")
        model = AdaptiveHorizonModel(horizon, n_states, config)
        model.initial_fit(
            scaled_inputs[origins],
            future_returns[origins],
            future_regimes[origins],
        )
        models[horizon] = model
        baselines[horizon] = ExpandingBaseline.fit(future_returns[origins])
        training_cutoffs[horizon] = int(origins[-1])
        initial_training_rows[horizon] = len(origins)

    records: list[dict[str, Any]] = []
    for current_position in range(test_start_position, len(dates)):
        for horizon in config.horizons:
            future_returns, future_regimes = targets[horizon]
            matured_origin = current_position - horizon
            if matured_origin > training_cutoffs[horizon]:
                matured_return = future_returns[matured_origin]
                if np.isfinite(matured_return):
                    models[horizon].update(
                        scaled_inputs[matured_origin],
                        float(matured_return),
                        int(future_regimes[matured_origin]),
                    )
                    baselines[horizon].update(float(matured_return))

        for horizon in config.horizons:
            future_returns, future_regimes = targets[horizon]
            predicted_return, up_probability, regime_probability = models[horizon].predict(
                scaled_inputs[current_position]
            )
            outcome_position = current_position + horizon
            resolved = outcome_position < len(dates)
            if resolved:
                forecast_for = dates[outcome_position]
                actual_return = float(future_returns[current_position])
                actual_regime = int(future_regimes[current_position])
            else:
                forecast_for = dates[current_position] + BDay(horizon)
                actual_return = np.nan
                actual_regime = -1
            record: dict[str, Any] = {
                "origin_date": dates[current_position],
                "forecast_for_date": forecast_for,
                "forecast_date_is_estimate": not resolved,
                "horizon_trading_days": horizon,
                "resolved": resolved,
                "predicted_cumulative_log_return": predicted_return,
                "baseline_predicted_cumulative_log_return": baselines[horizon].return_mean,
                "predicted_up_probability": up_probability,
                "baseline_up_probability": baselines[horizon].positive_probability,
                "predicted_regime": int(np.argmax(regime_probability)),
                "current_regime": int(regimes[current_position]),
                "actual_cumulative_log_return": actual_return,
                "actual_direction_up": int(actual_return > 0) if resolved else np.nan,
                "actual_regime": actual_regime if resolved else np.nan,
            }
            for state in range(n_states):
                record[f"predicted_regime_probability_{state}"] = regime_probability[state]
                record[f"persistence_regime_probability_{state}"] = probabilities.iloc[
                    current_position, state
                ]
            records.append(record)

    ledger = pd.DataFrame.from_records(records).sort_values(
        ["origin_date", "horizon_trading_days"], kind="stable"
    )
    ledger["return_error"] = (
        ledger["actual_cumulative_log_return"]
        - ledger["predicted_cumulative_log_return"]
    )
    ledger["absolute_return_error"] = ledger["return_error"].abs()
    return AdaptiveForecastResult(
        ledger=ledger.reset_index(drop=True),
        scaler=scaler,
        models=models,
        baselines=baselines,
        input_features=input_features,
        initial_training_rows=initial_training_rows,
    )


def _binary_log_loss(actual: FloatArray, probability: FloatArray) -> float:
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped)))


def evaluate_forecast_ledger(
    ledger: pd.DataFrame,
    n_states: int,
) -> dict[str, dict[str, float | int]]:
    """Compare adaptive forecasts with causal expanding and persistence baselines."""
    results: dict[str, dict[str, float | int]] = {}
    identity = np.eye(n_states)
    for horizon in sorted(ledger["horizon_trading_days"].unique()):
        resolved = ledger.loc[
            (ledger["horizon_trading_days"] == horizon) & ledger["resolved"]
        ].copy()
        actual_return = resolved["actual_cumulative_log_return"].to_numpy(dtype=float)
        predicted_return = resolved["predicted_cumulative_log_return"].to_numpy(dtype=float)
        baseline_return = resolved[
            "baseline_predicted_cumulative_log_return"
        ].to_numpy(dtype=float)
        actual_direction = resolved["actual_direction_up"].to_numpy(dtype=float)
        up_probability = resolved["predicted_up_probability"].to_numpy(dtype=float)
        baseline_up_probability = resolved["baseline_up_probability"].to_numpy(dtype=float)
        actual_regime = resolved["actual_regime"].to_numpy(dtype=int)
        regime_probability = resolved[
            [f"predicted_regime_probability_{state}" for state in range(n_states)]
        ].to_numpy(dtype=float)
        persistence_probability = resolved[
            [f"persistence_regime_probability_{state}" for state in range(n_states)]
        ].to_numpy(dtype=float)
        one_hot_regime = identity[actual_regime]
        return_correlation = (
            float(np.corrcoef(actual_return, predicted_return)[0, 1])
            if np.std(actual_return) > 0 and np.std(predicted_return) > 0
            else float("nan")
        )
        results[str(int(horizon))] = {
            "resolved_forecasts": len(resolved),
            "pending_forecasts": int(
                np.sum(
                    (ledger["horizon_trading_days"] == horizon)
                    & ~ledger["resolved"]
                )
            ),
            "return_mae": float(np.mean(np.abs(actual_return - predicted_return))),
            "baseline_return_mae": float(np.mean(np.abs(actual_return - baseline_return))),
            "return_rmse": float(np.sqrt(np.mean((actual_return - predicted_return) ** 2))),
            "baseline_return_rmse": float(
                np.sqrt(np.mean((actual_return - baseline_return) ** 2))
            ),
            "return_correlation": return_correlation,
            "directional_accuracy": float(
                np.mean((up_probability >= 0.5) == actual_direction)
            ),
            "baseline_directional_accuracy": float(
                np.mean((baseline_up_probability >= 0.5) == actual_direction)
            ),
            "direction_brier_score": float(np.mean((up_probability - actual_direction) ** 2)),
            "baseline_direction_brier_score": float(
                np.mean((baseline_up_probability - actual_direction) ** 2)
            ),
            "direction_log_loss": _binary_log_loss(actual_direction, up_probability),
            "baseline_direction_log_loss": _binary_log_loss(
                actual_direction, baseline_up_probability
            ),
            "regime_accuracy": float(
                np.mean(np.argmax(regime_probability, axis=1) == actual_regime)
            ),
            "persistence_regime_accuracy": float(
                np.mean(np.argmax(persistence_probability, axis=1) == actual_regime)
            ),
            "regime_brier_score": float(
                np.mean(np.sum((regime_probability - one_hot_regime) ** 2, axis=1))
            ),
            "persistence_regime_brier_score": float(
                np.mean(np.sum((persistence_probability - one_hot_regime) ** 2, axis=1))
            ),
        }
    return results
