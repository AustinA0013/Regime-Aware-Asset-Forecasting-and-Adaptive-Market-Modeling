"""Per-asset online learners with strictly delayed target release."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler

from market_regimes.asset_config import AnchoredFoldConfig, AssetOnlineConfig
from market_regimes.assets.targets import future_asset_targets

FloatArray = npt.NDArray[np.float64]


@dataclass
class ExpandingOutcomeBaseline:
    count: int
    mean: float
    positive_count: int

    @classmethod
    def fit(cls, outcomes: FloatArray) -> ExpandingOutcomeBaseline:
        values = np.asarray(outcomes, dtype=np.float64)
        if len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError("expanding baseline requires finite outcomes")
        return cls(len(values), float(values.mean()), int(np.sum(values > 0)))

    @property
    def positive_probability(self) -> float:
        return (self.positive_count + 1.0) / (self.count + 2.0)

    def update(self, outcome: float) -> None:
        if not np.isfinite(outcome):
            raise ValueError("baseline update requires a finite outcome")
        self.count += 1
        self.mean += (outcome - self.mean) / self.count
        self.positive_count += int(outcome > 0)


@dataclass
class RegimeConditionedBaseline:
    weight: FloatArray
    outcome_sum: FloatArray
    positive_sum: FloatArray

    @classmethod
    def fit(
        cls,
        outcomes: FloatArray,
        probabilities: FloatArray,
    ) -> RegimeConditionedBaseline:
        values = np.asarray(outcomes, dtype=np.float64)
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape[0] != len(values) or probs.ndim != 2:
            raise ValueError("regime baseline outcomes/probabilities do not align")
        return cls(
            weight=probs.sum(axis=0),
            outcome_sum=probs.T @ values,
            positive_sum=probs.T @ (values > 0).astype(float),
        )

    def predict(self, current_probabilities: FloatArray, fallback: float) -> float:
        state_means = np.divide(
            self.outcome_sum,
            self.weight,
            out=np.full_like(self.outcome_sum, fallback),
            where=self.weight > 0,
        )
        return float(np.asarray(current_probabilities) @ state_means)

    def positive_probability(
        self,
        current_probabilities: FloatArray,
        fallback: float,
    ) -> float:
        # Beta(1,1) smoothing in every state keeps probabilities away from 0/1.
        state_probability = (self.positive_sum + 1.0) / (self.weight + 2.0)
        value = float(np.asarray(current_probabilities) @ state_probability)
        return float(np.clip(value if np.isfinite(value) else fallback, 1e-6, 1 - 1e-6))

    def update(self, outcome: float, probabilities: FloatArray) -> None:
        probs = np.asarray(probabilities, dtype=np.float64)
        self.weight += probs
        self.outcome_sum += probs * outcome
        self.positive_sum += probs * int(outcome > 0)


class AssetHorizonModel:
    """Return, direction, excess-return, and outperform classifiers for one horizon."""

    def __init__(self, ticker: str, horizon: int, config: AssetOnlineConfig) -> None:
        stable_ticker_seed = zlib.crc32(ticker.encode("ascii")) % 100_000
        common: dict[str, Any] = {
            "penalty": "l2",
            "alpha": config.regularization,
            "learning_rate": "constant",
            "eta0": config.learning_rate,
            "random_state": config.random_seed + stable_ticker_seed + horizon,
        }
        self.target_multiplier = config.target_multiplier
        self.return_model = SGDRegressor(loss="huber", epsilon=1.35, **common)
        self.direction_model = SGDClassifier(loss="log_loss", **common)
        self.excess_model = SGDRegressor(loss="huber", epsilon=1.35, **common)
        self.outperform_model = SGDClassifier(loss="log_loss", **common)
        self.initial_epochs = config.initial_epochs
        self.online_updates = 0

    def initial_fit(
        self,
        features: FloatArray,
        returns: FloatArray,
        excess_returns: FloatArray,
    ) -> None:
        directions = (returns > 0).astype(np.int64)
        outperform = (excess_returns > 0).astype(np.int64)
        classes = np.array([0, 1], dtype=np.int64)
        for _ in range(self.initial_epochs):
            self.return_model.partial_fit(features, returns * self.target_multiplier)
            self.direction_model.partial_fit(features, directions, classes=classes)
            self.excess_model.partial_fit(features, excess_returns * self.target_multiplier)
            self.outperform_model.partial_fit(features, outperform, classes=classes)

    def update(self, features: FloatArray, future_return: float, future_excess: float) -> None:
        row = np.asarray(features, dtype=np.float64).reshape(1, -1)
        self.return_model.partial_fit(row, [future_return * self.target_multiplier])
        self.direction_model.partial_fit(row, [int(future_return > 0)])
        self.excess_model.partial_fit(row, [future_excess * self.target_multiplier])
        self.outperform_model.partial_fit(row, [int(future_excess > 0)])
        self.online_updates += 1

    def predict(self, features: FloatArray) -> tuple[float, float, float, float]:
        row = np.asarray(features, dtype=np.float64).reshape(1, -1)
        predicted_return = float(self.return_model.predict(row)[0] / self.target_multiplier)
        probability_up = float(self.direction_model.predict_proba(row)[0, 1])
        predicted_excess = float(self.excess_model.predict(row)[0] / self.target_multiplier)
        probability_outperform = float(self.outperform_model.predict_proba(row)[0, 1])
        return predicted_return, probability_up, predicted_excess, probability_outperform


@dataclass(frozen=True)
class AssetWalkForwardResult:
    ledger: pd.DataFrame
    scaler: StandardScaler
    models: dict[int, AssetHorizonModel]
    input_features: tuple[str, ...]
    initial_training_rows: dict[int, int]


def _validate_context(
    market_features: pd.DataFrame,
    probabilities: pd.DataFrame,
    asset_features: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    if not market_features.index.equals(probabilities.index):
        raise ValueError("market features and regime probabilities must share an index")
    if len(probabilities.columns) != 3:
        raise ValueError("asset layer currently requires the existing three-state HMM")
    probability_values = probabilities.to_numpy(dtype=float)
    if np.any(probability_values < 0) or not np.allclose(
        probability_values.sum(axis=1), 1.0
    ):
        raise ValueError("regime probabilities must be nonnegative and sum to one")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("price calendar must be strictly increasing")
    if asset_features.index.has_duplicates:
        raise ValueError("asset feature rows cannot have duplicate dates")


def walk_forward_asset_fold(
    *,
    ticker: str,
    benchmark: str,
    market_features: pd.DataFrame,
    probabilities: pd.DataFrame,
    asset_features: pd.DataFrame,
    prices: pd.DataFrame,
    fold: AnchoredFoldConfig,
    config: AssetOnlineConfig,
) -> AssetWalkForwardResult:
    """Train at an anchor, then predict/update chronologically inside one test era."""
    config.validate()
    _validate_context(market_features, probabilities, asset_features, prices)
    dates = pd.DatetimeIndex(prices.index)
    if ticker not in prices or benchmark not in prices:
        raise ValueError("asset and benchmark prices are required")
    market = market_features.reindex(dates).add_prefix("market__")
    regimes = probabilities.reindex(dates).copy()
    regimes.columns = [f"regime_probability_{index}" for index in range(regimes.shape[1])]
    local = asset_features.reindex(dates)
    inputs = pd.concat([market, regimes, local], axis=1)
    input_features = tuple(str(column) for column in inputs.columns)
    if len(input_features) != 30:
        raise ValueError(
            f"asset input contract requires 30 features; received {len(input_features)}"
        )
    input_values = inputs.to_numpy(dtype=float)
    finite_inputs = np.all(np.isfinite(input_values), axis=1)
    train_end = pd.Timestamp(fold.train_end)
    test_start = pd.Timestamp(fold.test_start)
    test_end = pd.Timestamp(fold.test_end)
    train_mask = (dates <= train_end) & finite_inputs
    test_positions = np.flatnonzero((dates >= test_start) & (dates <= test_end))
    if train_mask.sum() < config.min_initial_samples or len(test_positions) == 0:
        raise ValueError(
            f"{ticker}/{fold.fold_id} has insufficient train/test inputs: "
            f"train={train_mask.sum()} test={len(test_positions)}"
        )
    scaler = StandardScaler().fit(inputs.loc[train_mask])
    scaled = np.full_like(input_values, np.nan, dtype=np.float64)
    scaled[finite_inputs] = scaler.transform(inputs.loc[finite_inputs])
    target_results = {
        horizon: future_asset_targets(prices, benchmark=benchmark, horizon=horizon)
        for horizon in config.horizons
    }
    models: dict[int, AssetHorizonModel] = {}
    expanding_return: dict[int, ExpandingOutcomeBaseline] = {}
    expanding_excess: dict[int, ExpandingOutcomeBaseline] = {}
    regime_return: dict[int, RegimeConditionedBaseline] = {}
    regime_excess: dict[int, RegimeConditionedBaseline] = {}
    trained_origins: dict[int, set[int]] = {}
    initial_training_rows: dict[int, int] = {}
    probability_values = probabilities.reindex(dates).to_numpy(dtype=float)

    for horizon, target in target_results.items():
        future_return = target.cumulative_return[ticker].to_numpy(dtype=float)
        future_excess = target.cumulative_excess_return[ticker].to_numpy(dtype=float)
        target_dates = pd.to_datetime(target.target_dates).to_numpy()
        initial_mask = (
            finite_inputs
            & np.isfinite(future_return)
            & np.isfinite(future_excess)
            & (dates <= train_end)
            & (target_dates <= np.datetime64(train_end))
        )
        origins = np.flatnonzero(initial_mask)
        if len(origins) < config.min_initial_samples:
            raise ValueError(
                f"{ticker}/{fold.fold_id}/{horizon}d has only {len(origins)} matured training rows"
            )
        model = AssetHorizonModel(ticker, horizon, config)
        model.initial_fit(scaled[origins], future_return[origins], future_excess[origins])
        models[horizon] = model
        expanding_return[horizon] = ExpandingOutcomeBaseline.fit(future_return[origins])
        expanding_excess[horizon] = ExpandingOutcomeBaseline.fit(future_excess[origins])
        regime_return[horizon] = RegimeConditionedBaseline.fit(
            future_return[origins], probability_values[origins]
        )
        regime_excess[horizon] = RegimeConditionedBaseline.fit(
            future_excess[origins], probability_values[origins]
        )
        trained_origins[horizon] = set(int(value) for value in origins)
        initial_training_rows[horizon] = len(origins)

    records: list[dict[str, object]] = []
    for current_position in test_positions:
        # Settle exactly the target that matures now, before issuing today's prediction.
        for horizon, target in target_results.items():
            matured_origin = current_position - horizon
            if matured_origin < 0 or matured_origin in trained_origins[horizon]:
                continue
            future_return = float(target.cumulative_return[ticker].iloc[matured_origin])
            future_excess = float(
                target.cumulative_excess_return[ticker].iloc[matured_origin]
            )
            if not finite_inputs[matured_origin] or not (
                np.isfinite(future_return) and np.isfinite(future_excess)
            ):
                trained_origins[horizon].add(matured_origin)
                continue
            models[horizon].update(scaled[matured_origin], future_return, future_excess)
            expanding_return[horizon].update(future_return)
            expanding_excess[horizon].update(future_excess)
            regime_return[horizon].update(future_return, probability_values[matured_origin])
            regime_excess[horizon].update(future_excess, probability_values[matured_origin])
            trained_origins[horizon].add(matured_origin)

        if not finite_inputs[current_position]:
            continue
        current_probability = probability_values[current_position]
        for horizon, target in target_results.items():
            predicted_return, probability_up, predicted_excess, probability_outperform = (
                models[horizon].predict(scaled[current_position])
            )
            expanding_return_prediction = expanding_return[horizon].mean
            expanding_excess_prediction = expanding_excess[horizon].mean
            regime_return_prediction = regime_return[horizon].predict(
                current_probability, expanding_return_prediction
            )
            regime_excess_prediction = regime_excess[horizon].predict(
                current_probability, expanding_excess_prediction
            )
            outcome_position = current_position + horizon
            target_within_data = outcome_position < len(dates)
            target_date = (
                dates[outcome_position]
                if target_within_data
                else dates[current_position] + BDay(horizon)
            )
            resolved = bool(target_within_data and target_date <= test_end)
            actual_return = (
                float(target.cumulative_return[ticker].iloc[current_position])
                if resolved
                else np.nan
            )
            actual_excess = (
                float(target.cumulative_excess_return[ticker].iloc[current_position])
                if resolved
                else np.nan
            )
            actual_benchmark = (
                float(target.cumulative_return[benchmark].iloc[current_position])
                if resolved
                else np.nan
            )
            if resolved and not (
                np.isfinite(actual_return)
                and np.isfinite(actual_excess)
                and np.isfinite(actual_benchmark)
            ):
                resolved = False
                actual_return = actual_excess = actual_benchmark = np.nan
            record: dict[str, object] = {
                "fold_id": fold.fold_id,
                "origin_date": dates[current_position],
                "forecast_for_date": target_date,
                "forecast_date_is_estimate": not target_within_data,
                "ticker": ticker,
                "benchmark": benchmark,
                "horizon_trading_days": horizon,
                "resolved": resolved,
                "current_price": float(prices[ticker].iloc[current_position]),
                "predicted_cumulative_log_return": predicted_return,
                "expanding_baseline_return": expanding_return_prediction,
                "regime_baseline_return": regime_return_prediction,
                "predicted_up_probability": probability_up,
                "expanding_baseline_up_probability": expanding_return[
                    horizon
                ].positive_probability,
                "regime_baseline_up_probability": regime_return[
                    horizon
                ].positive_probability(
                    current_probability,
                    expanding_return[horizon].positive_probability,
                ),
                "predicted_excess_log_return": predicted_excess,
                "expanding_baseline_excess_return": expanding_excess_prediction,
                "regime_baseline_excess_return": regime_excess_prediction,
                "predicted_outperform_probability": probability_outperform,
                "expanding_baseline_outperform_probability": expanding_excess[
                    horizon
                ].positive_probability,
                "regime_baseline_outperform_probability": regime_excess[
                    horizon
                ].positive_probability(
                    current_probability,
                    expanding_excess[horizon].positive_probability,
                ),
                "actual_cumulative_log_return": actual_return,
                "actual_benchmark_log_return": actual_benchmark,
                "actual_excess_log_return": actual_excess,
                "actual_direction_up": int(actual_return > 0) if resolved else np.nan,
                "actual_outperformed": int(actual_excess > 0) if resolved else np.nan,
            }
            for state in range(probabilities.shape[1]):
                record[f"regime_probability_{state}"] = current_probability[state]
            for column in (
                "asset_realized_volatility_20d",
                "asset_drawdown_252d",
                "asset_relative_strength_60d",
            ):
                record[column] = float(local[column].iloc[current_position])
            records.append(record)

    ledger = pd.DataFrame.from_records(records).sort_values(
        ["origin_date", "ticker", "horizon_trading_days"], kind="stable"
    )
    return AssetWalkForwardResult(
        ledger=ledger.reset_index(drop=True),
        scaler=scaler,
        models=models,
        input_features=input_features,
        initial_training_rows=initial_training_rows,
    )
