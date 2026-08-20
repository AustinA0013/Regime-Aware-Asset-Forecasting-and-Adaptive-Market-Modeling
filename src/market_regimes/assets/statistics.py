"""Hard and probability-weighted asset behavior conditional on fitted HMM states."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from market_regimes.asset_config import AssetStatisticsConfig


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    order = np.argsort(values[valid])
    sorted_values = values[valid][order]
    sorted_weights = weights[valid][order]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * cumulative[-1]
    return float(sorted_values[min(np.searchsorted(cumulative, threshold), len(order) - 1)])


def _block_bootstrap_interval(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    samples: int,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    valid = np.isfinite(values) & np.isfinite(weights)
    x = values[valid]
    w = weights[valid]
    n = len(x)
    if n < block_length or np.sum(w) <= 0:
        return float("nan"), float("nan")
    estimates = np.empty(samples, dtype=np.float64)
    n_blocks = int(np.ceil(n / block_length))
    for sample in range(samples):
        starts = rng.integers(0, n, size=n_blocks)
        indices = np.concatenate(
            [(start + np.arange(block_length)) % n for start in starts]
        )[:n]
        sampled_weights = w[indices]
        denominator = sampled_weights.sum()
        estimates[sample] = (
            np.sum(sampled_weights * x[indices]) / denominator
            if denominator > 0
            else np.nan
        )
    finite = estimates[np.isfinite(estimates)]
    if len(finite) < max(20, samples // 2):
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))


def calculate_asset_state_statistics(
    asset_features: pd.DataFrame,
    probabilities: pd.DataFrame,
    state_labels: Mapping[int, str],
    *,
    drawdown_column: str,
    config: AssetStatisticsConfig,
    random_seed: int,
) -> pd.DataFrame:
    """Describe returns under the fitted global HMM using hard and soft membership."""
    config.validate()
    if not probabilities.index.is_monotonic_increasing:
        raise ValueError("probability index must be increasing")
    probability_values = probabilities.to_numpy(dtype=float)
    if np.any(probability_values < 0) or not np.allclose(
        probability_values.sum(axis=1), 1.0
    ):
        raise ValueError("probabilities must be nonnegative and sum to one")
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, object]] = []
    hard_states = np.argmax(probability_values, axis=1)
    for ticker, ticker_frame in asset_features.groupby("ticker", sort=True):
        indexed = ticker_frame.set_index("timestamp").reindex(probabilities.index)
        returns = indexed["asset_log_return_1d"].to_numpy(dtype=float)
        drawdown = indexed[drawdown_column].to_numpy(dtype=float)
        finite_return = np.isfinite(returns)
        for method in ("hard_argmax", "probability_weighted"):
            for state in sorted(state_labels):
                weights = (
                    (hard_states == state).astype(float)
                    if method == "hard_argmax"
                    else probability_values[:, state]
                )
                weights = weights * finite_return
                weight_sum = float(weights.sum())
                effective_n = (
                    float(weight_sum**2 / np.sum(weights**2))
                    if np.sum(weights**2) > 0
                    else 0.0
                )
                sufficient = effective_n >= config.min_state_samples
                mean = (
                    float(np.sum(weights * np.nan_to_num(returns)) / weight_sum)
                    if weight_sum > 0
                    else float("nan")
                )
                variance = (
                    float(np.sum(weights * (np.nan_to_num(returns) - mean) ** 2) / weight_sum)
                    if weight_sum > 0
                    else float("nan")
                )
                volatility = float(np.sqrt(max(variance, 0.0)))
                positive_frequency = (
                    float(np.sum(weights * (np.nan_to_num(returns) > 0)) / weight_sum)
                    if weight_sum > 0
                    else float("nan")
                )
                ci_low, ci_high = _block_bootstrap_interval(
                    returns,
                    weights,
                    samples=config.bootstrap_samples,
                    block_length=config.bootstrap_block_length,
                    rng=rng,
                )
                records.append(
                    {
                        "ticker": ticker,
                        "state": state,
                        "state_label": state_labels[state],
                        "membership_method": method,
                        "sample_count": int(np.sum((weights > 0) & finite_return)),
                        "weight_sum": weight_sum,
                        "effective_sample_size": effective_n,
                        "sufficient_sample": sufficient,
                        "mean_daily_return": mean,
                        "mean_return_ci_2_5": ci_low if sufficient else np.nan,
                        "mean_return_ci_97_5": ci_high if sufficient else np.nan,
                        "median_daily_return": _weighted_quantile(returns, weights, 0.5),
                        "daily_volatility": volatility,
                        "annualized_volatility": volatility * np.sqrt(252),
                        "positive_return_frequency": positive_frequency,
                        "representative_drawdown_5pct": _weighted_quantile(
                            drawdown, weights, 0.05
                        ),
                        "minimum_observed_drawdown": (
                            float(np.nanmin(drawdown[weights > 0]))
                            if np.any((weights > 0) & np.isfinite(drawdown))
                            else np.nan
                        ),
                        "annualized_sharpe_like": (
                            mean / volatility * np.sqrt(252)
                            if sufficient and volatility > 0
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)
