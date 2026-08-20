"""Headless diagnostics for adaptive out-of-sample forecasts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def create_adaptive_forecast_plots(
    ledger: pd.DataFrame,
    metrics: dict[str, dict[str, float | int]],
    state_labels: dict[int, str],
    *,
    rolling_window: int,
    output_dir: str | Path,
) -> list[Path]:
    """Create baseline comparison, rolling accuracy, and live regime forecast plots."""
    output = Path(output_dir)
    horizons = sorted(int(value) for value in metrics)
    x = np.arange(len(horizons))
    width = 0.36
    paths: list[Path] = []

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    panels = (
        (
            "return_mae",
            "baseline_return_mae",
            "Cumulative log-return MAE",
            "Absolute log return",
        ),
        (
            "directional_accuracy",
            "baseline_directional_accuracy",
            "Up/down accuracy",
            "Fraction correct",
        ),
        (
            "regime_accuracy",
            "persistence_regime_accuracy",
            "Future inferred-regime accuracy",
            "Fraction correct",
        ),
    )
    for axis, (model_key, baseline_key, title, ylabel) in zip(axes, panels, strict=True):
        model_values = [float(metrics[str(horizon)][model_key]) for horizon in horizons]
        baseline_values = [float(metrics[str(horizon)][baseline_key]) for horizon in horizons]
        axis.bar(x - width / 2, model_values, width, label="Adaptive ML", color="#457b9d")
        axis.bar(x + width / 2, baseline_values, width, label="Naive baseline", color="#b8b8b8")
        axis.set_xticks(x, [f"{horizon}d" for horizon in horizons])
        axis.set(title=title, xlabel="Forecast horizon", ylabel=ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend()
    paths.append(_save(figure, output / "adaptive_skill_vs_baselines.png"))

    figure, axes = plt.subplots(
        len(horizons),
        1,
        figsize=(13, 2.8 * len(horizons)),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, horizon in zip(axes_array, horizons, strict=True):
        resolved = ledger.loc[
            (ledger["horizon_trading_days"] == horizon) & ledger["resolved"]
        ].copy()
        resolved = resolved.sort_values("origin_date")
        actual = resolved["actual_direction_up"].astype(float)
        model_correct = (
            (resolved["predicted_up_probability"] >= 0.5).astype(float) == actual
        ).astype(float)
        baseline_correct = (
            (resolved["baseline_up_probability"] >= 0.5).astype(float) == actual
        ).astype(float)
        axis.plot(
            resolved["origin_date"],
            model_correct.rolling(rolling_window, min_periods=20).mean(),
            color="#457b9d",
            label="Adaptive ML",
        )
        axis.plot(
            resolved["origin_date"],
            baseline_correct.rolling(rolling_window, min_periods=20).mean(),
            color="#777777",
            label="Naive baseline",
        )
        axis.axhline(0.5, color="black", lw=0.8, ls="--")
        axis.set(
            title=f"{horizon}-trading-day direction: rolling {rolling_window}-forecast accuracy",
            ylabel="Fraction correct",
            ylim=(0.25, 0.8),
        )
        axis.grid(alpha=0.2)
    axes_array[0].legend(ncol=2)
    axes_array[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes_array[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    paths.append(_save(figure, output / "rolling_direction_accuracy.png"))

    latest_origin = pd.Timestamp(ledger["origin_date"].max())
    latest = ledger.loc[ledger["origin_date"] == latest_origin].sort_values(
        "horizon_trading_days"
    )
    ordered_states = sorted(state_labels)
    state_x = np.arange(len(ordered_states))
    group_width = 0.8 / len(latest)
    figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for position, row in enumerate(latest.itertuples(index=False)):
        values = [
            getattr(row, f"predicted_regime_probability_{state}") for state in ordered_states
        ]
        offset = (position - (len(latest) - 1) / 2) * group_width
        axis.bar(
            state_x + offset,
            values,
            group_width,
            label=f"{row.horizon_trading_days} trading days",
            alpha=0.85,
        )
    axis.set_xticks(state_x, [state_labels[state] for state in ordered_states])
    axis.set(
        title=f"Adaptive regime forecasts made {latest_origin.date()}",
        ylabel="Forecast probability",
        ylim=(0, 1),
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    paths.append(_save(figure, output / "latest_regime_forecast.png"))

    figure, axes = plt.subplots(
        1,
        len(horizons),
        figsize=(4.8 * len(horizons), 4.5),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    for axis, horizon in zip(axes_array, horizons, strict=True):
        resolved = ledger.loc[
            (ledger["horizon_trading_days"] == horizon) & ledger["resolved"]
        ]
        actual = 100 * resolved["actual_cumulative_log_return"]
        predicted = 100 * resolved["predicted_cumulative_log_return"]
        bound = max(1.0, float(np.nanmax(np.abs(np.concatenate([actual, predicted])))))
        axis.scatter(predicted, actual, s=7, alpha=0.35, color="#457b9d")
        axis.axhline(0, color="black", lw=0.7)
        axis.axvline(0, color="black", lw=0.7)
        axis.plot([-bound, bound], [-bound, bound], color="#777777", ls="--", lw=0.8)
        axis.set(
            title=f"{horizon}-day return forecasts",
            xlabel="Predicted cumulative log return (%)",
            ylabel="Realized cumulative log return (%)",
            xlim=(-bound, bound),
            ylim=(-bound, bound),
        )
        axis.grid(alpha=0.15)
    paths.append(_save(figure, output / "predicted_vs_realized_returns.png"))
    return paths
