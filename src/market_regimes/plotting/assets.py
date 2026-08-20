"""Diagnostic plots for individual-asset forecasting and ranking."""

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


def create_asset_forecast_plots(
    *,
    ledger: pd.DataFrame,
    metrics: pd.DataFrame,
    cross_sectional_daily: pd.DataFrame,
    rank_summary: pd.DataFrame,
    state_statistics: pd.DataFrame,
    asset_features: pd.DataFrame,
    probabilities: pd.DataFrame,
    state_labels: dict[int, str],
    rolling_window: int,
    output_dir: str | Path,
) -> list[Path]:
    output = Path(output_dir)
    paths: list[Path] = []
    resolved = ledger.loc[ledger["resolved"]].copy()
    aggregate = metrics.loc[metrics["scope"] == "all_folds"].copy()
    tickers = sorted(resolved["ticker"].unique())
    horizons = sorted(int(value) for value in resolved["horizon_trading_days"].unique())
    semantic_label_order = [
        "Risk-On-like",
        "Transitional-like",
        "Risk-Off-like",
    ]
    label_to_state = {label: state for state, label in state_labels.items()}
    ordered_labels = [
        label for label in semantic_label_order if label in label_to_state
    ]
    ordered_labels.extend(
        label for label in state_labels.values() if label not in ordered_labels
    )
    ordered_states = [label_to_state[label] for label in ordered_labels]

    # 1. Predicted versus realized return by asset at the middle horizon.
    middle_horizon = horizons[len(horizons) // 2]
    figure, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for axis, ticker in zip(axes.flat, tickers, strict=False):
        rows = resolved.loc[
            (resolved["ticker"] == ticker)
            & (resolved["horizon_trading_days"] == middle_horizon)
        ]
        axis.scatter(
            100 * rows["predicted_cumulative_log_return"],
            100 * rows["actual_cumulative_log_return"],
            s=7,
            alpha=0.28,
            color="#457b9d",
        )
        bound = max(
            1.0,
            float(
                np.nanmax(
                    np.abs(
                        np.concatenate(
                            [
                                rows["predicted_cumulative_log_return"].to_numpy(),
                                rows["actual_cumulative_log_return"].to_numpy(),
                            ]
                        )
                    )
                )
                * 100
            ),
        )
        axis.plot([-bound, bound], [-bound, bound], ls="--", lw=0.7, color="#777777")
        axis.axhline(0, color="black", lw=0.5)
        axis.axvline(0, color="black", lw=0.5)
        axis.set(title=ticker, xlim=(-bound, bound), ylim=(-bound, bound))
        axis.grid(alpha=0.15)
    for axis in axes.flat[len(tickers) :]:
        axis.set_visible(False)
    figure.supxlabel(f"Predicted {middle_horizon}d cumulative log return (%)")
    figure.supylabel("Realized cumulative log return (%)")
    figure.suptitle("Predicted versus realized return by asset", fontsize=14)
    paths.append(_save(figure, output / "predicted_vs_realized_return_by_asset.png"))

    # 2. ML versus both causal excess-return baselines.
    figure, axes = plt.subplots(1, len(horizons), figsize=(16, 5), constrained_layout=True)
    for axis, horizon in zip(np.atleast_1d(axes), horizons, strict=True):
        rows = aggregate.loc[aggregate["horizon_trading_days"] == horizon].sort_values(
            "ticker"
        )
        x = np.arange(len(rows))
        width = 0.26
        axis.bar(x - width, rows["excess_mae"], width, label="Adaptive ML")
        axis.bar(x, rows["expanding_excess_mae"], width, label="Expanding mean")
        axis.bar(x + width, rows["regime_excess_mae"], width, label="Regime mean")
        axis.set_xticks(x, rows["ticker"], rotation=45, ha="right")
        axis.set(title=f"{horizon}d excess-return MAE", ylabel="Absolute log return")
        axis.grid(axis="y", alpha=0.2)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    paths.append(_save(figure, output / "ml_vs_baseline_errors_by_asset.png"))

    # 3. Directional accuracy by asset.
    figure, axes = plt.subplots(1, len(horizons), figsize=(16, 5), constrained_layout=True)
    for axis, horizon in zip(np.atleast_1d(axes), horizons, strict=True):
        rows = aggregate.loc[aggregate["horizon_trading_days"] == horizon].sort_values(
            "ticker"
        )
        x = np.arange(len(rows))
        width = 0.26
        axis.bar(x - width, rows["directional_accuracy"], width, label="Adaptive ML")
        axis.bar(x, rows["expanding_directional_accuracy"], width, label="Expanding")
        axis.bar(x + width, rows["regime_directional_accuracy"], width, label="Regime")
        axis.axhline(0.5, color="black", ls="--", lw=0.7)
        axis.set_xticks(x, rows["ticker"], rotation=45, ha="right")
        axis.set(title=f"{horizon}d direction accuracy", ylim=(0.35, 0.72))
        axis.grid(axis="y", alpha=0.2)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    paths.append(_save(figure, output / "directional_accuracy_by_asset.png"))

    # 4. IC through time.
    figure, axes = plt.subplots(
        len(horizons), 1, figsize=(13, 8.5), sharex=True, constrained_layout=True
    )
    for axis, horizon in zip(np.atleast_1d(axes), horizons, strict=True):
        rows = cross_sectional_daily.loc[
            cross_sectional_daily["horizon_trading_days"] == horizon
        ].sort_values("origin_date")
        axis.plot(
            rows["origin_date"],
            rows["information_coefficient"].rolling(63, min_periods=20).mean(),
            color="#2a9d8f",
            label="63-origin rolling mean IC",
        )
        axis.axhline(0, color="black", lw=0.7)
        axis.set(title=f"{horizon}d cross-sectional Information Coefficient", ylim=(-0.5, 0.5))
        axis.grid(alpha=0.2)
    np.atleast_1d(axes)[-1].xaxis.set_major_locator(mdates.YearLocator())
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    paths.append(_save(figure, output / "information_coefficient_through_time.png"))

    # 5. Realized outcomes by within-origin predicted rank bucket.
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for horizon in horizons:
        rows = rank_summary.loc[rank_summary["horizon_trading_days"] == horizon]
        axis.plot(
            rows["predicted_rank_bucket"] + 1,
            100 * rows["mean_realized_excess"],
            marker="o",
            label=f"{horizon}d",
        )
    axis.axhline(0, color="black", lw=0.7)
    axis.set(
        title="Realized excess return by predicted-excess rank bucket",
        xlabel="Predicted bucket (1 = lowest, 5 = highest)",
        ylabel="Mean realized excess log return (%)",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    paths.append(_save(figure, output / "realized_performance_by_forecast_rank.png"))

    # 6. Probability-weighted state mean returns.
    weighted = state_statistics.loc[
        state_statistics["membership_method"] == "probability_weighted"
    ]
    matrix = weighted.pivot(index="ticker", columns="state_label", values="mean_daily_return")
    matrix = matrix.reindex(index=tickers, columns=ordered_labels)
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    image = axis.imshow(100 * matrix.to_numpy(), cmap="RdYlGn", aspect="auto")
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
    axis.set_title("Probability-weighted mean daily asset return by fitted HMM state")
    figure.colorbar(image, ax=axis, label="Mean daily log return (%)")
    paths.append(_save(figure, output / "asset_performance_by_hmm_regime.png"))

    # 7. Rolling absolute excess-return error versus expanding baseline.
    figure, axes = plt.subplots(
        len(horizons), 1, figsize=(13, 8.5), sharex=True, constrained_layout=True
    )
    for axis, horizon in zip(np.atleast_1d(axes), horizons, strict=True):
        rows = resolved.loc[resolved["horizon_trading_days"] == horizon].copy()
        daily = rows.groupby("origin_date").apply(
            lambda group: pd.Series(
                {
                    "ml": np.mean(
                        np.abs(
                            group["actual_excess_log_return"]
                            - group["predicted_excess_log_return"]
                        )
                    ),
                    "baseline": np.mean(
                        np.abs(
                            group["actual_excess_log_return"]
                            - group["expanding_baseline_excess_return"]
                        )
                    ),
                }
            ),
            include_groups=False,
        )
        axis.plot(
            daily.index,
            daily["ml"].rolling(rolling_window, min_periods=20).mean(),
            label="Adaptive ML",
            color="#457b9d",
        )
        axis.plot(
            daily.index,
            daily["baseline"].rolling(rolling_window, min_periods=20).mean(),
            label="Expanding baseline",
            color="#777777",
        )
        axis.set(title=f"{horizon}d rolling cross-asset excess-return MAE")
        axis.grid(alpha=0.2)
    np.atleast_1d(axes)[0].legend()
    paths.append(_save(figure, output / "rolling_forecast_performance.png"))

    # 8. Hard-state daily return distributions by asset.
    probability_argmax = pd.Series(
        np.argmax(probabilities.to_numpy(), axis=1), index=probabilities.index
    )
    figure, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for axis, ticker in zip(axes.flat, tickers, strict=False):
        rows = asset_features.loc[asset_features["ticker"] == ticker].set_index("timestamp")
        returns = rows["asset_log_return_1d"].reindex(probabilities.index)
        samples = [
            (100 * returns.loc[probability_argmax == state]).dropna().to_numpy()
            for state in ordered_states
        ]
        tick_labels = [
            label.removeprefix("Risk-").removesuffix("-like")
            for label in ordered_labels
        ]
        axis.boxplot(samples, showfliers=False, tick_labels=tick_labels)
        axis.axhline(0, color="black", lw=0.6)
        axis.set(title=ticker, ylabel="Daily log return (%)")
        axis.grid(axis="y", alpha=0.15)
    for axis in axes.flat[len(tickers) :]:
        axis.set_visible(False)
    figure.suptitle("Asset return distributions conditional on fitted hard HMM state", fontsize=14)
    paths.append(_save(figure, output / "regime_conditioned_return_distributions.png"))
    return paths
