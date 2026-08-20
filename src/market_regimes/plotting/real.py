"""Headless diagnostic plots for the real financial-data milestone."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STATE_COLORS = {
    "Risk-On-like": "#2a9d8f",
    "Transitional-like": "#e9c46a",
    "Risk-Off-like": "#e76f51",
}


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _date_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def create_real_diagnostic_plots(
    levels: pd.DataFrame,
    features: pd.DataFrame,
    probabilities: np.ndarray,
    state_labels: dict[int, str],
    transition_matrix: np.ndarray,
    standardized_centroids: pd.DataFrame,
    train_end: pd.Timestamp,
    output_dir: str | Path,
) -> list[Path]:
    """Create five plots separating model structure from financial interpretation."""
    output = Path(output_dir)
    timestamps = features.index
    levels = levels.reindex(timestamps)
    states = np.argmax(probabilities, axis=1)
    paths: list[Path] = []

    figure, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    axis.plot(timestamps, levels["price"], color="#555555", lw=0.7, alpha=0.45)
    for state, label in state_labels.items():
        mask = states == state
        axis.scatter(
            timestamps[mask],
            levels.loc[timestamps[mask], "price"],
            s=7,
            color=STATE_COLORS[label],
            label=label,
        )
    axis.axvline(train_end, color="black", ls="--", lw=1.1, label="Parameters frozen")
    axis.set(title="S&P 500 close colored by filtered HMM state", ylabel="Index level")
    _date_axis(axis)
    axis.legend(ncol=4, fontsize=8)
    paths.append(_save(figure, output / "sp500_filtered_states.png"))

    test_mask = timestamps > train_end
    figure, axis = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    for state, label in state_labels.items():
        axis.plot(
            timestamps[test_mask],
            probabilities[test_mask, state],
            label=label,
            color=STATE_COLORS[label],
            lw=1.0,
        )
    axis.set(
        title="Out-of-sample forward-filtered probabilities",
        ylabel="Probability",
        ylim=(-0.02, 1.02),
    )
    _date_axis(axis)
    axis.legend(ncol=3)
    paths.append(_save(figure, output / "test_filtered_probabilities.png"))

    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, constrained_layout=True)
    for axis, column, title in (
        (axes[0], "vix_level", "VIX level"),
        (axes[1], "credit_spread", "Baa corporate-minus-Treasury credit spread"),
    ):
        axis.plot(timestamps, features[column], color="#555555", lw=0.7, alpha=0.5)
        for state, label in state_labels.items():
            mask = states == state
            axis.scatter(
                timestamps[mask],
                features.loc[timestamps[mask], column],
                color=STATE_COLORS[label],
                s=6,
                label=label,
            )
        axis.axvline(train_end, color="black", ls="--", lw=1.0)
        axis.set(title=title)
    _date_axis(axes[-1])
    axes[0].legend(ncol=3, fontsize=8)
    paths.append(_save(figure, output / "stress_variables_by_state.png"))

    ordered_states = sorted(state_labels, key=lambda state: state_labels[state])
    matrix = transition_matrix[np.ix_(ordered_states, ordered_states)]
    ordered_labels = [state_labels[state] for state in ordered_states]
    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(ordered_labels)), ordered_labels, rotation=25, ha="right")
    axis.set_yticks(range(len(ordered_labels)), ordered_labels)
    axis.set(title="Fitted training-period transition matrix", xlabel="Next state")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, shrink=0.8)
    paths.append(_save(figure, output / "real_transition_matrix.png"))

    centroids = standardized_centroids.loc[ordered_states]
    figure, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    bound = max(1.0, float(np.nanmax(np.abs(centroids.to_numpy()))))
    image = axis.imshow(centroids.to_numpy(), cmap="RdBu_r", vmin=-bound, vmax=bound)
    axis.set_xticks(range(centroids.shape[1]), centroids.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(ordered_labels)), ordered_labels)
    axis.set(title="Training-period conditional feature means (standardized units)")
    figure.colorbar(image, ax=axis, label="Standard deviations from training mean")
    paths.append(_save(figure, output / "state_feature_centroids.png"))
    return paths
