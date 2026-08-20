"""Visual diagnostics for the synthetic validation milestone."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _save_figure(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_synthetic_diagnostic_plots(
    observations: FloatArray,
    true_states: IntArray,
    aligned_filtered_probabilities: FloatArray,
    true_transition: FloatArray,
    aligned_fitted_transition: FloatArray,
    true_means: FloatArray,
    aligned_fitted_means: FloatArray,
    output_dir: str | Path,
) -> list[Path]:
    """Write four deterministic plots and return their paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inferred_states = np.argmax(aligned_filtered_probabilities, axis=1)
    paths: list[Path] = []

    figure, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, constrained_layout=True)
    time = np.arange(observations.shape[0])
    axes[0].scatter(time, observations[:, 0], c=true_states, cmap="viridis", s=3)
    axes[0].set(title="Synthetic observations colored by true state", ylabel="Feature 0")
    axes[1].scatter(time, observations[:, 0], c=inferred_states, cmap="viridis", s=3)
    axes[1].set(title="Causal filtered state (aligned)", xlabel="Time", ylabel="Feature 0")
    path = output / "state_timeline.png"
    _save_figure(figure, path)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for state in range(aligned_filtered_probabilities.shape[1]):
        axis.plot(aligned_filtered_probabilities[:, state], lw=0.9, label=f"State {state}")
    axis.set(
        title="Forward-filtered regime probabilities",
        xlabel="Time",
        ylabel="P(state | observations through t)",
        ylim=(-0.02, 1.02),
    )
    axis.legend(ncol=aligned_filtered_probabilities.shape[1])
    path = output / "filtered_probabilities.png"
    _save_figure(figure, path)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, matrix, title in zip(
        axes,
        (true_transition, aligned_fitted_transition),
        ("True transition matrix", "Fitted transition matrix (aligned)"),
        strict=True,
    ):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        axis.set(title=title, xlabel="Next state", ylabel="Current state")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    figure.colorbar(image, ax=axes, shrink=0.8)
    path = output / "transition_matrices.png"
    _save_figure(figure, path)
    paths.append(path)

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.scatter(
        observations[:, 0], observations[:, 1], c=true_states, cmap="viridis", s=5, alpha=0.2
    )
    axis.scatter(
        true_means[:, 0], true_means[:, 1], c="black", marker="x", s=130, label="True means"
    )
    axis.scatter(
        aligned_fitted_means[:, 0],
        aligned_fitted_means[:, 1],
        facecolors="none",
        edgecolors="red",
        marker="o",
        s=130,
        label="Fitted means",
    )
    axis.set(title="Emission space and recovered centroids", xlabel="Feature 0", ylabel="Feature 1")
    axis.legend()
    path = output / "emission_space.png"
    _save_figure(figure, path)
    paths.append(path)
    return paths

