"""Animate the log-loss contour and the (intercept, weight) path of a one-feature model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from src import config
from src.models.logistic_regression import (
    LogisticRegression,
    _penalty_terms,
    _sample_weights,
    _softplus,
)


def plot_logreg_contour(model: LogisticRegression, X: np.ndarray, y: np.ndarray, output_path: Path | str | None = None) -> Path:
    """Save a GIF of the training path on the log-loss surface of a one-feature model.

    ``model`` must already be fit with ``record_history=True``. The horizontal
    axis is the intercept and the vertical axis is the single weight. The
    surface uses the same weighted mean log-loss and penalty as ``fit``,
    including ``class_weight``. The default file is
    ``output/figures/logreg_contour.gif``.
    """
    model._check_is_fitted()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError(f"X must have shape (n, 1), got {X.shape}")
    if model.coef_.shape[0] != 1:
        raise ValueError("plot_logreg_contour requires a one-feature model")
    if model.coef_history_ is None or model.intercept_history_ is None:
        raise ValueError("plot_logreg_contour requires a model fit with record_history=True")
    if len(X) != len(y):
        raise ValueError(f"Length mismatch: len(X)={len(X)} != len(y)={len(y)}")

    if output_path is None:
        output_path = config.FIGURE_DIR / "logreg_contour.gif"
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".gif":
        raise ValueError("plot_logreg_contour requires a .gif path")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_bin = (y == model.classes_[1]).astype(float)
    weights = _sample_weights(model.class_weight, model.classes_, y)
    l2, l1 = _penalty_terms(model.penalty, model.reg_lambda, model.l1_ratio)
    theta0 = np.asarray(model.intercept_history_, dtype=float)
    theta1 = np.asarray(model.coef_history_[:, 0], dtype=float)
    loss_grid, grid0, grid1 = _loss_grid(X[:, 0], y_bin, weights, theta0, theta1, l2, l1)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    fill = ax.contourf(grid0, grid1, loss_grid, levels=18, cmap="viridis")
    ax.contour(grid0, grid1, loss_grid, levels=10, colors="k", linewidths=0.35, alpha=0.35)
    fig.colorbar(fill, ax=ax, label="mean log-loss + penalty")
    (path_line,) = ax.plot([], [], color="white", lw=1.6)
    (point,) = ax.plot([], [], "o", color="#C0392B", ms=6)
    ax.set_xlabel(r"$\theta_0$ (intercept)")
    ax.set_ylabel(r"$\theta_1$ (weight)")
    ax.set_title("Log-loss contour and $(\\theta_0, \\theta_1)$ path")

    def _frame(index):
        path_line.set_data(theta0[: index + 1], theta1[: index + 1])
        point.set_data([theta0[index]], [theta1[index]])
        return path_line, point

    animation = FuncAnimation(fig, _frame, frames=len(theta0), interval=80, blit=False)
    animation.save(output_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    return output_path


def _loss_grid(x, y_bin, weights, theta0, theta1, l2, l1):
    """Loss on a grid that pads the recorded ``(intercept, weight)`` path.

    Returns the loss values, the intercept mesh, and the weight mesh. The
    path itself is not drawn here; the caller animates it on top.
    """
    pad0 = 0.2 * (float(theta0.max() - theta0.min()) + 1.0)
    pad1 = 0.2 * (float(theta1.max() - theta1.min()) + 1.0)
    axis0 = np.linspace(float(theta0.min()) - pad0, float(theta0.max()) + pad0, 70)
    axis1 = np.linspace(float(theta1.min()) - pad1, float(theta1.max()) + pad1, 70)
    grid0, grid1 = np.meshgrid(axis0, axis1)
    scores = x[:, None, None] * grid1[None, :, :] + grid0[None, :, :]
    weighted = weights[:, None, None] * (_softplus(scores) - y_bin[:, None, None] * scores)
    loss = weighted.mean(axis=0) + 0.5 * l2 * grid1**2 + l1 * np.abs(grid1)
    return loss, grid0, grid1
