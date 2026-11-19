from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.models.svm import SVM

_DEFAULT_KERNELS = (
    ("linear", {"kernel": "linear"}),
    ("poly", {"kernel": "poly", "degree": 3}),
    ("rbf", {"kernel": "rbf"}),
)


def plot_svm_boundary(
    model: SVM,
    X: np.ndarray,
    y: np.ndarray,
    ax: plt.Axes | None = None,
    title: str | None = None,
    resolution: int = 300,
    feature_names: tuple[str, str] = ("x1", "x2"),
) -> plt.Axes:
    """Gambar bidang pemisah f(x)=0, margin f(x)=+-1, dan support vector dari SVM 2D yang sudah di-fit."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"X harus berbentuk (n, 2), didapat {X.shape}")
    model._check_is_fitted()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    scores = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    limit = max(np.abs(scores).max(), 1.0)
    ax.contourf(xx, yy, scores, levels=np.linspace(-limit, limit, 21), cmap="RdBu_r", alpha=0.35)
    ax.contour(
        xx, yy, scores, levels=[-1.0, 0.0, 1.0], colors="k", linestyles=["--", "-", "--"], linewidths=[1, 2, 1]
    )

    negative, positive = y == model.classes_[0], y == model.classes_[1]
    ax.scatter(X[negative, 0], X[negative, 1], s=12, c="#2A5C8A", label=f"kelas {model.classes_[0]}")
    ax.scatter(X[positive, 0], X[positive, 1], s=12, c="#C0392B", label=f"kelas {model.classes_[1]}")
    sv = model.support_vectors_
    ax.scatter(
        sv[:, 0], sv[:, 1], s=60, facecolors="none", edgecolors="k", linewidths=0.8,
        label=f"support vector ({len(sv)})",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title or f"SVM kernel={model.kernel_}")
    ax.legend(loc="upper right", fontsize=8)
    return ax


def plot_svm_kernels(
    X: np.ndarray,
    y: np.ndarray,
    kernels: tuple[tuple[str, dict], ...] = _DEFAULT_KERNELS,
    common_params: dict | None = None,
    output_path: Path | str | None = None,
    feature_names: tuple[str, str] = ("x1", "x2"),
) -> Path:
    """Latih satu SVM per kernel pada data 2D dan simpan gambar berdampingan."""
    if output_path is None:
        output_path = config.FIGURE_DIR / "svm_kernels_2d.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    common_params = common_params or {}
    fig, axes = plt.subplots(1, len(kernels), figsize=(6 * len(kernels), 5))
    axes = np.atleast_1d(axes)
    for ax, (name, params) in zip(axes, kernels):
        model = SVM(**{**common_params, **params}).fit(X, y)
        acc = np.mean(model.predict(X) == y)
        plot_svm_boundary(model, X, y, ax=ax, title=f"{name} (akurasi latih {acc:.3f})", feature_names=feature_names)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
