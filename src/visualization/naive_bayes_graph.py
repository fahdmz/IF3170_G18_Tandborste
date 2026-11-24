from pathlib import Path

import matplotlib.pyplot as plt

from src import config
from src.data import CATEGORICAL_COLS, NUMERICAL_COLS


def plot_naive_bayes_graph(
    feature_names: list[str] | None = None,
    class_name: str = config.TARGET_COL,
    output_path: Path | str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Path:
    if feature_names is None:
        feature_names = NUMERICAL_COLS + CATEGORICAL_COLS

    if output_path is None:
        output_path = config.FIGURE_DIR / "naive_bayes_graph.png"
    else:
        output_path = Path(output_path)

    n_features = len(feature_names)
    if n_features == 0:
        raise ValueError("feature_names cannot be empty")

    if figsize is None:
        fig_width = max(10.0, n_features * 0.9)
        figsize = (fig_width, 6.5)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-1, n_features)
    ax.set_ylim(-0.5, 3.0)
    ax.axis("off")

    class_x = (n_features - 1) / 2.0
    class_y = 2.4

    ax.text(
        class_x,
        class_y,
        f"Class Node\n({class_name})",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor="#2A5C8A",
            edgecolor="#16324F",
            lw=2.0,
        ),
        color="white",
        fontsize=12,
        fontweight="bold",
    )

    for i, name in enumerate(feature_names):
        feat_x = float(i)
        feat_y = 0.5

        is_cat = name in CATEGORICAL_COLS
        node_color = "#E8F0FE" if not is_cat else "#FEF3D6"
        edge_color = "#3C4043"

        ax.text(
            feat_x,
            feat_y,
            name,
            ha="center",
            va="center",
            rotation=45,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor=node_color,
                edgecolor=edge_color,
                lw=1.2,
            ),
            fontsize=8.5,
        )

        ax.annotate(
            "",
            xy=(feat_x, feat_y + 0.35),
            xytext=(class_x, class_y - 0.35),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.2,
                color="#5F6368",
                shrinkA=8,
                shrinkB=8,
                mutation_scale=12,
            ),
        )

    plt.title(
        f"Naive Bayes Conditional Independence Graph ({class_name} -> Features)",
        fontsize=13,
        pad=20,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
