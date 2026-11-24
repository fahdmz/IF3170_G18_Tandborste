"""Headless PNG rendering for the fitted custom DecisionTree."""

from numbers import Integral
from pathlib import Path
from textwrap import fill

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot_decision_tree(model, path, top_n=None, feature_names=None):
    """Save a PNG with at most top_n BFS nodes, counting the root."""
    model._check_is_fitted()
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("plot_decision_tree requires a .png path")
    if top_n is not None and (
        isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n <= 0
    ):
        raise ValueError("top_n must be a positive integer or None")
    names = [f"x{i}" for i in range(model.n_features_in_)] if feature_names is None else list(feature_names)
    if len(names) != model.n_features_in_:
        raise ValueError("feature_names must match n_features_in_")

    nodes, edges, hidden = [(model.tree_, 0)], [], {}
    index = 0
    while index < len(nodes):
        node, level = nodes[index]
        if not node.is_leaf:
            if node.is_categorical:
                try:
                    keys = sorted(node.children)
                except TypeError:
                    keys = sorted(node.children, key=lambda key: (type(key).__name__, repr(key)))
            else:
                keys = [key for key in ("le", "gt") if key in node.children]
            for key in keys:
                if top_n is not None and len(nodes) >= top_n:
                    hidden[index] = hidden.get(index, 0) + 1
                    continue
                label = str(key) if node.is_categorical else ("<=" if key == "le" else ">")
                edges.append((index, len(nodes), label))
                nodes.append((node.children[key], level + 1))
        index += 1

    levels = {}
    for index, (_, level) in enumerate(nodes):
        levels.setdefault(level, []).append(index)
    positions = {
        index: ((column + 0.5) / len(indices), -level)
        for level, indices in levels.items() for column, index in enumerate(indices)
    }
    width = max(6, 2.7 * max(map(len, levels.values())))
    height = max(3, 2.2 * len(levels))
    figure = Figure(figsize=(width, height))
    FigureCanvasAgg(figure)
    try:
        axis = figure.add_subplot()
        axis.set_xlim(0, 1)
        axis.set_ylim(-len(levels) + 0.4, 0.6)
        axis.set_axis_off()
        for parent, child, label in edges:
            x0, y0 = positions[parent]
            x1, y1 = positions[child]
            axis.plot([x0, x1], [y0, y1], color="#64748b", linewidth=1, zorder=1)
            axis.text((x0 + x1) / 2, (y0 + y1) / 2,
                      fill(label, width=22), ha="center", va="center",
                      fontsize=8, parse_math=False, backgroundcolor="white", zorder=2)
        for index, (node, _) in enumerate(nodes):
            if node.is_leaf:
                title = f"prediction = {node.prediction}"
            elif node.is_categorical:
                title = f"{names[node.feature_index]} (categorical)"
            else:
                title = f"{names[node.feature_index]} <= {node.threshold:.6g}"
            counts = ", ".join(f"{count:.6g}" for count in node.class_distribution)
            label = f"{title}\nmass = {node.n_samples:.6g}\ncounts = [{counts}]"
            if index in hidden:
                label += f"\n... {hidden[index]} branches hidden"
            label = "\n".join(fill(line, width=28) for line in label.splitlines())
            axis.text(*positions[index], label, ha="center", va="center",
                      fontsize=8, parse_math=False, zorder=3,
                      bbox=dict(boxstyle="round,pad=0.5", edgecolor="#64748b",
                                facecolor="#dcfce7" if node.is_leaf else "#dbeafe"))
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, format="png", dpi=150, bbox_inches="tight", pad_inches=0.25)
    finally:
        figure.clear()
