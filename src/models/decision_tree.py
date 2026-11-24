from dataclasses import dataclass, field
from math import expm1, log, sqrt
from numbers import Integral, Real
from statistics import NormalDist

import numpy as np

from src.models.base import BaseClassifier


_EPS = 1e-12
_PRUNING_CF = 0.25
_PRUNING_Z = NormalDist().inv_cdf(1 - _PRUNING_CF)


def _estimated_errors(mass, error):
    """Total pessimistic error E(N,e), approximated for effective float mass."""
    if mass <= _EPS:
        return 0.0
    error = min(mass, max(0.0, error))
    if error == 0:
        estimate = -mass * expm1(log(_PRUNING_CF) / mass)
    elif error < min(1.0, mass):
        base = _estimated_errors(mass, 0.0)
        endpoint = min(1.0, mass)
        upper = mass if mass < 1.0 else _estimated_errors(mass, 1.0)
        estimate = base + (error / endpoint) * (upper - base)
    elif error + 0.5 >= mass:
        estimate = mass
    else:
        z = _PRUNING_Z
        fraction = (error + 0.5) / mass
        estimate = mass * (
            fraction + z * z / (2 * mass)
            + z * sqrt(fraction * (1 - fraction) / mass + z * z / (4 * mass * mass))
        ) / (1 + z * z / mass)
    return min(mass, max(error, estimate))


def _is_missing(value):
    return value is None or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    )


def _entropy(counts):
    """Entropy along the last axis, including fractional and zero counts."""
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    probabilities = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    logs = np.zeros_like(probabilities)
    np.log2(probabilities, out=logs, where=probabilities > 0)
    return -(probabilities * logs).sum(axis=-1)


def _split_statistics(branch_counts, total_weight):
    """Return adjusted gain, split info (with unknown bucket), and ratio."""
    counts = np.asarray(branch_counts, dtype=float)
    if total_weight <= 0:
        zeros = np.zeros(counts.shape[:-2])
        return zeros, zeros, zeros
    masses = counts.sum(axis=-1)
    known = masses.sum(axis=-1)
    proportions = np.divide(
        masses, known[..., None], out=np.zeros_like(masses), where=known[..., None] > 0
    )
    gain = (known / total_weight) * (
        _entropy(counts.sum(axis=-2)) - (proportions * _entropy(counts)).sum(axis=-1)
    )
    gain = np.maximum(gain, 0.0)
    unknown = np.maximum(total_weight - known, 0.0)
    split_info = _entropy(np.concatenate([masses, unknown[..., None]], axis=-1))
    ratio = np.divide(gain, split_info, out=np.zeros_like(gain), where=split_info > _EPS)
    return gain, split_info, ratio


@dataclass
class TreeNode:
    is_leaf: bool
    prediction: int = None
    class_distribution: object = None
    feature_index: int = None
    threshold: float = None
    is_categorical: bool = False
    children: dict = field(default_factory=dict)
    n_samples: float = 0.0
    impurity: float = 0.0
    gain: float = 0.0
    depth: int = 0


class DecisionTree(BaseClassifier):
    """C4.5-style tree with fractional missing values and pessimistic pruning."""

    def __init__(
        self,
        criterion="gain_ratio",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_gain=0.0,
        categorical_features=None,
        random_state=42,
    ):
        super().__init__(random_state)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.categorical_features = categorical_features

    def _numeric_split(self, values, y, weights, feature_index):
        values = np.asarray(values, dtype=float)
        known = ~np.isnan(values) & (weights > 0)
        values, labels, known_weights = values[known], y[known], weights[known]
        if len(values) < 2:
            return None
        order = np.argsort(values, kind="stable")
        values, labels, known_weights = values[order], labels[order], known_weights[order]
        boundaries = np.flatnonzero(values[:-1] < values[1:])
        if not len(boundaries):
            return None
        weighted_counts = np.zeros((len(values), 2))
        weighted_counts[np.arange(len(values)), labels] = known_weights
        prefix = weighted_counts.cumsum(axis=0)
        left = prefix[boundaries]
        right = np.maximum(prefix[-1] - left, 0.0)
        counts = np.stack([left, right], axis=1)
        masses = counts.sum(axis=-1)
        total, known_mass = float(weights.sum()), float(prefix[-1].sum())
        valid = np.all(masses > 0, axis=1) & np.all(
            total * (masses / known_mass) >= self.min_samples_leaf - _EPS, axis=1
        )
        gains, infos, ratios = _split_statistics(counts, total)
        valid &= (gains > self.min_gain + _EPS) & (infos > _EPS)
        if not valid.any():
            return None
        best_gain = gains[valid].max()
        best = np.flatnonzero(valid & (gains >= best_gain - _EPS))[0]
        low, high = values[boundaries[best]], values[boundaries[best] + 1]
        # Halving first avoids overflow, including opposite-sign extremes.
        threshold = low / 2 + high / 2
        if not low <= threshold < high:
            threshold = low
        return {
            "feature_index": feature_index, "threshold": float(threshold),
            "is_categorical": False, "gain": float(gains[best]),
            "split_info": float(infos[best]), "gain_ratio": float(ratios[best]),
        }

    def _categorical_split(self, values, y, weights, feature_index):
        groups = {}
        for value, label, weight in zip(values, y, weights):
            if weight > 0 and not _is_missing(value):
                if value not in groups:
                    groups[value] = np.zeros(2)
                groups[value][label] += weight
        if len(groups) < 2:
            return None
        counts = np.array(list(groups.values()))
        masses = counts.sum(axis=1)
        total = float(weights.sum())
        if np.any(total * (masses / masses.sum()) < self.min_samples_leaf - _EPS):
            return None
        gain, info, ratio = _split_statistics(counts, total)
        if gain <= self.min_gain + _EPS or info <= _EPS:
            return None
        return {
            "feature_index": feature_index, "threshold": None,
            "is_categorical": True, "gain": float(gain),
            "split_info": float(info), "gain_ratio": float(ratio),
        }

    def _best_split(self, X, y, weights, categorical_features):
        candidates = []
        for feature in range(X.shape[1]):
            method = self._categorical_split if feature in categorical_features else self._numeric_split
            candidate = method(X[:, feature], y, weights, feature)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return None
        average_gain = sum(candidate["gain"] for candidate in candidates) / len(candidates)
        best = None
        for candidate in candidates:
            if candidate["gain"] < average_gain - _EPS:
                continue
            if best is None or candidate["gain_ratio"] > best["gain_ratio"] + _EPS:
                best = candidate
            elif abs(candidate["gain_ratio"] - best["gain_ratio"]) <= _EPS:
                if candidate["gain"] > best["gain"] + _EPS:
                    best = candidate
        return best

    def _validate_parameters(self, n_features):
        if self.criterion != "gain_ratio":
            raise ValueError("criterion must be 'gain_ratio'")
        for name, minimum in (("max_depth", 0), ("min_samples_split", 2), ("min_samples_leaf", 1)):
            value = getattr(self, name)
            if name == "max_depth" and value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (
            isinstance(self.min_gain, (bool, np.bool_))
            or not isinstance(self.min_gain, Real)
            or not np.isfinite(self.min_gain)
            or self.min_gain < 0
        ):
            raise ValueError("min_gain must be finite and >= 0")
        if self.categorical_features is None:
            return frozenset()
        try:
            indices = list(self.categorical_features)
        except TypeError as exc:
            raise ValueError("categorical_features must contain unique integer indices") from exc
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, Integral)
            or not 0 <= index < n_features
            for index in indices
        ) or len(set(indices)) != len(indices):
            raise ValueError("categorical_features must contain unique integer indices within X")
        return frozenset(indices)

    def _validate_X(self, X, categorical_features):
        try:
            X = np.array(X, dtype=object, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must be a two-dimensional array") from exc
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        for feature in range(X.shape[1]):
            for row, value in enumerate(X[:, feature]):
                if _is_missing(value):
                    X[row, feature] = np.nan
                elif feature in categorical_features:
                    try:
                        if np.ndim(value) != 0:
                            raise TypeError
                        hash(value)
                    except TypeError as exc:
                        raise ValueError(f"categorical feature {feature} must contain scalar hashable values") from exc
                else:
                    try:
                        if np.ndim(value) != 0 or isinstance(value, (complex, np.complexfloating)):
                            raise ValueError
                        number = float(value)
                        if not np.isfinite(number):
                            raise ValueError
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ValueError(f"numeric feature {feature} must contain finite numbers or None/NaN") from exc
                    X[row, feature] = number
        return X

    def _grow_tree(self, X, y, weights, categorical_features, depth):
        root = TreeNode(is_leaf=True)
        pending = [(root, X, y, weights, depth)]
        while pending:
            node, X, y, weights, depth = pending.pop()
            counts = np.bincount(y, weights=weights, minlength=2)
            mass = float(counts.sum())
            node.prediction = int(np.argmax(counts))
            node.class_distribution = counts
            node.n_samples = mass
            node.impurity = float(_entropy(counts))
            node.depth = depth
            if (
                np.count_nonzero(counts > 0) < 2
                or (self.max_depth is not None and depth >= self.max_depth)
                or mass < self.min_samples_split - _EPS
            ):
                continue
            split = self._best_split(X, y, weights, categorical_features)
            if split is None:
                continue
            feature = split["feature_index"]
            values = X[:, feature]
            unknown = np.array([_is_missing(value) for value in values])
            known = ~unknown & (weights > 0)
            if split["is_categorical"]:
                keys = set(values[known])
                try:
                    keys = sorted(keys)
                except TypeError:
                    keys = sorted(keys, key=lambda value: (type(value).__name__, repr(value)))
                branches = [(key, known & (values == key)) for key in keys]
            else:
                numbers = np.asarray(values, dtype=float)
                branches = [
                    ("le", known & (numbers <= split["threshold"])),
                    ("gt", known & (numbers > split["threshold"])),
                ]
            known_mass = float(weights[known].sum())
            # Unknown rows send weight w*q_b to every child.
            for key, mask in branches:
                prior = float(weights[mask].sum()) / known_mass
                child_weights = weights * np.where(unknown, prior, mask.astype(float))
                active = child_weights > 0
                child = TreeNode(is_leaf=True)
                node.children[key] = child
                pending.append((child, X[active], y[active], child_weights[active], depth + 1))
            node.is_leaf = False
            node.feature_index = feature
            node.threshold = split["threshold"]
            node.is_categorical = split["is_categorical"]
            node.gain = split["gain"]
        return root

    def _prune_tree(self, root):
        """Replace subtrees bottom-up; return the surviving leaf error sum."""
        pending = [(root, False)]
        errors = []
        while pending:
            node, visited = pending.pop()
            if not node.is_leaf and not visited:
                pending.append((node, True))
                pending.extend((child, False) for child in node.children.values())
                continue
            leaf_error = _estimated_errors(
                node.n_samples, node.n_samples - max(node.class_distribution)
            )
            if node.is_leaf:
                errors.append(leaf_error)
                continue
            child_count = len(node.children)
            subtree_error = sum(errors[-child_count:])
            del errors[-child_count:]
            if leaf_error <= subtree_error + _EPS:
                node.is_leaf = True
                node.children.clear()
                node.feature_index = None
                node.threshold = None
                node.is_categorical = False
                node.gain = 0.0
                errors.append(leaf_error)
            else:
                errors.append(subtree_error)
        return errors[0]

    def fit(self, X, y):
        shape = np.shape(X)
        if len(shape) != 2:
            raise ValueError("X must be a two-dimensional array")
        if shape[0] == 0:
            raise ValueError("X training data must not be empty")
        if shape[1] == 0:
            raise ValueError("X must have at least one feature")
        categorical_features = self._validate_parameters(shape[1])
        X = self._validate_X(X, categorical_features)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = np.ravel(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional with the same row count as X")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y must contain only 0 and 1 without missing values")
        y = y.astype(np.intp)
        tree = self._grow_tree(X, y, np.ones(len(y)), categorical_features, 0)
        self._prune_tree(tree)
        self.tree_ = tree
        self.n_features_in_ = X.shape[1]
        self._categorical_features_ = categorical_features
        self.classes_ = np.array([0, 1])
        return self

    def _predict_row(self, row, node):
        fallback = node.class_distribution / node.n_samples
        probabilities = np.zeros(2)
        pending = [(node, 1.0)]
        while pending:
            current, path_weight = pending.pop()
            if current.is_leaf:
                probabilities += path_weight * (current.class_distribution / current.n_samples)
                continue
            value = row[current.feature_index]
            if not _is_missing(value):
                key = value if current.is_categorical else ("le" if value <= current.threshold else "gt")
                if key in current.children:
                    pending.append((current.children[key], path_weight))
                    continue
            total = sum(child.n_samples for child in current.children.values())
            if total <= 0 or not np.isfinite(total):
                probabilities += path_weight * (current.class_distribution / current.n_samples)
                continue
            for child in current.children.values():
                pending.append((child, path_weight * (child.n_samples / total)))
        probabilities = np.maximum(probabilities, 0.0)
        mass = probabilities.sum()
        if mass <= 0 or not np.isfinite(probabilities).all():
            return fallback
        return probabilities / mass

    def predict_proba(self, X):
        self._check_is_fitted()
        X = self._validate_X(X, self._categorical_features_)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have {self.n_features_in_} features")
        probabilities = np.empty((len(X), 2))
        for index, row in enumerate(X):
            probabilities[index] = self._predict_row(row, self.tree_)
        return probabilities

    def predict(self, X):
        self._check_is_fitted()
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_depth(self):
        self._check_is_fitted()
        nodes = [self.tree_]
        depth = 0
        while nodes:
            node = nodes.pop()
            depth = max(depth, node.depth)
            if not node.is_leaf:
                nodes.extend(node.children.values())
        return depth

    def get_n_leaves(self):
        self._check_is_fitted()
        nodes = [self.tree_]
        leaves = 0
        while nodes:
            node = nodes.pop()
            if node.is_leaf:
                leaves += 1
            else:
                nodes.extend(node.children.values())
        return leaves

    def export_graph(self, path, top_n=None, feature_names=None):
        """Write UTF-8 DOT; top_n limits actual BFS nodes, including the root."""
        from pathlib import Path

        self._check_is_fitted()
        path = Path(path)
        if path.suffix.lower() != ".dot":
            raise ValueError("export_graph requires a .dot path")
        if top_n is not None and (
            isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n <= 0
        ):
            raise ValueError("top_n must be a positive integer or None")
        names = [f"x{i}" for i in range(self.n_features_in_)] if feature_names is None else list(feature_names)
        if len(names) != self.n_features_in_:
            raise ValueError("feature_names must match n_features_in_")

        def escape(value):
            return str(value).replace("\\", "\\\\").replace('"', '\\"').replace(
                "\r\n", "\n"
            ).replace("\r", "\n").replace("\n", "\\n")

        nodes, edges, hidden = [self.tree_], [], {}
        index = 0
        while index < len(nodes):
            node = nodes[index]
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
                    nodes.append(node.children[key])
            index += 1

        lines = ["digraph DecisionTree {", "  node [shape=box];"]
        for index, node in enumerate(nodes):
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
            lines.append(f'  n{index} [label="{escape(label)}"];')
        for parent, child, label in edges:
            lines.append(f'  n{parent} -> n{child} [label="{escape(label)}"];')
        lines.append("}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
