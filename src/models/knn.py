from math import isfinite
from numbers import Integral, Real

import numpy as np

from src.config import SEED
from src.models.base import BaseClassifier


class KNN(BaseClassifier):
    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        p: float = 2.0,
        weights: str = "uniform",
        batch_size: int = 1024,
        random_state: int = SEED,
    ):
        super().__init__(random_state)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        self.batch_size = batch_size

    def _validate_params(self) -> None:
        if not isinstance(self.n_neighbors, Integral) or self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be an integer >= 1, got {self.n_neighbors}")
        if self.metric not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError(
                f"metric must be 'euclidean', 'manhattan', or 'minkowski', got '{self.metric}'"
            )
        if self.metric == "minkowski" and (
            not isinstance(self.p, Real) or not isfinite(self.p) or self.p < 1.0
        ):
            raise ValueError(f"p must be a number >= 1.0 and finite for Minkowski distance, got {self.p}")
        if self.weights not in ("uniform", "distance"):
            raise ValueError(
                f"weights must be 'uniform' or 'distance', got '{self.weights}'"
            )
        if not isinstance(self.batch_size, Integral) or self.batch_size < 1:
            raise ValueError(f"batch_size must be an integer >= 1, got {self.batch_size}")

    def fit(self, X, y):
        """Fit the classifier on at least two distinct classes."""
        self._validate_params()
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).ravel()

        if len(X_arr) != len(y_arr):
            raise ValueError(f"Length mismatch: len(X)={len(X_arr)} != len(y)={len(y_arr)}")
        if len(X_arr) == 0:
            raise ValueError("X and y cannot be empty")
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X_arr.ndim}D")
        if int(self.n_neighbors) > len(X_arr):
            raise ValueError(
                f"n_neighbors={self.n_neighbors} cannot be greater than training sample count {len(X_arr)}"
            )

        classes = np.unique(y_arr)
        if len(classes) < 2:
            raise ValueError("y must contain at least two distinct classes")
        self.classes_ = classes
        self.X_train_ = np.ascontiguousarray(X_arr)
        self.y_train_ = y_arr
        return self

    def _compute_distances(self, X_batch: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean" or (self.metric == "minkowski" and float(self.p) == 2.0):
            dist2 = np.zeros((len(X_batch), len(self.X_train_)), dtype=float)
            for d in range(X_batch.shape[1]):
                diff = X_batch[:, d : d + 1] - self.X_train_[:, d]
                dist2 += diff * diff
            return np.sqrt(dist2)
        elif self.metric == "manhattan" or (self.metric == "minkowski" and float(self.p) == 1.0):
            dist = np.zeros((len(X_batch), len(self.X_train_)), dtype=float)
            for d in range(X_batch.shape[1]):
                dist += np.abs(X_batch[:, d : d + 1] - self.X_train_[:, d])
            return dist
        elif self.metric == "minkowski":
            dist = np.zeros((len(X_batch), len(self.X_train_)), dtype=float)
            p = float(self.p)
            for d in range(X_batch.shape[1]):
                dist += np.abs(X_batch[:, d : d + 1] - self.X_train_[:, d]) ** p
            return dist ** (1.0 / p)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def predict_proba(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if len(X_arr) == 0:
            return np.empty((0, len(self.classes_)), dtype=float)
        if X_arr.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"Feature count mismatch: expected {self.X_train_.shape[1]}, got {X_arr.shape[1]}"
            )

        n_samples = len(X_arr)
        n_classes = len(self.classes_)
        k = int(self.n_neighbors)
        proba = np.zeros((n_samples, n_classes), dtype=float)

        batch_size = int(self.batch_size)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_arr[start:end]
            batch_len = end - start

            dists = self._compute_distances(X_batch)

            if k < len(self.X_train_):
                top_k_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
                row_idx = np.arange(batch_len)[:, None]
                top_k_dists = dists[row_idx, top_k_idx]
                sort_order = np.argsort(top_k_dists, axis=1, kind="stable")
                neighbor_idx = top_k_idx[row_idx, sort_order]
                neighbor_dists = top_k_dists[row_idx, sort_order]
            else:
                sort_order = np.argsort(dists, axis=1, kind="stable")
                neighbor_idx = sort_order[:, :k]
                neighbor_dists = np.take_along_axis(dists, neighbor_idx, axis=1)

            neighbor_labels = self.y_train_[neighbor_idx]

            if self.weights == "uniform":
                weights_arr = np.ones((batch_len, k), dtype=float)
            elif self.weights == "distance":
                with np.errstate(divide="ignore"):
                    inv_dists = 1.0 / neighbor_dists
                inf_mask = np.isinf(inv_dists)
                inf_row = np.any(inf_mask, axis=1)
                weights_arr = inv_dists.copy()
                if np.any(inf_row):
                    weights_arr[inf_row] = inf_mask[inf_row].astype(float)
            else:
                raise ValueError(f"Unknown weights: {self.weights}")

            batch_votes = np.zeros((batch_len, n_classes), dtype=float)
            for c_idx, c_val in enumerate(self.classes_):
                mask = (neighbor_labels == c_val)
                batch_votes[:, c_idx] = np.sum(weights_arr * mask, axis=1)

            total_votes = np.sum(batch_votes, axis=1, keepdims=True)
            batch_proba = np.where(total_votes > 0, batch_votes / total_votes, 1.0 / n_classes)
            proba[start:end] = batch_proba

        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        if len(proba) == 0:
            return np.empty(0, dtype=self.classes_.dtype)
        best_idx = np.argmax(proba, axis=1)
        return self.classes_[best_idx]
