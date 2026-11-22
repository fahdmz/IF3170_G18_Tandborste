import numpy as np

from src.config import SEED
from src.preprocessing.base import BaseResampler


def _validate(X: np.ndarray, y: np.ndarray, sampling_strategy: float) -> None:
    if not 0.0 < sampling_strategy <= 1.0:
        raise ValueError("sampling_strategy harus di antara 0 (eksklusif) dan 1")
    if X.ndim != 2:
        raise ValueError("X harus berupa array 2 dimensi")
    if len(X) != len(y):
        raise ValueError("X dan y harus memiliki jumlah baris yang sama")


def _class_indices(y: np.ndarray) -> tuple[np.generic, np.ndarray, np.ndarray]:
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) != 2:
        raise ValueError("Resampling membutuhkan tepat dua kelas pada y")
    minority = labels[int(np.argmin(counts))]
    majority = labels[int(np.argmax(counts))]
    if minority == majority:
        minority, majority = labels[0], labels[1]
    return minority, np.flatnonzero(y == minority), np.flatnonzero(y == majority)


class RandomUnderSampler(BaseResampler):
    def __init__(self, sampling_strategy: float = 1.0, random_state: int = SEED) -> None:
        super().__init__(random_state)
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        y = np.asarray(y)
        _validate(X, y, self.sampling_strategy)
        rng = self._rng()
        _, minority_idx, majority_idx = _class_indices(y)
        target = int(round(len(minority_idx) / self.sampling_strategy))
        if target < len(majority_idx):
            majority_idx = rng.choice(majority_idx, size=target, replace=False)
        idx = np.concatenate([minority_idx, majority_idx])
        idx = idx[rng.permutation(len(idx))]
        return X[idx], y[idx]


class RandomOverSampler(BaseResampler):
    def __init__(self, sampling_strategy: float = 1.0, random_state: int = SEED) -> None:
        super().__init__(random_state)
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X)
        y = np.asarray(y)
        _validate(X, y, self.sampling_strategy)
        rng = self._rng()
        _, minority_idx, majority_idx = _class_indices(y)
        n_new = int(round(self.sampling_strategy * len(majority_idx))) - len(minority_idx)
        idx = np.arange(len(y))
        if n_new > 0:
            extra = rng.choice(minority_idx, size=n_new, replace=True)
            idx = np.concatenate([idx, extra])
        idx = idx[rng.permutation(len(idx))]
        return X[idx], y[idx]


class SMOTE(BaseResampler):
    def __init__(
        self,
        sampling_strategy: float = 1.0,
        k_neighbors: int = 5,
        batch_size: int = 1024,
        random_state: int = SEED,
    ) -> None:
        super().__init__(random_state)
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size

    def _nearest_neighbors(self, M: np.ndarray, rows: np.ndarray, k: int) -> np.ndarray:
        neighbors = np.zeros((len(M), k), dtype=np.int64)
        sq_norms = (M**2).sum(axis=1)
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            dist = sq_norms[batch][:, None] + sq_norms[None, :] - 2.0 * (M[batch] @ M.T)
            dist[np.arange(len(batch)), batch] = np.inf
            nearest = np.argpartition(dist, k - 1, axis=1)[:, :k]
            order = np.argsort(np.take_along_axis(dist, nearest, axis=1), axis=1, kind="stable")
            neighbors[batch] = np.take_along_axis(nearest, order, axis=1)
        return neighbors

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype="float64")
        y = np.asarray(y)
        _validate(X, y, self.sampling_strategy)
        if not np.all(np.isfinite(X)):
            raise ValueError("SMOTE membutuhkan nilai numerik yang finite, tanpa NaN atau inf")
        if self.k_neighbors < 1 or self.batch_size < 1:
            raise ValueError("k_neighbors dan batch_size harus >= 1")
        rng = self._rng()
        label, minority_idx, majority_idx = _class_indices(y)
        n_new = int(round(self.sampling_strategy * len(majority_idx))) - len(minority_idx)
        if n_new <= 0:
            order = rng.permutation(len(y))
            return X[order], y[order]
        if len(minority_idx) < 2:
            raise ValueError("SMOTE membutuhkan minimal 2 sampel minoritas")

        M = X[minority_idx]
        k = min(self.k_neighbors, len(M) - 1)
        base = rng.integers(0, len(M), size=n_new)
        neighbors = self._nearest_neighbors(M, np.unique(base), k)
        chosen = neighbors[base, rng.integers(0, k, size=n_new)]
        gap = rng.random((n_new, 1))
        synthetic = M[base] + gap * (M[chosen] - M[base])

        X_all = np.vstack([X, synthetic])
        y_all = np.concatenate([y, np.full(n_new, label, dtype=y.dtype)])
        order = rng.permutation(len(y_all))
        return X_all[order], y_all[order]
