from collections.abc import Callable, Iterator
import copy
import itertools

import numpy as np
import pandas as pd

from src.config import SEED
from src.evaluation.metrics import roc_auc
from src.models.base import BaseClassifier
from src.preprocessing.base import BaseResampler, Pipeline


def _clone_estimator(model: BaseClassifier):
    if hasattr(model, "clone") and callable(getattr(model, "clone")):
        return model.clone()
    from sklearn.base import clone as sk_clone

    return sk_clone(model)


def _clone_resampler(resampler: BaseResampler | None) -> BaseResampler | None:
    if resampler is None:
        return None
    clone = getattr(resampler, "clone", None)
    if callable(clone):
        cloned = clone()
        try:
            cloned = copy.deepcopy(cloned)
        except Exception as exc:
            if cloned is resampler:
                raise TypeError(
                    "Resampler cloning must produce an independent instance; "
                    "implement clone() or __deepcopy__()."
                ) from exc
    else:
        try:
            from sklearn.base import clone as sk_clone

            cloned = sk_clone(resampler)
        except Exception:
            cloned = copy.deepcopy(resampler)

    if cloned is resampler:
        cloned = copy.deepcopy(resampler)
        if cloned is resampler:
            raise TypeError(
                "Resampler cloning must produce an independent instance; "
                "implement clone() or __deepcopy__()."
            )
    return cloned


def _check_label_metric(metric: Callable[[np.ndarray, np.ndarray], float]) -> None:
    if metric is roc_auc:
        raise ValueError(
            "roc_auc requires continuous scores, but cross_validate and grid_search "
            "pass predicted labels to metrics. Compute ROC AUC separately on "
            "validation folds using predict_proba(X)[:, 1] or decision_function(X)."
        )

def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split rows with a test count of round(len(X) * test_size).

    Stratification allocates that count proportionally while keeping every
    class represented in both splits. An infeasible allocation raises
    ValueError instead of changing the requested total.
    """
    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError(f"Length mismatch: len(X)={n_samples} != len(y)={len(y)}")
    if n_samples == 0:
        raise ValueError("X and y cannot be empty")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be strictly between 0.0 and 1.0")

    y_arr = np.asarray(y).ravel()
    rng = np.random.default_rng(random_state)
    n_test = int(round(n_samples * test_size))

    if stratify:
        unique_classes, class_inverse, class_counts = np.unique(
            y_arr, return_inverse=True, return_counts=True
        )
        if len(unique_classes) < 2:
            raise ValueError("Stratified split requires at least 2 distinct classes in y")

        min_class_count = int(class_counts.min())
        if min_class_count < 2:
            raise ValueError(
                f"The least populated class in y has only {min_class_count} member(s), "
                "which is too few for stratified splitting."
            )

        n_classes = len(unique_classes)
        if n_test < n_classes or n_samples - n_test < n_classes:
            raise ValueError(
                f"Stratification is infeasible: {n_test} test sample(s) and "
                f"{n_samples - n_test} training sample(s) cannot each contain "
                f"all {n_classes} classes."
            )

        quotas = class_counts * (n_test / n_samples)
        test_counts = np.clip(np.floor(quotas).astype(int), 1, class_counts - 1)
        remaining = n_test - int(test_counts.sum())
        while remaining > 0:
            priority = np.where(
                test_counts < class_counts - 1, quotas - test_counts, -np.inf
            )
            test_counts[np.argmax(priority)] += 1
            remaining -= 1
        while remaining < 0:
            priority = np.where(test_counts > 1, test_counts - quotas, -np.inf)
            test_counts[np.argmax(priority)] -= 1
            remaining += 1

        test_indices: list[int] = []
        train_indices: list[int] = []
        for class_index, n_cls_test in enumerate(test_counts):
            cls_idx = np.flatnonzero(class_inverse == class_index)
            shuffled = rng.permutation(cls_idx)
            test_indices.extend(shuffled[:n_cls_test].tolist())
            train_indices.extend(shuffled[n_cls_test:].tolist())

        test_idx = rng.permutation(test_indices)
        train_idx = rng.permutation(train_indices)
    else:
        if n_test <= 0:
            raise ValueError(
                f"test_size={test_size} is too small for {n_samples} samples, resulting in an empty test set."
            )
        if n_test >= n_samples:
            raise ValueError(
                f"test_size={test_size} is too large for {n_samples} samples, resulting in an empty training set."
            )
        perm = rng.permutation(n_samples)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

    X_train = X.iloc[train_idx] if isinstance(X, (pd.DataFrame, pd.Series)) else X[train_idx]
    X_test = X.iloc[test_idx] if isinstance(X, (pd.DataFrame, pd.Series)) else X[test_idx]
    y_train = y.iloc[train_idx] if isinstance(y, (pd.DataFrame, pd.Series)) else y[train_idx]
    y_test = y.iloc[test_idx] if isinstance(y, (pd.DataFrame, pd.Series)) else y[test_idx]

    return X_train, X_test, y_train, y_test


class StratifiedKFold:
    def __init__(
        self, n_splits: int = 5, shuffle: bool = True, random_state: int = SEED
    ) -> None:
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer >= 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        if len(y) != n_samples:
            raise ValueError(f"Length mismatch: len(X)={n_samples} != len(y)={len(y)}")
        if n_samples < self.n_splits:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than sample count {n_samples}"
            )

        y_arr = np.asarray(y).ravel()
        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2:
            raise ValueError("Stratified split requires at least 2 distinct classes in y")

        rng = np.random.default_rng(self.random_state) if self.shuffle else None

        for cls in unique_classes:
            cls_count = int(np.sum(y_arr == cls))
            if cls_count < self.n_splits:
                raise ValueError(
                    f"n_splits={self.n_splits} cannot be greater than class count "
                    f"({cls_count} for class {cls})"
                )

        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        y_encoded = np.array([class_to_idx[c] for c in y_arr], dtype=int)
        n_classes = len(unique_classes)

        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [
                np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                for i in range(self.n_splits)
            ]
        )

        test_folds = np.empty(n_samples, dtype=int)
        for k, cls in enumerate(unique_classes):
            cls_indices = np.where(y_arr == cls)[0]
            if rng is not None:
                cls_indices = rng.permutation(cls_indices)
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
            test_folds[cls_indices] = folds_for_class

        for f in range(self.n_splits):
            val_idx = np.where(test_folds == f)[0]
            train_idx = np.where(test_folds != f)[0]
            yield train_idx, val_idx

def cross_validate(
    model: BaseClassifier,
    pipeline_factory: Callable[[], Pipeline] | None,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    resampler: BaseResampler | None = None,
) -> dict[str, list[float]]:
    """Evaluate label metrics on each fold using independently cloned resamplers.

    Custom clone() implementations must preserve configuration and return an
    independent, unfitted resampler. Resamplers without a clone protocol are
    deep-copied and should be unfitted templates. Compute ROC AUC separately
    from validation-fold scores.
    The pipeline receives the original DataFrames; model fit and prediction
    receive NumPy arrays after preprocessing and any training-only resampling.
    """
    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError(f"Length mismatch: len(X)={n_samples} != len(y)={len(y)}")
    if n_samples == 0:
        raise ValueError("X and y cannot be empty")
    if not metrics:
        raise ValueError("metrics dictionary cannot be empty")
    for metric in metrics.values():
        _check_label_metric(metric)

    results: dict[str, list[float]] = {name: [] for name in metrics.keys()}

    for train_idx, val_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        pipeline = pipeline_factory() if pipeline_factory is not None else None
        if pipeline is not None:
            X_tr_proc = pipeline.fit_transform(X_train, y_train)
            X_val_proc = pipeline.transform(X_val)
        else:
            X_tr_proc = X_train
            X_val_proc = X_val

        y_train_arr = np.asarray(y_train).ravel()
        y_val_arr = np.asarray(y_val).ravel()

        if resampler is not None:
            fold_resampler = _clone_resampler(resampler)
            X_tr_fit, y_tr_fit = fold_resampler.fit_resample(
                np.asarray(X_tr_proc), y_train_arr
            )
        else:
            X_tr_fit = X_tr_proc
            y_tr_fit = y_train_arr

        fold_model = _clone_estimator(model)
        fold_model.fit(np.asarray(X_tr_fit), np.asarray(y_tr_fit).ravel())

        y_pred = fold_model.predict(np.asarray(X_val_proc))
        y_pred_arr = np.asarray(y_pred).ravel()

        for name, metric_fn in metrics.items():
            score = metric_fn(y_val_arr, y_pred_arr)
            results[name].append(float(score))

    return results


def grid_search(
    model: BaseClassifier,
    param_grid: dict[str, list],
    pipeline_factory: Callable[[], Pipeline] | None,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    resampler: BaseResampler | None = None,
) -> tuple[dict, list[dict]]:
    """Search parameters using the label-metric contract of cross_validate."""
    n_samples = len(X)
    if len(y) != n_samples:
        raise ValueError(f"Length mismatch: len(X)={n_samples} != len(y)={len(y)}")
    if n_samples == 0:
        raise ValueError("X and y cannot be empty")
    if not isinstance(param_grid, dict):
        raise TypeError("param_grid must be a dictionary")
    if not param_grid:
        raise ValueError("param_grid cannot be empty")
    if not callable(scoring):
        raise TypeError("scoring must be a callable metric function")
    _check_label_metric(scoring)

    keys = list(param_grid.keys())
    for k in keys:
        v = param_grid[k]
        if not isinstance(v, (list, tuple, np.ndarray, range)):
            raise ValueError(
                f"param_grid values must be sequences, got {type(v).__name__} for key '{k}'"
            )
        if len(v) == 0:
            raise ValueError(f"param_grid['{k}'] cannot be empty")

    combinations = [
        dict(zip(keys, prod))
        for prod in itertools.product(*(param_grid[k] for k in keys))
    ]

    cv_results: list[dict] = []

    for combo in combinations:
        candidate_model = _clone_estimator(model)
        candidate_model.set_params(**combo)

        cv_out = cross_validate(
            model=candidate_model,
            pipeline_factory=pipeline_factory,
            X=X,
            y=y,
            cv=cv,
            metrics={"score": scoring},
            resampler=resampler,
        )

        scores = cv_out["score"]
        mean_score = float(np.mean(scores))

        cv_results.append(
            {
                "params": dict(combo),
                "scores": scores,
                "mean_score": mean_score,
            }
        )

    best_candidate = max(cv_results, key=lambda item: item["mean_score"])
    best_params = dict(best_candidate["params"])

    return best_params, cv_results
