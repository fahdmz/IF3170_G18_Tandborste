from numbers import Real

import numpy as np

from src.config import SEED
from src.models.base import BaseClassifier


def _is_missing_scalar(val) -> bool:
    if val is None:
        return True
    if isinstance(val, (float, np.floating)):
        return bool(np.isnan(val))
    return False


def _constant_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = ~np.isnan(X)
    first_values = X[np.argmax(observed, axis=0), np.arange(X.shape[1])]
    constant = np.all(~observed | (X == first_values), axis=0)
    return constant, first_values


class NaiveBayes(BaseClassifier):
    def __init__(
        self,
        categorical_features=None,
        alpha: float = 1.0,
        var_smoothing: float = 1e-9,
        random_state: int = SEED,
    ):
        super().__init__(random_state)
        self.categorical_features = categorical_features
        self.alpha = alpha
        self.var_smoothing = var_smoothing

    def _validate_params(self) -> None:
        if not isinstance(self.alpha, Real) or self.alpha < 0:
            raise ValueError(f"alpha must be a non-negative number, got {self.alpha}")
        if not isinstance(self.var_smoothing, Real) or self.var_smoothing < 0:
            raise ValueError(f"var_smoothing must be a non-negative number, got {self.var_smoothing}")

    def _resolve_feature_indices(self, n_features: int) -> tuple[list[int], list[int]]:
        if self.categorical_features is None:
            cat_indices = []
        elif isinstance(self.categorical_features, (list, tuple, np.ndarray, set)):
            cat_indices = []
            for item in self.categorical_features:
                if isinstance(item, (bool, np.bool_)):
                    raise TypeError(
                        f"Boolean masks are not supported in categorical_features, got {item}"
                    )
                if not isinstance(item, (int, np.integer)):
                    raise TypeError(
                        f"Categorical feature indices must be integers, got {type(item).__name__} ({item})"
                    )
                idx = int(item)
                if idx < 0 or idx >= n_features:
                    raise ValueError(
                        f"Categorical feature index {idx} out of range [0, {n_features - 1}]"
                    )
                cat_indices.append(idx)
        else:
            raise TypeError(
                f"categorical_features must be an iterable of integer indices or None, got {type(self.categorical_features).__name__}"
            )

        cat_set = set(cat_indices)
        cat_indices = sorted(list(cat_set))
        num_indices = [i for i in range(n_features) if i not in cat_set]
        return cat_indices, num_indices

    def fit(self, X, y):
        self._validate_params()

        X_arr = np.asarray(X)
        if hasattr(X_arr, "ndim") and X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X_arr.ndim}D")

        n_samples, n_features = X_arr.shape
        if n_samples == 0:
            raise ValueError("X and y cannot be empty")
        if n_features == 0:
            raise ValueError("X must have at least one feature")

        y_arr = np.asarray(y).ravel()
        if len(y_arr) != n_samples:
            raise ValueError(f"Length mismatch: len(X)={n_samples} != len(y)={len(y_arr)}")

        unique_y = np.unique(y_arr)
        if not np.array_equal(np.sort(unique_y), np.array([0, 1])):
            raise ValueError(
                f"y must contain binary labels (0 and 1), got {unique_y}"
            )

        self.classes_ = np.array([0, 1], dtype=int)
        n_classes = 2
        self.n_features_in_ = n_features
        self.categorical_indices_, self.numerical_indices_ = self._resolve_feature_indices(n_features)

        # Class counts and priors
        self.class_count_ = np.array([np.sum(y_arr == c) for c in self.classes_], dtype=float)
        self.class_prior_ = self.class_count_ / float(n_samples)
        self.class_log_prior_ = np.log(self.class_prior_)

        # Numerical Gaussian fitting
        n_num = len(self.numerical_indices_)
        self.theta_ = np.full((n_classes, n_num), np.nan, dtype=float)
        self.var_ = np.full((n_classes, n_num), np.nan, dtype=float)
        self.globally_constant_ = np.zeros(n_num, dtype=bool)
        self.numerical_observed_ = np.zeros(n_num, dtype=bool)
        if n_num > 0:
            X_num = np.asarray(X_arr[:, self.numerical_indices_], dtype=float)
            self.numerical_observed_ = np.any(~np.isnan(X_num), axis=0)

        if np.any(self.numerical_observed_):
            X_num = X_num[:, self.numerical_observed_]

            if self.var_smoothing == 0.0:
                self.epsilon_ = 0.0
            else:
                with np.errstate(all="ignore"):
                    var_all = np.nanvar(X_num, axis=0)
                valid_vars = var_all[~np.isnan(var_all)]
                max_var = float(np.max(valid_vars)) if len(valid_vars) > 0 else 0.0
                if max_var > 0.0:
                    self.epsilon_ = float(self.var_smoothing * max_var)
                else:
                    self.epsilon_ = float(self.var_smoothing)

            with np.errstate(all="ignore"):
                global_mean = np.nanmean(X_num, axis=0)
                global_var = np.nanvar(X_num, axis=0)
            global_mean = np.nan_to_num(global_mean, nan=0.0)
            global_var = np.nan_to_num(global_var, nan=self.epsilon_)

            if self.epsilon_ == 0.0:
                globally_constant, first_values = _constant_columns(X_num)
                global_mean = np.where(
                    globally_constant & ~np.isnan(first_values), first_values, global_mean
                )
                global_var = np.where(globally_constant, 0.0, global_var)
            else:
                globally_constant = (global_var == 0.0)
            self.globally_constant_[self.numerical_observed_] = globally_constant

            for i, c in enumerate(self.classes_):
                mask = (y_arr == c)
                X_c = X_num[mask]
                with np.errstate(all="ignore"):
                    mean_c = np.nanmean(X_c, axis=0)
                    var_c = np.nanvar(X_c, axis=0)

                if self.epsilon_ == 0.0:
                    constant, first_values = _constant_columns(X_c)
                    constant &= ~np.isnan(first_values)
                    mean_c = np.where(constant, first_values, mean_c)
                    var_c = np.where(constant, 0.0, var_c)

                mean_c = np.where(np.isnan(mean_c), global_mean, mean_c)
                var_c = np.where(np.isnan(var_c), global_var, var_c)

                self.theta_[i, self.numerical_observed_] = mean_c
                self.var_[i, self.numerical_observed_] = var_c + self.epsilon_
        else:
            self.epsilon_ = 0.0

        # Categorical Laplace fitting
        self.categories_ = []
        self.cat_to_idx_ = []
        self.category_count_ = []
        self.feature_log_prob_ = []

        if len(self.categorical_indices_) > 0:
            for col_idx in self.categorical_indices_:
                col = X_arr[:, col_idx]

                valid_mask = np.array([not _is_missing_scalar(v) for v in col], dtype=bool)
                if np.any(valid_mask):
                    valid_vals = list(dict.fromkeys(col[valid_mask]))
                    try:
                        valid_vals = sorted(valid_vals)
                    except TypeError:
                        pass
                    unique_cats = np.array(valid_vals, dtype=object)
                else:
                    unique_cats = np.array([], dtype=object)

                cat_to_idx = {val: idx for idx, val in enumerate(unique_cats)}
                K_j = len(unique_cats)
                counts = np.zeros((n_classes, K_j), dtype=float)

                for i, c in enumerate(self.classes_):
                    class_mask = (y_arr == c) & valid_mask
                    col_c = col[class_mask]
                    for val in col_c:
                        counts[i, cat_to_idx[val]] += 1.0

                if K_j > 0:
                    smoothed_cat_count = counts + self.alpha
                    smoothed_class_count = np.sum(smoothed_cat_count, axis=1, keepdims=True)
                    with np.errstate(divide="ignore"):
                        log_prob = np.where(
                            smoothed_cat_count > 0,
                            np.log(smoothed_cat_count) - np.log(smoothed_class_count),
                            -np.inf,
                        )
                else:
                    log_prob = np.zeros((n_classes, 0), dtype=float)

                self.categories_.append(unique_cats)
                self.cat_to_idx_.append(cat_to_idx)
                self.category_count_.append(counts)
                self.feature_log_prob_.append(log_prob)

        return self

    def _check_and_prepare_X(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = np.asarray(X)
        if hasattr(X_arr, "ndim") and X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        elif not hasattr(X_arr, "shape") and len(X_arr) > 0 and not isinstance(X_arr[0], (list, tuple, np.ndarray)):
            X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X_arr.ndim}D")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature count mismatch: expected {self.n_features_in_}, got {X_arr.shape[1]}"
            )
        return X_arr

    def _joint_log_likelihood(self, X_arr: np.ndarray) -> np.ndarray:
        n_samples = len(X_arr)
        n_classes = len(self.classes_)

        if n_samples == 0:
            return np.empty((0, n_classes), dtype=float)

        jll = np.zeros((n_samples, n_classes), dtype=float)
        for i in range(n_classes):
            jll[:, i] = self.class_log_prior_[i]

        point_matches = None
        if np.any(self.var_ == 0.0):
            point_matches = np.zeros((n_samples, n_classes), dtype=int)
            support = np.ones((n_samples, n_classes), dtype=bool)

        # Numerical likelihood
        numerical_positions = np.flatnonzero(self.numerical_observed_)
        if len(numerical_positions) > 0:
            numerical_indices = [self.numerical_indices_[f_idx] for f_idx in numerical_positions]
            X_num = np.asarray(X_arr[:, numerical_indices], dtype=float)
            nan_mask = np.isnan(X_num)

            for i in range(n_classes):
                theta_i = self.theta_[i]
                var_i = self.var_[i]

                feat_ll = np.zeros((n_samples, len(numerical_positions)), dtype=float)
                for active_idx, f_idx in enumerate(numerical_positions):
                    if self.globally_constant_[f_idx]:
                        continue

                    col_vals = X_num[:, active_idx]
                    col_nan = nan_mask[:, active_idx]
                    v = var_i[f_idx]
                    t = theta_i[f_idx]

                    if v == 0.0:
                        match = col_vals == t
                        point_matches[:, i] += match & ~col_nan
                        support[:, i] &= col_nan | match
                        feat_ll[:, active_idx] = np.where(col_nan | match, 0.0, -np.inf)
                    else:
                        with np.errstate(all="ignore"):
                            term1 = -0.5 * (np.log(2.0 * np.pi) + np.log(v))
                            diff = np.where(col_nan, 0.0, col_vals - t)
                            scaled_diff = diff / np.sqrt(v)
                            term2 = -0.5 * scaled_diff * scaled_diff
                            feat_ll[:, active_idx] = np.where(col_nan, 0.0, term1 + term2)

                jll[:, i] += np.sum(feat_ll, axis=1)

        # Categorical likelihood
        if len(self.categorical_indices_) > 0:
            for k, col_idx in enumerate(self.categorical_indices_):
                col = X_arr[:, col_idx]
                cat_to_idx = self.cat_to_idx_[k]
                log_prob = self.feature_log_prob_[k]

                if len(cat_to_idx) == 0:
                    continue

                mapped = np.full(n_samples, -1, dtype=int)
                for s_idx, val in enumerate(col):
                    if not _is_missing_scalar(val) and val in cat_to_idx:
                        mapped[s_idx] = cat_to_idx[val]

                valid = (mapped >= 0)
                if np.any(valid):
                    valid_idx = mapped[valid]
                    for i in range(n_classes):
                        cat_ll = log_prob[i, valid_idx]
                        jll[valid, i] += cat_ll
                        if point_matches is not None:
                            support[valid, i] &= ~np.isneginf(cat_ll)

        if point_matches is not None:
            # Matched point masses dominate densities in the zero-variance limit.
            possible_matches = np.where(support, point_matches, -1)
            max_matches = np.max(possible_matches, axis=1, keepdims=True)
            eligible = support & (point_matches == max_matches)
            jll[~eligible] = -np.inf
            # A sole supported class remains certain even if its log density overflows.
            only_eligible = np.sum(eligible, axis=1, keepdims=True) == 1
            jll[eligible & only_eligible & np.isneginf(jll)] = 0.0

        return jll

    def predict_proba(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = self._check_and_prepare_X(X)
        n_samples = len(X_arr)
        if n_samples == 0:
            return np.empty((0, len(self.classes_)), dtype=float)

        jll = self._joint_log_likelihood(X_arr)

        all_neginf = np.all(np.isneginf(jll), axis=1)
        proba = np.zeros_like(jll, dtype=float)

        if np.any(all_neginf):
            proba[all_neginf] = self.class_prior_

        not_all_neginf = ~all_neginf
        if np.any(not_all_neginf):
            sub_jll = jll[not_all_neginf]
            sub_max = np.max(sub_jll, axis=1, keepdims=True)
            sub_max = np.where(np.isneginf(sub_max), 0.0, sub_max)
            exp_jll = np.exp(sub_jll - sub_max)
            row_sums = np.sum(exp_jll, axis=1, keepdims=True)
            proba[not_all_neginf] = exp_jll / row_sums

        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        if len(proba) == 0:
            return np.empty(0, dtype=self.classes_.dtype)
        best_idx = np.argmax(proba, axis=1)
        return self.classes_[best_idx]
