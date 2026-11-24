"""Binary logistic regression with batch and mini-batch gradient descent."""

import warnings
from numbers import Integral, Real

import numpy as np

from src.models.base import BaseClassifier

_PENALTIES = (None, "none", "l1", "l2", "elasticnet")


def _is_number(value, kind):
    return isinstance(value, kind) and not isinstance(value, bool)


def _sigmoid(z):
    """Sigmoid of ``z``, with the logit clipped so ``exp`` cannot overflow."""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softplus(z):
    """``log(1 + exp(z))`` written so large positive or negative logits stay finite."""
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def _soft_threshold(coef, threshold):
    """Proximal map of ``threshold * ||w||_1``."""
    return np.sign(coef) * np.maximum(np.abs(coef) - threshold, 0.0)


def _validate_class_weight(class_weight):
    if class_weight is None or class_weight == "balanced":
        return
    if not isinstance(class_weight, dict):
        raise ValueError(
            f"class_weight must be None, 'balanced', or a dict, got {class_weight!r}"
        )
    for label, weight in class_weight.items():
        if not _is_number(weight, Real) or not np.isfinite(weight) or weight < 0:
            raise ValueError(
                f"class weight for {label!r} must be a finite number >= 0, got {weight!r}"
            )


def _sample_weights(class_weight, classes, y):
    """One weight per training row."""
    if class_weight is None:
        table = {label: 1.0 for label in classes}
    elif class_weight == "balanced":
        table = {label: len(y) / (2.0 * np.sum(y == label)) for label in classes}
    else:
        table = {label: float(class_weight.get(label, 1.0)) for label in classes}
    return np.fromiter((table[label] for label in y), dtype=float, count=len(y))


def _penalty_terms(penalty, reg_lambda, l1_ratio):
    """Return ``(l2_strength, l1_strength)`` for ``(l2/2)||w||^2 + l1||w||_1``."""
    if penalty is None or penalty == "none":
        return 0.0, 0.0
    if penalty == "l2":
        return float(reg_lambda), 0.0
    if penalty == "l1":
        return 0.0, float(reg_lambda)
    return (1.0 - float(l1_ratio)) * float(reg_lambda), float(l1_ratio) * float(reg_lambda)


def _objective(X, y, weights, coef, intercept, l2, l1):
    """Weighted mean log-loss plus the L2 and L1 penalties on ``coef``."""
    z = X @ coef + intercept
    with np.errstate(over="ignore", invalid="ignore"):
        data = np.mean(weights * (_softplus(z) - y * z))
        penalty = 0.5 * l2 * np.dot(coef, coef) + l1 * np.sum(np.abs(coef))
    if not np.isfinite(data) or not np.isfinite(penalty):
        return float("inf")
    return float(data + penalty)


def _gradients(X, y, weights, coef, intercept):
    """Mean gradient of the weighted log-loss for the weights and the intercept."""
    residual = weights * (_sigmoid(X @ coef + intercept) - y)
    scale = float(len(y))
    return (X.T @ residual) / scale, float(residual.sum() / scale)


class _GradientDescent:
    """One proximal gradient step at a fixed learning rate."""

    def __init__(self, learning_rate, fit_intercept):
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept

    def update(self, coef, intercept, grad_w, grad_b, l2, l1):
        """Step the smooth loss and L2 term, then soft-threshold the L1 part."""
        coef = coef - self.learning_rate * (grad_w + l2 * coef)
        if l1 > 0.0:
            coef = _soft_threshold(coef, self.learning_rate * l1)
        if self.fit_intercept:
            intercept = intercept - self.learning_rate * grad_b
        return coef, float(intercept)


class _Adam:
    """Bias-corrected Adam step, with L1 applied by a scaled soft-threshold."""

    def __init__(self, learning_rate, fit_intercept, beta1, beta2, epsilon, n_features):
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_w = np.zeros(n_features)
        self.v_w = np.zeros(n_features)
        self.m_b = 0.0
        self.v_b = 0.0
        self.t = 0

    def update(self, coef, intercept, grad_w, grad_b, l2, l1):
        """Adam update of the smooth loss. The L1 threshold uses the second moment."""
        self.t += 1
        grad_w = grad_w + l2 * coef
        self.m_w = self.beta1 * self.m_w + (1.0 - self.beta1) * grad_w
        self.v_w = self.beta2 * self.v_w + (1.0 - self.beta2) * (grad_w * grad_w)
        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t
        denom = np.sqrt(self.v_w / bias2) + self.epsilon
        coef = coef - self.learning_rate * (self.m_w / bias1) / denom
        if l1 > 0.0:
            coef = _soft_threshold(coef, self.learning_rate * l1 / denom)
        if self.fit_intercept:
            self.m_b = self.beta1 * self.m_b + (1.0 - self.beta1) * grad_b
            self.v_b = self.beta2 * self.v_b + (1.0 - self.beta2) * (grad_b * grad_b)
            intercept = intercept - self.learning_rate * (self.m_b / bias1) / (
                np.sqrt(self.v_b / bias2) + self.epsilon
            )
        return coef, float(intercept)


class LogisticRegression(BaseClassifier):
    """Binary logistic regression fit by proximal gradient descent."""

    def __init__(
        self,
        penalty="l2",
        reg_lambda=0.01,
        l1_ratio=0.5,
        learning_rate=0.01,
        n_iter=1000,
        batch_size=None,
        tol=1e-6,
        fit_intercept=True,
        class_weight=None,
        record_history=False,
        random_state=42,
    ):
        super().__init__(random_state)
        self.penalty = penalty
        self.reg_lambda = reg_lambda
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.record_history = record_history

    def _validate_params(self):
        if self.penalty not in _PENALTIES:
            raise ValueError(
                f"penalty must be None, 'none', 'l1', 'l2', or 'elasticnet', got {self.penalty!r}"
            )
        if not _is_number(self.reg_lambda, Real) or not np.isfinite(self.reg_lambda) or self.reg_lambda < 0:
            raise ValueError(f"reg_lambda must be a finite number >= 0, got {self.reg_lambda!r}")
        if not _is_number(self.l1_ratio, Real) or not np.isfinite(self.l1_ratio) or not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be a number in [0, 1], got {self.l1_ratio!r}")
        if (
            not _is_number(self.learning_rate, Real)
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(f"learning_rate must be a finite number > 0, got {self.learning_rate!r}")
        if not _is_number(self.n_iter, Integral) or self.n_iter < 1:
            raise ValueError(f"n_iter must be an integer >= 1, got {self.n_iter!r}")
        if self.batch_size is not None and (
            not _is_number(self.batch_size, Integral) or self.batch_size < 1
        ):
            raise ValueError(f"batch_size must be None or an integer >= 1, got {self.batch_size!r}")
        if not _is_number(self.tol, Real) or not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError(f"tol must be a finite number > 0, got {self.tol!r}")
        if not isinstance(self.fit_intercept, bool):
            raise ValueError(f"fit_intercept must be a bool, got {self.fit_intercept!r}")
        if not isinstance(self.record_history, bool):
            raise ValueError(f"record_history must be a bool, got {self.record_history!r}")
        _validate_class_weight(self.class_weight)

    def _optimizer(self, n_features):
        return _GradientDescent(self.learning_rate, self.fit_intercept)

    def _prepare_training(self, X, y):
        """Validate inputs and return ``(X, classes, y_positive, sample_weights)``."""
        self._validate_params()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D")
        if len(X) == 0:
            raise ValueError("X and y cannot be empty")
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: len(X)={len(X)} != len(y)={len(y)}")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or inf")
        if y.dtype.kind in "fc" and not np.all(np.isfinite(y)):
            raise ValueError("y must not contain NaN or inf")
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"y must contain exactly two classes, got {len(classes)}")
        y_bin = (y == classes[1]).astype(float)
        weights = _sample_weights(self.class_weight, classes, y)
        return X, classes, y_bin, weights

    def fit(self, X, y):
        """Fit on a finite numeric matrix and exactly two classes."""
        X, classes, y_bin, weights = self._prepare_training(X, y)
        n_samples, n_features = X.shape
        l2, l1 = _penalty_terms(self.penalty, self.reg_lambda, self.l1_ratio)
        coef = np.zeros(n_features)
        intercept = 0.0
        optimizer = self._optimizer(n_features)
        rng = self._rng()
        loss_history = [_objective(X, y_bin, weights, coef, intercept, l2, l1)]
        coef_history = [coef.copy()] if self.record_history else None
        intercept_history = [0.0] if self.record_history else None

        converged = False
        n_epochs = 0
        previous = loss_history[-1]
        batch_size = n_samples if self.batch_size is None else int(self.batch_size)
        for _ in range(int(self.n_iter)):
            if batch_size >= n_samples:
                splits = (slice(None),)
            else:
                order = rng.permutation(n_samples)
                splits = tuple(order[start : start + batch_size] for start in range(0, n_samples, batch_size))
            for rows in splits:
                grad_w, grad_b = _gradients(X[rows], y_bin[rows], weights[rows], coef, intercept)
                coef, intercept = optimizer.update(coef, intercept, grad_w, grad_b, l2, l1)
                if coef_history is not None:
                    coef_history.append(coef.copy())
                    intercept_history.append(intercept)
            n_epochs += 1
            loss = _objective(X, y_bin, weights, coef, intercept, l2, l1)
            loss_history.append(loss)
            if not np.isfinite(loss):
                break
            if abs(loss - previous) < self.tol:
                converged = True
                break
            previous = loss

        self.classes_ = classes
        self.coef_ = coef
        self.intercept_ = float(intercept)
        self.n_iter_ = n_epochs
        self.converged_ = converged
        self.loss_history_ = np.asarray(loss_history, dtype=float)
        if self.record_history:
            self.coef_history_ = np.vstack(coef_history)
            self.intercept_history_ = np.asarray(intercept_history, dtype=float)
        else:
            self.coef_history_ = None
            self.intercept_history_ = None

        if not converged:
            warnings.warn(
                f"Logistic regression did not converge after {n_epochs} iterations; "
                "increase n_iter or tol, or reduce learning_rate",
                RuntimeWarning,
                stacklevel=2,
            )
        return self

    def _validate_X(self, X):
        """Return ``X`` as a finite 2-D array with the fitted number of columns."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got {X.ndim}D")
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fit with {self.coef_.shape[0]}"
            )
        if len(X) and not np.all(np.isfinite(X)):
            raise ValueError("X must not contain NaN or inf")
        return X

    def decision_function(self, X):
        """Logit of the positive class, ``X @ coef_ + intercept_``."""
        self._check_is_fitted()
        X = self._validate_X(X)
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        """Class probabilities with columns in ``classes_`` order. Rows sum to 1."""
        self._check_is_fitted()
        positive = _sigmoid(self.decision_function(X))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X):
        """Return ``classes_[1]`` when its probability is at least 0.5."""
        self._check_is_fitted()
        positive = _sigmoid(self.decision_function(X))
        return np.where(positive >= 0.5, self.classes_[1], self.classes_[0])


class AdamLogisticRegression(LogisticRegression):
    """Logistic regression trained with bias-corrected Adam."""

    def __init__(
        self,
        penalty="l2",
        reg_lambda=0.01,
        l1_ratio=0.5,
        learning_rate=0.01,
        n_iter=1000,
        batch_size=None,
        tol=1e-6,
        fit_intercept=True,
        class_weight=None,
        record_history=False,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        random_state=42,
    ):
        super().__init__(
            penalty=penalty,
            reg_lambda=reg_lambda,
            l1_ratio=l1_ratio,
            learning_rate=learning_rate,
            n_iter=n_iter,
            batch_size=batch_size,
            tol=tol,
            fit_intercept=fit_intercept,
            class_weight=class_weight,
            record_history=record_history,
            random_state=random_state,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _validate_params(self):
        super()._validate_params()
        if not _is_number(self.beta1, Real) or not np.isfinite(self.beta1) or not 0.0 < self.beta1 < 1.0:
            raise ValueError(f"beta1 must be a number in (0, 1), got {self.beta1!r}")
        if not _is_number(self.beta2, Real) or not np.isfinite(self.beta2) or not 0.0 < self.beta2 < 1.0:
            raise ValueError(f"beta2 must be a number in (0, 1), got {self.beta2!r}")
        if not _is_number(self.epsilon, Real) or not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError(f"epsilon must be a finite number > 0, got {self.epsilon!r}")

    def _optimizer(self, n_features):
        return _Adam(
            self.learning_rate,
            self.fit_intercept,
            self.beta1,
            self.beta2,
            self.epsilon,
            n_features,
        )
