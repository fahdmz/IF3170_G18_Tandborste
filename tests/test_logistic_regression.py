import ast
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from src.models.logistic_regression import AdamLogisticRegression, LogisticRegression

PENALTIES = [None, "l2", "l1", "elasticnet"]


def _sklearn_model(penalty, n_train, reg_lambda=0.01, l1_ratio=0.5, class_weight=None):
    common = dict(solver="saga", max_iter=8000, tol=1e-8, random_state=42, class_weight=class_weight)
    if penalty in (None, "none"):
        return SkLogisticRegression(C=np.inf, **common)
    ratio = {"l2": 0.0, "l1": 1.0, "elasticnet": l1_ratio}[penalty]
    return SkLogisticRegression(C=1.0 / (reg_lambda * n_train), l1_ratio=ratio, **common)


@pytest.fixture(scope="module")
def dataset():
    X, y = make_classification(n_samples=2000, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(y))
    train, test = order[:1500], order[1500:]
    return X[train], y[train], X[test], y[test]


def _fit(penalty, X, y):
    return LogisticRegression(
        penalty=penalty,
        reg_lambda=0.01,
        l1_ratio=0.5,
        learning_rate=0.5,
        n_iter=1000,
        tol=1e-6,
    ).fit(X, y)


@pytest.mark.parametrize("penalty", PENALTIES)
def test_matches_sklearn(dataset, penalty):
    X_tr, y_tr, X_te, y_te = dataset
    own = _fit(penalty, X_tr, y_tr)
    ref = _sklearn_model(penalty, len(y_tr)).fit(X_tr, y_tr)

    own_acc = np.mean(own.predict(X_te) == y_te)
    ref_acc = np.mean(ref.predict(X_te) == y_te)
    assert abs(own_acc - ref_acc) < 0.03
    assert own.converged_
    correlation = np.corrcoef(own.decision_function(X_te), ref.decision_function(X_te))[0, 1]
    assert correlation > 0.99


def test_none_penalty_alias_matches(dataset):
    X_tr, y_tr, X_te, _ = dataset
    omitted = _fit(None, X_tr, y_tr)
    named = _fit("none", X_tr, y_tr)
    np.testing.assert_allclose(omitted.decision_function(X_te), named.decision_function(X_te))


def test_probabilities_follow_classes(dataset):
    X_tr, y_tr, X_te, _ = dataset
    model = _fit("l2", X_tr, y_tr)
    assert model.fit(X_tr, y_tr) is model
    np.testing.assert_array_equal(model.classes_, [0, 1])
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    expected = np.where(proba[:, 1] >= 0.5, model.classes_[1], model.classes_[0])
    np.testing.assert_array_equal(model.predict(X_te), expected)
    np.testing.assert_allclose(model.coef_.shape, (X_tr.shape[1],))


def test_full_and_mini_batch_fit_and_repeat(dataset):
    X_tr, y_tr, X_te, y_te = dataset
    full = _fit("l2", X_tr, y_tr)
    params = dict(penalty="l2", learning_rate=0.5, n_iter=80, batch_size=64, tol=1e-4, random_state=7)
    first = LogisticRegression(**params).fit(X_tr, y_tr)
    second = LogisticRegression(**params).fit(X_tr, y_tr)
    np.testing.assert_allclose(first.decision_function(X_te), second.decision_function(X_te))
    assert np.mean(full.predict(X_te) == y_te) > 0.8
    assert np.mean(first.predict(X_te) == y_te) > 0.8


def test_early_stopping_sets_converged(dataset):
    X_tr, y_tr, _, _ = dataset
    model = LogisticRegression(learning_rate=0.5, n_iter=5000, tol=1e-4).fit(X_tr, y_tr)
    assert model.converged_
    assert model.n_iter_ < 5000
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_max_iter_warns_when_not_converged(dataset):
    X_tr, y_tr, _, _ = dataset
    with pytest.warns(RuntimeWarning, match="did not converge"):
        model = LogisticRegression(learning_rate=0.5, n_iter=1, tol=1e-12).fit(X_tr, y_tr)
    assert not model.converged_
    assert model.n_iter_ == 1


def test_class_weight_increases_minority_recall():
    X, y = make_classification(n_samples=1500, weights=[0.9, 0.1], random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    params = dict(penalty="l2", reg_lambda=0.1, learning_rate=0.5, n_iter=1000)
    plain = LogisticRegression(**params).fit(X, y)
    balanced = LogisticRegression(class_weight="balanced", **params).fit(X, y)
    heavy = LogisticRegression(class_weight={0: 1.0, 1: 5.0}, **params).fit(X, y)
    recall = lambda model: np.mean(model.predict(X)[y == 1] == 1)
    assert recall(balanced) > recall(plain)
    assert recall(heavy) > recall(plain)


def test_history_and_save_load(dataset, tmp_path):
    X_tr, y_tr, X_te, _ = dataset
    model = LogisticRegression(learning_rate=0.5, n_iter=200, tol=1e-6, record_history=True).fit(X_tr, y_tr)
    assert model.loss_history_.shape == (model.n_iter_ + 1,)
    assert model.coef_history_.shape == (model.n_iter_ + 1, X_tr.shape[1])
    assert model.intercept_history_.shape == (model.n_iter_ + 1,)
    assert model.loss_history_[-1] < model.loss_history_[0]

    silent = LogisticRegression(learning_rate=0.5, n_iter=50, tol=1e-3, record_history=False).fit(X_tr, y_tr)
    assert silent.loss_history_.shape[0] == silent.n_iter_ + 1
    assert silent.coef_history_ is None
    assert silent.intercept_history_ is None

    path = tmp_path / "logistic_regression.pkl"
    model.save(path)
    loaded = LogisticRegression.load(path)
    np.testing.assert_array_equal(model.predict(X_te), loaded.predict(X_te))
    np.testing.assert_allclose(model.decision_function(X_te), loaded.decision_function(X_te))


def test_rejects_bad_input(dataset):
    X_tr, y_tr, _, _ = dataset
    with pytest.raises(ValueError):
        LogisticRegression().fit(np.array([[0.0, 1.0], [np.nan, 2.0]]), np.array([0, 1]))
    with pytest.raises(ValueError):
        LogisticRegression().fit(np.zeros((4, 2)), np.zeros(4))
    with pytest.raises(ValueError):
        LogisticRegression().fit(X_tr, y_tr[:-1])
    with pytest.raises(RuntimeError):
        LogisticRegression().predict(X_tr[:2])
    with pytest.raises(RuntimeError):
        LogisticRegression().predict_proba(X_tr[:2])
    with pytest.raises(RuntimeError):
        LogisticRegression().decision_function(X_tr[:2])

    fitted = _fit("l2", X_tr, y_tr)
    with pytest.raises(ValueError):
        fitted.predict(X_tr[:, :1])
    empty = fitted.predict(X_tr[:0])
    assert empty.shape == (0,)
    assert fitted.predict_proba(X_tr[:0]).shape == (0, 2)


@pytest.mark.parametrize(
    "params",
    [
        {"penalty": "ridge"},
        {"reg_lambda": -1},
        {"l1_ratio": 1.5},
        {"learning_rate": 0},
        {"n_iter": 0},
        {"batch_size": 0},
        {"tol": 0},
        {"class_weight": "auto"},
        {"fit_intercept": 1},
    ],
)
def test_invalid_params_raise(dataset, params):
    X_tr, y_tr, _, _ = dataset
    with pytest.raises(ValueError):
        LogisticRegression(**params).fit(X_tr, y_tr)


def test_only_allowed_imports():
    source = (Path(__file__).resolve().parents[1] / "src" / "models" / "logistic_regression.py").read_text(
        encoding="utf-8"
    )
    modules = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module.split(".")[0])
    assert modules <= {"numpy", "math", "warnings", "numbers", "src"}


def test_adam_loss_decreases_and_matches_sklearn(dataset):
    X_tr, y_tr, X_te, y_te = dataset
    model = AdamLogisticRegression(
        penalty="l2", reg_lambda=0.01, learning_rate=0.1, n_iter=400, tol=1e-8
    ).fit(X_tr, y_tr)
    ref = _sklearn_model("l2", len(y_tr)).fit(X_tr, y_tr)
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert model.converged_
    own_acc = np.mean(model.predict(X_te) == y_te)
    ref_acc = np.mean(ref.predict(X_te) == y_te)
    assert abs(own_acc - ref_acc) < 0.03
