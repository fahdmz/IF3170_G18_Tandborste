import ast
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from src.models.svm import SVM, PegasosSVM

KERNELS = ["linear", "poly", "rbf"]


@pytest.fixture(scope="module")
def dataset():
    X, y = make_classification(n_samples=2000, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(y))
    train, test = order[:1500], order[1500:]
    return X[train], y[train], X[test], y[test]


@pytest.mark.parametrize("kernel", KERNELS)
def test_matches_sklearn(dataset, kernel):
    X_tr, y_tr, X_te, y_te = dataset
    own = SVM(kernel=kernel).fit(X_tr, y_tr)
    ref = SVC(kernel=kernel, coef0=1.0).fit(X_tr, y_tr)

    own_acc = np.mean(own.predict(X_te) == y_te)
    ref_acc = np.mean(ref.predict(X_te) == y_te)
    assert abs(own_acc - ref_acc) < 0.03
    assert abs(len(own.support_) - len(ref.support_)) <= 0.05 * len(ref.support_)
    assert np.corrcoef(own.decision_function(X_te), ref.decision_function(X_te))[0, 1] > 0.99


def test_kernel_aliases(dataset):
    X_tr, y_tr, X_te, _ = dataset
    rbf = SVM(kernel="rbf").fit(X_tr, y_tr)
    gaussian = SVM(kernel="gaussian").fit(X_tr, y_tr)
    np.testing.assert_allclose(rbf.decision_function(X_te), gaussian.decision_function(X_te))


def test_fitted_attributes(dataset):
    X_tr, y_tr, _, _ = dataset
    model = SVM(kernel="linear").fit(X_tr, y_tr)
    assert model.fit(X_tr, y_tr) is model
    np.testing.assert_array_equal(model.classes_, [0, 1])
    assert model.support_vectors_.shape == (len(model.support_), X_tr.shape[1])
    assert model.dual_coef_.shape == (len(model.support_),)
    assert model.n_support_.sum() == len(model.support_)
    assert model.coef_.shape == (X_tr.shape[1],)
    assert model.converged_
    assert not hasattr(SVM(kernel="rbf").fit(X_tr, y_tr), "coef_")


def test_dual_constraints(dataset):
    X_tr, y_tr, _, _ = dataset
    model = SVM(kernel="rbf", C=0.5).fit(X_tr, y_tr)
    alpha = np.abs(model.dual_coef_)
    assert np.all(alpha > 0) and np.all(alpha <= 0.5 + 1e-12)
    assert abs(model.dual_coef_.sum()) < 1e-8


def test_predict_matches_decision_sign(dataset):
    X_tr, y_tr, X_te, _ = dataset
    model = SVM(kernel="poly").fit(X_tr, y_tr)
    expected = (model.decision_function(X_te) > 0).astype(int)
    np.testing.assert_array_equal(model.predict(X_te), expected)


def test_batch_size_does_not_change_scores(dataset):
    X_tr, y_tr, X_te, _ = dataset
    small = SVM(kernel="rbf", batch_size=7).fit(X_tr, y_tr)
    large = SVM(kernel="rbf", batch_size=10_000).fit(X_tr, y_tr)
    np.testing.assert_allclose(small.decision_function(X_te), large.decision_function(X_te))


def test_string_labels(dataset):
    X_tr, y_tr, X_te, _ = dataset
    labels = np.array(["tidak", "batal"])[y_tr]
    model = SVM(kernel="rbf").fit(X_tr, labels)
    assert set(model.predict(X_te)) <= {"tidak", "batal"}


def test_class_weight_increases_minority_recall():
    X, y = make_classification(n_samples=1500, weights=[0.8, 0.2], random_state=42)
    plain = SVM(kernel="rbf").fit(X, y)
    weighted = SVM(kernel="rbf", class_weight="balanced").fit(X, y)
    heavy = SVM(kernel="rbf", class_weight={0: 1.0, 1: 5.0}).fit(X, y)
    recall = lambda m: np.mean(m.predict(X)[y == 1] == 1)
    assert recall(weighted) > recall(plain)
    assert recall(heavy) > recall(plain)


def test_class_weight_matches_sklearn(dataset):
    X_tr, y_tr, X_te, _ = dataset
    own = SVM(kernel="rbf", class_weight={0: 1.0, 1: 3.0}).fit(X_tr, y_tr)
    ref = SVC(kernel="rbf", class_weight={0: 1.0, 1: 3.0}).fit(X_tr, y_tr)
    assert np.mean(own.predict(X_te) == ref.predict(X_te)) > 0.98


def test_subsampling_is_stratified_and_reproducible(dataset):
    X_tr, y_tr, X_te, _ = dataset
    a = SVM(kernel="rbf", max_train_samples=500, random_state=7).fit(X_tr, y_tr)
    b = SVM(kernel="rbf", max_train_samples=500, random_state=7).fit(X_tr, y_tr)
    np.testing.assert_array_equal(a.support_, b.support_)
    np.testing.assert_allclose(a.decision_function(X_te), b.decision_function(X_te))

    X_sub, y_sub = a._subsample(X_tr, y_tr)
    assert len(y_sub) == 500
    assert abs(y_sub.mean() - y_tr.mean()) < 0.01


def test_subsample_size_is_exact_for_rare_class():
    X = np.zeros((10_000, 2))
    y = np.zeros(10_000, dtype=int)
    y[:3] = 1
    model = SVM(max_train_samples=10)
    model.classes_ = np.array([0, 1])
    X_sub, y_sub = model._subsample(X, y)
    assert len(y_sub) == 10
    assert set(y_sub) == {0, 1}


def test_refit_with_other_kernel_drops_stale_state(dataset):
    X_tr, y_tr, X_te, _ = dataset
    model = SVM(kernel="linear").fit(X_tr, y_tr)
    model.set_params(kernel="rbf").fit(X_tr, y_tr)
    fresh = SVM(kernel="rbf").fit(X_tr, y_tr)
    assert not hasattr(model, "coef_")
    np.testing.assert_allclose(model.decision_function(X_te), fresh.decision_function(X_te))


def test_set_params_after_fit_does_not_change_predictions(dataset):
    X_tr, y_tr, X_te, _ = dataset
    model = SVM(kernel="poly", degree=3).fit(X_tr, y_tr)
    before = model.decision_function(X_te)
    model.set_params(kernel="rbf", degree=2, coef0=5.0, gamma=10.0)
    np.testing.assert_array_equal(model.decision_function(X_te), before)


def test_intercept_finite_without_free_support_vectors():
    y = np.array([1.0, 1.0, -1.0, -1.0])
    C = np.ones(4)
    alpha = np.ones(4)
    grad = np.array([-0.5, -0.2, 0.3, 0.1])
    assert np.isfinite(SVM._intercept(alpha, grad, y, C))
    assert np.isfinite(SVM._intercept(np.array([1.0, 1.0, 0.0, 0.0]), grad, y, C))


def test_save_load_identical(dataset, tmp_path):
    X_tr, y_tr, X_te, _ = dataset
    model = SVM(kernel="rbf", max_train_samples=800).fit(X_tr, y_tr)
    path = tmp_path / "svm.pkl"
    model.save(path)
    loaded = SVM.load(path)
    np.testing.assert_array_equal(model.predict(X_te), loaded.predict(X_te))
    np.testing.assert_array_equal(model.decision_function(X_te), loaded.decision_function(X_te))


@pytest.mark.parametrize("method", ["predict", "decision_function"])
def test_predict_before_fit_raises(method):
    with pytest.raises(RuntimeError):
        getattr(SVM(), method)(np.zeros((2, 3)))


@pytest.mark.parametrize(
    "params",
    [
        {"kernel": "sigmoid"},
        {"C": 0},
        {"C": True},
        {"gamma": "x"},
        {"gamma": -1.0},
        {"gamma": True},
        {"degree": 0},
        {"degree": True},
        {"max_iter": 0},
        {"max_iter": True},
    ],
)
def test_invalid_params_raise(dataset, params):
    X_tr, y_tr, _, _ = dataset
    with pytest.raises(ValueError):
        SVM(**params).fit(X_tr, y_tr)


def test_rejects_nan_and_single_class():
    X = np.array([[0.0, 1.0], [np.nan, 2.0]])
    with pytest.raises(ValueError):
        SVM().fit(X, np.array([0, 1]))
    with pytest.raises(ValueError):
        SVM().fit(np.zeros((3, 2)), np.zeros(3))


def test_max_iter_warns_when_not_converged(dataset):
    X_tr, y_tr, _, _ = dataset
    with pytest.warns(RuntimeWarning):
        model = SVM(kernel="linear", max_iter=1, tol=1e-9).fit(X_tr[:300], y_tr[:300])
    assert not model.converged_


def test_only_allowed_imports():
    source = (Path(__file__).resolve().parents[1] / "src" / "models" / "svm.py").read_text(encoding="utf-8")
    modules = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module.split(".")[0])
    assert modules <= {"numpy", "math", "warnings", "collections", "numbers", "src"}


def test_pegasos_matches_sklearn_linear(dataset):
    X_tr, y_tr, X_te, y_te = dataset
    C = 1.0
    own = PegasosSVM(lam=1.0 / (C * len(y_tr)), n_epochs=30).fit(X_tr, y_tr)
    ref = SVC(kernel="linear", C=C).fit(X_tr, y_tr)
    assert abs(np.mean(own.predict(X_te) == y_te) - np.mean(ref.predict(X_te) == y_te)) < 0.03
    assert np.mean(own.predict(X_te) == ref.predict(X_te)) > 0.95


def test_pegasos_objective_decreases(dataset):
    X_tr, y_tr, _, _ = dataset
    model = PegasosSVM(n_epochs=20).fit(X_tr, y_tr)
    assert model.objective_history_.shape == (20,)
    assert model.objective_history_[-1] < model.objective_history_[0]
    assert model.n_iter_ == 20 * int(np.ceil(len(y_tr) / model.batch_size))


def test_pegasos_reproducible_and_save_load(dataset, tmp_path):
    X_tr, y_tr, X_te, _ = dataset
    a = PegasosSVM(random_state=3).fit(X_tr, y_tr)
    b = PegasosSVM(random_state=3).fit(X_tr, y_tr)
    np.testing.assert_array_equal(a.coef_, b.coef_)
    a.save(tmp_path / "pegasos.pkl")
    loaded = PegasosSVM.load(tmp_path / "pegasos.pkl")
    np.testing.assert_array_equal(a.decision_function(X_te), loaded.decision_function(X_te))


def test_pegasos_class_weight_increases_minority_recall():
    X, y = make_classification(n_samples=1500, weights=[0.8, 0.2], random_state=42)
    plain = PegasosSVM().fit(X, y)
    weighted = PegasosSVM(class_weight="balanced").fit(X, y)
    recall = lambda m: np.mean(m.predict(X)[y == 1] == 1)
    assert recall(weighted) > recall(plain)


def test_pegasos_interface():
    model = PegasosSVM()
    assert model.clone().get_params() == model.get_params()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((2, 3)))
    for params in ({"lam": 0}, {"n_epochs": 0}, {"batch_size": True}, {"class_weight": "x"}):
        with pytest.raises(ValueError):
            PegasosSVM(**params).fit(np.eye(4), np.array([0, 1, 0, 1]))
