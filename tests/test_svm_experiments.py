import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, roc_auc_score

from src.experiments.svm_experiments import (
    best_threshold,
    compare_with_sklearn,
    cv_scores,
    macro_f1,
    roc_auc,
    stratified_folds,
)
from src.evaluation.metrics import macro_f1 as team_macro_f1
from src.evaluation.model_selection import StratifiedKFold, cross_validate
from src.models.svm import SVM, PegasosSVM


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    score = np.round(rng.normal(size=500) + y, 1)  # dibulatkan agar ada ties
    pred = (score > 0.5).astype(int)
    assert np.isclose(macro_f1(y, pred), f1_score(y, pred, average="macro"))
    assert np.isclose(roc_auc(y, score), roc_auc_score(y, score))


def test_stratified_folds_partition_and_balance():
    y = np.array([0] * 70 + [1] * 30)
    folds = stratified_folds(y, n_splits=5, seed=1)
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(100))
    assert all(len(f) == 20 and y[f].sum() == 6 for f in folds)
    assert all(np.array_equal(a, b) for a, b in zip(folds, stratified_folds(y, n_splits=5, seed=1)))


def test_best_threshold_improves_macro_f1():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(int)
    score = rng.normal(size=400) + 1.5 * y - 1.0
    t = best_threshold(y, score)
    assert macro_f1(y, (score > t).astype(int)) >= macro_f1(y, (score > 0).astype(int))


def _toy_folds():
    X, y = make_classification(n_samples=600, random_state=42)
    folds = stratified_folds(y, n_splits=3)
    out = []
    for va in folds:
        tr = np.setdiff1d(np.arange(len(y)), va)
        out.append((X[tr], y[tr], X[va], y[va]))
    return out


def test_cv_scores_keys_and_ranges():
    result = cv_scores(SVM(kernel="rbf"), _toy_folds())
    assert set(result) >= {"macro_f1_thr0", "macro_f1_tuned_thr", "roc_auc", "mean_threshold", "fit_sec"}
    assert 0.5 < result["macro_f1_thr0"] <= 1.0 and 0.5 < result["roc_auc"] <= 1.0


def test_compare_with_sklearn_agrees():
    models = {"rbf": SVM(kernel="rbf"), "pegasos": PegasosSVM(lam=1e-3)}
    per_fold, summary = compare_with_sklearn(models, _toy_folds())
    assert len(per_fold) == 2 * 2 * 3
    assert per_fold.loc[per_fold.config == "rbf", "agreement"].min() == 1.0
    assert per_fold.loc[per_fold.config == "pegasos", "agreement"].min() > 0.9


def test_svm_works_with_team_cross_validate():
    X, y = make_classification(n_samples=400, random_state=42)
    scores = cross_validate(
        SVM(kernel="rbf"), None, pd.DataFrame(X), pd.Series(y), StratifiedKFold(3), {"macro_f1": team_macro_f1}
    )
    assert len(scores["macro_f1"]) == 3 and min(scores["macro_f1"]) > 0.7
