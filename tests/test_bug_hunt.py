"""Bug-hunt checks across the project.

Assertions describe the behavior the shared interface asks for. A failure is
evidence; these tests do not weaken the expected result to match the code.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from src.data import CATEGORICAL_COLS, NUMERICAL_COLS
from src.evaluation.cost_sensitive import tune_threshold
from src.evaluation.metrics import (
    accuracy,
    confusion_matrix,
    f1_score,
    precision,
    recall,
    roc_auc,
)
from src.evaluation.model_selection import StratifiedKFold, cross_validate, grid_search, train_test_split
from src.models.decision_tree import DecisionTree
from src.models.knn import KNN
from src.models.naive_bayes import NaiveBayes
from src.models.svm import SVM
from src.preprocessing.cleaning import clean_train
from src.preprocessing.presets import make_numeric_pipeline, make_tree_pipeline, to_model_input
from src.preprocessing.resampling import RandomOverSampler, RandomUnderSampler, SMOTE
from src.submission import make_submission


def _stay_matrix(n=36, seed=4):
    rng = np.random.default_rng(seed)
    nights = rng.integers(0, 6, size=n).astype(float)
    rate = rng.normal(1.2, 0.35, size=n)
    lead = rng.uniform(1, 30, size=n)
    lead[2] = 180.0
    rate[5] = 0.0
    nights[7] = nights[1]
    signal = 0.9 * rate + 0.04 * lead - 0.2 * nights
    y = (signal + rng.normal(0, 0.4, size=n) > 0.8).astype(int)
    y[0], y[1] = 0, 1
    X = np.column_stack([nights, rate, lead, nights + rng.normal(0, 0.02, size=n)])
    center, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (X - center) / scale, y


def test_knn_fit_snapshots_training_rows():
    X, y = _stay_matrix()
    X = np.ascontiguousarray(X)
    y = np.array(y)
    query = X[:6].copy()
    model = KNN(n_neighbors=3, weights="distance").fit(X, y)
    before = model.predict_proba(query)
    X += 3.5
    y[:] = 1 - y
    after = model.predict_proba(query)
    np.testing.assert_array_equal(before, after)
    assert not np.shares_memory(model.X_train_, X)
    assert not np.shares_memory(model.y_train_, y)


def test_knn_rejects_hidden_nonfinite_values():
    X, y = _stay_matrix()
    broken = X.copy()
    broken[4, 1] = np.nan
    broken[9, 2] = np.inf
    with pytest.raises(ValueError):
        KNN(n_neighbors=3).fit(broken, y)


def test_knn_rejects_boolean_neighbor_count():
    X, y = _stay_matrix()
    with pytest.raises(ValueError):
        KNN(n_neighbors=True, batch_size=True).fit(X, y)


def test_knn_distance_weights_match_sklearn_with_an_exact_neighbor():
    X, y = _stay_matrix(n=40, seed=9)
    query = np.vstack([X[:8], X[3:4]])
    own = KNN(n_neighbors=5, weights="distance", metric="manhattan").fit(X, y)
    ref = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="manhattan", p=1).fit(X, y)
    np.testing.assert_allclose(own.predict_proba(query), ref.predict_proba(query), atol=1e-6)
    assert own.predict(query).shape == (len(query),)


def test_naive_bayes_accepts_float_zero_one_labels():
    X, y = _stay_matrix()
    model = NaiveBayes(var_smoothing=1e-9).fit(X, y.astype(float))
    np.testing.assert_array_equal(model.classes_, [0, 1])
    proba = model.predict_proba(X[:5])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.isfinite(proba).all()


def test_naive_bayes_skips_missing_numeric_and_unknown_category():
    rng = np.random.default_rng(3)
    numeric = rng.normal(size=(30, 1))
    category = rng.choice(["Breakfast", "No Meal"], size=30)
    y = np.array([0, 1] * 15)
    X = np.column_stack([numeric, category])
    full = NaiveBayes(categorical_features=[1], alpha=1.0).fit(X, y)
    only_category = NaiveBayes(categorical_features=[0], alpha=1.0).fit(category.reshape(-1, 1), y)

    missing_numeric = np.array([[np.nan, "Breakfast"]], dtype=object)
    known = np.array([[0.0, "Breakfast"]], dtype=object)
    unknown = np.array([[0.2, "Ultra Plan"]], dtype=object)
    blank = np.array([[0.2, None]], dtype=object)
    assert not np.allclose(full.predict_proba(missing_numeric), full.predict_proba(known))
    np.testing.assert_allclose(
        full.predict_proba(missing_numeric),
        only_category.predict_proba(np.array([["Breakfast"]], dtype=object)),
    )
    np.testing.assert_allclose(full.predict_proba(unknown), full.predict_proba(blank))


def test_naive_bayes_does_not_learn_pandas_na_as_a_category():
    category = np.array(["Breakfast", "No Meal", pd.NA, "Breakfast", "No Meal", pd.NA], dtype=object)
    y = np.array([0, 0, 0, 1, 1, 1])
    X = np.column_stack([np.linspace(-1, 1, 6), category])
    model = NaiveBayes(categorical_features=[1], alpha=0.5).fit(X, y)
    learned = list(model.categories_[0])
    assert not any(value is pd.NA for value in learned)
    assert set(map(str, learned)) == {"Breakfast", "No Meal"}


def test_naive_bayes_rejects_boolean_smoothing():
    X, y = _stay_matrix(n=20)
    with pytest.raises(ValueError):
        NaiveBayes(alpha=True, var_smoothing=False).fit(X, y)


def test_decision_tree_accepts_column_shaped_labels():
    X, y = _stay_matrix(n=24)
    tree = DecisionTree(max_depth=3, min_samples_leaf=2)
    tree.fit(X, y.reshape(-1, 1))
    pred = tree.predict(X)
    assert pred.shape == (len(y),)
    assert set(pred).issubset({0, 1})


def test_metrics_match_sklearn_when_scores_tie():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    scores = np.array([0.2, 0.2, 0.2, 0.8, 0.5, 0.5, 0.9, 0.1])
    assert roc_auc(y, scores) == pytest.approx(roc_auc_score(y, scores))
    pred = (scores >= 0.5).astype(int)
    matrix = confusion_matrix(y, pred)
    assert matrix.sum() == len(y)
    assert accuracy(y, pred) == pytest.approx(np.mean(y == pred))
    assert 0.0 <= precision(y, pred) <= 1.0
    assert f1_score(y, pred) == pytest.approx(
        0.0 if precision(y, pred) + recall(y, pred) == 0
        else 2 * precision(y, pred) * recall(y, pred) / (precision(y, pred) + recall(y, pred))
    )


def test_threshold_sweep_matches_explicit_costs():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    scores = np.array([0.15, 0.42, 0.42, 0.61, 0.2, 0.9, 0.73, 0.55])
    chosen = tune_threshold(y, scores, fp_cost=2.0, fn_cost=1.0)
    candidates = np.unique(np.append(scores, np.inf))

    def cost(threshold):
        pred = scores >= threshold
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        return 2 * fp + fn

    best = min(candidates, key=lambda threshold: (cost(threshold), threshold if np.isfinite(threshold) else 1e9))
    if np.isfinite(chosen):
        assert cost(chosen) == cost(best)
    else:
        assert cost(np.inf) == cost(best)


def test_stratified_split_and_folds_cover_every_row_once():
    rng = np.random.default_rng(2)
    n = 37
    X = pd.DataFrame({"rate": rng.normal(size=n), "nights": rng.integers(0, 5, size=n)})
    y = pd.Series(np.array([0] * 25 + [1] * 12))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=True, random_state=8)
    assert len(X_train) + len(X_test) == n
    assert set(y_train) == {0, 1} and set(y_test) == {0, 1}
    assert abs(len(X_test) / n - 0.3) < 0.05

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=8)
    seen = []
    for train_idx, val_idx in cv.split(X, y):
        assert len(np.intersect1d(train_idx, val_idx)) == 0
        assert set(y.iloc[val_idx]) == {0, 1}
        seen.append(val_idx)
    covered = np.sort(np.concatenate(seen))
    np.testing.assert_array_equal(covered, np.arange(n))


def test_grid_search_does_not_mutate_the_template_model():
    X, y = _stay_matrix(n=30)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    labels = pd.Series(y)
    template = KNN(n_neighbors=3, weights="uniform")
    before = template.get_params()
    best, rows = grid_search(
        template,
        {"n_neighbors": [1, 3], "weights": ["uniform", "distance"]},
        None,
        frame,
        labels,
        StratifiedKFold(2, True, 1),
        scoring=accuracy,
    )
    assert template.get_params() == before
    assert best["n_neighbors"] in (1, 3)
    assert len(rows) == 4
    scores = cross_validate(
        KNN(**best), None, frame, labels, StratifiedKFold(2, True, 1), {"accuracy": accuracy}
    )
    assert len(scores["accuracy"]) == 2
    assert all(np.isfinite(scores["accuracy"]))


def test_resamplers_on_imbalanced_stays_with_a_fractional_ratio():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(40, 3))
    y = np.array([0] * 30 + [1] * 10)
    X[3, 0] = 0.0
    under_x, under_y = RandomUnderSampler(sampling_strategy=0.5, random_state=6).fit_resample(X, y)
    n_min = np.sum(under_y == 1)
    n_maj = np.sum(under_y == 0)
    assert n_min == 10
    assert n_maj == int(round(10 / 0.5))
    over_x, over_y = RandomOverSampler(sampling_strategy=0.5, random_state=6).fit_resample(X, y)
    assert np.sum(over_y == 1) == int(round(0.5 * 30))
    assert len(over_x) == len(over_y)
    smote_x, smote_y = SMOTE(sampling_strategy=0.5, k_neighbors=3, random_state=6).fit_resample(X, y)
    assert np.sum(smote_y == 1) == int(round(0.5 * 30))
    assert np.isfinite(smote_x).all()

    dirty = X.copy()
    dirty[8, 1] = np.nan
    with pytest.raises(ValueError):
        SMOTE(sampling_strategy=0.5, k_neighbors=2).fit_resample(dirty, y)


def test_cleaning_and_submission_keep_awkward_hotel_rows(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    rows = 28
    frame = pd.DataFrame(
        {
            "num_adults": rng.integers(1, 4, rows),
            "num_children": rng.integers(0, 2, rows).astype(float),
            "weekend_nights": rng.integers(0, 3, rows),
            "weekday_nights": rng.integers(1, 4, rows),
            "needs_parking": rng.integers(0, 2, rows),
            "days_before_arrival": rng.integers(2, 40, rows).astype(float),
            "arrival_year": 2024,
            "arrival_month": rng.integers(1, 13, rows),
            "arrival_day": rng.integers(1, 28, rows),
            "is_returning_guest": rng.integers(0, 2, rows),
            "prior_cancellations": rng.integers(0, 2, rows),
            "prior_completed_bookings": rng.integers(0, 3, rows),
            "nightly_rate_idr": rng.integers(400_000, 1_800_000, rows).astype(float),
            "special_requests": rng.integers(0, 3, rows).astype(float),
            "meal_package": rng.choice(["Breakfast", "No Meal"], rows),
            "room_category": rng.choice(["Type A", "Type B"], rows),
            "booking_channel": rng.choice(["Corporate", "Offline Agent"], rows),
        }
    )
    y = pd.Series(np.array([0, 1] * (rows // 2)), name="is_canceled")
    frame.loc[2, "num_children"] = np.nan
    frame.loc[4, "nightly_rate_idr"] = 0.0
    frame.loc[6] = frame.loc[0]
    y.loc[6] = y.loc[0]
    frame.loc[7, ["num_adults", "num_children"]] = 0.0
    different_label = frame.loc[1].copy()
    frame.loc[8] = different_label
    y.loc[8] = 1 - y.loc[1]

    cleaned_x, cleaned_y = clean_train(frame, y)
    assert len(cleaned_x) == rows - 2
    assert ((cleaned_x["num_adults"] == 0) & (cleaned_x["num_children"] == 0)).sum() == 0

    pipeline = make_numeric_pipeline(NUMERICAL_COLS, CATEGORICAL_COLS, engineer_features=False)
    transformed = pipeline.fit_transform(cleaned_x, cleaned_y)
    features, _ = to_model_input(transformed, CATEGORICAL_COLS)
    model = KNN(n_neighbors=3).fit(features, cleaned_y.to_numpy())

    test = frame.iloc[:5].copy()
    test.loc[test.index[0], "meal_package"] = "Ultra Plan"
    test.loc[test.index[1], "nightly_rate_idr"] = np.nan
    ids = pd.Series([f"B{i}" for i in range(len(test))])
    sample = pd.DataFrame({"id": ids, "is_canceled": 0})
    sample_path = tmp_path / "sample_submission.csv"
    sample.to_csv(sample_path, index=False)
    monkeypatch.setattr("src.submission.config.SAMPLE_SUBMISSION_PATH", sample_path)
    monkeypatch.setattr("src.config.SAMPLE_SUBMISSION_PATH", sample_path)

    path = tmp_path / "submission.csv"
    submission = make_submission(model, pipeline, test, ids, path)
    assert len(submission) == len(test)
    assert set(submission["is_canceled"]).issubset({0, 1})


def test_svm_and_tree_on_the_same_scaled_stays():
    X, y = _stay_matrix(n=50, seed=15)
    svm = SVM(kernel="linear", C=1.0, max_iter=30).fit(X, y)
    tree = DecisionTree(max_depth=4, min_samples_leaf=3).fit(X, y)
    tree_input = X.copy()
    tree_input[1, 0] = np.nan
    assert svm.predict(X).shape == (len(y),)
    assert tree.predict(tree_input).shape == (len(y),)
    proba = tree.predict_proba(tree_input)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(svm.decision_function(X)).all()
