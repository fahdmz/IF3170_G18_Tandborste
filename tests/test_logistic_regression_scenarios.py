"""Scenario tests for Anggota 4."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from src.data import CATEGORICAL_COLS, NUMERICAL_COLS
from src.evaluation.cost_sensitive import cost_sensitive_class_weight
from src.evaluation.model_selection import StratifiedKFold, cross_validate
from src.evaluation.metrics import macro_f1
from src.experiments.logistic_regression import (
    _params_from_row,
    _sklearn_model,
    adam_grid,
    compare_with_sklearn,
    logreg_grid,
    prepare_folds,
    tune,
)
from src.models.logistic_regression import AdamLogisticRegression, LogisticRegression
from src.preprocessing.cleaning import clean_train
from src.preprocessing.presets import make_numeric_pipeline, to_model_input
from src.visualization.logreg_contour import plot_logreg_contour

_PARAM_FIELDS = [
    "penalty",
    "reg_lambda",
    "l1_ratio",
    "learning_rate",
    "n_iter",
    "batch_size",
    "tol",
    "class_weight",
    "random_state",
]


def _guest_book(n=48, seed=7):
    """Numeric stays. Rates are in rupiah, lead time has one far outlier, one rate is 0."""
    rng = np.random.default_rng(seed)
    nights = rng.integers(0, 8, size=n).astype(float)
    nights[3] = 0.0
    guests = rng.integers(1, 5, size=n).astype(float)
    rate = rng.uniform(0.45, 2.4, size=n) * 1_000_000.0
    rate[5] = 0.0
    lead = rng.uniform(2.0, 45.0, size=n)
    lead[8] = 365.0
    parking = rng.integers(0, 2, size=n).astype(float)
    requests = np.zeros(n)
    requests[rng.choice(n, size=5, replace=False)] = rng.integers(1, 4, size=5)
    redundant = nights + rng.normal(0.0, 0.08, size=n)
    logit = (
        -1.25
        + 0.028 * (lead - 18.0)
        + 0.55 * (rate / 1_000_000.0 - 1.1)
        + 0.12 * nights
        - 0.35 * requests
    )
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    y[[0, 4, 12]] = 0
    y[[1, 6, 9, 15]] = 1
    features = np.column_stack([nights, guests, rate, lead, parking, requests, redundant])
    return features, y


def _standardize(train, test):
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return (train - center) / scale, (test - center) / scale


def _split(features, labels, test_size=12, seed=11):
    """Keep the free stay, the zero-night stay, and the 365-day lead in the training rows."""
    rng = np.random.default_rng(seed)
    pinned = np.array([3, 5, 8])
    rest = np.setdiff1d(np.arange(len(labels)), pinned)
    test = rng.choice(rest, size=test_size, replace=False)
    train = np.setdiff1d(np.arange(len(labels)), test)
    return features[train], labels[train], features[test], labels[test]


def _fit_kwargs(**overrides):
    params = dict(
        penalty="elasticnet",
        reg_lambda=0.037,
        l1_ratio=0.35,
        learning_rate=0.45,
        n_iter=700,
        batch_size=None,
        tol=1e-7,
        class_weight={0: 1.7, 1: 0.85},
        random_state=11,
    )
    params.update(overrides)
    return params


def _sklearn_from(params, n_train, **extra):
    penalty = params["penalty"]
    common = dict(
        solver="saga",
        max_iter=8000,
        tol=1e-8,
        random_state=0,
        class_weight=params.get("class_weight"),
        fit_intercept=params.get("fit_intercept", True),
    )
    common.update(extra)
    if penalty in (None, "none") or params["reg_lambda"] == 0:
        return SkLogisticRegression(C=np.inf, **common)
    ratio = {"l2": 0.0, "l1": 1.0, "elasticnet": params["l1_ratio"]}[penalty]
    return SkLogisticRegression(
        C=1.0 / (params["reg_lambda"] * n_train), l1_ratio=ratio, **common
    )


@pytest.fixture(scope="module")
def guest_split():
    features, labels = _guest_book()
    raw_train, y_tr, raw_test, y_te = _split(features, labels)
    x_tr, x_te = _standardize(raw_train, raw_test)
    return x_tr, y_tr, x_te, y_te, raw_train, raw_test


def test_elasticnet_on_messy_stays_matches_sklearn_and_keeps_every_row(guest_split):
    x_tr, y_tr, x_te, y_te, raw_train, raw_test = guest_split
    assert raw_train.shape[1] == 7
    assert np.any(raw_train[:, 2] == 0.0)
    assert np.any(raw_train[:, 3] > 300.0)
    assert abs(np.corrcoef(raw_train[:, 0], raw_train[:, 6])[0, 1]) > 0.95
    assert min(np.bincount(y_tr)) >= 4

    params = _fit_kwargs()
    model = LogisticRegression(**params).fit(x_tr, y_tr.reshape(-1, 1))
    ref = _sklearn_from(params, len(y_tr)).fit(x_tr, y_tr)

    own_scores = model.decision_function(x_te)
    ref_scores = ref.decision_function(x_te)
    assert np.corrcoef(own_scores, ref_scores)[0, 1] > 0.98
    assert abs(np.mean(model.predict(x_te) == y_te) - np.mean(ref.predict(x_te) == y_te)) < 0.08
    assert model.converged_
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_)

    proba = model.predict_proba(x_te)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_allclose(proba[:, 1], 1.0 / (1.0 + np.exp(-np.clip(own_scores, -500, 500))))
    assert set(np.unique(model.predict(x_te))).issubset({0, 1})
    assert len(model.predict(x_te)) == len(raw_test)


def test_minibatch_seed_dataframe_and_incomplete_last_batch(guest_split):
    x_tr, y_tr, x_te, y_te, _, _ = guest_split
    frame = pd.DataFrame(x_tr, index=np.arange(30, 30 + len(x_tr)))
    labels = pd.Series(y_tr, index=frame.index, name="is_canceled")
    assert len(y_tr) % 10 != 0

    params = _fit_kwargs(batch_size=10, n_iter=2, tol=1e-15, record_history=True, learning_rate=0.35)
    with pytest.warns(RuntimeWarning):
        first = LogisticRegression(**params).fit(frame, labels)
        second = LogisticRegression(**params).fit(frame.to_numpy().tolist(), labels.to_numpy())
        other_seed = LogisticRegression(**{**params, "random_state": 99}).fit(frame, labels)
    np.testing.assert_allclose(first.decision_function(x_te), second.decision_function(x_te))
    assert not np.allclose(first.coef_, other_seed.coef_)
    batches = int(np.ceil(len(y_tr) / 10))
    assert first.coef_history_.shape == (1 + first.n_iter_ * batches, x_tr.shape[1])
    assert first.loss_history_.shape == (first.n_iter_ + 1,)
    learned = LogisticRegression(
        penalty="l2", reg_lambda=0.02, learning_rate=0.5, n_iter=400, batch_size=10, tol=1e-6, random_state=11
    ).fit(x_tr, y_tr)
    assert learned.converged_
    assert learned.loss_history_[-1] < learned.loss_history_[0]
    assert np.mean(learned.predict(x_tr) == y_tr) > 0.75

    full = LogisticRegression(**_fit_kwargs(batch_size=None, random_state=3)).fit(x_tr, y_tr)
    oversized = LogisticRegression(**_fit_kwargs(batch_size=len(y_tr) + 17, random_state=8)).fit(x_tr, y_tr)
    np.testing.assert_allclose(full.decision_function(x_te), oversized.decision_function(x_te))
    same = LogisticRegression(**_fit_kwargs(batch_size=None, random_state=8)).fit(x_tr, y_tr)
    np.testing.assert_allclose(full.coef_, same.coef_)


def test_penalty_boundaries_and_intercept_switch(guest_split):
    x_tr, y_tr, x_te, _, _, _ = guest_split
    shared = dict(learning_rate=0.5, n_iter=1200, tol=1e-5, random_state=4)
    none = LogisticRegression(penalty=None, reg_lambda=0.02, **shared).fit(x_tr, y_tr)
    zero_lambda = LogisticRegression(penalty="l2", reg_lambda=0.0, **shared).fit(x_tr, y_tr)
    np.testing.assert_allclose(
        none.decision_function(x_te), zero_lambda.decision_function(x_te), atol=1e-5
    )

    l2 = LogisticRegression(penalty="l2", reg_lambda=0.041, l1_ratio=0.2, **shared).fit(x_tr, y_tr)
    as_l2 = LogisticRegression(penalty="elasticnet", reg_lambda=0.041, l1_ratio=0.0, **shared).fit(x_tr, y_tr)
    as_l1 = LogisticRegression(penalty="elasticnet", reg_lambda=0.041, l1_ratio=1.0, **shared).fit(x_tr, y_tr)
    pure_l1 = LogisticRegression(penalty="l1", reg_lambda=0.041, l1_ratio=0.2, **shared).fit(x_tr, y_tr)
    np.testing.assert_allclose(l2.decision_function(x_te), as_l2.decision_function(x_te), atol=1e-5)
    np.testing.assert_allclose(pure_l1.decision_function(x_te), as_l1.decision_function(x_te), atol=1e-5)
    assert abs(l2.intercept_) > 1e-4

    no_intercept = LogisticRegression(
        penalty="l2", reg_lambda=0.041, fit_intercept=False, **shared
    ).fit(x_tr, y_tr)
    assert no_intercept.intercept_ == 0.0
    ref = _sklearn_from(
        {"penalty": "l2", "reg_lambda": 0.041, "l1_ratio": 0.0, "class_weight": None, "fit_intercept": False},
        len(y_tr),
    ).fit(x_tr, y_tr)
    assert np.corrcoef(no_intercept.decision_function(x_te), ref.decision_function(x_te))[0, 1] > 0.98
    assert abs(float(ref.intercept_[0])) < 1e-8


def test_string_labels_missing_weight_and_cost_sensitive(guest_split):
    x_tr, y_tr, x_te, y_te, _, _ = guest_split
    named = np.where(y_tr == 1, "batal", "tinggal")
    named_te = np.where(y_te == 1, "batal", "tinggal")
    partial = LogisticRegression(
        penalty="l2", reg_lambda=0.05, learning_rate=0.5, n_iter=500, class_weight={"batal": 3.5}
    ).fit(x_tr, named)
    explicit = LogisticRegression(
        penalty="l2",
        reg_lambda=0.05,
        learning_rate=0.5,
        n_iter=500,
        class_weight={"batal": 3.5, "tinggal": 1.0},
    ).fit(x_tr, named)
    np.testing.assert_allclose(partial.decision_function(x_te), explicit.decision_function(x_te))
    assert set(partial.classes_) == {"tinggal", "batal"}
    assert set(partial.predict(x_te)).issubset({"tinggal", "batal"})
    proba = partial.predict_proba(x_te)
    batal_column = int(np.where(partial.classes_ == "batal")[0][0])
    assert proba.shape == (len(y_te), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert proba[named_te == "batal", batal_column].mean() > proba[named_te == "tinggal", batal_column].mean()
    assert np.mean(partial.predict(x_te) == named_te) > 0.5

    plain = LogisticRegression(penalty="l2", reg_lambda=0.08, learning_rate=0.5, n_iter=500).fit(x_tr, y_tr)
    costly = LogisticRegression(
        penalty="l2",
        reg_lambda=0.08,
        learning_rate=0.5,
        n_iter=500,
        class_weight=cost_sensitive_class_weight(),
    ).fit(x_tr, y_tr)
    plain_positive = np.mean(plain.predict(x_tr) == 1)
    costly_positive = np.mean(costly.predict(x_tr) == 1)
    assert costly_positive < plain_positive
    assert costly.class_weight == {0: 2.0, 1: 1.0}


def test_refit_failure_keeps_state_and_params_wait_for_refit(guest_split, tmp_path):
    x_tr, y_tr, x_te, _, _, _ = guest_split
    model = LogisticRegression(**_fit_kwargs(record_history=True)).fit(x_tr, y_tr)
    original = model.decision_function(x_te).copy()
    original_coef = model.coef_.copy()

    poisoned = x_tr.copy()
    poisoned[4, 2] = np.inf
    poisoned[9, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        model.fit(poisoned, y_tr)
    np.testing.assert_allclose(model.decision_function(x_te), original)
    np.testing.assert_allclose(model.coef_, original_coef)
    assert model.coef_history_ is not None

    model.set_params(penalty="l1", reg_lambda=4.5, learning_rate=0.05, record_history=False)
    np.testing.assert_allclose(model.decision_function(x_te), original)
    unfitted = model.clone()
    with pytest.raises(RuntimeError):
        unfitted.predict_proba(x_te)
    assert unfitted.get_params()["penalty"] == "l1"
    assert unfitted.get_params()["reg_lambda"] == 4.5

    refit = model.fit(x_tr, y_tr)
    assert refit is model
    assert model.coef_history_ is None
    assert model.penalty == "l1"
    assert not np.allclose(model.coef_, original_coef)

    path = tmp_path / "nested" / "lr.pkl"
    model.save(path)
    loaded = LogisticRegression.load(path)
    np.testing.assert_array_equal(loaded.predict(x_te), model.predict(x_te))
    with pytest.raises(TypeError):
        AdamLogisticRegression.load(path)

    bool_labels = y_tr.astype(bool)
    boolean_model = LogisticRegression(learning_rate=0.5, n_iter=200, reg_lambda=0.03).fit(x_tr, bool_labels)
    assert set(boolean_model.classes_.tolist()) == {False, True}
    assert set(np.unique(boolean_model.predict(x_te))).issubset({False, True})


def test_corruptions_of_the_same_table_are_rejected(guest_split):
    x_tr, y_tr, x_te, _, _, _ = guest_split
    with pytest.raises(ValueError):
        LogisticRegression().fit(x_tr, np.append(y_tr, 0))
    with pytest.raises(ValueError, match="empty"):
        LogisticRegression().fit(np.zeros((0, x_tr.shape[1])), np.array([]))
    with pytest.raises(ValueError, match="2D"):
        LogisticRegression().fit(x_tr[:, 0], y_tr)
    three = y_tr.copy()
    three[2] = 2
    with pytest.raises(ValueError, match="exactly two"):
        LogisticRegression().fit(x_tr, three)
    missing_label = y_tr.astype(float)
    missing_label[6] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        LogisticRegression().fit(x_tr, missing_label)

    fitted = LogisticRegression(**_fit_kwargs()).fit(x_tr, y_tr)
    bad_score = x_te.copy()
    bad_score[1, 4] = np.inf
    with pytest.raises(ValueError, match="NaN or inf"):
        fitted.decision_function(bad_score)
    with pytest.raises(ValueError, match="2D"):
        fitted.predict(x_te[:, 0])
    with pytest.raises(ValueError, match="features"):
        fitted.predict(np.column_stack([x_te, x_te[:, 0]]))

    with pytest.warns(RuntimeWarning, match="did not converge"):
        exploded = LogisticRegression(learning_rate=1e308, n_iter=3, tol=1e-12).fit(x_tr, y_tr)
    assert not exploded.converged_
    assert exploded.n_iter_ >= 1
    assert hasattr(exploded, "classes_")


@pytest.mark.parametrize(
    "params",
    [
        {"n_iter": True, "batch_size": 8, "learning_rate": 0.2},
        {"batch_size": True, "n_iter": 10},
        {"learning_rate": np.inf, "n_iter": 10},
        {"reg_lambda": np.nan, "n_iter": 10},
        {"l1_ratio": -0.15, "penalty": "elasticnet", "n_iter": 10},
        {"penalty": "L2", "n_iter": 10},
        {"record_history": "yes", "n_iter": 10},
        {"class_weight": {0: 1.5, 1: -0.2}, "n_iter": 10},
        {"tol": np.inf, "n_iter": 10},
    ],
)
def test_nontrivial_invalid_settings_raise(guest_split, params):
    x_tr, y_tr, _, _, _, _ = guest_split
    with pytest.raises(ValueError):
        LogisticRegression(**params).fit(x_tr, y_tr)


def test_adam_on_the_same_stays(guest_split, tmp_path):
    x_tr, y_tr, x_te, y_te, _, _ = guest_split
    params = _fit_kwargs(
        penalty="elasticnet",
        l1_ratio=0.3,
        reg_lambda=0.028,
        learning_rate=0.12,
        n_iter=600,
        batch_size=None,
        class_weight="balanced",
        record_history=True,
        random_state=5,
        tol=1e-6,
    )
    model = AdamLogisticRegression(beta1=0.85, beta2=0.99, epsilon=1e-7, **params).fit(x_tr, y_tr)
    ref = _sklearn_from(params, len(y_tr)).fit(x_tr, y_tr)
    assert model.converged_
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert np.isfinite(model.coef_).all()
    assert abs(np.mean(model.predict(x_te) == y_te) - np.mean(ref.predict(x_te) == y_te)) < 0.12
    assert np.corrcoef(model.decision_function(x_te), ref.decision_function(x_te))[0, 1] > 0.9

    mini_params = {**params, "batch_size": 7, "n_iter": 40, "tol": 1e-8}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mini = AdamLogisticRegression(beta1=0.85, beta2=0.99, epsilon=1e-7, **mini_params).fit(x_tr, y_tr)
        mini_again = AdamLogisticRegression(beta1=0.85, beta2=0.99, epsilon=1e-7, **mini_params).fit(x_tr, y_tr)
    np.testing.assert_allclose(mini.decision_function(x_te), mini_again.decision_function(x_te))
    assert np.isfinite(mini.coef_).all()
    assert mini.coef_history_.shape[0] > mini.loss_history_.shape[0]
    assert len(y_tr) % 7 != 0

    path = tmp_path / "adam.pkl"
    model.save(path)
    loaded = AdamLogisticRegression.load(path)
    np.testing.assert_allclose(loaded.decision_function(x_te), model.decision_function(x_te))
    assert loaded.get_params()["beta1"] == 0.85

    with pytest.raises(ValueError):
        AdamLogisticRegression(beta1=1.0, beta2=0.4, epsilon=1e-8, n_iter=5).fit(x_tr, y_tr)
    with pytest.raises(ValueError):
        AdamLogisticRegression(beta1=0.4, beta2=0.0, epsilon=1e-8, n_iter=5).fit(x_tr, y_tr)
    with pytest.raises(ValueError):
        AdamLogisticRegression(beta1=0.4, beta2=0.4, epsilon=0.0, n_iter=5).fit(x_tr, y_tr)


def _hotel_frames():
    rng = np.random.default_rng(19)
    n = 42

    def _frame(rows, random_seed):
        local = np.random.default_rng(random_seed)
        meals = np.array(["Breakfast", "Half Board", "No Meal"])
        rooms = np.array(["Type A", "Type B", "Type C", "Type D"])
        channels = np.array(["Online Travel Agent", "Offline Agent", "Corporate"])
        return pd.DataFrame(
            {
                "num_adults": local.integers(1, 4, rows),
                "num_children": local.integers(0, 3, rows).astype(float),
                "weekend_nights": local.integers(0, 3, rows),
                "weekday_nights": local.integers(1, 6, rows),
                "needs_parking": local.integers(0, 2, rows),
                "days_before_arrival": local.integers(3, 50, rows).astype(float),
                "arrival_year": np.full(rows, 2024),
                "arrival_month": local.integers(1, 13, rows),
                "arrival_day": local.integers(1, 28, rows),
                "is_returning_guest": local.integers(0, 2, rows),
                "prior_cancellations": local.integers(0, 3, rows),
                "prior_completed_bookings": local.integers(0, 5, rows),
                "nightly_rate_idr": local.integers(350_000, 2_400_000, rows).astype(float),
                "special_requests": local.integers(0, 4, rows).astype(float),
                "meal_package": meals[local.integers(0, 3, rows)],
                "room_category": rooms[local.integers(0, 4, rows)],
                "booking_channel": channels[local.integers(0, 3, rows)],
            }
        )

    train = _frame(n, 19)
    train.loc[2, "num_children"] = np.nan
    train.loc[7, "nightly_rate_idr"] = np.nan
    train.loc[9, "meal_package"] = np.nan
    train.loc[11, "special_requests"] = np.nan
    train.loc[4, "nightly_rate_idr"] = 0.0
    train.loc[6, "days_before_arrival"] = 420.0
    train.loc[14, "meal_package"] = "Full Board"
    train.loc[15, "room_category"] = "Type E"
    train.loc[20] = train.loc[0]
    train.loc[21, ["num_adults", "num_children"]] = 0.0
    lead = train["days_before_arrival"].fillna(20.0)
    prior = train["prior_cancellations"].astype(float)
    requests = train["special_requests"].fillna(0.0)
    rate = train["nightly_rate_idr"].fillna(1_000_000.0) / 1_000_000.0
    logit = -1.05 + 0.018 * lead + 0.45 * prior - 0.28 * requests + 0.22 * rate
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit.to_numpy()))).astype(int)
    y[20] = y[0]
    y[[0, 5, 8]] = 0
    y[[1, 3, 10, 12]] = 1
    labels = pd.Series(y, name="is_canceled")

    test = _frame(11, 23)
    test.loc[0, "meal_package"] = "Ultra Plan"
    test.loc[1, "nightly_rate_idr"] = np.nan
    test.loc[2, "num_children"] = np.nan
    test.loc[3, "room_category"] = "Type E"
    return train, labels, test


def test_pipeline_keeps_messy_hotel_rows_and_matches_saga():
    raw_train, raw_labels, raw_test = _hotel_frames()
    assert raw_train.isna().any().any()
    assert (raw_train["nightly_rate_idr"] == 0).any()
    cleaned_x, cleaned_y = clean_train(raw_train, raw_labels)
    assert len(cleaned_x) < len(raw_train)
    assert set(cleaned_y.unique()) == {0, 1}

    pipeline = make_numeric_pipeline(NUMERICAL_COLS, CATEGORICAL_COLS)
    transformed = pipeline.fit_transform(cleaned_x, cleaned_y)
    held_out = pipeline.transform(raw_test)
    assert len(held_out) == len(raw_test)
    features, categorical_index = to_model_input(transformed, CATEGORICAL_COLS)
    holdout_features, _ = to_model_input(held_out, CATEGORICAL_COLS)
    assert categorical_index == []
    assert np.isfinite(features).all() and np.isfinite(holdout_features).all()

    params = _fit_kwargs(
        penalty="elasticnet",
        reg_lambda=0.018,
        l1_ratio=0.4,
        learning_rate=0.5,
        n_iter=250,
        batch_size=9,
        class_weight=cost_sensitive_class_weight(),
        tol=1e-6,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model = LogisticRegression(**params).fit(features, cleaned_y.to_numpy())
    assert len(cleaned_y) % 9 != 0
    predicted = model.predict(holdout_features)
    assert predicted.shape == (len(raw_test),)
    assert set(np.unique(predicted)).issubset({0, 1})
    proba = model.predict_proba(holdout_features)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert np.isfinite(proba).all()

    full_batch = LogisticRegression(**{**params, "batch_size": None, "n_iter": 400}).fit(
        features, cleaned_y.to_numpy()
    )
    reference = _sklearn_from(params, len(cleaned_y)).fit(features, cleaned_y.to_numpy())
    assert np.corrcoef(full_batch.decision_function(holdout_features), reference.decision_function(holdout_features))[0, 1] > 0.95


def test_experiment_helpers_on_the_small_hotel():
    raw_train, raw_labels, _ = _hotel_frames()
    cleaned_x, cleaned_y = clean_train(raw_train, raw_labels)
    folds = prepare_folds(cleaned_x, cleaned_y, NUMERICAL_COLS, CATEGORICAL_COLS, n_splits=2)
    assert len(folds) == 2
    assigned = sum(len(fold[3]) for fold in folds)
    assert assigned == len(cleaned_y)
    for features_train, y_train, features_val, y_val in folds:
        assert np.isfinite(features_train).all() and np.isfinite(features_val).all()
        assert len(y_train) + len(y_val) == len(cleaned_y)
        assert set(np.unique(y_train)) == {0, 1}

    params = _fit_kwargs(penalty="l2", reg_lambda=0.02, l1_ratio=0.0, n_iter=80, batch_size=None, class_weight=None, tol=1e-5)
    per_fold, summary = compare_with_sklearn({"small": params}, folds)
    assert set(per_fold["impl"]) == {"scratch", "sklearn"}
    assert per_fold["roc_auc"].between(0.0, 1.0).all()
    scratch = float(summary.loc[("small", "scratch"), "accuracy"])
    official = float(summary.loc[("small", "sklearn"), "accuracy"])
    assert abs(scratch - official) < 0.2

    factory = lambda: make_numeric_pipeline(NUMERICAL_COLS, CATEGORICAL_COLS)
    cv = StratifiedKFold(2, True, 19)
    grid = [
        _fit_kwargs(penalty="l2", reg_lambda=0.01, n_iter=200, class_weight=None, tol=1e-4),
        _fit_kwargs(penalty="l2", reg_lambda=0.2, n_iter=200, class_weight="balanced", tol=1e-4),
    ]
    tuned = tune(grid, LogisticRegression, cleaned_x, cleaned_y, factory, cv)
    assert list(tuned["macro_f1"]) == sorted(tuned["macro_f1"], reverse=True)
    team = cross_validate(LogisticRegression(**grid[0]), factory, cleaned_x, cleaned_y, cv, {"macro_f1": macro_f1})
    assert len(team["macro_f1"]) == 2
    assert all(np.isfinite(team["macro_f1"]))

    restored = _params_from_row(pd.DataFrame(logreg_grid()).iloc[0], _PARAM_FIELDS)
    LogisticRegression(**restored)
    saga = _sklearn_model({"penalty": "elasticnet", "reg_lambda": 0.02, "l1_ratio": 0.35, "class_weight": {0: 2.0, 1: 1.0}}, 37)
    assert saga.C == pytest.approx(1.0 / (0.02 * 37))
    assert saga.l1_ratio == pytest.approx(0.35)
    assert saga.class_weight == {0: 2.0, 1: 1.0}
    unpenalized = _sklearn_model({"penalty": "none", "reg_lambda": 0.02, "class_weight": None}, 37)
    assert np.isinf(unpenalized.C)

    penalties = {row["penalty"] for row in logreg_grid()}
    assert penalties == {"l1", "l2", "elasticnet"}
    assert {row["class_weight"] for row in logreg_grid()} == {None, "balanced"}
    assert {row["batch_size"] for row in logreg_grid()} == {None, 512}
    assert {row["learning_rate"] for row in logreg_grid()} >= {0.2, 0.5}
    assert {row["class_weight"] for row in adam_grid()} == {None, "balanced"}
    assert all(row["penalty"] == "l2" for row in adam_grid())


def test_contour_of_a_single_messy_feature(guest_split, tmp_path):
    _, y_tr, _, _, raw_train, _ = guest_split
    rate = raw_train[:, 2:3]
    rate = (rate - rate.mean()) / rate.std()
    assert np.any(raw_train[:, 2] == 0.0)
    model = LogisticRegression(
        penalty="elasticnet",
        reg_lambda=0.04,
        l1_ratio=0.25,
        learning_rate=0.7,
        n_iter=18,
        tol=1e-8,
        class_weight={0: 2.0, 1: 1.0},
        record_history=True,
        random_state=6,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(rate, y_tr)
    gif = plot_logreg_contour(model, rate, y_tr, tmp_path / "stay_rate.gif")
    assert gif.stat().st_size > 0
    with Image.open(gif) as image:
        frames = getattr(image, "n_frames", 1)
    assert frames >= 2
    np.testing.assert_allclose(model.coef_history_[-1, 0], model.coef_[0])
    np.testing.assert_allclose(model.intercept_history_[-1], model.intercept_)
    assert model.loss_history_[-1] <= model.loss_history_[0] + 1e-8

    with pytest.raises(RuntimeError):
        plot_logreg_contour(LogisticRegression(), rate, y_tr, tmp_path / "unfitted.gif")
    png = tmp_path / "contour.png"
    with pytest.raises(ValueError, match="gif"):
        plot_logreg_contour(model, rate, y_tr, png)
    assert not png.exists()
    with pytest.raises(ValueError, match="Length mismatch"):
        plot_logreg_contour(model, rate, y_tr[:-3], tmp_path / "short.gif")
    wide = LogisticRegression(n_iter=40, tol=1e-4, record_history=True, learning_rate=0.6).fit(
        np.column_stack([rate, rate]), y_tr
    )
    with pytest.raises(ValueError, match="one-feature"):
        plot_logreg_contour(wide, rate, y_tr, tmp_path / "wide.gif")
    assert not Path(tmp_path / "short.gif").exists()
