import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.config import SEED
from src.data import CATEGORICAL_COLS, NUMERICAL_COLS
from src.models.decision_tree import DecisionTree
from src.preprocessing.presets import make_tree_pipeline, to_model_input
from src.preprocessing.transformation import FeatureEngineer


@pytest.fixture
def hotel_frames():
    # 32 observations per category keep common categories above min_count=20.
    codes = np.arange(96) % 3
    train = pd.DataFrame(1.0, index=np.arange(96), columns=NUMERICAL_COLS)
    train["num_adults"], train["num_children"] = 2.0, 0.0
    train["weekday_nights"], train["nightly_rate_idr"] = 2.0, 500_000.0
    train["arrival_year"], train["arrival_month"], train["arrival_day"] = 2026, 11, 2
    categories = {
        "meal_package": ["Room Only", "Breakfast", "Half Board"],
        "room_category": ["Type A", "Type B", "Type C"],
        "booking_channel": ["Direct", "Online", "Corporate"],
    }
    for col, values in categories.items():
        train[col] = np.asarray(values, dtype=object)[codes]
    train.loc[0, "meal_package"] = None
    train.loc[1, "room_category"] = pd.NA
    train.loc[2, "booking_channel"] = np.nan
    train.loc[3, "num_children"] = np.nan
    train.loc[4, "nightly_rate_idr"] = np.nan
    validation = train.iloc[:9].copy(deep=True)
    validation.loc[0, CATEGORICAL_COLS] = "ValidationOnly"
    validation.loc[1, "days_before_arrival"] = np.nan
    return train, (codes > 0).astype(int), validation


@pytest.mark.parametrize(
    "options", [{}, {"engineer_features": False}, {"group_rare": False},
                {"engineer_features": False, "group_rare": False}],
)
def test_a1_pipeline_uses_actual_category_schema_without_mutating_raw_input(hotel_frames, options):
    train, y, validation = hotel_frames
    original_train, original_validation = train.copy(deep=True), validation.copy(deep=True)
    original_y = y.copy()
    pipeline = make_tree_pipeline(NUMERICAL_COLS, CATEGORICAL_COLS, **options)
    transformed_train = pipeline.fit_transform(train, y)
    # Transform only: validation categories must never enter the fitted schema.
    transformed_validation = pipeline.transform(validation)
    X_train, categorical = to_model_input(transformed_train, CATEGORICAL_COLS)
    X_validation, validation_categorical = to_model_input(transformed_validation, CATEGORICAL_COLS)
    engineered = FeatureEngineer.ENGINEERED_COLS if options.get("engineer_features", True) else []
    expected_columns = NUMERICAL_COLS + engineered + CATEGORICAL_COLS
    assert list(transformed_train.columns) == list(transformed_validation.columns) == expected_columns
    assert categorical == validation_categorical == [expected_columns.index(c) for c in CATEGORICAL_COLS]
    raw_indices = [train.columns.get_loc(c) for c in CATEGORICAL_COLS]
    assert categorical == [i + len(engineered) for i in raw_indices]
    if engineered:
        assert categorical != raw_indices
    assert transformed_train[NUMERICAL_COLS].isna().any().any()
    assert transformed_train[CATEGORICAL_COLS].isna().any().any()
    for col in CATEGORICAL_COLS:
        assert transformed_train[col].nunique() >= 3
        expected_unknown = "Other" if options.get("group_rare", True) else "ValidationOnly"
        assert transformed_validation.loc[0, col] == expected_unknown
    if options.get("group_rare", True):
        grouper = dict(pipeline.steps)["rare_grouper"]
        assert all("ValidationOnly" not in values for values in grouper.frequent_.values())
    original_array = pd.DataFrame(X_train.copy())
    model = DecisionTree(categorical_features=categorical, random_state=SEED)
    assert model.fit(X_train, y) is model
    predictions = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)
    assert predictions.shape == (len(validation),) and set(predictions) <= {0, 1}
    assert probabilities.shape == (len(validation), 2)
    assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert model.tree_.is_categorical and len(model.tree_.children) >= 3
    pd.testing.assert_frame_equal(pd.DataFrame(X_train), original_array)
    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(validation, original_validation)
    np.testing.assert_array_equal(y, original_y)


@pytest.fixture(params=["numeric", "mixed_missing"])
def model_case(request):
    if request.param == "numeric":
        X, y = make_classification(n_samples=96, n_features=4, n_informative=3,
                                   n_redundant=0, random_state=SEED)
        queries = np.vstack([X[:6], [np.nan] * 4, [None] * 4])
        categories = None
    else:
        X = np.array([[0, "A"]] * 12 + [[1, "B"]] * 8 + [[2, "C"]] * 8
                     + [[None, "A"], [np.nan, "B"], [1, None], [2, np.nan]], dtype=object)
        y = np.array([0] * 12 + [1] * 16 + [0, 1, 0, 1])
        queries = np.array([[0, "A"], [1, "B"], [2, "C"], [None, "new"],
                            [np.nan, None], [1, "new"]], dtype=object)
        categories = [1]
    return X, y, queries, dict(categorical_features=categories, max_depth=5, random_state=SEED)


def _tree_state(root):
    state, pending = [], [((), root)]
    while pending:
        path, node = pending.pop()
        state.append((path, node.is_leaf, node.prediction, tuple(node.class_distribution),
                      node.n_samples, node.impurity, node.depth, node.feature_index,
                      node.threshold, node.is_categorical, node.gain, tuple(node.children)))
        pending.extend((path + (key,), child) for key, child in node.children.items())
    return state


def test_inherited_save_load_roundtrip_preserves_fitted_model(tmp_path, model_case):
    X, y, queries, params = model_case
    model = DecisionTree(**params).fit(X, y)
    path = tmp_path / "nested" / "decision_tree.pkl"
    model.save(path)
    restored = DecisionTree.load(path)
    assert restored is not model and restored.get_params() == model.get_params()
    np.testing.assert_array_equal(restored.classes_, [0, 1])
    assert restored.n_features_in_ == X.shape[1]
    assert restored.get_depth() == model.get_depth()
    assert restored.get_n_leaves() == model.get_n_leaves()
    assert _tree_state(restored.tree_) == _tree_state(model.tree_)
    np.testing.assert_array_equal(restored.predict(queries), model.predict(queries))
    np.testing.assert_array_equal(restored.predict_proba(queries), model.predict_proba(queries))


def test_clone_is_unfitted_and_refit_replaces_old_nodes(model_case):
    X, y, queries, params = model_case
    model = DecisionTree(**params).fit(X, y)
    clone = model.clone()
    assert clone.get_params() == model.get_params()
    assert not hasattr(clone, "tree_") and not hasattr(clone, "classes_")
    for method in (clone.predict, clone.predict_proba):
        with pytest.raises(RuntimeError, match="fit"):
            method(queries)
    clone.fit(X, y)
    assert clone.tree_ is not model.tree_
    assert _tree_state(clone.tree_) == _tree_state(model.tree_)
    np.testing.assert_array_equal(clone.predict_proba(queries), model.predict_proba(queries))
    old_root = model.tree_
    model.fit(X[:8], np.ones(8, dtype=int))
    assert model.tree_ is not old_root and model.tree_.n_samples == 8
    assert model.tree_.is_leaf and not model.tree_.children
    assert model.get_depth() == 0 and model.get_n_leaves() == 1
    np.testing.assert_array_equal(model.predict(queries), np.ones(len(queries), dtype=int))
    np.testing.assert_array_equal(model.predict_proba(queries), [[0, 1]] * len(queries))


def test_same_seed_fit_and_refit_reproduce_tree_and_probabilities(model_case):
    X, y, queries, params = model_case
    first = DecisionTree(**params).fit(X, y)
    state, probabilities, labels = _tree_state(first.tree_), first.predict_proba(queries), first.predict(queries)
    second = DecisionTree(**params).fit(X, y)
    first.fit(X, y)
    for model in (first, second):
        assert model.get_params() == params | {
            "criterion": "gain_ratio", "min_samples_split": 2, "min_samples_leaf": 1, "min_gain": 0.0,
        }
        assert _tree_state(model.tree_) == state
        np.testing.assert_array_equal(model.predict_proba(queries), probabilities)
        np.testing.assert_array_equal(model.predict(queries), labels)


def test_raw_unknown_category_uses_child_mixture_independently_of_rare_grouper():
    X = np.array([["A"]] * 30 + [["B"]] * 10, dtype=object)
    model = DecisionTree(categorical_features=[0], random_state=SEED).fit(X, [0] * 30 + [1] * 10)
    assert "never-seen" not in model.tree_.children
    queries = np.array([["never-seen"], [None], [np.nan]], dtype=object)
    np.testing.assert_allclose(model.predict_proba(queries), [[0.75, 0.25]] * 3)
    np.testing.assert_array_equal(model.predict(queries), [0, 0, 0])
