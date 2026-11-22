import numpy as np
import pytest

from src.models import decision_tree as dt
from src.models.decision_tree import DecisionTree, TreeNode


@pytest.mark.parametrize(
    "counts, expected",
    [([4, 0], 0), ([1, 1], 1), ([3, 1], 0.8112781244591328), ([0, 0], 0)],
)
def test_entropy_from_weighted_counts(counts, expected):
    assert dt._entropy(counts) == pytest.approx(expected)


def test_split_math_uses_mass_instead_of_row_count():
    # Parent [1.5, 1.5]; left H=H(3/4,1/4), right H=0.
    gain, info, ratio = dt._split_statistics([[1.5, 0.5], [0, 1]], 3)
    assert gain == pytest.approx(0.4591479170272448)
    assert info == pytest.approx(0.9182958340544896)
    assert ratio == pytest.approx(0.5)


@pytest.mark.parametrize("counts, total", [([[0, 0], [0, 0]], 0), ([[2, 1]], 3)])
def test_degenerate_split_has_zero_ratio(counts, total):
    gain, info, ratio = dt._split_statistics(counts, total)
    assert (gain, info, ratio) == pytest.approx((0, 0, 0))


def test_missing_gain_and_split_info_use_known_entropy_and_unknown_bucket():
    # W=6, K=4, known counts [3,1], and branch masses [3,1,2].
    gain, info, ratio = dt._split_statistics([[3, 0], [0, 1]], 6)
    assert gain == pytest.approx(0.5408520829727552)
    assert info == pytest.approx(1.4591479170272448)
    assert ratio == pytest.approx(0.3706629579231731)
    sparse_gain, _, _ = dt._split_statistics([[3, 0], [0, 1]], 60)
    assert sparse_gain == pytest.approx(gain / 10)


def test_numeric_split_is_separable():
    split = DecisionTree()._numeric_split(
        np.array([0, 1, 2, 3]), np.array([0, 0, 1, 1]), np.ones(4), 0
    )
    assert split["threshold"] == 1.5
    assert [split[key] for key in ("gain", "split_info", "gain_ratio")] == pytest.approx([1, 1, 1])


def test_weighted_numeric_split_ignores_zero_weight_and_repeated_values():
    split = DecisionTree()._numeric_split(
        np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0]), np.array([1.5, 0.5, 1, 0]), 0
    )
    assert split["threshold"] == 0.5
    assert split["gain"] == pytest.approx(0.4591479170272448)


def test_category_split_groups_actual_values_and_ignores_zero_mass():
    split = DecisionTree()._categorical_split(
        np.array(["A", "A", "B", "B", "C"]),
        np.array([0, 0, 1, 1, 0]), np.array([1, 1, 1, 1, 0]), 0,
    )
    assert split["is_categorical"]
    assert split["gain"] == pytest.approx(1)
    assert split["gain_ratio"] == pytest.approx(1)


@pytest.mark.parametrize("categorical", [False, True])
def test_constant_or_too_small_branch_has_no_candidate(categorical):
    model = DecisionTree(min_samples_leaf=2)
    method = model._categorical_split if categorical else model._numeric_split
    y, weights = np.array([0, 0, 1]), np.ones(3)
    assert method(np.array([0, 0, 0]), y, weights, 0) is None
    assert method(np.array([0, 0, 1]), y, weights, 0) is None


def test_leaf_mass_limit_is_applied_after_fractional_distribution():
    split = DecisionTree(min_samples_leaf=2)._numeric_split(
        np.array([0, 1, np.nan, np.nan]), np.array([0, 1, 0, 1]), np.ones(4), 0
    )
    assert split is not None  # Known masses 1+1 become effective masses 2+2.
    assert split["gain"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "values", [[1e308, 1.7e308], [-1.7e308, 1.7e308], [1.0, np.nextafter(1.0, 2.0)]]
)
def test_numeric_midpoint_is_finite_and_separates_neighbors(values):
    split = DecisionTree()._numeric_split(np.array(values), np.array([0, 1]), np.ones(2), 0)
    assert np.isfinite(split["threshold"])
    assert values[0] <= split["threshold"] < values[1]


def test_numeric_threshold_maximizes_gain_before_ratio():
    # Split after row 0 has larger ratio, but split after row 2 has larger gain.
    split = DecisionTree()._numeric_split(
        np.arange(8), np.array([0, 1, 0, 1, 1, 1, 1, 1]), np.ones(8), 0
    )
    assert split["threshold"] == 2.5


def test_equal_gain_threshold_and_feature_ties_are_deterministic():
    X = np.column_stack([np.arange(3), np.arange(3)])
    y = np.array([0, 1, 0])
    split = DecisionTree()._best_split(X, y, np.ones(3), frozenset())
    assert split["feature_index"] == 0
    assert split["threshold"] == 0.5


def test_average_gain_guard_rejects_high_ratio_low_gain_feature():
    # Feature 0 isolates one positive; feature 1 provides much more information.
    y = np.array([0] * 50 + [1] * 50)
    X = np.column_stack([
        [0] * 99 + [1],
        [0] * 34 + [1] * 16 + [0] * 16 + [1] * 34,
    ])
    model = DecisionTree()
    rare = model._numeric_split(X[:, 0], y, np.ones(100), 0)
    useful = model._numeric_split(X[:, 1], y, np.ones(100), 1)
    assert rare["gain"] < useful["gain"]
    assert rare["gain_ratio"] > useful["gain_ratio"]
    assert model._best_split(X, y, np.ones(100), frozenset())["feature_index"] == 1


def test_fit_predict_numeric_and_introspection():
    X, y = np.arange(4).reshape(-1, 1), np.array([0, 0, 1, 1])
    model = DecisionTree()
    assert model.fit(X, y) is model
    np.testing.assert_array_equal(model.classes_, [0, 1])
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_allclose(model.predict_proba(X), [[1, 0], [1, 0], [0, 1], [0, 1]])
    assert model.n_features_in_ == 1
    assert model.tree_.threshold == 1.5
    assert model.tree_.gain == pytest.approx(1)
    assert set(model.tree_.children) == {"le", "gt"}
    assert model.get_depth() == 1
    assert model.get_n_leaves() == 2


def test_three_categories_form_multiway_split_in_deterministic_order():
    X = np.array([["C"], ["B"], ["A"], ["C"], ["A"], ["B"]])
    y = np.array([1, 1, 0, 1, 0, 1])
    model = DecisionTree(categorical_features=[0]).fit(X, y)
    assert list(model.tree_.children) == ["A", "B", "C"]
    assert model.get_n_leaves() == 3
    np.testing.assert_array_equal(model.predict(X), y)


def test_numeric_features_can_split_again_on_the_same_path():
    X, y = np.arange(6).reshape(-1, 1), np.array([0, 0, 1, 1, 0, 0])
    model = DecisionTree().fit(X, y)
    assert model.get_depth() == 2
    assert model.get_n_leaves() == 3
    np.testing.assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_has_two_probability_columns(label):
    model = DecisionTree().fit([[0], [1]], [label, label])
    np.testing.assert_array_equal(model.classes_, [0, 1])
    np.testing.assert_array_equal(model.predict([[0], [2]]), [label, label])
    np.testing.assert_array_equal(model.predict_proba([[2]])[0], [1 - label, label])
    assert model.get_depth() == 0
    assert model.get_n_leaves() == 1


@pytest.mark.parametrize(
    "params", [{"max_depth": 0}, {"min_samples_split": 5}, {"min_samples_leaf": 3}, {"min_gain": 1}]
)
def test_stopping_controls_create_root_leaf(params):
    model = DecisionTree(**params).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    assert model.tree_.is_leaf
    assert model.get_depth() == 0
    np.testing.assert_allclose(model.predict_proba([[0]]), [[0.5, 0.5]])
    np.testing.assert_array_equal(model.predict([[0]]), [0])


def test_constant_features_and_class_tie_produce_majority_leaf():
    model = DecisionTree().fit([[3], [3]], [1, 0])
    assert model.tree_.is_leaf
    assert model.tree_.prediction == 0
    assert model.tree_.impurity == pytest.approx(1)


@pytest.mark.parametrize("method", ["predict", "predict_proba", "get_depth", "get_n_leaves"])
def test_public_inference_requires_fit(method):
    args = ([[0]],) if method.startswith("predict") else ()
    with pytest.raises(RuntimeError, match="fit"):
        getattr(DecisionTree(), method)(*args)


@pytest.mark.parametrize(
    "params, match",
    [
        ({"criterion": "gini"}, "criterion"),
        ({"max_depth": -1}, "max_depth"), ({"max_depth": 1.5}, "max_depth"),
        ({"max_depth": True}, "max_depth"),
        ({"min_samples_split": 1}, "min_samples_split"),
        ({"min_samples_split": 2.5}, "min_samples_split"),
        ({"min_samples_split": False}, "min_samples_split"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"min_samples_leaf": 1.5}, "min_samples_leaf"),
        ({"min_samples_leaf": True}, "min_samples_leaf"),
        ({"min_gain": -0.1}, "min_gain"), ({"min_gain": np.nan}, "min_gain"),
        ({"min_gain": np.inf}, "min_gain"), ({"min_gain": "0"}, "min_gain"),
        ({"categorical_features": [0, 0]}, "categorical_features"),
        ({"categorical_features": [-1]}, "categorical_features"),
        ({"categorical_features": [1]}, "categorical_features"),
        ({"categorical_features": [0.0]}, "categorical_features"),
        ({"categorical_features": [True]}, "categorical_features"),
        ({"categorical_features": 0}, "categorical_features"),
        ({"categorical_features": "0"}, "categorical_features"),
    ],
)
def test_invalid_parameters_are_rejected(params, match):
    with pytest.raises(ValueError, match=match):
        DecisionTree(**params).fit([[0], [1]], [0, 1])


@pytest.mark.parametrize(
    "X, y, match",
    [
        ([0, 1], [0, 1], "X"), (np.empty((0, 1)), [], "empty"),
        (np.empty((2, 0)), [0, 1], "feature"), ([[0], [1]], [0], "y"),
        ([[0], [1]], [0, 2], "0.*1"),
        ([[0], [1]], [0, None], "0.*1"), ([[0], [1]], [0, np.nan], "0.*1"),
        ([[0], [1]], ["0", "1"], "0.*1"),
        ([[0], [np.inf]], [0, 1], "numeric"),
        ([[0], [-np.inf]], [0, 1], "numeric"),
        ([[0], ["invalid"]], [0, 1], "numeric"),
        ([[0], ["nan"]], [0, 1], "numeric"),
        ([[0], [1j]], [0, 1], "numeric"),
    ],
)
def test_invalid_training_data_is_rejected(X, y, match):
    model = DecisionTree()
    with pytest.raises(ValueError, match=match):
        model.fit(X, y)
    assert not hasattr(model, "classes_")


@pytest.mark.parametrize("X", [[0], [[0, 1]], [[[0]]], [[np.inf]], [["invalid"]]])
def test_predict_validates_shape_width_and_numeric_values(X):
    model = DecisionTree().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        model.predict_proba(X)


def test_empty_prediction_batch_shapes():
    model = DecisionTree().fit([[0], [1]], [0, 1])
    assert model.predict_proba(np.empty((0, 1))).shape == (0, 2)
    assert model.predict(np.empty((0, 1))).shape == (0,)


def test_mixed_input_is_unchanged_and_constructor_params_roundtrip():
    X = np.array([[0, "A"], [1, "B"], [2, "A"], [3, "B"]], dtype=object)
    y = np.array([0, 1, 0, 1])
    original_X, original_y = X.copy(), y.copy()
    categories = [np.int64(1)]
    model = DecisionTree(categorical_features=categories, max_depth=np.int64(3))
    params = model.get_params()
    model.fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(y, original_y)
    assert model.categorical_features is categories
    assert model.get_params() == params
    clone = model.clone()
    assert clone.get_params() == params
    assert not hasattr(clone, "tree_")
    with pytest.raises(RuntimeError):
        clone.predict(X)


def test_refit_replaces_tree_schema_and_predictions():
    model = DecisionTree().fit([[0], [1]], [0, 1])
    previous = model.tree_
    model.set_params(categorical_features=[1])
    model.fit([[1, "A"], [2, "B"]], [1, 1])
    assert model.tree_ is not previous
    assert model.n_features_in_ == 2
    np.testing.assert_array_equal(model.predict([[1, "C"]]), [1])


def test_failed_refit_does_not_leave_half_updated_fitted_state():
    model = DecisionTree().fit([[0], [1]], [0, 1])
    tree = model.tree_
    with pytest.raises(ValueError):
        model.fit([["bad", 0]], [1])
    assert model.tree_ is tree
    assert model.n_features_in_ == 1
    np.testing.assert_array_equal(model.predict([[0], [1]]), [0, 1])


@pytest.mark.parametrize("missing", [None, np.nan])
@pytest.mark.parametrize("categorical", [False, True])
def test_fractional_missing_hand_calculated_fixture(missing, categorical):
    values = ["A", "A", "A", "B"] if categorical else [0, 0, 0, 1]
    X = np.array(values + [missing, missing], dtype=object).reshape(-1, 1)
    model = DecisionTree(max_depth=1, categorical_features=[0] if categorical else None)
    model.fit(X, [0, 0, 0, 1, 0, 1])
    root = model.tree_
    first, second = list(root.children.values())
    assert root.n_samples == pytest.approx(6)
    assert root.gain == pytest.approx(0.5408520829727552)
    assert first.n_samples == pytest.approx(4.5)
    assert second.n_samples == pytest.approx(1.5)
    np.testing.assert_allclose(first.class_distribution, [3.75, 0.75])
    np.testing.assert_allclose(second.class_distribution, [0.25, 1.25])
    np.testing.assert_allclose(model.predict_proba([[missing]]), [[2 / 3, 1 / 3]])
    np.testing.assert_allclose(model.predict_proba([[values[0]]]), [[5 / 6, 1 / 6]])
    np.testing.assert_allclose(model.predict_proba([[values[3]]]), [[1 / 6, 5 / 6]])
    if categorical:
        assert list(root.children) == ["A", "B"]  # No explicit missing child.
        np.testing.assert_allclose(model.predict_proba([["unseen"]]), [[2 / 3, 1 / 3]])


@pytest.mark.parametrize("categorical", [False, True])
def test_all_missing_features_create_distribution_leaf(categorical):
    X = np.array([[None, np.nan], [np.nan, None], [None, None]], dtype=object)
    model = DecisionTree(categorical_features=[0, 1] if categorical else None).fit(X, [0, 1, 1])
    assert model.tree_.is_leaf
    np.testing.assert_allclose(model.predict_proba(X), [[1 / 3, 2 / 3]] * 3)


def test_all_missing_column_does_not_win_split():
    model = DecisionTree(categorical_features=[0]).fit(
        [[None, 0], [np.nan, 1], [None, 2], [None, 3]], [0, 0, 1, 1]
    )
    assert model.tree_.feature_index == 1


def test_literal_nan_string_is_a_category_and_mixed_category_types_are_safe():
    X = np.array([["nan"], ["nan"], [1], [1], [None], [np.nan]], dtype=object)
    model = DecisionTree(categorical_features=[0]).fit(X, [0, 0, 1, 1, 0, 1])
    assert set(model.tree_.children) == {"nan", 1}
    np.testing.assert_allclose(model.predict_proba([["nan"], [1], ["new"]]),
                               [[5 / 6, 1 / 6], [1 / 6, 5 / 6], [0.5, 0.5]])


@pytest.fixture
def two_level_missing_tree():
    # Known A=[8,2], B=[2,8]. Unknown root rows each send weight 1/2.
    X = np.array(
        [["A", 0]] * 6 + [["A", 1]] * 4 + [["B", 0]] * 4 + [["B", 1]] * 6
        + [[None, 0], [None, 1]], dtype=object,
    )
    y = np.array([0] * 6 + [0, 0, 1, 1] + [0, 0, 1, 1] + [1] * 6 + [1, 0])
    model = DecisionTree(categorical_features=[0], max_depth=2).fit(X, y)
    # Isolate growth/multilevel routing: these weak descendants are correctly
    # collapsed by pruning. No unpruned copy is retained by production fit.
    model.tree_ = model._grow_tree(X, y, np.ones(len(y)), frozenset([0]), 0)
    return model


def test_multilevel_mixture_uses_known_descendant_feature(two_level_missing_tree):
    model = two_level_missing_tree
    assert model.tree_.feature_index == 0
    assert model.get_depth() == 2
    assert model.get_n_leaves() == 4
    X = np.array([[None, 0], ["new", 0], [np.nan, 1], [None, None]], dtype=object)
    # For z=0: (1/2)*[12/13,1/13] + (1/2)*[4/9,5/9].
    expected = [[80 / 117, 37 / 117], [80 / 117, 37 / 117],
                [37 / 117, 80 / 117], [0.5, 0.5]]
    np.testing.assert_allclose(model.predict_proba(X), expected)
    assert X[0, 0] is None and X[1, 0] == "new" and np.isnan(X[2, 0])


def test_fractional_mass_and_class_counts_conserved_at_every_level(two_level_missing_tree):
    def check(node):
        assert node.n_samples == pytest.approx(np.sum(node.class_distribution))
        if not node.is_leaf:
            children = list(node.children.values())
            assert sum(child.n_samples for child in children) == pytest.approx(node.n_samples)
            np.testing.assert_allclose(np.sum([c.class_distribution for c in children], axis=0),
                                       node.class_distribution)
            for child in children:
                assert child.depth == node.depth + 1
                check(child)
    root = two_level_missing_tree.tree_
    np.testing.assert_allclose(root.class_distribution, [11, 11])
    np.testing.assert_allclose(root.children["A"].class_distribution, [8.5, 2.5])
    np.testing.assert_allclose(root.children["B"].class_distribution, [2.5, 8.5])
    check(root)


def test_split_stopping_uses_effective_mass_not_fractional_row_count():
    node = DecisionTree(min_samples_split=3)._grow_tree(
        np.arange(4).reshape(-1, 1), np.array([0, 0, 1, 1]), np.full(4, 0.6), frozenset(), 0
    )
    assert node.n_samples == pytest.approx(2.4)
    assert node.is_leaf


def test_small_weights_are_retained_in_counts_without_rounding():
    node = DecisionTree()._grow_tree(
        np.array([[0], [1]]), np.array([0, 1]), np.array([1e-14, 2e-14]), frozenset(), 0
    )
    np.testing.assert_allclose(node.class_distribution, [1e-14, 2e-14], rtol=1e-14, atol=0)
    assert node.n_samples == pytest.approx(3e-14, rel=1e-14, abs=0)


def test_predictions_and_refit_are_deterministic(two_level_missing_tree):
    first = two_level_missing_tree
    X = np.array([[None, 0], ["A", 1], ["unknown", None]], dtype=object)
    expected = first.predict_proba(X)
    other = first.clone()
    training = [["A", 0], ["A", 1], ["B", 0], ["B", 1], [None, 1]]
    labels = [0, 1, 0, 1, 0]
    first.fit(training, labels)
    result = first.predict_proba(X)
    first.fit(training, labels)
    other.fit(training, labels)
    np.testing.assert_array_equal(first.predict_proba(X), result)
    np.testing.assert_array_equal(other.predict_proba(X), result)
    assert not np.allclose(expected, result)
    assert np.isfinite(result).all() and (result >= 0).all()
    np.testing.assert_allclose(result.sum(axis=1), 1)


def test_deep_tree_fit_does_not_depend_on_python_recursion_limit():
    # Three observations per pure branch keep the splits useful after pruning;
    # one alternating observation per branch is correctly pruned as noise.
    X = np.repeat(np.arange(1100), 3).reshape(-1, 1)
    y = np.repeat(np.arange(1100) % 2, 3)
    model = DecisionTree().fit(X, y)
    assert model.get_depth() > 1000
    assert model.get_n_leaves() == 1100
    np.testing.assert_array_equal(model.predict(X), y)


def test_missing_prediction_traverses_deep_tree_without_recursion():
    X = np.repeat(np.arange(600), 3).reshape(-1, 1)
    y = np.repeat(np.arange(600) % 2, 3)
    model = DecisionTree().fit(X, y)
    assert model.get_depth() > 500
    np.testing.assert_allclose(model.predict_proba([[None], [np.nan]]), [[0.5, 0.5]] * 2)
    np.testing.assert_array_equal(model.predict([[599]]), [1])


@pytest.mark.parametrize(
    "mass, error, expected",
    [
        (0, 0, 0), (1e-14, 0, 0),
        (10, 0, 1.294494367038759), (10, 10, 10),
        (10, 2, 3.518577506615758),
        (10, 0.4, 1.741721220627051),
        (0.5, 0, 0.46875), (0.5, 0.2, 0.48125),
        (0.75, 0.25, 0.6712549343815704),
        (1, 0.5, 0.875), (1, 1, 1), (2, 1.5, 2),
    ],
)
def test_pessimistic_error_estimate_hand_calculated(mass, error, expected):
    # CF=.25: E(10,0)=10*(1-.25**.1); fractional e interpolates total E.
    # For N=.5, interpolate (.0,.46875) to (.5,.5), never to e=1.
    actual = dt._estimated_errors(mass, error)
    assert actual == pytest.approx(expected)
    assert np.isfinite(actual) and 0 <= actual <= mass
    if mass > 1e-12:
        assert actual >= error


def _leaf(counts, depth=1):
    counts = np.asarray(counts, dtype=float)
    return TreeNode(
        is_leaf=True, prediction=int(np.argmax(counts)), class_distribution=counts,
        n_samples=float(counts.sum()), impurity=float(dt._entropy(counts)), depth=depth,
    )


def _split_node(children, categorical=False, depth=0):
    node = _leaf(np.sum([c.class_distribution for c in children.values()], axis=0), depth)
    node.is_leaf = False
    node.feature_index = 0
    node.threshold = None if categorical else 0.5
    node.is_categorical = categorical
    node.children = children
    node.gain = 0.1
    return node


@pytest.mark.parametrize("categorical", [False, True])
def test_pruning_collapses_uninformative_split_and_preserves_node_statistics(categorical):
    # E(10,2)=3.51858 < 2*E(5,1)=4.50067: replacing this split is cheaper.
    root = _split_node({"le": _leaf([4, 1]), "gt": _leaf([4, 1])}, categorical)
    before = (root.prediction, root.n_samples, root.impurity, root.depth)
    counts = root.class_distribution.copy()
    error = DecisionTree()._prune_tree(root)
    assert error == pytest.approx(3.518577506615758)
    assert root.is_leaf and root.children == {}
    assert root.feature_index is None and root.threshold is None
    assert not root.is_categorical and root.gain == 0
    assert (root.prediction, root.n_samples, root.impurity, root.depth) == before
    np.testing.assert_array_equal(root.class_distribution, counts)


def test_pruning_keeps_informative_split():
    # E(20,10)=11.97839 > 2*E(10,0)=2.58899: retain the pure branches.
    root = _split_node({"le": _leaf([10, 0]), "gt": _leaf([0, 10])})
    error = DecisionTree()._prune_tree(root)
    assert error == pytest.approx(2.588988734077517)
    assert not root.is_leaf and len(root.children) == 2
    np.testing.assert_array_equal(root.class_distribution, [10, 10])


def test_pruning_exact_tie_prefers_leaf():
    # A redundant one-child fixture has exactly E(10,2) on both sides.
    root = _split_node({"only": _leaf([8, 2])})
    DecisionTree()._prune_tree(root)
    assert root.is_leaf and not root.children


def test_pruning_is_bottom_up_and_introspection_ignores_removed_nodes():
    left = _split_node({"le": _leaf([3, 2], 2), "gt": _leaf([3, 2], 2)}, depth=1)
    right = _split_node({"le": _leaf([2, 3], 2), "gt": _leaf([2, 3], 2)}, depth=1)
    root = _split_node({"le": left, "gt": right})
    model = DecisionTree().fit([[0]], [0])
    model.tree_ = root
    # 2*E(10,4) < E(20,10) < 4*E(5,2): comparing the original leaves
    # would incorrectly collapse the root; compare the pruned children first.
    model._prune_tree(root)
    assert not root.is_leaf and left.is_leaf and right.is_leaf
    assert not left.children and not right.children
    assert model.get_depth() == 1 and model.get_n_leaves() == 2
    np.testing.assert_array_equal(root.class_distribution, [10, 10])
    np.testing.assert_array_equal(left.class_distribution, [6, 4])
    np.testing.assert_array_equal(right.class_distribution, [4, 6])


@pytest.mark.parametrize("counts", [[0, 0], [0.1, 0], [0.3, 0.2], [1e-14, 2e-14]])
def test_pruning_pure_or_fractional_leaf_preserves_counts(counts):
    leaf = _leaf(counts, depth=0)
    error = DecisionTree()._prune_tree(leaf)
    assert np.isfinite(error) and 0 <= error <= leaf.n_samples
    assert leaf.is_leaf and not leaf.children
    np.testing.assert_array_equal(leaf.class_distribution, counts)


def test_fit_prunes_after_growth_and_missing_uses_surviving_child_masses():
    model = DecisionTree(categorical_features=[0], max_depth=1)
    X = np.array([["A"]] * 30 + [["B"]] * 10, dtype=object)
    y = np.array([0] * 27 + [1] * 3 + [0] + [1] * 9)
    model.fit(X, y)
    assert not model.tree_.is_leaf
    np.testing.assert_allclose(model.predict_proba([[None], ["unseen"], ["A"], ["B"]]),
                               [[0.7, 0.3], [0.7, 0.3], [0.9, 0.1], [0.1, 0.9]])
    # A weak real split grows, but pessimistic error prefers its parent leaf.
    X, y = np.arange(5).reshape(-1, 1), np.array([0, 0, 1, 0, 0])
    grown = DecisionTree(max_depth=1)._grow_tree(X, y, np.ones(5), frozenset(), 0)
    assert not grown.is_leaf
    model = DecisionTree(max_depth=1).fit(X, y)
    assert model.tree_.is_leaf and model.get_depth() == 0 and model.get_n_leaves() == 1
    np.testing.assert_allclose(model.predict_proba([[None], [2]]), [[0.8, 0.2]] * 2)


def test_missing_multilevel_prediction_after_pruning_retains_known_descendant_feature():
    X = np.array([["A", 0]] * 18 + [["A", 1]] * 12 + [["B", 0]] * 10
                 + [["B", 1]] * 20 + [[None, 0], [None, 1]], dtype=object)
    y = [0] * 18 + [0] * 2 + [1] * 10 + [1] * 30 + [0, 1]
    model = DecisionTree(categorical_features=[0], max_depth=2).fit(X, y)
    root = model.tree_
    assert root.feature_index == 0 and model.get_depth() == 2
    assert not root.children["A"].is_leaf and root.children["B"].is_leaf
    assert model.get_n_leaves() == 3
    np.testing.assert_allclose([child.n_samples for child in root.children.values()], [31, 31])
    # A retains [18.5,0] and [2,10.5]; B collapses to [.5,30.5].
    # Unknown root mixes each with prior 1/2 while preserving numeric z.
    queries = [[None, 0], ["unseen", 0], [np.nan, 1], [None, None]]
    np.testing.assert_allclose(model.predict_proba(queries),
                               [[63 / 124, 61 / 124], [63 / 124, 61 / 124],
                                [273 / 3100, 2827 / 3100], [21 / 62, 41 / 62]])
