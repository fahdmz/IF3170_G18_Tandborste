import json
from time import perf_counter

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import SEED
from src.models.decision_tree import DecisionTree


def test_numeric_sanity_against_sklearn_entropy():
    X, y = make_classification(n_samples=2000, random_state=SEED)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    original_train, original_validation = X_train.copy(), X_validation.copy()
    majority = int(np.argmax(np.bincount(y_train, minlength=2)))
    baseline = float(np.mean(y_validation == majority))
    models = {
        "c45": DecisionTree(criterion="gain_ratio", max_depth=12,
                            min_samples_leaf=5, random_state=SEED),
        "sklearn": DecisionTreeClassifier(criterion="entropy", max_depth=12,
                                           min_samples_leaf=5, ccp_alpha=0.0, random_state=SEED),
    }
    results = {}
    for name, model in models.items():
        started = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - started
        labels, probabilities = model.predict(X_validation), model.predict_proba(X_validation)
        results[name] = {
            "accuracy": float(accuracy_score(y_validation, labels)),
            "f1": float(f1_score(y_validation, labels)),
            "confusion_matrix": confusion_matrix(y_validation, labels, labels=[0, 1]).tolist(),
            "fit_seconds": fit_seconds,
        }
        assert labels.shape == (400,) and set(labels) == {0, 1}
        assert probabilities.shape == (400, 2)
        assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(labels, np.argmax(probabilities, axis=1))
        assert results[name]["accuracy"] > baseline
        assert 1 <= model.get_depth() <= 12 and model.get_n_leaves() >= 2
    tree = models["c45"].tree_
    pending = [tree]
    while pending:
        node = pending.pop()
        assert node.n_samples >= 5
        if not node.is_leaf:
            assert set(node.children) == {"le", "gt"} and node.gain > 0
            assert np.isfinite(node.threshold)
            pending.extend(node.children.values())
    np.testing.assert_array_equal(X_train, original_train)
    np.testing.assert_array_equal(X_validation, original_validation)
    # Gain ratio, pessimistic pruning, multiway categories and missing routing
    # differ from sklearn. The 3% target is diagnostic, never an equality gate.
    gap = abs(results["c45"]["accuracy"] - results["sklearn"]["accuracy"])
    print(json.dumps({"seed": SEED, "train_rows": len(y_train), "validation_rows": len(y_validation),
                      "majority_accuracy": baseline, "accuracy_gap": gap, "models": results}, sort_keys=True))
