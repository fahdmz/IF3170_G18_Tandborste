import warnings

import pytest
from sklearn.datasets import make_classification

from src.models.logistic_regression import LogisticRegression
from src.visualization.logreg_contour import plot_logreg_contour


def _one_feature():
    X, y = make_classification(
        n_samples=80,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        random_state=42,
    )
    X = (X - X.mean()) / X.std()
    return X, y


def test_plot_logreg_contour_writes_gif(tmp_path):
    X, y = _one_feature()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model = LogisticRegression(
            penalty="l2", learning_rate=0.8, n_iter=40, tol=1e-4, record_history=True
        ).fit(X, y)
    path = plot_logreg_contour(model, X, y, tmp_path / "logreg_contour.gif")
    assert path.exists() and path.stat().st_size > 0
    assert path.suffix == ".gif"


def test_plot_logreg_contour_requires_history(tmp_path):
    X, y = _one_feature()
    model = LogisticRegression(learning_rate=0.8, n_iter=40, tol=1e-3, record_history=False).fit(X, y)
    with pytest.raises(ValueError, match="record_history"):
        plot_logreg_contour(model, X, y, tmp_path / "missing.gif")


def test_plot_logreg_contour_rejects_two_features(tmp_path):
    X, y = make_classification(n_samples=60, n_features=2, n_redundant=0, random_state=42)
    model = LogisticRegression(learning_rate=1.0, n_iter=40, tol=1e-3, record_history=True).fit(X, y)
    with pytest.raises(ValueError, match="shape"):
        plot_logreg_contour(model, X, y, tmp_path / "wide.gif")
