import numpy as np
from sklearn.datasets import make_classification

from src.models.svm import SVM
from src.visualization.svm_boundary import plot_svm_boundary, plot_svm_kernels


def _data():
    X, y = make_classification(
        n_samples=150, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42
    )
    return X, y


def test_plot_svm_kernels_writes_png(tmp_path):
    X, y = _data()
    path = plot_svm_kernels(X, y, output_path=tmp_path / "svm.png")
    assert path.exists() and path.stat().st_size > 0


def test_plot_svm_boundary_rejects_non_2d():
    X, y = _data()
    model = SVM(kernel="rbf").fit(np.c_[X, X[:, :1]], y)
    try:
        plot_svm_boundary(model, np.c_[X, X[:, :1]], y)
    except ValueError:
        return
    raise AssertionError("ValueError diharapkan untuk X dengan 3 fitur")
