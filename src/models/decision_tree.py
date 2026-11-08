from dataclasses import dataclass, field

from src.models.base import BaseClassifier


@dataclass
class TreeNode:
    is_leaf: bool
    prediction: int = None
    class_distribution: object = None
    feature_index: int = None
    threshold: float = None
    is_categorical: bool = False
    children: dict = field(default_factory=dict)
    n_samples: float = 0.0
    impurity: float = 0.0
    gain: float = 0.0
    depth: int = 0


class DecisionTree(BaseClassifier):
    def __init__(
        self,
        criterion="gain_ratio",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_gain=0.0,
        categorical_features=None,
        random_state=42,
    ):
        super().__init__(random_state)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.categorical_features = categorical_features

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def get_depth(self):
        raise NotImplementedError

    def get_n_leaves(self):
        raise NotImplementedError

    def export_graph(self, path, top_n=None, feature_names=None):
        raise NotImplementedError
