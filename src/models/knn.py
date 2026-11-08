from src.models.base import BaseClassifier


class KNN(BaseClassifier):
    def __init__(
        self,
        n_neighbors=5,
        metric="euclidean",
        p=2.0,
        weights="uniform",
        batch_size=1024,
        random_state=42,
    ):
        super().__init__(random_state)
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.weights = weights
        self.batch_size = batch_size

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
