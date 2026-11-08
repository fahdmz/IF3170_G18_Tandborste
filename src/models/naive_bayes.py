from src.models.base import BaseClassifier


class NaiveBayes(BaseClassifier):
    def __init__(
        self,
        categorical_features=None,
        alpha=1.0,
        var_smoothing=1e-9,
        random_state=42,
    ):
        super().__init__(random_state)
        self.categorical_features = categorical_features
        self.alpha = alpha
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
