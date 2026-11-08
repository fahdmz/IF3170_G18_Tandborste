from src.models.base import BaseClassifier


class LogisticRegression(BaseClassifier):
    def __init__(
        self,
        penalty="l2",
        reg_lambda=0.01,
        l1_ratio=0.5,
        learning_rate=0.01,
        n_iter=1000,
        batch_size=None,
        tol=1e-6,
        fit_intercept=True,
        class_weight=None,
        record_history=False,
        random_state=42,
    ):
        super().__init__(random_state)
        self.penalty = penalty
        self.reg_lambda = reg_lambda
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.class_weight = class_weight
        self.record_history = record_history

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError
