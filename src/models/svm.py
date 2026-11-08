from src.models.base import BaseClassifier


class SVM(BaseClassifier):
    def __init__(
        self,
        kernel="linear",
        C=1.0,
        degree=3,
        gamma="scale",
        coef0=1.0,
        tol=1e-3,
        max_iter=1000,
        max_train_samples=None,
        batch_size=1024,
        class_weight=None,
        random_state=42,
    ):
        super().__init__(random_state)
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.max_train_samples = max_train_samples
        self.batch_size = batch_size
        self.class_weight = class_weight

    def fit(self, X, y):
        raise NotImplementedError

    def decision_function(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
