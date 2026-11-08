import inspect
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseClassifier(ABC):
    def __init__(self, random_state=42):
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError(f"{type(self).__name__} tidak mendukung predict_proba")

    def decision_function(self, X):
        raise NotImplementedError(f"{type(self).__name__} tidak mendukung decision_function")

    def get_params(self):
        signature = inspect.signature(type(self).__init__)
        return {name: getattr(self, name) for name in signature.parameters if name != "self"}

    def set_params(self, **params):
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"Parameter tidak dikenal: {key}")
            setattr(self, key, value)
        return self

    def clone(self):
        return type(self)(**self.get_params())

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"File berisi {type(model).__name__}, bukan {cls.__name__}")
        return model

    def _rng(self):
        return np.random.default_rng(self.random_state)

    def _check_is_fitted(self):
        if not hasattr(self, "classes_"):
            raise RuntimeError(f"{type(self).__name__} belum di-fit")

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({params})"
