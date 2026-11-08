from abc import ABC, abstractmethod

import numpy as np


class BaseTransformer(ABC):
    @abstractmethod
    def fit(self, X, y=None):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class BaseResampler(ABC):
    def __init__(self, random_state=42):
        self.random_state = random_state

    @abstractmethod
    def fit_resample(self, X, y):
        raise NotImplementedError

    def _rng(self):
        return np.random.default_rng(self.random_state)


class Pipeline(BaseTransformer):
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        for _, step in self.steps:
            X = step.fit_transform(X, y)
        return X

    def transform(self, X):
        for _, step in self.steps:
            X = step.transform(X)
        return X
