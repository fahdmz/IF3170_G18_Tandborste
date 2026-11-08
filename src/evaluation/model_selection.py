from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

from src.models.base import BaseClassifier
from src.preprocessing.base import BaseResampler, Pipeline


def train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    raise NotImplementedError


class StratifiedKFold:
    def __init__(
        self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError


def cross_validate(
    model: BaseClassifier,
    pipeline_factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    resampler: BaseResampler | None = None,
) -> dict[str, list[float]]:
    raise NotImplementedError


def grid_search(
    model: BaseClassifier,
    param_grid: dict[str, list],
    pipeline_factory: Callable[[], Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    scoring: Callable[[np.ndarray, np.ndarray], float],
    resampler: BaseResampler | None = None,
) -> tuple[dict, list[dict]]:
    raise NotImplementedError
