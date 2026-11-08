from pathlib import Path

import pandas as pd

from src.models.base import BaseClassifier
from src.preprocessing.base import Pipeline


def make_submission(
    model: BaseClassifier,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    ids: pd.Series,
    path: str | Path,
) -> pd.DataFrame:
    raise NotImplementedError
