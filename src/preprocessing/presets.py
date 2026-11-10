import numpy as np
import pandas as pd

from src.preprocessing.base import Pipeline
from src.preprocessing.transformation import (
    MissingValueImputer,
    OneHotEncoder,
    StandardScaler,
    TreeInputFormatter,
)


def make_numeric_pipeline(
    numerical_cols: list[str], categorical_cols: list[str], use_pca: bool = False
) -> Pipeline:
    if use_pca:
        raise NotImplementedError("PCA belum tersedia di baseline v0")
    return Pipeline(
        [
            ("imputer", MissingValueImputer(numerical_cols, categorical_cols)),
            ("scaler", StandardScaler(numerical_cols)),
            ("encoder", OneHotEncoder(categorical_cols)),
        ]
    )


def make_tree_pipeline(
    numerical_cols: list[str], categorical_cols: list[str]
) -> Pipeline:
    return Pipeline([("formatter", TreeInputFormatter(numerical_cols, categorical_cols))])


def to_model_input(
    X: pd.DataFrame, categorical_cols: list[str]
) -> tuple[np.ndarray, list[int]]:
    wanted = set(categorical_cols)
    indices = [i for i, col in enumerate(X.columns) if col in wanted]
    if indices:
        return X.to_numpy(dtype=object), indices
    return X.to_numpy(dtype=float), []
