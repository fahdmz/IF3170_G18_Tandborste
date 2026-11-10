import numpy as np
import pandas as pd

from src.preprocessing.base import BaseTransformer


def _check_fitted(obj: BaseTransformer, attr: str) -> None:
    if not hasattr(obj, attr):
        raise RuntimeError(
            f"{type(obj).__name__} belum di-fit. Panggil fit() sebelum transform()."
        )


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _to_str_object(s: pd.Series) -> pd.Series:
    values = [np.nan if pd.isna(v) else str(v) for v in s]
    return pd.Series(values, index=s.index, dtype=object)


class MissingValueImputer(BaseTransformer):
    def __init__(
        self,
        numerical_cols: list[str],
        categorical_cols: list[str],
        numerical_strategy: str = "median",
        categorical_fill: str = "Missing",
    ) -> None:
        if numerical_strategy not in ("median", "mean"):
            raise ValueError("numerical_strategy harus 'median' atau 'mean'")
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.numerical_strategy = numerical_strategy
        self.categorical_fill = categorical_fill

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MissingValueImputer":
        fill_values: dict[str, float] = {}
        for col in self.numerical_cols:
            s = _to_float(X[col])
            value = s.median() if self.numerical_strategy == "median" else s.mean()
            fill_values[col] = 0.0 if pd.isna(value) else float(value)
        self.fill_values_ = fill_values
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_fitted(self, "fill_values_")
        X = X.copy()
        for col in self.numerical_cols:
            X[col] = _to_float(X[col]).fillna(self.fill_values_[col])
        for col in self.categorical_cols:
            s = X[col]
            X[col] = s.where(s.notna(), self.categorical_fill)
        return X


class StandardScaler(BaseTransformer):
    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "StandardScaler":
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for col in self.cols:
            s = _to_float(X[col])
            mean = s.mean()
            std = s.std(ddof=0)
            means[col] = 0.0 if pd.isna(mean) else float(mean)
            stds[col] = 1.0 if (pd.isna(std) or std == 0) else float(std)
        self.mean_ = means
        self.std_ = stds
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_fitted(self, "mean_")
        X = X.copy()
        for col in self.cols:
            X[col] = (_to_float(X[col]) - self.mean_[col]) / self.std_[col]
        return X


class OneHotEncoder(BaseTransformer):
    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OneHotEncoder":
        categories: dict[str, list[str]] = {}
        for col in self.cols:
            values = {str(v) for v in X[col] if not pd.isna(v)}
            categories[col] = sorted(values)
        self.categories_ = categories
        self.feature_names_ = [
            f"{col}__{cat}" for col in self.cols for cat in categories[col]
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_fitted(self, "categories_")
        X = X.copy()
        encoded: dict[str, np.ndarray] = {}
        for col in self.cols:
            values = _to_str_object(X[col]).to_numpy(dtype=object)
            for cat in self.categories_[col]:
                encoded[f"{col}__{cat}"] = (values == cat).astype("float64")
        X = X.drop(columns=self.cols)
        new = pd.DataFrame(encoded, index=X.index, columns=self.feature_names_)
        return pd.concat([X, new], axis=1)


class TreeInputFormatter(BaseTransformer):
    def __init__(self, numerical_cols: list[str], categorical_cols: list[str]) -> None:
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TreeInputFormatter":
        self.columns_ = list(self.numerical_cols) + list(self.categorical_cols)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_fitted(self, "columns_")
        X = X.copy()
        out: dict[str, pd.Series] = {}
        for col in self.numerical_cols:
            out[col] = _to_float(X[col])
        for col in self.categorical_cols:
            out[col] = _to_str_object(X[col])
        return pd.DataFrame(out, index=X.index, columns=self.columns_)
