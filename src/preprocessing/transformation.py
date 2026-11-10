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


_MISSING_LABEL = "Missing"


def _numeric_matrix(X: pd.DataFrame, name: str) -> np.ndarray:
    try:
        A = X.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} hanya menerima input numerik penuh") from exc
    if np.isnan(A).any():
        raise ValueError(f"{name} hanya menerima input tanpa NaN")
    return A


def _discretize(values: np.ndarray) -> np.ndarray:
    uniques = np.unique(values)
    if len(uniques) <= 10:
        return np.searchsorted(uniques, values)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 11)[1:-1]))
    return np.searchsorted(edges, values, side="right")


def _mutual_information(x_codes: np.ndarray, y_codes: np.ndarray, n_y: int) -> float:
    n = len(x_codes)
    n_x = int(x_codes.max()) + 1
    joint = np.bincount(x_codes * n_y + y_codes, minlength=n_x * n_y).reshape(n_x, n_y) / n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    expected = (px @ py)[nonzero]
    return max(0.0, float((joint[nonzero] * np.log(joint[nonzero] / expected)).sum()))


def _abs_target_correlation(A: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered = A - A.mean(axis=0)
    t = target - target.mean()
    denom = np.sqrt((centered**2).sum(axis=0) * float((t**2).sum()))
    safe = np.where(denom > 0, denom, 1.0)
    return np.where(denom > 0, np.abs(centered.T @ t) / safe, 0.0)


class FeatureEngineer(BaseTransformer):
    REQUIRED_COLS: list[str] = [
        "weekend_nights",
        "weekday_nights",
        "num_adults",
        "num_children",
        "nightly_rate_idr",
        "arrival_year",
        "arrival_month",
        "arrival_day",
    ]
    ENGINEERED_COLS: list[str] = [
        "total_nights",
        "total_guests",
        "has_children",
        "is_free",
        "total_cost",
        "arrival_dayofweek",
        "arrival_is_weekend",
    ]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        missing = [c for c in self.REQUIRED_COLS if c not in X.columns]
        if missing:
            raise ValueError(f"FeatureEngineer membutuhkan kolom: {missing}")
        self.feature_names_ = list(self.ENGINEERED_COLS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_fitted(self, "feature_names_")
        X = X.copy()
        weekend = _to_float(X["weekend_nights"])
        weekday = _to_float(X["weekday_nights"])
        adults = _to_float(X["num_adults"])
        children = _to_float(X["num_children"])
        rate = _to_float(X["nightly_rate_idr"])

        total_nights = weekend + weekday
        dates = pd.to_datetime(
            pd.DataFrame(
                {
                    "year": _to_float(X["arrival_year"]),
                    "month": _to_float(X["arrival_month"]),
                    "day": _to_float(X["arrival_day"]),
                }
            ),
            errors="coerce",
        )
        dayofweek = dates.dt.day_of_week.astype("float64")

        X["total_nights"] = total_nights
        X["total_guests"] = adults + children
        X["has_children"] = (children > 0).astype("float64").where(children.notna())
        X["is_free"] = (rate == 0).astype("float64").where(rate.notna())
        X["total_cost"] = rate * total_nights
        X["arrival_dayofweek"] = dayofweek
        X["arrival_is_weekend"] = dayofweek.isin([5.0, 6.0]).astype("float64").where(
            dayofweek.notna()
        )
        return X


class MissingIndicator(BaseTransformer):
    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MissingIndicator":
        self.missing_cols_ = [col for col in self.cols if bool(X[col].isna().any())]
        self.feature_names_ = [f"{col}__was_missing" for col in self.missing_cols_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_fitted(self, "missing_cols_")
        X = X.copy()
        for col, name in zip(self.missing_cols_, self.feature_names_):
            X[name] = X[col].isna().astype("float64")
        return X


class OutlierClipper(BaseTransformer):
    def __init__(
        self, cols: list[str], lower_quantile: float = 0.0, upper_quantile: float = 0.99
    ) -> None:
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("Harus berlaku 0 <= lower_quantile < upper_quantile <= 1")
        self.cols = cols
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "OutlierClipper":
        lower: dict[str, float] = {}
        upper: dict[str, float] = {}
        for col in self.cols:
            s = _to_float(X[col]).dropna()
            if s.empty:
                lower[col], upper[col] = -np.inf, np.inf
            else:
                lower[col] = float(s.quantile(self.lower_quantile))
                upper[col] = float(s.quantile(self.upper_quantile))
        self.lower_ = lower
        self.upper_ = upper
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_fitted(self, "lower_")
        X = X.copy()
        for col in self.cols:
            X[col] = _to_float(X[col]).clip(self.lower_[col], self.upper_[col])
        return X


class RareCategoryGrouper(BaseTransformer):
    def __init__(
        self, cols: list[str], min_count: int = 20, other_label: str = "Other"
    ) -> None:
        self.cols = cols
        self.min_count = min_count
        self.other_label = other_label

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RareCategoryGrouper":
        frequent: dict[str, list[str]] = {}
        for col in self.cols:
            counts = _to_str_object(X[col]).value_counts(dropna=True)
            frequent[col] = sorted(
                str(cat) for cat, n in counts.items() if n >= self.min_count
            )
        self.frequent_ = frequent
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_fitted(self, "frequent_")
        X = X.copy()
        for col in self.cols:
            keep = set(self.frequent_[col])
            keep.add(_MISSING_LABEL)
            values = [
                v if (pd.isna(v) or v in keep) else self.other_label
                for v in _to_str_object(X[col])
            ]
            X[col] = pd.Series(values, index=X.index, dtype=object)
        return X


class FeatureSelector(BaseTransformer):
    def __init__(
        self,
        method: str = "mutual_info",
        k: int | None = None,
        variance_threshold: float = 0.0,
        correlation_threshold: float | None = 0.95,
    ) -> None:
        if method not in ("mutual_info", "correlation"):
            raise ValueError("method harus 'mutual_info' atau 'correlation'")
        if k is not None and k < 1:
            raise ValueError("k harus >= 1 atau None")
        self.method = method
        self.k = k
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureSelector":
        if y is None:
            raise ValueError("FeatureSelector.fit membutuhkan y")
        A = _numeric_matrix(X, "FeatureSelector")
        target = np.asarray(y, dtype="float64")
        if len(target) != len(A):
            raise ValueError("X dan y harus memiliki jumlah baris yang sama")
        columns = list(X.columns)
        target_corr = _abs_target_correlation(A, target)

        kept = np.flatnonzero(A.var(axis=0) > self.variance_threshold)

        if self.correlation_threshold is not None and len(kept) > 1:
            corr = np.nan_to_num(np.abs(np.corrcoef(A[:, kept], rowvar=False)))
            order = sorted(range(len(kept)), key=lambda i: (-target_corr[kept[i]], i))
            accepted: list[int] = []
            for i in order:
                if all(corr[i, j] <= self.correlation_threshold for j in accepted):
                    accepted.append(i)
            kept = kept[sorted(accepted)]

        if self.method == "mutual_info":
            y_values, y_codes = np.unique(target, return_inverse=True)
            scores = np.array(
                [
                    _mutual_information(_discretize(A[:, j]), y_codes, len(y_values))
                    for j in kept
                ]
            )
        else:
            scores = target_corr[kept]
        self.scores_ = {columns[j]: float(s) for j, s in zip(kept, scores)}

        selected = list(range(len(kept)))
        if self.k is not None and self.k < len(kept):
            ranked = sorted(range(len(kept)), key=lambda i: (-scores[i], i))
            selected = sorted(ranked[: self.k])
        self.selected_features_ = [columns[kept[i]] for i in selected]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_fitted(self, "selected_features_")
        X = X.copy()
        return X[self.selected_features_]


class PCA(BaseTransformer):
    def __init__(self, n_components: int | float | None = None) -> None:
        if isinstance(n_components, bool):
            raise ValueError("n_components tidak boleh bool")
        if isinstance(n_components, (int, np.integer)) and n_components < 1:
            raise ValueError("n_components bertipe int harus >= 1")
        if isinstance(n_components, (float, np.floating)) and not 0.0 < n_components <= 1.0:
            raise ValueError("n_components bertipe float harus di antara 0 dan 1")
        self.n_components = n_components

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PCA":
        A = _numeric_matrix(X, "PCA")
        n_samples, n_features = A.shape
        max_components = min(n_samples, n_features)
        mean = A.mean(axis=0)
        _, singular, vt = np.linalg.svd(A - mean, full_matrices=False)

        variance = singular**2 / max(n_samples - 1, 1)
        total = float(variance.sum())
        ratio = variance / total if total > 0 else np.zeros_like(variance)

        if self.n_components is None:
            n_keep = max_components
        elif isinstance(self.n_components, (int, np.integer)):
            if self.n_components > max_components:
                raise ValueError(
                    f"n_components={self.n_components} melebihi min(n_samples, n_features)={max_components}"
                )
            n_keep = self.n_components
        else:
            cumulative = np.cumsum(ratio)
            n_keep = min(int(np.searchsorted(cumulative, self.n_components, side="left")) + 1, len(ratio))

        components = vt[:n_keep].copy()
        for i in range(n_keep):
            if components[i, np.argmax(np.abs(components[i]))] < 0:
                components[i] = -components[i]

        self.columns_ = list(X.columns)
        self.mean_ = mean
        self.components_ = components
        self.explained_variance_ = variance[:n_keep]
        self.explained_variance_ratio_ = ratio[:n_keep]
        self.n_components_ = n_keep
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _check_fitted(self, "components_")
        X = X.copy()
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"PCA membutuhkan kolom: {missing}")
        A = _numeric_matrix(X[self.columns_], "PCA")
        projected = (A - self.mean_) @ self.components_.T
        names = [f"pc{i + 1}" for i in range(self.n_components_)]
        return pd.DataFrame(projected, index=X.index, columns=names)
