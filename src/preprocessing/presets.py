import numpy as np
import pandas as pd

from src.preprocessing.base import Pipeline
from src.preprocessing.transformation import (
    PCA,
    FeatureEngineer,
    FeatureSelector,
    MissingIndicator,
    MissingValueImputer,
    OneHotEncoder,
    OutlierClipper,
    RareCategoryGrouper,
    StandardScaler,
    TreeInputFormatter,
)

_CLIP_COLS = [
    "nightly_rate_idr",
    "days_before_arrival",
    "weekend_nights",
    "weekday_nights",
    "num_children",
    "num_adults",
    "prior_cancellations",
    "prior_completed_bookings",
    "total_nights",
    "total_guests",
    "total_cost",
]


def _engineered_cols(numerical_cols: list[str], engineer_features: bool) -> list[str]:
    if engineer_features and set(FeatureEngineer.REQUIRED_COLS) <= set(numerical_cols):
        return list(FeatureEngineer.ENGINEERED_COLS)
    return []


def make_numeric_pipeline(
    numerical_cols: list[str],
    categorical_cols: list[str],
    use_pca: bool = False,
    engineer_features: bool = True,
    add_missing_indicators: bool = True,
    clip_outliers: bool = True,
    group_rare: bool = True,
    select_k: int | None = None,
    selection_method: str = "mutual_info",
    pca_components: int | float | None = 0.95,
) -> Pipeline:
    engineered = _engineered_cols(numerical_cols, engineer_features)
    all_numeric = list(numerical_cols) + engineered

    steps: list[tuple[str, object]] = []
    if engineered:
        steps.append(("engineer", FeatureEngineer()))
    if add_missing_indicators:
        steps.append(("missing_indicator", MissingIndicator(numerical_cols)))
    steps.append(("imputer", MissingValueImputer(all_numeric, categorical_cols)))
    if clip_outliers:
        clip_cols = [c for c in _CLIP_COLS if c in all_numeric]
        if clip_cols:
            steps.append(("clipper", OutlierClipper(clip_cols)))
    if group_rare and categorical_cols:
        steps.append(("rare_grouper", RareCategoryGrouper(categorical_cols)))
    steps.append(("scaler", StandardScaler(all_numeric)))
    steps.append(("encoder", OneHotEncoder(categorical_cols)))
    if select_k is not None:
        steps.append(
            (
                "selector",
                FeatureSelector(
                    method=selection_method, k=select_k, correlation_threshold=0.95
                ),
            )
        )
    if use_pca:
        steps.append(("pca", PCA(n_components=pca_components)))
    return Pipeline(steps)


def make_tree_pipeline(
    numerical_cols: list[str],
    categorical_cols: list[str],
    engineer_features: bool = True,
    group_rare: bool = True,
) -> Pipeline:
    engineered = _engineered_cols(numerical_cols, engineer_features)

    steps: list[tuple[str, object]] = []
    if engineered:
        steps.append(("engineer", FeatureEngineer()))
    if group_rare and categorical_cols:
        steps.append(("rare_grouper", RareCategoryGrouper(categorical_cols)))
    steps.append(
        ("formatter", TreeInputFormatter(list(numerical_cols) + engineered, categorical_cols))
    )
    return Pipeline(steps)


def to_model_input(
    X: pd.DataFrame, categorical_cols: list[str]
) -> tuple[np.ndarray, list[int]]:
    wanted = set(categorical_cols)
    indices = [i for i, col in enumerate(X.columns) if col in wanted]
    if indices:
        return X.to_numpy(dtype=object), indices
    return X.to_numpy(dtype=float), []
