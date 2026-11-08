import pandas as pd

from src import config

NUMERICAL_COLS = [
    "num_adults",
    "num_children",
    "weekend_nights",
    "weekday_nights",
    "needs_parking",
    "days_before_arrival",
    "arrival_year",
    "arrival_month",
    "arrival_day",
    "is_returning_guest",
    "prior_cancellations",
    "prior_completed_bookings",
    "nightly_rate_idr",
    "special_requests",
]
CATEGORICAL_COLS = ["meal_package", "room_category", "booking_channel"]


def load_train() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(config.TRAIN_PATH)
    X = df.drop(columns=[config.ID_COL, config.TARGET_COL])
    y = df[config.TARGET_COL]
    return X, y


def load_test() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(config.TEST_PATH)
    X = df.drop(columns=[config.ID_COL])
    ids = df[config.ID_COL]
    return X, ids


def get_feature_types() -> tuple[list[str], list[str]]:
    return NUMERICAL_COLS, CATEGORICAL_COLS
