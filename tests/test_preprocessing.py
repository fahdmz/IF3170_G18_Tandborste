import numpy as np
import pandas as pd
import pytest

from sklearn.decomposition import PCA as SklearnPCA

from src.data import CATEGORICAL_COLS, NUMERICAL_COLS
from src.preprocessing.cleaning import clean_train
from src.preprocessing.presets import (
    make_numeric_pipeline,
    make_tree_pipeline,
    to_model_input,
)
from src.preprocessing.resampling import RandomOverSampler, RandomUnderSampler, SMOTE
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

NUM = ["a", "b", "c"]
CAT = ["m", "r"]


@pytest.fixture
def train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [10, 20, 30, 40, 50],
            "c": [3.0, 3.0, 3.0, 3.0, 3.0],
            "m": pd.Series(["x", "y", None, "x", "z"], dtype="string"),
            "r": pd.Series(["p", "q", "p", pd.NA, "q"], dtype="string"),
        }
    )


@pytest.fixture
def test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [np.nan, 100.0, 200.0],
            "b": [np.nan, 5, 6],
            "c": [3.0, np.nan, 9.0],
            "m": pd.Series(["w", "x", None], dtype="string"),
            "r": pd.Series(["p", "q", "q"], dtype="string"),
        }
    )


def _all_transformers() -> list:
    return [
        MissingValueImputer(NUM, CAT),
        StandardScaler(NUM),
        OneHotEncoder(CAT),
        TreeInputFormatter(NUM, CAT),
    ]


def test_imputer_no_nan(train: pd.DataFrame, test: pd.DataFrame) -> None:
    imp = MissingValueImputer(NUM, CAT).fit(train)
    for df in (imp.transform(train), imp.transform(test)):
        assert int(df.isna().sum().sum()) == 0


def test_imputer_uses_train_median(train: pd.DataFrame, test: pd.DataFrame) -> None:
    imp = MissingValueImputer(NUM, CAT).fit(train)
    out = imp.transform(test)
    assert out.loc[0, "a"] == 3.0
    assert out.loc[0, "b"] == 30.0
    assert out.loc[1, "c"] == 3.0
    assert out.loc[2, "m"] == "Missing"


def test_imputer_mean_strategy(train: pd.DataFrame) -> None:
    imp = MissingValueImputer(NUM, CAT, numerical_strategy="mean").fit(train)
    assert imp.transform(train).loc[2, "a"] == pytest.approx(3.0)


def test_scaler_train_mean_zero_and_zero_std(train: pd.DataFrame, test: pd.DataFrame) -> None:
    filled = MissingValueImputer(NUM, CAT).fit_transform(train)
    sc = StandardScaler(NUM).fit(filled)
    out = sc.transform(filled)
    for col in ("a", "b"):
        assert out[col].mean() == pytest.approx(0.0, abs=1e-9)
        assert out[col].std(ddof=0) == pytest.approx(1.0)
    assert np.isfinite(out["c"].to_numpy()).all()
    assert (out["c"] == 0.0).all()
    out_test = sc.transform(MissingValueImputer(NUM, CAT).fit(train).transform(test))
    assert np.isfinite(out_test[NUM].to_numpy()).all()


def test_scaler_only_touches_given_cols(train: pd.DataFrame) -> None:
    out = StandardScaler(["a"]).fit_transform(train)
    assert (out["b"] == train["b"]).all()


def test_encoder_columns_and_unknown(train: pd.DataFrame, test: pd.DataFrame) -> None:
    filled = MissingValueImputer(NUM, CAT).fit_transform(train)
    enc = OneHotEncoder(CAT).fit(filled)
    out_train = enc.transform(filled)
    out_test = enc.transform(test)
    assert list(out_train.columns) == list(out_test.columns)
    expected = [
        "m__Missing",
        "m__x",
        "m__y",
        "m__z",
        "r__Missing",
        "r__p",
        "r__q",
    ]
    assert [c for c in out_train.columns if "__" in c] == expected
    assert "m" not in out_train.columns and "r" not in out_train.columns
    m_cols = [c for c in out_test.columns if c.startswith("m__")]
    assert (out_test.loc[0, m_cols] == 0.0).all()
    assert (out_test.loc[2, m_cols] == 0.0).all()
    assert out_test.loc[1, "m__x"] == 1.0
    for c in expected:
        assert out_train[c].dtype == np.float64
        assert set(out_train[c].unique()) <= {0.0, 1.0}


def test_encoder_each_row_sums_to_one_on_known(train: pd.DataFrame) -> None:
    filled = MissingValueImputer(NUM, CAT).fit_transform(train)
    out = OneHotEncoder(CAT).fit_transform(filled)
    assert (out[[c for c in out.columns if c.startswith("m__")]].sum(axis=1) == 1.0).all()


@pytest.mark.parametrize("idx", range(4))
def test_transformers_preserve_rows_and_input(
    idx: int, train: pd.DataFrame, test: pd.DataFrame
) -> None:
    t = _all_transformers()[idx]
    t.fit(train)
    before = train.copy(deep=True)
    before_test = test.copy(deep=True)
    out_train = t.transform(train)
    out_test = t.transform(test)
    assert len(out_train) == len(train)
    assert len(out_test) == len(test)
    pd.testing.assert_frame_equal(train, before)
    pd.testing.assert_frame_equal(test, before_test)


@pytest.mark.parametrize("idx", range(4))
def test_transformers_require_fit(idx: int, train: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError):
        _all_transformers()[idx].transform(train)


def test_fit_returns_self(train: pd.DataFrame) -> None:
    for t in _all_transformers():
        assert t.fit(train) is t


def test_numeric_pipeline(train: pd.DataFrame, test: pd.DataFrame) -> None:
    p = make_numeric_pipeline(NUM, CAT)
    a = p.fit_transform(train)
    b = p.transform(test)
    assert int(a.isna().sum().sum()) == 0
    assert int(b.isna().sum().sum()) == 0
    assert all(dt == np.float64 for dt in a.dtypes)
    assert all(dt == np.float64 for dt in b.dtypes)
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(train) and len(b) == len(test)


def test_tree_pipeline(train: pd.DataFrame, test: pd.DataFrame) -> None:
    shuffled = train[["m", "c", "r", "b", "a"]]
    p = make_tree_pipeline(NUM, CAT)
    a = p.fit_transform(shuffled)
    b = p.transform(test)
    assert list(a.columns) == NUM + CAT
    assert list(b.columns) == NUM + CAT
    assert int(a.isna().sum().sum()) > 0
    assert np.isnan(a.loc[2, "a"])
    assert a["m"].dtype == object and a["r"].dtype == object
    assert a["a"].dtype == np.float64 and a["b"].dtype == np.float64
    for col in CAT:
        assert all(isinstance(v, str) or v is np.nan or (isinstance(v, float) and np.isnan(v)) for v in a[col])
        assert not any(v is pd.NA for v in a[col])
    assert a.loc[2, "m"] is np.nan or np.isnan(a.loc[2, "m"])


def test_to_model_input_with_categorical(train: pd.DataFrame) -> None:
    X = make_tree_pipeline(NUM, CAT).fit_transform(train)
    arr, idx = to_model_input(X, CAT)
    assert arr.dtype == object
    assert idx == [3, 4]
    assert arr.shape == X.shape


def test_to_model_input_without_categorical(train: pd.DataFrame) -> None:
    X = make_numeric_pipeline(NUM, CAT).fit_transform(train)
    arr, idx = to_model_input(X, CAT)
    assert arr.dtype == np.float64
    assert idx == []
    assert arr.shape == X.shape


def _hotel_frame(n: int, seed: int, with_rare: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "num_adults": rng.choice([1.0, 2.0, 3.0], size=n),
            "num_children": rng.choice([0.0, 0.0, 0.0, 1.0, 2.0], size=n),
            "weekend_nights": rng.integers(0, 3, size=n).astype(float),
            "weekday_nights": rng.integers(0, 6, size=n).astype(float),
            "needs_parking": rng.integers(0, 2, size=n).astype(float),
            "days_before_arrival": rng.integers(0, 300, size=n).astype(float),
            "arrival_year": rng.choice([2023.0, 2024.0], size=n),
            "arrival_month": rng.integers(1, 13, size=n).astype(float),
            "arrival_day": rng.integers(1, 29, size=n).astype(float),
            "is_returning_guest": rng.integers(0, 2, size=n).astype(float),
            "prior_cancellations": rng.choice([0.0, 0.0, 0.0, 1.0, 5.0], size=n),
            "prior_completed_bookings": rng.choice([0.0, 0.0, 1.0, 2.0], size=n),
            "nightly_rate_idr": rng.integers(500, 3000, size=n) * 1000.0,
            "special_requests": rng.integers(0, 4, size=n).astype(float),
            "meal_package": rng.choice(["Breakfast", "Half Board", "Room Only"], size=n).astype(object),
            "room_category": rng.choice(["Type A", "Type B", "Type C"], size=n).astype(object),
            "booking_channel": rng.choice(["Airline", "Corporate", "Offline Agent"], size=n).astype(object),
        }
    )
    df.loc[0, "nightly_rate_idr"] = 0.0
    df.loc[1, "nightly_rate_idr"] = 9e9
    for col, rows in [
        ("num_adults", [2, 5]),
        ("num_children", [3, 7, 9]),
        ("nightly_rate_idr", [4]),
        ("special_requests", [6, 8, 10]),
        ("meal_package", [11, 12]),
        ("room_category", [13]),
    ]:
        df.loc[rows, col] = np.nan
    if with_rare:
        df.loc[[14, 15], "meal_package"] = "Full Board"
        df.loc[[16, 17, 18], "room_category"] = "Type E"
    return df


@pytest.fixture
def hotel_train() -> pd.DataFrame:
    return _hotel_frame(300, seed=1)


@pytest.fixture
def hotel_test() -> pd.DataFrame:
    df = _hotel_frame(120, seed=2, with_rare=False)
    df.loc[0, "room_category"] = "Type Z"
    df.loc[1, "meal_package"] = "Brunch"
    df.loc[2, ["days_before_arrival", "needs_parking", "prior_cancellations"]] = np.nan
    return df


@pytest.fixture
def hotel_target() -> pd.Series:
    return pd.Series(np.random.default_rng(3).integers(0, 2, size=300))


def test_clean_train_removes_duplicates_and_zero_guests() -> None:
    X = pd.DataFrame(
        {
            "num_adults": [2.0, 2.0, 2.0, 0.0, 0.0, np.nan, 1.0, 1.0],
            "num_children": [1.0, 1.0, 1.0, 0.0, np.nan, np.nan, 0.0, 0.0],
            "nightly_rate_idr": [100.0, 100.0, 100.0, 50.0, 70.0, np.nan, np.nan, np.nan],
            "meal_package": ["Breakfast", "Breakfast", "Breakfast", "Breakfast", None, None, None, None],
        },
        index=[10, 11, 12, 13, 14, 15, 16, 17],
    )
    y = pd.Series([0, 0, 1, 0, 1, 1, 0, 0], index=X.index, name="is_canceled")
    Xc, yc = clean_train(X, y)
    assert len(Xc) == len(yc) == 5
    assert list(Xc.index) == list(range(5)) and list(yc.index) == list(range(5))
    assert yc.name == "is_canceled"
    assert yc.tolist() == [0, 1, 1, 1, 0]
    assert not ((Xc["num_adults"] == 0) & (Xc["num_children"] == 0)).any()
    assert len(X) == 8 and list(X.index) == [10, 11, 12, 13, 14, 15, 16, 17]


def test_clean_train_keeps_nan_guest_rows_and_checks_length() -> None:
    X = pd.DataFrame({"num_adults": [np.nan, 0.0], "num_children": [np.nan, np.nan]})
    Xc, yc = clean_train(X, pd.Series([0, 1]))
    assert len(Xc) == 2 and len(yc) == 2
    with pytest.raises(ValueError):
        clean_train(X, pd.Series([0, 1, 1]))


def _booking_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "weekend_nights": [1.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "weekday_nights": [2.0, 3.0, 0.0, 1.0, np.nan, 1.0, 1.0, 1.0],
            "num_adults": [2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            "num_children": [1.0, 0.0, 0.0, 2.0, 0.0, np.nan, 0.0, 0.0],
            "nightly_rate_idr": [1000.0, 0.0, 500.0, 200.0, 300.0, 400.0, np.nan, 600.0],
            "arrival_year": [2024.0, 2024.0, 2023.0, 2023.0, 2024.0, 2024.0, 2024.0, 2024.0],
            "arrival_month": [4.0, 1.0, 8.0, 8.0, 4.0, 4.0, 4.0, 4.0],
            "arrival_day": [3.0, 1.0, 5.0, 6.0, 3.0, 3.0, 3.0, np.nan],
        }
    )


def test_feature_engineer_values() -> None:
    X = _booking_frame()
    fe = FeatureEngineer().fit(X)
    out = fe.transform(X)
    assert list(out.columns) == list(X.columns) + FeatureEngineer.ENGINEERED_COLS
    assert out.loc[0, "total_nights"] == 3.0
    assert out.loc[1, "total_nights"] == 5.0
    assert out.loc[0, "total_guests"] == 3.0
    assert out.loc[2, "total_guests"] == 2.0
    assert out["has_children"].tolist()[:4] == [1.0, 0.0, 0.0, 1.0]
    assert out["is_free"].tolist()[:4] == [0.0, 1.0, 0.0, 0.0]
    assert out.loc[0, "total_cost"] == 3000.0
    assert out.loc[1, "total_cost"] == 0.0
    assert out.loc[3, "total_cost"] == 400.0
    assert out["arrival_dayofweek"].tolist()[:4] == [2.0, 0.0, 5.0, 6.0]
    assert out["arrival_is_weekend"].tolist()[:4] == [0.0, 0.0, 1.0, 1.0]
    for col in FeatureEngineer.ENGINEERED_COLS:
        assert out[col].dtype == np.float64


def test_feature_engineer_nan_inputs_give_nan() -> None:
    X = _booking_frame()
    out = FeatureEngineer().fit(X).transform(X)
    assert np.isnan(out.loc[4, "total_nights"]) and np.isnan(out.loc[4, "total_cost"])
    assert not np.isnan(out.loc[4, "total_guests"])
    assert np.isnan(out.loc[5, "total_guests"]) and np.isnan(out.loc[5, "has_children"])
    assert np.isnan(out.loc[6, "is_free"]) and np.isnan(out.loc[6, "total_cost"])
    assert np.isnan(out.loc[7, "arrival_dayofweek"]) and np.isnan(out.loc[7, "arrival_is_weekend"])
    assert out.loc[6, "total_nights"] == 2.0


def test_feature_engineer_missing_columns_raises() -> None:
    with pytest.raises(ValueError):
        FeatureEngineer().fit(pd.DataFrame({"a": [1.0]}))


def test_missing_indicator_only_for_cols_missing_in_train(train: pd.DataFrame, test: pd.DataFrame) -> None:
    mi = MissingIndicator(["a", "b", "c", "m", "r"]).fit(train)
    assert mi.missing_cols_ == ["a", "m", "r"]
    assert mi.feature_names_ == ["a__was_missing", "m__was_missing", "r__was_missing"]
    out_train = mi.transform(train)
    out_test = mi.transform(test)
    assert list(out_train.columns) == list(out_test.columns)
    assert "b__was_missing" not in out_test.columns
    assert out_train["a__was_missing"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert out_train["m__was_missing"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert out_test["a__was_missing"].tolist() == [1.0, 0.0, 0.0]
    assert out_test["m__was_missing"].tolist() == [0.0, 0.0, 1.0]
    assert out_test["a__was_missing"].dtype == np.float64
    assert out_test["a"].isna().sum() == 1


def test_outlier_clipper_bounds_from_train_and_nan_kept() -> None:
    train_a = pd.DataFrame({"a": list(range(101)) + [np.nan], "b": [1.0] * 102})
    clipper = OutlierClipper(["a"], lower_quantile=0.05, upper_quantile=0.9).fit(train_a)
    assert clipper.lower_["a"] == pytest.approx(5.0)
    assert clipper.upper_["a"] == pytest.approx(90.0)
    test_a = pd.DataFrame({"a": [-100.0, 50.0, 1000.0, np.nan], "b": [500.0] * 4})
    out = clipper.transform(test_a)
    assert out["a"].iloc[:3].tolist() == [5.0, 50.0, 90.0]
    assert np.isnan(out["a"].iloc[3])
    assert (out["b"] == 500.0).all()


def test_outlier_clipper_default_quantiles() -> None:
    s = pd.DataFrame({"a": np.arange(1000, dtype=float)})
    clipper = OutlierClipper(["a"]).fit(s)
    assert clipper.lower_["a"] == 0.0
    assert clipper.upper_["a"] == pytest.approx(np.quantile(np.arange(1000), 0.99))
    assert clipper.transform(s)["a"].max() == pytest.approx(clipper.upper_["a"])


def test_rare_category_grouper() -> None:
    train_m = pd.DataFrame(
        {"m": ["x"] * 25 + ["y"] * 25 + ["z"] * 3 + ["Missing"] * 2 + [None] * 2}
    )
    g = RareCategoryGrouper(["m"], min_count=20).fit(train_m)
    test_m = pd.DataFrame({"m": ["x", "z", "w", "Missing", None, "y"]})
    out = g.transform(test_m)["m"].tolist()
    assert out[:4] == ["x", "Other", "Other", "Missing"]
    assert pd.isna(out[4])
    assert out[5] == "y"
    assert g.transform(train_m)["m"].tolist().count("Other") == 3
    custom = RareCategoryGrouper(["m"], min_count=20, other_label="Langka").fit(train_m)
    assert "Langka" in custom.transform(test_m)["m"].tolist()


def test_new_transformers_fit_rules(hotel_train: pd.DataFrame) -> None:
    factories = [
        FeatureEngineer,
        lambda: MissingIndicator(["num_adults", "special_requests"]),
        lambda: OutlierClipper(["nightly_rate_idr", "num_children"]),
        lambda: RareCategoryGrouper(CATEGORICAL_COLS),
    ]
    for factory in factories:
        with pytest.raises(RuntimeError):
            factory().transform(hotel_train)
        t = factory()
        assert t.fit(hotel_train) is t
        before = hotel_train.copy(deep=True)
        out = t.transform(hotel_train)
        assert len(out) == len(hotel_train)
        pd.testing.assert_frame_equal(hotel_train, before)


def _selector_data(n: int = 600) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=n)
    informative = y + 0.5 * rng.normal(size=n)
    X = pd.DataFrame(
        {
            "noise1": rng.normal(size=n),
            "informative": informative,
            "const": np.full(n, 3.0),
            "noise2": rng.normal(size=n),
            "informative_dup": informative.copy(),
            "noise3": rng.normal(size=n),
        }
    )
    return X, pd.Series(y)


def test_feature_selector_drops_constant_and_duplicate() -> None:
    X, y = _selector_data()
    sel = FeatureSelector().fit(X, y)
    assert "const" not in sel.selected_features_
    assert ("informative" in sel.selected_features_) != ("informative_dup" in sel.selected_features_)
    assert set(sel.selected_features_) == {"noise1", "informative", "noise2", "noise3"}
    assert list(sel.transform(X).columns) == sel.selected_features_
    assert set(sel.scores_) == set(sel.selected_features_)


def test_feature_selector_correlation_filter_keeps_higher_target_correlation() -> None:
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.integers(0, 2, size=n)
    a = y + 0.5 * rng.normal(size=n)
    b = a - 0.3 * y
    X = pd.DataFrame({"b": b, "a": a})
    assert abs(np.corrcoef(a, b)[0, 1]) > 0.95
    assert abs(np.corrcoef(a, y)[0, 1]) > abs(np.corrcoef(b, y)[0, 1])
    sel = FeatureSelector(correlation_threshold=0.95).fit(X, pd.Series(y))
    assert sel.selected_features_ == ["a"]
    keep_both = FeatureSelector(correlation_threshold=None).fit(X, pd.Series(y))
    assert keep_both.selected_features_ == ["b", "a"]


def test_feature_selector_variance_threshold() -> None:
    X, y = _selector_data()
    X["tiny"] = X["noise1"] * 0.01
    sel = FeatureSelector(variance_threshold=0.001, correlation_threshold=None).fit(X, y)
    assert "tiny" not in sel.selected_features_
    assert "const" not in sel.selected_features_


@pytest.mark.parametrize("method", ["mutual_info", "correlation"])
def test_feature_selector_k_and_informative_first(method: str) -> None:
    X, y = _selector_data()
    sel = FeatureSelector(method=method, k=1).fit(X, y)
    assert len(sel.selected_features_) == 1
    assert sel.selected_features_[0] in ("informative", "informative_dup")
    sel3 = FeatureSelector(method=method, k=3).fit(X, y)
    assert len(sel3.selected_features_) == 3
    order = [c for c in X.columns if c in sel3.selected_features_]
    assert sel3.selected_features_ == order
    assert sel3.scores_["informative" if "informative" in sel3.scores_ else "informative_dup"] == max(
        sel3.scores_.values()
    )
    assert len(FeatureSelector(method=method, k=100).fit(X, y).selected_features_) == 4


def test_feature_selector_mutual_info_catches_nonlinear() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    x = rng.normal(size=n)
    X = pd.DataFrame({"noise": rng.normal(size=n), "quad": x})
    y = pd.Series((np.abs(x) > 1.0).astype(int))
    sel = FeatureSelector(method="mutual_info", k=1).fit(X, y)
    assert sel.selected_features_ == ["quad"]
    assert sel.scores_["quad"] > 5 * sel.scores_["noise"]


def test_feature_selector_errors() -> None:
    X, y = _selector_data()
    with pytest.raises(ValueError):
        FeatureSelector().fit(X, None)
    with pytest.raises(ValueError):
        FeatureSelector().fit(X.assign(noise1=np.nan), y)
    with pytest.raises(ValueError):
        FeatureSelector(method="chi2")
    with pytest.raises(RuntimeError):
        FeatureSelector().transform(X)


def _pca_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(5)
    mixing = rng.normal(size=(6, 6))
    train_x = pd.DataFrame(rng.normal(size=(200, 6)) @ mixing + 5.0, columns=list("abcdef"))
    test_x = pd.DataFrame(rng.normal(size=(30, 6)) @ mixing + 7.0, columns=list("abcdef"), index=range(100, 130))
    return train_x, test_x


def test_pca_matches_sklearn() -> None:
    train_x, test_x = _pca_data()
    ours = PCA(n_components=4).fit(train_x)
    ref = SklearnPCA(n_components=4).fit(train_x.to_numpy())
    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_)
    np.testing.assert_allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_)
    np.testing.assert_allclose(ours.mean_, ref.mean_)
    signs = np.sign((ours.components_ * ref.components_).sum(axis=1))
    np.testing.assert_allclose(ours.components_, ref.components_ * signs[:, None], atol=1e-8)
    assert ours.n_components_ == 4
    out = ours.transform(test_x)
    np.testing.assert_allclose(out.to_numpy(), ref.transform(test_x.to_numpy()) * signs, atol=1e-8)
    assert list(out.columns) == ["pc1", "pc2", "pc3", "pc4"]
    assert list(out.index) == list(test_x.index)


def test_pca_transform_uses_train_mean() -> None:
    train_x, test_x = _pca_data()
    pca = PCA().fit(train_x)
    expected = (test_x.to_numpy() - train_x.to_numpy().mean(axis=0)) @ pca.components_.T
    np.testing.assert_allclose(pca.transform(test_x).to_numpy(), expected)
    assert pca.n_components_ == 6
    assert not np.allclose(pca.mean_, test_x.to_numpy().mean(axis=0))


def test_pca_float_components_and_deterministic_sign() -> None:
    train_x, _ = _pca_data()
    pca = PCA(n_components=0.9).fit(train_x)
    full = PCA().fit(train_x)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    n = pca.n_components_
    assert cumulative[n - 1] >= 0.9
    assert n == 1 or cumulative[n - 2] < 0.9
    assert pca.components_.shape == (n, 6)
    assert pca.explained_variance_ratio_.shape == (n,)
    for comp in full.components_:
        assert comp[np.argmax(np.abs(comp))] > 0
    again = PCA(n_components=0.9).fit(train_x.sample(frac=1.0, random_state=0))
    np.testing.assert_allclose(again.components_, pca.components_, atol=1e-8)
    assert PCA(n_components=1.0).fit(train_x).n_components_ == 6


def test_pca_errors() -> None:
    train_x, _ = _pca_data()
    with pytest.raises(RuntimeError):
        PCA().transform(train_x)
    with pytest.raises(ValueError):
        PCA().fit(train_x.where(train_x["a"] > 0))
    with pytest.raises(ValueError):
        PCA(n_components=10).fit(train_x)
    with pytest.raises(ValueError):
        PCA(n_components=0)
    with pytest.raises(ValueError):
        PCA(n_components=1.5)
    before = train_x.copy(deep=True)
    PCA(n_components=2).fit(train_x).transform(train_x)
    pd.testing.assert_frame_equal(train_x, before)


def _imbalanced(n_major: int = 100, n_minor: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n_major + n_minor, 3))
    y = np.array([0] * n_major + [1] * n_minor)
    return X, y


def _row_set(X: np.ndarray) -> set[tuple[float, ...]]:
    return {tuple(row) for row in X}


@pytest.mark.parametrize("strategy", [1.0, 0.5])
def test_under_sampler_ratio(strategy: float) -> None:
    X, y = _imbalanced()
    Xr, yr = RandomUnderSampler(sampling_strategy=strategy).fit_resample(X, y)
    assert int((yr == 1).sum()) == 20
    assert int((yr == 0).sum()) == int(round(20 / strategy))
    assert len(Xr) == len(yr)
    assert _row_set(X[y == 1]) <= _row_set(Xr)
    assert _row_set(Xr) <= _row_set(X)
    assert len(_row_set(Xr)) == len(Xr)


@pytest.mark.parametrize("strategy", [1.0, 0.5])
def test_over_sampler_ratio(strategy: float) -> None:
    X, y = _imbalanced()
    Xr, yr = RandomOverSampler(sampling_strategy=strategy).fit_resample(X, y)
    assert int((yr == 0).sum()) == 100
    assert int((yr == 1).sum()) == int(round(strategy * 100))
    assert _row_set(X) <= _row_set(Xr)
    assert _row_set(Xr) <= _row_set(X)
    for row, label in zip(Xr, yr):
        assert label == y[np.flatnonzero((X == row).all(axis=1))[0]]


@pytest.mark.parametrize("strategy", [1.0, 0.5])
def test_smote_ratio_and_originals_kept(strategy: float) -> None:
    X, y = _imbalanced()
    Xr, yr = SMOTE(sampling_strategy=strategy).fit_resample(X, y)
    assert int((yr == 0).sum()) == 100
    assert int((yr == 1).sum()) == int(round(strategy * 100))
    assert _row_set(X) <= _row_set(Xr)
    assert Xr.dtype == np.float64 and Xr.shape[1] == 3
    synthetic = [row for row in Xr if tuple(row) not in _row_set(X)]
    assert len(synthetic) == int(round(strategy * 100)) - 20


@pytest.mark.parametrize("cls", [RandomUnderSampler, RandomOverSampler, SMOTE])
def test_resamplers_deterministic(cls: type) -> None:
    X, y = _imbalanced()
    a = cls(random_state=3).fit_resample(X, y)
    b = cls(random_state=3).fit_resample(X, y)
    c = cls(random_state=4).fit_resample(X, y)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])
    assert not np.array_equal(a[1], np.sort(a[1]))


@pytest.mark.parametrize("cls", [RandomUnderSampler, RandomOverSampler, SMOTE])
def test_resamplers_do_not_modify_input_and_validate(cls: type) -> None:
    X, y = _imbalanced()
    X0, y0 = X.copy(), y.copy()
    cls().fit_resample(X, y)
    np.testing.assert_array_equal(X, X0)
    np.testing.assert_array_equal(y, y0)
    with pytest.raises(ValueError):
        cls(sampling_strategy=0.0).fit_resample(X, y)
    with pytest.raises(ValueError):
        cls().fit_resample(X, np.zeros(len(y), dtype=int))


def test_smote_synthetic_between_two_minority_samples() -> None:
    minority = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    X = np.concatenate([minority, np.full(30, 100.0)]).reshape(-1, 1)
    y = np.array([1] * 6 + [0] * 30)
    Xr, yr = SMOTE(k_neighbors=2).fit_resample(X, y)
    assert int((yr == 1).sum()) == 30 and int((yr == 0).sum()) == 30
    values = Xr[yr == 1, 0]
    synthetic = values[~np.isin(values, minority)]
    assert len(synthetic) == 24
    in_left = (synthetic >= 0.0) & (synthetic <= 2.0)
    in_right = (synthetic >= 10.0) & (synthetic <= 12.0)
    assert (in_left | in_right).all()
    assert in_left.any() and in_right.any()
    assert (Xr[yr == 0, 0] == 100.0).all()


def test_smote_batching_does_not_change_result() -> None:
    X, y = _imbalanced()
    big = SMOTE(batch_size=1024, random_state=1).fit_resample(X, y)
    small = SMOTE(batch_size=7, random_state=1).fit_resample(X, y)
    np.testing.assert_allclose(big[0], small[0])
    np.testing.assert_array_equal(big[1], small[1])


def test_smote_fewer_minority_than_k_and_too_few() -> None:
    X = np.arange(33, dtype=float).reshape(-1, 1)
    y = np.array([1] * 3 + [0] * 30)
    Xr, yr = SMOTE(k_neighbors=5).fit_resample(X, y)
    assert int((yr == 1).sum()) == 30
    with pytest.raises(ValueError):
        SMOTE().fit_resample(X, np.array([1] + [0] * 32))


NUMERIC_PIPELINE_FLAGS = [
    {},
    {"engineer_features": False},
    {"add_missing_indicators": False},
    {"clip_outliers": False},
    {"group_rare": False},
    {
        "engineer_features": False,
        "add_missing_indicators": False,
        "clip_outliers": False,
        "group_rare": False,
    },
    {"select_k": 10},
    {"select_k": 5, "selection_method": "correlation"},
    {"use_pca": True},
    {"use_pca": True, "pca_components": 3},
    {"use_pca": True, "pca_components": None},
    {"select_k": 12, "use_pca": True, "pca_components": 0.9},
]


@pytest.mark.parametrize("flags", NUMERIC_PIPELINE_FLAGS)
def test_numeric_pipeline_flag_combinations(
    flags: dict, hotel_train: pd.DataFrame, hotel_test: pd.DataFrame, hotel_target: pd.Series
) -> None:
    p = make_numeric_pipeline(NUMERICAL_COLS, CATEGORICAL_COLS, **flags)
    a = p.fit_transform(hotel_train, hotel_target)
    b = p.transform(hotel_test)
    assert int(a.isna().sum().sum()) == 0
    assert int(b.isna().sum().sum()) == 0
    assert all(dt == np.float64 for dt in a.dtypes)
    assert all(dt == np.float64 for dt in b.dtypes)
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(hotel_train) and len(b) == len(hotel_test)
    arr, idx = to_model_input(a, CATEGORICAL_COLS)
    assert arr.dtype == np.float64 and idx == [] and arr.shape == a.shape
    if flags.get("select_k") is not None and not flags.get("use_pca"):
        assert a.shape[1] <= flags["select_k"]
    if flags.get("use_pca"):
        assert all(c.startswith("pc") for c in a.columns)
        if flags.get("pca_components") == 3:
            assert a.shape[1] == 3


def test_numeric_pipeline_default_columns(
    hotel_train: pd.DataFrame, hotel_test: pd.DataFrame, hotel_target: pd.Series
) -> None:
    p = make_numeric_pipeline(NUMERICAL_COLS, CATEGORICAL_COLS)
    a = p.fit_transform(hotel_train, hotel_target)
    cols = list(a.columns)
    for col in FeatureEngineer.ENGINEERED_COLS:
        assert col in cols
    assert "num_adults__was_missing" in cols and "special_requests__was_missing" in cols
    assert "needs_parking__was_missing" not in cols
    assert "meal_package__Other" in cols and "room_category__Other" in cols
    assert "meal_package__Full Board" not in cols and "room_category__Type E" not in cols
    assert "meal_package__Missing" in cols
    for col in ["total_cost", "total_nights", "nightly_rate_idr"]:
        assert a[col].mean() == pytest.approx(0.0, abs=1e-9)
    for col in [c for c in cols if c.endswith("__was_missing")]:
        assert set(a[col].unique()) <= {0.0, 1.0}
    b = p.transform(hotel_test)
    assert b.loc[0, "room_category__Other"] == 1.0
    assert b.loc[1, "meal_package__Other"] == 1.0


def test_numeric_pipeline_clips_with_train_bounds(hotel_train: pd.DataFrame, hotel_target: pd.Series) -> None:
    clipped = make_numeric_pipeline(NUMERICAL_COLS, CATEGORICAL_COLS).fit_transform(hotel_train, hotel_target)
    raw = make_numeric_pipeline(NUMERICAL_COLS, CATEGORICAL_COLS, clip_outliers=False).fit_transform(
        hotel_train, hotel_target
    )
    assert clipped["nightly_rate_idr"].max() < raw["nightly_rate_idr"].max()


def test_numeric_pipeline_does_not_mutate_input(
    hotel_train: pd.DataFrame, hotel_test: pd.DataFrame, hotel_target: pd.Series
) -> None:
    before_train, before_test = hotel_train.copy(deep=True), hotel_test.copy(deep=True)
    p = make_numeric_pipeline(NUMERICAL_COLS, CATEGORICAL_COLS, select_k=10)
    p.fit_transform(hotel_train, hotel_target)
    p.transform(hotel_test)
    pd.testing.assert_frame_equal(hotel_train, before_train)
    pd.testing.assert_frame_equal(hotel_test, before_test)


def test_tree_pipeline_keeps_nan_and_groups_rare(
    hotel_train: pd.DataFrame, hotel_test: pd.DataFrame, hotel_target: pd.Series
) -> None:
    p = make_tree_pipeline(NUMERICAL_COLS, CATEGORICAL_COLS)
    a = p.fit_transform(hotel_train, hotel_target)
    b = p.transform(hotel_test)
    expected = NUMERICAL_COLS + FeatureEngineer.ENGINEERED_COLS + CATEGORICAL_COLS
    assert list(a.columns) == expected and list(b.columns) == expected
    assert int(a.isna().sum().sum()) > 0 and int(b.isna().sum().sum()) > 0
    assert a["meal_package"].isna().sum() == 2
    assert set(a["meal_package"].dropna()) == {"Breakfast", "Half Board", "Room Only", "Other"}
    assert set(a["room_category"].dropna()) == {"Type A", "Type B", "Type C", "Other"}
    assert (a.loc[[14, 15], "meal_package"] == "Other").all()
    assert (a.loc[[16, 17, 18], "room_category"] == "Other").all()
    assert b.loc[0, "room_category"] == "Other" and b.loc[1, "meal_package"] == "Other"
    assert np.isnan(a.loc[4, "total_cost"]) and np.isnan(a.loc[3, "total_guests"])
    assert all(a[c].dtype == np.float64 for c in NUMERICAL_COLS + FeatureEngineer.ENGINEERED_COLS)
    assert "Missing" not in set(a["meal_package"].dropna())
    arr, idx = to_model_input(a, CATEGORICAL_COLS)
    assert arr.dtype == object and idx == [len(expected) - 3, len(expected) - 2, len(expected) - 1]


def test_tree_pipeline_flags_off(hotel_train: pd.DataFrame) -> None:
    p = make_tree_pipeline(NUMERICAL_COLS, CATEGORICAL_COLS, engineer_features=False, group_rare=False)
    a = p.fit_transform(hotel_train)
    assert list(a.columns) == NUMERICAL_COLS + CATEGORICAL_COLS
    assert "Full Board" in set(a["meal_package"].dropna())
