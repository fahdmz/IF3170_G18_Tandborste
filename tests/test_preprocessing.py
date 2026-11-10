import numpy as np
import pandas as pd
import pytest

from src.preprocessing.presets import (
    make_numeric_pipeline,
    make_tree_pipeline,
    to_model_input,
)
from src.preprocessing.transformation import (
    MissingValueImputer,
    OneHotEncoder,
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


def test_numeric_pipeline_pca_not_available() -> None:
    with pytest.raises(NotImplementedError):
        make_numeric_pipeline(NUM, CAT, use_pca=True)


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
