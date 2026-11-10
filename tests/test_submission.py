from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.submission import make_submission, validate_submission


class FakePipeline:
    def __init__(self) -> None:
        self.fit_calls = 0
        self.transform_calls = 0

    def fit(self, X, y=None):
        self.fit_calls += 1
        return self

    def fit_transform(self, X, y=None):
        self.fit_calls += 1
        return X

    def transform(self, X):
        self.transform_calls += 1
        return X[["a", "b"]].astype(float)


class FakeModel:
    def __init__(self, output=None) -> None:
        self.output = output
        self.seen: np.ndarray | None = None

    def predict(self, X):
        self.seen = X
        if self.output is not None:
            return np.asarray(self.output)
        return (X[:, 0] > 1.5).astype(float)


@pytest.fixture
def X_test() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3, 0], "b": [5, 6, 7, 8], "c": ["x", "y", "z", "w"]})


@pytest.fixture
def ids() -> pd.Series:
    return pd.Series(["id1", "id2", "id3", "id4"], name=config.ID_COL)


def test_make_submission_columns_and_dtype(tmp_path: Path, X_test, ids) -> None:
    path = tmp_path / "sub" / "nested" / "out.csv"
    result = make_submission(FakeModel(), FakePipeline(), X_test, ids, path)

    assert list(result.columns) == [config.ID_COL, config.TARGET_COL]
    assert pd.api.types.is_integer_dtype(result[config.TARGET_COL])
    assert result[config.TARGET_COL].tolist() == [0, 1, 1, 0]
    assert result[config.ID_COL].tolist() == ids.tolist()

    assert path.exists()
    saved = pd.read_csv(path)
    assert list(saved.columns) == [config.ID_COL, config.TARGET_COL]
    assert pd.api.types.is_integer_dtype(saved[config.TARGET_COL])
    assert saved.equals(result)
    assert "Unnamed: 0" not in saved.columns


def test_make_submission_passes_float_array_to_model(tmp_path: Path, X_test, ids) -> None:
    model = FakeModel()
    make_submission(model, FakePipeline(), X_test, ids, tmp_path / "out.csv")
    assert isinstance(model.seen, np.ndarray)
    assert model.seen.shape == (4, 2)
    assert model.seen.dtype == np.float64


def test_make_submission_does_not_refit_pipeline(tmp_path: Path, X_test, ids) -> None:
    pipeline = FakePipeline()
    make_submission(FakeModel(), pipeline, X_test, ids, tmp_path / "out.csv")
    assert pipeline.fit_calls == 0
    assert pipeline.transform_calls == 1


def test_make_submission_accepts_int_predictions(tmp_path: Path, X_test, ids) -> None:
    result = make_submission(
        FakeModel([1, 0, 1, 0]), FakePipeline(), X_test, ids, tmp_path / "out.csv"
    )
    assert result[config.TARGET_COL].tolist() == [1, 0, 1, 0]


@pytest.mark.parametrize("bad", [[0, 1, 2, 0], [0, 1, -1, 0], [0.0, 0.5, 1.0, 0.0], [0, 1, np.nan, 0]])
def test_make_submission_rejects_out_of_range_predictions(
    tmp_path: Path, X_test, ids, bad
) -> None:
    path = tmp_path / "out.csv"
    with pytest.raises(ValueError):
        make_submission(FakeModel(bad), FakePipeline(), X_test, ids, path)
    assert not path.exists()


@pytest.mark.parametrize("bad", [[0, 1, 1], [0, 1, 1, 0, 1]])
def test_make_submission_rejects_wrong_length(tmp_path: Path, X_test, ids, bad) -> None:
    path = tmp_path / "out.csv"
    with pytest.raises(ValueError):
        make_submission(FakeModel(bad), FakePipeline(), X_test, ids, path)
    assert not path.exists()


@pytest.fixture
def sample_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        {config.ID_COL: ["a1", "b2", "c3"], config.TARGET_COL: [0, 0, 0]}
    ).to_csv(path, index=False)
    monkeypatch.setattr(config, "SAMPLE_SUBMISSION_PATH", path)
    return path


def _write(tmp_path: Path, frame: pd.DataFrame) -> Path:
    path = tmp_path / "candidate.csv"
    frame.to_csv(path, index=False)
    return path


def test_validate_submission_accepts_valid(tmp_path: Path, sample_path: Path) -> None:
    frame = pd.DataFrame({config.ID_COL: ["a1", "b2", "c3"], config.TARGET_COL: [1, 0, 1]})
    assert validate_submission(_write(tmp_path, frame)) is None


def test_validate_submission_accepts_output_of_make_submission(
    tmp_path: Path, sample_path: Path
) -> None:
    X = pd.DataFrame({"a": [2, 0, 3], "b": [1, 1, 1]})
    ids = pd.Series(["a1", "b2", "c3"])
    path = tmp_path / "made.csv"
    make_submission(FakeModel(), FakePipeline(), X, ids, path)
    validate_submission(path)


def test_validate_submission_rejects_different_columns(tmp_path: Path, sample_path: Path) -> None:
    frame = pd.DataFrame({config.ID_COL: ["a1", "b2", "c3"], "label": [1, 0, 1]})
    with pytest.raises(ValueError):
        validate_submission(_write(tmp_path, frame))


def test_validate_submission_rejects_swapped_column_order(
    tmp_path: Path, sample_path: Path
) -> None:
    frame = pd.DataFrame({config.TARGET_COL: [1, 0, 1], config.ID_COL: ["a1", "b2", "c3"]})
    with pytest.raises(ValueError):
        validate_submission(_write(tmp_path, frame))


def test_validate_submission_rejects_different_row_count(
    tmp_path: Path, sample_path: Path
) -> None:
    frame = pd.DataFrame({config.ID_COL: ["a1", "b2"], config.TARGET_COL: [1, 0]})
    with pytest.raises(ValueError):
        validate_submission(_write(tmp_path, frame))


def test_validate_submission_rejects_different_id_order(
    tmp_path: Path, sample_path: Path
) -> None:
    frame = pd.DataFrame({config.ID_COL: ["b2", "a1", "c3"], config.TARGET_COL: [1, 0, 1]})
    with pytest.raises(ValueError):
        validate_submission(_write(tmp_path, frame))


@pytest.mark.parametrize("bad", [[1, 0, 2], [1, 0, -1], [1.0, 0.5, 0.0], [1, 0, None]])
def test_validate_submission_rejects_non_binary_values(
    tmp_path: Path, sample_path: Path, bad
) -> None:
    frame = pd.DataFrame({config.ID_COL: ["a1", "b2", "c3"], config.TARGET_COL: bad})
    with pytest.raises(ValueError):
        validate_submission(_write(tmp_path, frame))
