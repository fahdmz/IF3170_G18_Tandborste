from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data import CATEGORICAL_COLS
from src.models.base import BaseClassifier
from src.preprocessing.base import Pipeline
from src.preprocessing.presets import to_model_input


def make_submission(
    model: BaseClassifier,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    ids: pd.Series,
    path: str | Path,
) -> pd.DataFrame:
    X_transformed = pipeline.transform(X_test)
    X_input, _ = to_model_input(X_transformed, CATEGORICAL_COLS)
    predictions = np.asarray(model.predict(X_input)).ravel()

    if len(predictions) != len(ids):
        raise ValueError(
            f"Jumlah prediksi ({len(predictions)}) tidak sama dengan jumlah id ({len(ids)})"
        )
    if not np.isin(predictions, [0, 1]).all():
        raise ValueError("Semua prediksi harus bernilai 0 atau 1")

    submission = pd.DataFrame(
        {
            config.ID_COL: np.asarray(ids),
            config.TARGET_COL: predictions.astype(int),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return submission


def validate_submission(path: str | Path) -> None:
    submission = pd.read_csv(path, dtype={config.ID_COL: str})
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_PATH, dtype={config.ID_COL: str})

    if list(submission.columns) != list(sample.columns):
        raise ValueError(
            f"Kolom submission {list(submission.columns)} berbeda dari sample {list(sample.columns)}"
        )
    if len(submission) != len(sample):
        raise ValueError(
            f"Jumlah baris submission ({len(submission)}) berbeda dari sample ({len(sample)})"
        )
    if not (submission[config.ID_COL].to_numpy() == sample[config.ID_COL].to_numpy()).all():
        raise ValueError("Urutan id submission berbeda dari sample")
    if not submission[config.TARGET_COL].isin([0, 1]).all():
        raise ValueError(f"Kolom {config.TARGET_COL} hanya boleh bernilai 0 atau 1")
