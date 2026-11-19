"""Reproducible Decision Tree (C4.5) comparison and tuning runner for A3.

Run from repository root:
    python -m src.experiments.decision_tree --search none
    python -m src.experiments.decision_tree --search grid
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import ExitStack
import csv
import json
import pickle
import platform
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.tree import DecisionTreeClassifier

from src import config
from src import data as data_module
from src.evaluation.metrics import (
    accuracy,
    confusion_matrix,
    f1_score,
    macro_f1,
    misclassification_cost,
    precision,
    recall,
    roc_auc,
)
from src.evaluation.model_selection import StratifiedKFold, cross_validate, grid_search
from src.models.decision_tree import DecisionTree
from src.preprocessing.base import Pipeline
from src.preprocessing.cleaning import clean_train
from src.preprocessing.presets import (
    make_numeric_pipeline,
    make_tree_pipeline,
    to_model_input,
)
from src.visualization.decision_tree_graph import plot_decision_tree

DEFAULT_DT_PARAMS: dict[str, Any] = {
    "criterion": "gain_ratio",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_gain": 0.0,
}

DEFAULT_SK_PARAMS: dict[str, Any] = {
    "criterion": "entropy",
    "ccp_alpha": 0.0,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
}

GRID_PARAM_SPACE: dict[str, list[int]] = {
    "max_depth": [8, 12],
    "min_samples_leaf": [5, 10],
}

THRESHOLD_CANDIDATES: tuple[float, ...] = tuple(
    round(step / 100.0, 2) for step in range(5, 96)
)

CV_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1_score,
    "macro_f1": macro_f1,
}

TEAM_COMPARISON_COLUMNS: list[str] = [
    "model",
    "implementation",
    "hyperparameters",
    "pipeline",
    "pipeline_options",
    "resampling",
    "sampling_strategy",
    "resampling_seed",
    "threshold",
    "fold_1_macro_f1",
    "fold_2_macro_f1",
    "fold_3_macro_f1",
    "fold_4_macro_f1",
    "fold_5_macro_f1",
    "cv_mean_macro_f1",
    "cv_std_macro_f1",
    "cv_mean_macro_f1_at_0_50",
    "holdout_macro_f1",
    "holdout_accuracy",
    "holdout_precision",
    "holdout_recall",
    "holdout_positive_f1",
    "holdout_roc_auc",
]

REFERENCES: list[dict[str, str]] = [
    {
        "name": "Weka InfoGainSplitCrit",
        "url": "https://raw.githubusercontent.com/Waikato/weka-3.8/master/weka/src/main/java/weka/classifiers/trees/j48/InfoGainSplitCrit.java",
        "role": "Missing-value adjusted information gain formula reference",
    },
    {
        "name": "Weka GainRatioSplitCrit",
        "url": "https://raw.githubusercontent.com/Waikato/weka-3.8/master/weka/src/main/java/weka/classifiers/trees/j48/GainRatioSplitCrit.java",
        "role": "Split info with unknown bucket and gain ratio reference",
    },
    {
        "name": "Weka C45ModelSelection",
        "url": "https://raw.githubusercontent.com/Waikato/weka-3.8/master/weka/src/main/java/weka/classifiers/trees/j48/C45ModelSelection.java",
        "role": "Average information gain guard reference",
    },
    {
        "name": "Weka Stats",
        "url": "https://raw.githubusercontent.com/Waikato/weka-3.8/master/weka/src/main/java/weka/classifiers/trees/j48/Stats.java",
        "role": "Upper-confidence pessimistic pruning math reference",
    },
    {
        "name": "scikit-learn DecisionTreeClassifier & User Guide",
        "url": "https://scikit-learn.org/stable/modules/tree.html",
        "role": "Baseline comparator and documented limits on native categorical/missing handling",
    },
]


def infer_tree_schema(
    tree_factory: Callable[[], Pipeline],
) -> tuple[list[str], list[str], list[int]]:
    """Derive output column order and categorical indices from unfitted formatter config."""
    pipeline = tree_factory()
    if not getattr(pipeline, "steps", None):
        raise ValueError("tree_factory pipeline must define at least one step")
    _, formatter = pipeline.steps[-1]
    if not hasattr(formatter, "numerical_cols") or not hasattr(
        formatter, "categorical_cols"
    ):
        raise ValueError(
            "Last step of tree_factory pipeline must expose numerical_cols and categorical_cols"
        )
    numerical = list(formatter.numerical_cols)
    categorical = list(formatter.categorical_cols)
    columns = numerical + categorical
    categorical_indices = [columns.index(col) for col in categorical]
    return columns, categorical, categorical_indices


class _SchemaCheckedPipeline:
    """Thin guard ensuring tree pipeline output schema and row counts stay invariant across folds."""

    def __init__(
        self,
        inner: Pipeline,
        expected_columns: list[str],
        categorical_cols: list[str],
        expected_cat_indices: list[int],
    ) -> None:
        self._inner = inner
        self._expected_columns = list(expected_columns)
        self._categorical_cols = list(categorical_cols)
        self._expected_cat_indices = list(expected_cat_indices)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_SchemaCheckedPipeline":
        self._inner.fit(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        out = self._inner.fit_transform(X, y)
        self._verify(out, expected_rows=len(X), check_indices=True)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._inner.transform(X)
        self._verify(out, expected_rows=len(X), check_indices=False)
        return out

    def _verify(self, out: pd.DataFrame, expected_rows: int, check_indices: bool) -> None:
        if len(out) != expected_rows:
            raise ValueError(
                f"Pipeline changed row count during transform: expected {expected_rows}, got {len(out)}"
            )
        actual_columns = list(out.columns)
        if actual_columns != self._expected_columns:
            raise ValueError(
                f"Tree pipeline schema mismatch across folds: expected {self._expected_columns}, "
                f"got {actual_columns}"
            )
        if check_indices:
            _, actual_indices = to_model_input(out, self._categorical_cols)
            if actual_indices != self._expected_cat_indices:
                raise ValueError(
                    f"Categorical indices mismatch: expected {self._expected_cat_indices}, "
                    f"got {actual_indices}"
                )


class _RowPreservingPipeline:
    """Thin guard ensuring numeric pipeline preserves row counts during transform."""

    def __init__(self, inner: Pipeline) -> None:
        self._inner = inner

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_RowPreservingPipeline":
        self._inner.fit(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        out = self._inner.fit_transform(X, y)
        if len(out) != len(X):
            raise ValueError(
                f"Numeric pipeline changed row count on fit_transform: expected {len(X)}, got {len(out)}"
            )
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._inner.transform(X)
        if len(out) != len(X):
            raise ValueError(
                f"Numeric pipeline changed row count on transform: expected {len(X)}, got {len(out)}"
            )
        return out


def _positive_class_column(model: Any) -> int:
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(f"Expected binary classes containing 1, got {classes.tolist()}")
    return int(matches[0])


def _predict_positive_proba(model: Any, X: np.ndarray) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X), dtype=float)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"Expected 2D probability output, got shape {proba.shape}")
    pos_scores = proba[:, _positive_class_column(model)]
    if not np.all(np.isfinite(pos_scores)):
        raise ValueError("predict_proba produced non-finite probabilities")
    return pos_scores


def select_macro_f1_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    candidates: tuple[float, ...] = THRESHOLD_CANDIDATES,
    tie_tol: float = 1e-12,
) -> dict[str, Any]:
    """Select Macro F1 threshold from OOF class-1 probabilities with deterministic tie-breaking.

    Prediction rule: y_pred = 1 if P(is_canceled=1) >= threshold else 0.
    Tie-breaker: closest to 0.50 first, then smaller threshold.
    """
    y_arr = np.asarray(y_true).ravel().astype(int)
    p_arr = np.asarray(y_proba, dtype=float).ravel()
    if len(y_arr) != len(p_arr) or len(y_arr) == 0:
        raise ValueError("y_true and y_proba must be non-empty and have matching lengths")
    if not np.all(np.isfinite(p_arr)):
        raise ValueError("y_proba must contain only finite probabilities")

    candidate_list = [float(t) for t in candidates]
    if not candidate_list:
        raise ValueError("candidates must not be empty")

    sweep: list[dict[str, float]] = []
    for threshold in candidate_list:
        preds = (p_arr >= threshold).astype(int)
        score = float(macro_f1(y_arr, preds))
        sweep.append({"threshold": threshold, "macro_f1": score})

    best_score = max(item["macro_f1"] for item in sweep)
    tied = [item for item in sweep if item["macro_f1"] >= best_score - tie_tol]
    winner = min(
        tied,
        key=lambda item: (round(abs(item["threshold"] - 0.50), 10), item["threshold"]),
    )

    score_at_0_50 = float(macro_f1(y_arr, (p_arr >= 0.50).astype(int)))
    return {
        "threshold": float(winner["threshold"]),
        "oof_macro_f1": float(winner["macro_f1"]),
        "oof_macro_f1_at_0_50": score_at_0_50,
        "tie_rule": "max_macro_f1 -> min_abs_diff_to_0_50 -> smaller_threshold",
        "prediction_rule": "P(is_canceled=1) >= threshold",
        "n_candidates": len(sweep),
        "sweep": sweep,
    }


def _tree_probability_folds(model, pipeline_factory, X, y, cv):
    """Fit once per fold and yield the validation rows and class-1 probabilities."""
    for train_idx, val_idx in cv.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        pipeline = pipeline_factory()
        train = np.asarray(pipeline.fit_transform(X_tr, y_tr))
        validation = np.asarray(pipeline.transform(X.iloc[val_idx]))
        fitted = model.clone().fit(train, y_tr.to_numpy())
        yield val_idx, _predict_positive_proba(fitted, validation)


def collect_oof_probabilities(model, pipeline_factory, X, y, cv) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for val_idx, probabilities in _tree_probability_folds(model, pipeline_factory, X, y, cv):
        oof[val_idx] = probabilities
    if np.isnan(oof).any():
        raise RuntimeError("OOF probability collection left unassigned rows")
    return oof


def _threshold_cv(model, pipeline_factory, X, y, cv, threshold):
    tuned = {name: [] for name in CV_METRICS}
    at_half = {name: [] for name in CV_METRICS}
    for val_idx, probabilities in _tree_probability_folds(model, pipeline_factory, X, y, cv):
        actual = y.iloc[val_idx].to_numpy()
        for cutoff, scores in ((threshold, tuned), (0.50, at_half)):
            labels = (probabilities >= cutoff).astype(int)
            for name, score in CV_METRICS.items():
                scores[name].append(float(score(actual, labels)))
    return _summarize_cv(tuned), _summarize_cv(at_half)


def validate_and_load_datasets() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    list[str],
    list[str],
]:
    """Verify dataset files exist and meet required column/target contracts before loading."""
    required_files = [
        ("train.csv", Path(config.TRAIN_PATH)),
        ("test.csv", Path(config.TEST_PATH)),
        ("sample_submission.csv", Path(config.SAMPLE_SUBMISSION_PATH)),
    ]
    missing_files = [
        f"{label} ({path})" for label, path in required_files if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Dataset dependency incomplete. Missing file(s): " + ", ".join(missing_files)
        )

    numerical_cols, categorical_cols = data_module.get_feature_types()
    expected_features = list(numerical_cols) + list(categorical_cols)

    X_train, y_train = data_module.load_train()
    X_test, test_ids = data_module.load_test()
    sample_submission = pd.read_csv(
        config.SAMPLE_SUBMISSION_PATH, dtype={config.ID_COL: str}
    )

    if len(X_train) == 0 or len(X_test) == 0 or len(sample_submission) == 0:
        raise ValueError("Train, test, and sample_submission datasets must be non-empty")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Train X/y length mismatch: {len(X_train)} vs {len(y_train)}"
        )
    if len(X_test) != len(test_ids):
        raise ValueError(
            f"Test X/ids length mismatch: {len(X_test)} vs {len(test_ids)}"
        )

    missing_train_cols = [c for c in expected_features if c not in X_train.columns]
    if missing_train_cols:
        raise ValueError(f"train.csv missing required feature columns: {missing_train_cols}")
    missing_test_cols = [c for c in expected_features if c not in X_test.columns]
    if missing_test_cols:
        raise ValueError(f"test.csv missing required feature columns: {missing_test_cols}")

    sample_cols = list(sample_submission.columns)
    expected_sample_cols = [config.ID_COL, config.TARGET_COL]
    if sample_cols != expected_sample_cols:
        raise ValueError(
            f"sample_submission.csv columns mismatch: expected {expected_sample_cols}, got {sample_cols}"
        )

    y_arr = np.asarray(y_train).ravel()
    if pd.isna(y_arr).any():
        raise ValueError("Target column contains missing values")
    unique_targets = set(np.unique(y_arr).tolist())
    if unique_targets != {0, 1}:
        raise ValueError(f"Target column must be binary {{0, 1}}, got {sorted(unique_targets)}")

    return (
        X_train,
        y_train.astype(int),
        X_test,
        test_ids,
        sample_submission,
        list(numerical_cols),
        list(categorical_cols),
    )


def evaluate_holdout(
    model: Any,
    X_val_arr: np.ndarray,
    y_val_arr: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate fitted classifier on holdout split using label metrics, ROC AUC, CM, and size."""
    y_true = np.asarray(y_val_arr).ravel().astype(int)
    pos_proba = _predict_positive_proba(model, X_val_arr)
    if threshold is None:
        y_pred = np.asarray(model.predict(X_val_arr)).ravel().astype(int)
        decision_rule = "model.predict (argmax, tie->0)"
    else:
        y_pred = (pos_proba >= threshold).astype(int)
        decision_rule = f"P(is_canceled=1) >= {threshold:.2f}"

    cm = confusion_matrix(y_true, y_pred)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    auc_val = float(roc_auc(y_true, pos_proba))
    if not np.isfinite(auc_val):
        raise ValueError("ROC AUC on holdout is non-finite (check class balance in holdout)")

    return {
        "decision_rule": decision_rule,
        "threshold": None if threshold is None else float(threshold),
        "macro_f1": float(macro_f1(y_true, y_pred)),
        "accuracy": float(accuracy(y_true, y_pred)),
        "precision": float(precision(y_true, y_pred)),
        "recall": float(recall(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": auc_val,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "misclassification_cost": float(misclassification_cost(y_true, y_pred)),
        "depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
    }


def _summarize_cv(cv_scores: dict[str, list[float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric_name, values in cv_scores.items():
        floats = [float(v) for v in values]
        summary[metric_name] = {
            "folds": floats,
            "mean": float(np.mean(floats)),
            "std": float(np.std(floats)),
        }
    return summary


def _git_output(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=config.ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _collect_metadata(
    search: str,
    raw_train_rows: int,
    outer_train_raw_rows: int,
    X_clean: pd.DataFrame,
    y_clean: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    tree_columns: list[str],
    categorical_cols: list[str],
    categorical_indices: list[int],
    numeric_feature_count: int,
    tree_factory_options: dict[str, Any],
    numeric_factory_options: dict[str, Any],
) -> dict[str, Any]:
    return {
        "git_sha": _git_output(["rev-parse", "HEAD"]),
        "git_branch": _git_output(["branch", "--show-current"]),
        "seed": int(config.SEED),
        "search_mode": search,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "counts": {
            "raw_train_rows": int(raw_train_rows),
            "outer_train_raw_rows": int(outer_train_raw_rows),
            "outer_train_clean_rows": int(len(X_clean)),
            "outer_train_dropped_rows": int(outer_train_raw_rows - len(X_clean)),
            "holdout_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "outer_train_clean_class_counts": {
                "0": int((y_clean == 0).sum()),
                "1": int((y_clean == 1).sum()),
            },
            "holdout_class_counts": {
                "0": int((y_val == 0).sum()),
                "1": int((y_val == 1).sum()),
            },
        },
        "schema": {
            "tree_columns": list(tree_columns),
            "categorical_cols": list(categorical_cols),
            "categorical_indices": list(categorical_indices),
            "tree_feature_count": len(tree_columns),
            "numeric_feature_count_outer_train": int(numeric_feature_count),
        },
        "factory_options": {
            "tree_factory": dict(tree_factory_options),
            "numeric_factory": dict(numeric_factory_options),
        },
        "preprocessing_difference_note": (
            "C4.5 uses make_tree_pipeline (engineered features, rare category grouping, "
            "native missing values & categorical multiway splits). scikit-learn "
            "DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.0) uses "
            "make_numeric_pipeline (imputation, outlier clipping, scaling, one-hot encoding) "
            "because sklearn does not implement native C4.5 gain ratio, multiway categorical "
            "splits, or fractional missing instances."
        ),
        "holdout_note": (
            "The 80/20 stratified holdout split (seed=42) matches Fahd's main.ipynb baseline split."
        ),
        "sources": REFERENCES,
    }


def _append_cv_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    protocol: str,
    cv_summary: dict[str, Any],
) -> None:
    for metric_name, stats in cv_summary.items():
        for idx, val in enumerate(stats["folds"], start=1):
            rows.append(
                {
                    "model": model_name,
                    "protocol": protocol,
                    "fold": str(idx),
                    "metric": metric_name,
                    "value": float(val),
                }
            )
        rows.append(
            {
                "model": model_name,
                "protocol": protocol,
                "fold": "mean",
                "metric": metric_name,
                "value": float(stats["mean"]),
            }
        )
        rows.append(
            {
                "model": model_name,
                "protocol": protocol,
                "fold": "std",
                "metric": metric_name,
                "value": float(stats["std"]),
            }
        )


def _append_holdout_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    holdout: dict[str, Any],
) -> None:
    keys = [
        "macro_f1",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "misclassification_cost",
        "depth",
        "n_leaves",
    ]
    for key in keys:
        rows.append(
            {
                "model": model_name,
                "protocol": "holdout",
                "fold": "holdout",
                "metric": key,
                "value": float(holdout[key]),
            }
        )


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "protocol", "fold", "metric", "value"]
        )
        writer.writeheader()
        for row in rows:
            val = float(row["value"])
            if not np.isfinite(val):
                raise ValueError(f"Non-finite value in comparison CSV row: {row}")
            writer.writerow(
                {
                    "model": row["model"],
                    "protocol": row["protocol"],
                    "fold": row["fold"],
                    "metric": row["metric"],
                    "value": repr(val),
                }
            )


def _build_team_summary_row(
    *,
    model_label: str,
    implementation: str,
    hyperparameters: dict[str, Any],
    pipeline_name: str,
    pipeline_options: dict[str, Any],
    threshold: float | None,
    cv_summary: dict[str, Any],
    cv_mean_at_0_50: float | None,
    holdout: dict[str, Any],
) -> dict[str, Any]:
    folds = cv_summary["macro_f1"]["folds"]
    row: dict[str, Any] = {
        "model": model_label,
        "implementation": implementation,
        "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
        "pipeline": pipeline_name,
        "pipeline_options": json.dumps(pipeline_options, sort_keys=True),
        "resampling": "none",
        "sampling_strategy": "",
        "resampling_seed": "",
        "threshold": "" if threshold is None else repr(float(threshold)),
        "cv_mean_macro_f1": repr(float(cv_summary["macro_f1"]["mean"])),
        "cv_std_macro_f1": repr(float(cv_summary["macro_f1"]["std"])),
        "cv_mean_macro_f1_at_0_50": (
            "" if cv_mean_at_0_50 is None else repr(float(cv_mean_at_0_50))
        ),
        "holdout_macro_f1": repr(float(holdout["macro_f1"])),
        "holdout_accuracy": repr(float(holdout["accuracy"])),
        "holdout_precision": repr(float(holdout["precision"])),
        "holdout_recall": repr(float(holdout["recall"])),
        "holdout_positive_f1": repr(float(holdout["f1"])),
        "holdout_roc_auc": repr(float(holdout["roc_auc"])),
    }
    for idx in range(1, 6):
        row[f"fold_{idx}_macro_f1"] = (
            repr(float(folds[idx - 1])) if idx <= len(folds) else ""
        )
    return row


def _write_team_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TEAM_COMPARISON_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def _run_grid(templates, factories, X, y, cv, param_grid, timings, rows):
    search_results = {}
    for name in ("c45", "sklearn"):
        start = time.perf_counter()
        best_params, trials = grid_search(
            model=templates[name],
            param_grid=param_grid,
            pipeline_factory=factories[name],
            X=X,
            y=y,
            cv=cv,
            scoring=macro_f1,
        )
        duration = round(time.perf_counter() - start, 4)
        timings[f"{name}_grid_search_3fold_seconds"] = duration
        for trial in trials:
            summary = _summarize_cv({"macro_f1": trial["scores"]})
            trial["std_score"] = summary["macro_f1"]["std"]
            tag = ",".join(f"{key}={value}" for key, value in sorted(trial["params"].items()))
            _append_cv_rows(rows, f"{name}_grid[{tag}]", "grid_3fold", summary)
        search_results[name] = {
            "best_params": best_params, "duration_seconds": duration, "trials": trials,
        }
    return search_results


def _append_threshold_rows(rows, selection):
    scores = {
        f"macro_f1@t={item['threshold']:.2f}": item["macro_f1"]
        for item in selection["sweep"]
    }
    scores.update(
        selected_threshold=selection["threshold"],
        oof_macro_f1_selected=selection["oof_macro_f1"],
        oof_macro_f1_at_0_50=selection["oof_macro_f1_at_0_50"],
    )
    for metric, value in scores.items():
        rows.append({
            "model": "c45_tuned", "protocol": "oof_threshold_sweep", "fold": "oof",
            "metric": metric, "value": value,
        })


def _holdout_inputs(pipeline, X_train, y_train, X_val):
    train = pipeline.fit_transform(X_train, y_train)
    validation = pipeline.transform(X_val)
    if len(train) != len(X_train) or len(validation) != len(X_val):
        raise ValueError("Pipeline changed row count during outer-train or holdout transform")
    return train, validation


def _verify_saved_model(
    model, pipeline, model_file, pipeline_file, X_val, val_array, X_test, categorical_cols,
):
    model.save(model_file)
    with pipeline_file.open("wb") as stream:
        pickle.dump(pipeline, stream)
    reloaded = DecisionTree.load(model_file)
    with pipeline_file.open("rb") as stream:
        reloaded_pipeline = pickle.load(stream)

    expected_threshold = model.decision_threshold_
    threshold = getattr(reloaded, "decision_threshold_", None)
    if threshold != expected_threshold:
        raise RuntimeError(
            f"Reloaded tuned model threshold mismatch: expected {expected_threshold}, got {threshold}"
        )
    restored_val, _ = to_model_input(reloaded_pipeline.transform(X_val), categorical_cols)
    original_scores = _predict_positive_proba(model, val_array)
    restored_scores = _predict_positive_proba(reloaded, restored_val)
    same_probabilities = bool(np.array_equal(original_scores, restored_scores))
    same_labels = bool(np.array_equal(original_scores >= expected_threshold, restored_scores >= threshold))
    if not (same_probabilities and same_labels):
        raise RuntimeError("Reloaded tuned model/pipeline predictions differ from in-memory model")

    test_frame = reloaded_pipeline.transform(X_test)
    if not test_frame.index.equals(X_test.index):
        raise RuntimeError("Reloaded pipeline changed the number or order of test rows")
    test_array, _ = to_model_input(test_frame, categorical_cols)
    test_scores = _predict_positive_proba(reloaded, test_array)
    if test_scores.shape != (len(X_test),):
        raise RuntimeError("Test prediction count does not match test rows")
    labels = (test_scores >= threshold).astype(int)
    return {
        "holdout_probabilities_exact": same_probabilities,
        "holdout_labels_exact": same_labels,
        "saved_decision_threshold_": threshold,
    }, labels


def _publish_artifacts(staged_files: dict[Path, Path], scratch: Path) -> None:
    """Replace completed outputs, restoring the preceding run if a replacement fails."""
    lock = scratch.parent / ".decision-tree-publish"
    lock.mkdir()
    try:
        backups = {}
        for index, destination in enumerate(staged_files):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                backup = scratch / f"previous-{index}"
                shutil.copy2(destination, backup)
                backups[destination] = backup

        with ExitStack() as rollback:
            for destination, staged in staged_files.items():
                staged.replace(destination)
                if destination in backups:
                    rollback.callback(backups[destination].replace, destination)
                else:
                    rollback.callback(destination.unlink, missing_ok=True)
            rollback.pop_all()
    finally:
        lock.rmdir()


def _team_rows(results, pipeline_options):
    rows = []
    for name in ("c45_default", "c45_tuned", "sklearn_default", "sklearn_tuned"):
        result = results[name]
        is_tree = name.startswith("c45")
        pipeline = "tree" if is_tree else "numeric"
        params = {key: value for key, value in result["params"].items() if key != "categorical_features"}
        cv_at_half = result.get("cv_5fold_at_0_50")
        rows.append(_build_team_summary_row(
            model_label=f"Decision Tree ({name.split('_')[-1]})",
            implementation="from-scratch" if is_tree else "scikit-learn",
            hyperparameters=params,
            pipeline_name=pipeline,
            pipeline_options=pipeline_options[pipeline],
            threshold=result.get("locked_threshold"),
            cv_summary=result["cv_5fold"],
            cv_mean_at_0_50=cv_at_half["macro_f1"]["mean"] if cv_at_half else None,
            holdout=result["holdout"],
        ))
    return rows


def run_experiment(
    search: str = "none",
    *,
    grid_param_space: dict[str, list[Any]] | None = None,
    cv_grid_splits: int = 3,
    cv_eval_splits: int = 5,
    top_n: int = 30,
) -> dict[str, Any]:
    """Run the baseline alone or compare it with the locked tuned configuration."""
    if search not in ("none", "grid"):
        raise ValueError(f"Unsupported search mode: {search!r}. Expected 'none' or 'grid'.")
    start = time.perf_counter()
    X, y, X_test, test_ids, sample, numerical, categorical = validate_and_load_datasets()
    X_train, X_val, y_train, y_val = sk_train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.SEED,
    )
    X_clean, y_clean = clean_train(X_train, y_train)

    def tree_factory():
        return make_tree_pipeline(numerical, categorical)

    def numeric_factory():
        return make_numeric_pipeline(numerical, categorical)

    columns, categorical_cols, categorical_indices = infer_tree_schema(tree_factory)

    def checked_tree_factory():
        return _SchemaCheckedPipeline(tree_factory(), columns, categorical_cols, categorical_indices)

    def checked_numeric_factory():
        return _RowPreservingPipeline(numeric_factory())

    factories = {"c45": checked_tree_factory, "sklearn": checked_numeric_factory}
    model_classes = {"c45": DecisionTree, "sklearn": DecisionTreeClassifier}
    params = {
        "c45_default": {
            **DEFAULT_DT_PARAMS, "categorical_features": categorical_indices, "random_state": config.SEED,
        },
        "sklearn_default": {**DEFAULT_SK_PARAMS, "random_state": config.SEED},
    }
    timings = {}
    comparison_rows = []
    results = {}
    locked_threshold = None
    tuned_model = None

    # Lock the hyperparameters and threshold using outer-train folds only.
    if search == "grid":
        param_grid = dict(GRID_PARAM_SPACE if grid_param_space is None else grid_param_space)
        grid_cv = StratifiedKFold(n_splits=cv_grid_splits, shuffle=True, random_state=config.SEED)
        templates = {name: cls(**params[f"{name}_default"]) for name, cls in model_classes.items()}
        grid_results = _run_grid(
            templates, factories, X_clean, y_clean, grid_cv, param_grid, timings, comparison_rows,
        )
        for name, grid_result in grid_results.items():
            full_params = {**params[f"{name}_default"], **grid_result["best_params"]}
            params[f"{name}_tuned"] = full_params
            grid_result["full_params"] = full_params

        template = DecisionTree(**params["c45_tuned"])
        threshold_start = time.perf_counter()
        probabilities = collect_oof_probabilities(template, checked_tree_factory, X_clean, y_clean, grid_cv)
        threshold_selection = select_macro_f1_threshold(y_clean.to_numpy(), probabilities)
        locked_threshold = threshold_selection["threshold"]
        timings["c45_oof_threshold_seconds"] = round(time.perf_counter() - threshold_start, 4)
        _append_threshold_rows(comparison_rows, threshold_selection)

    cv = StratifiedKFold(n_splits=cv_eval_splits, shuffle=True, random_state=config.SEED)
    for name, model_params in params.items():
        family, variant = name.split("_")
        factory = factories[family]
        model = model_classes[family](**model_params)
        threshold = locked_threshold if name == "c45_tuned" else None
        tuned = variant == "tuned"
        protocol = "cv_5fold_descriptive" if tuned else "cv_5fold"
        cv_start = time.perf_counter()
        result = {"params": model_params}
        if threshold is None:
            scores = cross_validate(model, factory, X_clean, y_clean, cv, CV_METRICS)
            cv_summary = _summarize_cv(scores)
        else:
            cv_summary, cv_at_half = _threshold_cv(model, factory, X_clean, y_clean, cv, threshold)
            result.update(locked_threshold=threshold, cv_5fold_at_0_50=cv_at_half)
            _append_cv_rows(comparison_rows, "c45_tuned_at_0_50", protocol, cv_at_half)
        timings[f"{name}_cv5_seconds"] = round(time.perf_counter() - cv_start, 4)
        result["cv_5fold"] = cv_summary
        if tuned:
            selected = "hyperparameters and threshold" if threshold is not None else "hyperparameters"
            result["cv_5fold_protocol"] = f"descriptive ({selected} selected on cleaned outer train)"
        _append_cv_rows(comparison_rows, name, protocol, cv_summary)
        results[name] = result

    tree_pipeline = tree_factory()
    tree_train, tree_val = _holdout_inputs(tree_pipeline, X_clean, y_clean, X_val)
    tree_train_array, final_indices = to_model_input(tree_train, categorical_cols)
    if (
        list(tree_train.columns) != columns
        or list(tree_val.columns) != columns
        or final_indices != categorical_indices
    ):
        raise ValueError("Final tree pipeline schema differs from the CV schema")
    tree_val_array, _ = to_model_input(tree_val, categorical_cols)
    numeric_train, numeric_val = _holdout_inputs(numeric_factory(), X_clean, y_clean, X_val)
    matrices = {
        "c45": (tree_train_array, tree_val_array),
        "sklearn": (np.asarray(numeric_train, dtype=float), np.asarray(numeric_val, dtype=float)),
    }
    actual_holdout = y_val.to_numpy()
    for name, result in results.items():
        family = name.split("_")[0]
        train, validation = matrices[family]
        fit_start = time.perf_counter()
        model = model_classes[family](**result["params"]).fit(train, y_clean.to_numpy())
        timings[f"{name}_holdout_fit_seconds"] = round(time.perf_counter() - fit_start, 4)
        threshold = result.get("locked_threshold")
        result["holdout"] = evaluate_holdout(model, validation, actual_holdout, threshold)
        _append_holdout_rows(comparison_rows, name, result["holdout"])
        if name == "c45_tuned":
            tuned_model = model
            tuned_model.decision_threshold_ = threshold
            result["holdout_at_0_50"] = evaluate_holdout(model, validation, actual_holdout, 0.50)
            _append_holdout_rows(comparison_rows, "c45_tuned_at_0_50", result["holdout_at_0_50"])

    pipeline_options = {
        "tree": {"engineer_features": True, "group_rare": True},
        "numeric": {
            "use_pca": False, "engineer_features": True, "add_missing_indicators": True,
            "clip_outliers": True, "group_rare": True, "select_k": None, "pca_components": None,
        },
    }
    metadata = _collect_metadata(
        search, len(X), len(X_train), X_clean, y_clean, X_val, y_val, X_test,
        columns, categorical_cols, final_indices, numeric_train.shape[1],
        pipeline_options["tree"], pipeline_options["numeric"],
    )
    output = Path(config.OUTPUT_DIR)
    report_dir = output / "decision_tree"
    if search == "none":
        report_dir /= "baseline"
    paths = {"comparison_csv": report_dir / "comparison.csv"}
    payload = {
        "search_mode": search,
        "scoring": "macro_f1",
        "metadata": metadata,
        "bonus_status": {
            "visualization_top_n": "BASELINE_AVAILABLE_IN_MAIN",
            "optimization": "DEFERRED",
        },
        "baselines": {name: result for name, result in results.items() if name.endswith("_default")},
        "timings": timings,
    }

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".dtl-", dir=output) as temp_dir:
        scratch = Path(temp_dir)
        if tuned_model is not None:
            if len(X_test) != len(sample):
                raise RuntimeError(f"Test row count ({len(X_test)}) != sample_submission ({len(sample)})")
            if not np.array_equal(test_ids.astype(str).to_numpy(), sample[config.ID_COL].astype(str).to_numpy()):
                raise RuntimeError("Test ID order does not match sample_submission.csv")

            paths.update(
                team_summary_csv=output / "dtl_tuning_comparison.csv",
                tuned_model_pkl=Path(config.MODEL_DIR) / "decision_tree_tuned.pkl",
                tuned_pipeline_pkl=Path(config.MODEL_DIR) / "decision_tree_tuned_pipeline.pkl",
                tuned_tree_dot=Path(config.FIGURE_DIR) / "decision_tree_tuned.dot",
                tuned_tree_png=Path(config.FIGURE_DIR) / "decision_tree_tuned.png",
            )
            verification, test_labels = _verify_saved_model(
                tuned_model, tree_pipeline, scratch / paths["tuned_model_pkl"].name,
                scratch / paths["tuned_pipeline_pkl"].name, X_val, tree_val_array, X_test, categorical_cols,
            )
            tuned_model.export_graph(scratch / paths["tuned_tree_dot"].name, top_n=top_n, feature_names=columns)
            plot_decision_tree(tuned_model, scratch / paths["tuned_tree_png"].name, top_n=top_n, feature_names=columns)
            _write_team_summary_csv(scratch / paths["team_summary_csv"].name, _team_rows(results, pipeline_options))

            delta = results["c45_tuned"]["holdout"]["macro_f1"] - results["c45_default"]["holdout"]["macro_f1"]
            payload["bonus_status"]["visualization_top_n"] = f"SELECTED_AND_VERIFIED (top_n={top_n})"
            payload.update(
                grid_search={
                    "cv_splits": cv_grid_splits, "seed": config.SEED,
                    "scoring": "macro_f1", "param_grid": param_grid, **grid_results,
                },
                threshold_selection=threshold_selection,
                tuned={name: result for name, result in results.items() if name.endswith("_tuned")},
                persistence_verification={
                    "model_path": str(paths["tuned_model_pkl"]),
                    "pipeline_path": str(paths["tuned_pipeline_pkl"]),
                    **verification,
                },
                kaggle_readiness={
                    "status": "READY", "submission_written": False,
                    "test_rows": len(X_test), "transformed_rows": len(test_labels),
                    "sample_submission_rows": len(sample), "id_order_matches_sample": True,
                    "predicted_class_counts": {str(label): int((test_labels == label).sum()) for label in (0, 1)},
                    "predicted_positive_rate": float(np.mean(test_labels)),
                },
                handoff={
                    "recommended_option": "c45_tuned" if delta > 0.0 else "c45_default",
                    "holdout_macro_f1_delta_tuned_vs_default": delta,
                    "best_c45_params": params["c45_tuned"],
                    "locked_threshold": locked_threshold,
                    "prediction_rule_tuned": f"P(is_canceled=1) >= {locked_threshold:.2f}",
                    "prediction_rule_default": "model.predict(X) (argmax; tie->0)",
                    "best_sklearn_params": params["sklearn_tuned"],
                    "prediction_example": (
                        'X_model, _ = to_model_input(fitted_pipeline.transform(X_raw), CATEGORICAL_COLS)\n'
                        'positive_class = list(model.classes_).index(1)\n'
                        f'labels = (model.predict_proba(X_model)[:, positive_class] >= {locked_threshold!r}).astype(int)'
                    ),
                    "notes": [
                        "Fit the tree pipeline on training data and derive categorical indices from to_model_input.",
                        "Tuned 5-fold scores are descriptive because grid and threshold selection used the same outer train.",
                    ],
                },
            )
        paths["run_json"] = report_dir / "run.json"
        payload["artifacts"] = {name: str(path) for name, path in paths.items()}
        timings["total_seconds"] = round(time.perf_counter() - start, 4)
        _write_comparison_csv(scratch / paths["comparison_csv"].name, comparison_rows)
        _write_json(scratch / paths["run_json"].name, payload)
        _publish_artifacts({path: scratch / path.name for path in paths.values()}, scratch)
    return payload


def _format_summary(payload: dict[str, Any]) -> str:
    mode = payload["search_mode"]
    c45_def = payload["baselines"]["c45_default"]
    sk_def = payload["baselines"]["sklearn_default"]
    lines = [
        f"DTL Experiment complete (search={mode}, seed={payload['metadata']['seed']})",
        (
            f"  C4.5 default   : CV5 Macro F1 = {c45_def['cv_5fold']['macro_f1']['mean']:.4f} "
            f"(std {c45_def['cv_5fold']['macro_f1']['std']:.4f}) | "
            f"Holdout Macro F1 = {c45_def['holdout']['macro_f1']:.4f} | "
            f"depth={c45_def['holdout']['depth']}, leaves={c45_def['holdout']['n_leaves']}"
        ),
        (
            f"  sklearn default: CV5 Macro F1 = {sk_def['cv_5fold']['macro_f1']['mean']:.4f} "
            f"(std {sk_def['cv_5fold']['macro_f1']['std']:.4f}) | "
            f"Holdout Macro F1 = {sk_def['holdout']['macro_f1']:.4f} | "
            f"depth={sk_def['holdout']['depth']}, leaves={sk_def['holdout']['n_leaves']}"
        ),
    ]
    if mode == "grid":
        c45_tuned = payload["tuned"]["c45_tuned"]
        sk_tuned = payload["tuned"]["sklearn_tuned"]
        lines.extend(
            [
                (
                    f"  C4.5 tuned     : params={payload['grid_search']['c45']['best_params']}, "
                    f"threshold={c45_tuned['locked_threshold']:.2f} | "
                    f"CV5(desc) Macro F1 = {c45_tuned['cv_5fold']['macro_f1']['mean']:.4f} "
                    f"(std {c45_tuned['cv_5fold']['macro_f1']['std']:.4f}, "
                    f"@0.50={c45_tuned['cv_5fold_at_0_50']['macro_f1']['mean']:.4f}) | "
                    f"Holdout Macro F1 = {c45_tuned['holdout']['macro_f1']:.4f} | "
                    f"depth={c45_tuned['holdout']['depth']}, leaves={c45_tuned['holdout']['n_leaves']}"
                ),
                (
                    f"  sklearn tuned  : params={payload['grid_search']['sklearn']['best_params']} | "
                    f"CV5(desc) Macro F1 = {sk_tuned['cv_5fold']['macro_f1']['mean']:.4f} "
                    f"(std {sk_tuned['cv_5fold']['macro_f1']['std']:.4f}) | "
                    f"Holdout Macro F1 = {sk_tuned['holdout']['macro_f1']:.4f} | "
                    f"depth={sk_tuned['holdout']['depth']}, leaves={sk_tuned['holdout']['n_leaves']}"
                ),
                f"  Artifacts      : {payload['artifacts']['run_json']}",
            ]
        )
    else:
        lines.append(f"  Artifacts      : {payload['artifacts']['run_json']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproducible Decision Tree (C4.5) hotel comparison and tuning runner."
    )
    parser.add_argument(
        "--search",
        choices=("none", "grid"),
        default="none",
        help="Experiment mode: 'none' (baseline only) or 'grid' (baseline + 3-fold grid & OOF threshold).",
    )
    args = parser.parse_args(argv)
    payload = run_experiment(search=args.search)
    print(_format_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
