"""Logistic regression experiments on the hotel-booking dataset."""

import itertools
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from src import config, data
from src.evaluation.metrics import (
    accuracy,
    confusion_matrix,
    f1_score,
    macro_f1,
    precision,
    recall,
    roc_auc,
)
from src.evaluation.model_selection import StratifiedKFold, cross_validate
from src.models.logistic_regression import AdamLogisticRegression, LogisticRegression
from src.preprocessing.cleaning import clean_train
from src.preprocessing.presets import make_numeric_pipeline, to_model_input
from src.visualization.logreg_contour import plot_logreg_contour

OUTPUT_DIR = config.OUTPUT_DIR / "logreg_comparison"
LABEL_METRICS = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1_score,
    "macro_f1": macro_f1,
}


def load_clean():
    """Return cleaned training features, labels, and the numeric and categorical column lists."""
    X, y = data.load_train()
    numerical, categorical = data.get_feature_types()
    return (*clean_train(X, y), numerical, categorical)


def _config(penalty, reg_lambda, learning_rate, batch_size, class_weight) -> dict:
    """One ``LogisticRegression`` parameter dict with the shared training budget."""
    return {
        "penalty": penalty,
        "reg_lambda": reg_lambda,
        "l1_ratio": 0.5,
        "learning_rate": learning_rate,
        "n_iter": 400,
        "batch_size": batch_size,
        "tol": 1e-5,
        "class_weight": class_weight,
        "random_state": config.SEED,
    }


def logreg_grid() -> list[dict]:
    """Small grid over penalty, ``reg_lambda``, learning rate, batch size, and class weight."""
    grid = [
        _config(penalty, reg_lambda, 0.5, None, class_weight)
        for penalty, reg_lambda, class_weight in itertools.product(
            ["l2", "l1", "elasticnet"],
            [1e-3, 1e-2, 1e-1],
            [None, "balanced"],
        )
    ]
    for learning_rate, batch_size in ((0.2, None), (0.5, 512)):
        for class_weight in (None, "balanced"):
            grid.append(_config("l2", 1e-2, learning_rate, batch_size, class_weight))
    return grid


def adam_grid() -> list[dict]:
    """L2 Adam settings: two penalty strengths, two learning rates, and both class weights."""
    grid = []
    for reg_lambda, learning_rate, class_weight in itertools.product(
        [1e-3, 1e-2],
        [0.05, 0.1],
        [None, "balanced"],
    ):
        grid.append(
            {
                "penalty": "l2",
                "reg_lambda": reg_lambda,
                "learning_rate": learning_rate,
                "n_iter": 250,
                "batch_size": None,
                "tol": 1e-5,
                "class_weight": class_weight,
                "random_state": config.SEED,
            }
        )
    return grid


def _score_row(params, scores, elapsed):
    """Flatten one configuration into means and standard deviations across folds."""
    row = dict(params)
    for name, values in scores.items():
        row[name] = float(np.mean(values))
        row[f"{name}_std"] = float(np.std(values))
    row["fit_sec"] = elapsed / 5.0
    return row


def tune(grid, model_cls, X, y, pipeline_factory, cv) -> pd.DataFrame:
    """Score every configuration with the team's ``cross_validate`` and sort by macro F1."""
    rows = []
    for index, params in enumerate(grid, start=1):
        model = model_cls(**params)
        started = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scores = cross_validate(model, pipeline_factory, X, y, cv, LABEL_METRICS)
        rows.append(_score_row(params, scores, time.time() - started))
        print(f"  [{index}/{len(grid)}] macro_f1={rows[-1]['macro_f1']:.4f} {params['penalty']} "
              f"lambda={params['reg_lambda']} lr={params['learning_rate']} batch={params['batch_size']} "
              f"class_weight={params['class_weight']}")
    frame = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    return frame.reset_index(drop=True)


def prepare_folds(X, y, numerical, categorical, n_splits=5):
    """Return ``(X_train, y_train, X_val, y_val)`` for each stratified fold."""
    folds = []
    cv = StratifiedKFold(n_splits, True, config.SEED)
    for train_idx, val_idx in cv.split(X, y):
        pipeline = make_numeric_pipeline(numerical, categorical)
        transformed_train = pipeline.fit_transform(X.iloc[train_idx], y.iloc[train_idx])
        transformed_val = pipeline.transform(X.iloc[val_idx])
        features_train, _ = to_model_input(transformed_train, categorical)
        features_val, _ = to_model_input(transformed_val, categorical)
        folds.append(
            (
                features_train,
                np.asarray(y.iloc[train_idx]).ravel(),
                features_val,
                np.asarray(y.iloc[val_idx]).ravel(),
            )
        )
    return folds


def _sklearn_model(params, n_train):
    """Saga logistic regression whose penalty matches ``params`` on ``n_train`` rows."""
    shared = dict(
        solver="saga",
        max_iter=5000,
        tol=1e-4,
        random_state=config.SEED,
        class_weight=params["class_weight"],
        fit_intercept=params.get("fit_intercept", True),
    )
    penalty = params["penalty"]
    reg_lambda = float(params["reg_lambda"])
    if penalty in (None, "none") or reg_lambda == 0.0:
        return SkLogisticRegression(C=np.inf, **shared)
    ratio = {"l2": 0.0, "l1": 1.0, "elasticnet": float(params.get("l1_ratio", 0.5))}[penalty]
    return SkLogisticRegression(C=1.0 / (reg_lambda * n_train), l1_ratio=ratio, **shared)


def _evaluate_split(y_true, y_pred, y_score):
    """Label metrics, ROC AUC from continuous scores, and the confusion counts."""
    matrix = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "roc_auc": roc_auc(y_true, y_score),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }


def compare_with_sklearn(configs: dict[str, dict], folds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare each scratch configuration with saga on the same folds."""
    rows = []
    for name, params in configs.items():
        for impl in ("scratch", "sklearn"):
            for fold, (features_train, y_train, features_val, y_val) in enumerate(folds):
                if impl == "scratch":
                    model = LogisticRegression(**params)
                else:
                    model = _sklearn_model(params, len(y_train))
                started = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    model.fit(features_train, y_train)
                elapsed = time.time() - started
                predicted = np.asarray(model.predict(features_val)).ravel()
                scores = np.asarray(model.decision_function(features_val), dtype=float).ravel()
                rows.append(
                    {
                        "config": name,
                        "impl": impl,
                        "fold": fold,
                        "fit_sec": elapsed,
                        "n_train": len(y_train),
                        **_evaluate_split(y_val, predicted, scores),
                    }
                )
    per_fold = pd.DataFrame(rows)
    summary = (
        per_fold.groupby(["config", "impl"], sort=False)
        .agg(
            n_train=("n_train", "mean"),
            fit_sec=("fit_sec", "mean"),
            accuracy=("accuracy", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            macro_f1=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            roc_auc=("roc_auc", "mean"),
            tn=("tn", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tp=("tp", "sum"),
        )
        .round(4)
    )
    return per_fold, summary


def _params_from_row(row, fields):
    """Turn one tuning row back into constructor arguments."""
    params = {}
    for field in fields:
        value = row[field]
        if field == "class_weight" and (value is None or (isinstance(value, float) and np.isnan(value))):
            value = None
        if field == "batch_size" and (value is None or (isinstance(value, float) and np.isnan(value))):
            value = None
        elif field == "batch_size" and value is not None:
            value = int(value)
        if field in {"n_iter", "random_state"}:
            value = int(value)
        params[field] = value
    return params


def save_best_model(params, X, y, numerical, categorical):
    """Fit ``params`` on the full cleaned training matrix and pickle the model."""
    pipeline = make_numeric_pipeline(numerical, categorical)
    features, _ = to_model_input(pipeline.fit_transform(X, y), categorical)
    model = LogisticRegression(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(features, np.asarray(y).ravel())
    path = config.MODEL_DIR / "logistic_regression.pkl"
    model.save(path)
    return path, model


def save_contour_gif():
    """Fit a one-feature L2 model with history and write the contour GIF."""
    features, labels = make_classification(
        n_samples=300,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        random_state=config.SEED,
    )
    features = (features - features.mean()) / features.std()
    model = LogisticRegression(
        penalty="l2",
        reg_lambda=0.01,
        learning_rate=0.8,
        n_iter=40,
        tol=1e-8,
        fit_intercept=True,
        record_history=True,
        random_state=config.SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(features, labels)
    return plot_logreg_contour(model, features, labels, config.FIGURE_DIR / "logreg_contour.gif")


def main() -> None:
    """Tune both solvers, write the comparison tables, and save the model and GIF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, numerical, categorical = load_clean()
    pipeline_factory = lambda: make_numeric_pipeline(numerical, categorical)
    cv = StratifiedKFold(5, True, config.SEED)

    print("Tuning logistic regression")
    tuned = tune(logreg_grid(), LogisticRegression, X, y, pipeline_factory, cv)
    tuned.to_csv(OUTPUT_DIR / "logreg_tuning_cv.csv", index=False)
    print(tuned.head(8).to_string(index=False))

    fields = [
        "penalty",
        "reg_lambda",
        "l1_ratio",
        "learning_rate",
        "n_iter",
        "batch_size",
        "tol",
        "class_weight",
        "random_state",
    ]
    best = _params_from_row(tuned.iloc[0], fields)
    print("Best params:", best)

    print("Tuning Adam")
    adam = tune(adam_grid(), AdamLogisticRegression, X, y, pipeline_factory, cv)
    adam.to_csv(OUTPUT_DIR / "adam_tuning_cv.csv", index=False)
    print(adam.head(8).to_string(index=False))

    print("Scratch versus sklearn")
    folds = prepare_folds(X, y, numerical, categorical)
    per_fold, summary = compare_with_sklearn({"best": best}, folds)
    per_fold.to_csv(OUTPUT_DIR / "logreg_scratch_vs_sklearn_folds.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "logreg_scratch_vs_sklearn_summary.csv")
    print(summary.to_string())

    team_scores = cross_validate(LogisticRegression(**best), pipeline_factory, X, y, cv, LABEL_METRICS)
    team_macro = float(np.mean(team_scores["macro_f1"]))
    scratch_macro = float(summary.loc[("best", "scratch"), "macro_f1"])
    print(f"cross_validate macro_f1={team_macro:.4f}  fold-loop macro_f1={scratch_macro:.4f}")

    model_path, _ = save_best_model(best, X, y, numerical, categorical)
    gif_path = save_contour_gif()
    print("Saved:", model_path.relative_to(config.ROOT_DIR))
    print("Saved:", gif_path.relative_to(config.ROOT_DIR))


if __name__ == "__main__":
    main()
