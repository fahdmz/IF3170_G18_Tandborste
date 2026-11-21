import csv
import importlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src import data as data_module
from src.experiments import decision_tree as dtl_exp
from src.models.decision_tree import DecisionTree
from src.preprocessing.base import BaseTransformer, Pipeline
from src.preprocessing.presets import make_tree_pipeline, to_model_input
from src.preprocessing.transformation import FeatureEngineer


def _make_synthetic_hotel_data(
    n_train: int = 180, n_test: int = 30, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    def _build_features(n: int) -> pd.DataFrame:
        df = pd.DataFrame(index=np.arange(n))
        df["num_adults"] = rng.choice([1.0, 2.0, 3.0], size=n)
        df["num_children"] = rng.choice([0.0, 0.0, 1.0, 2.0], size=n)
        df["weekend_nights"] = rng.integers(0, 3, size=n).astype(float)
        df["weekday_nights"] = rng.integers(1, 5, size=n).astype(float)
        df["needs_parking"] = rng.choice([0.0, 1.0], size=n, p=[0.85, 0.15])
        df["days_before_arrival"] = rng.integers(0, 240, size=n).astype(float)
        df["arrival_year"] = rng.choice([2025.0, 2026.0], size=n)
        df["arrival_month"] = rng.integers(1, 13, size=n).astype(float)
        df["arrival_day"] = rng.integers(1, 28, size=n).astype(float)
        df["is_returning_guest"] = rng.choice([0.0, 1.0], size=n, p=[0.9, 0.1])
        df["prior_cancellations"] = rng.choice([0.0, 0.0, 1.0, 2.0], size=n)
        df["prior_completed_bookings"] = rng.integers(0, 4, size=n).astype(float)
        df["nightly_rate_idr"] = rng.normal(900_000.0, 250_000.0, size=n).clip(150_000.0)
        df["special_requests"] = rng.choice([0.0, 1.0, 2.0], size=n, p=[0.5, 0.35, 0.15])
        df["meal_package"] = pd.Series(
            rng.choice(["Room Only", "Breakfast", "Half Board"], size=n), dtype=object
        )
        df["room_category"] = pd.Series(
            rng.choice(["Type A", "Type B", "Type C"], size=n), dtype=object
        )
        df["booking_channel"] = pd.Series(
            rng.choice(["Online", "Direct", "Corporate"], size=n), dtype=object
        )
        # Inject missing values across numeric and categorical columns
        df.loc[0, "nightly_rate_idr"] = np.nan
        df.loc[1, "num_children"] = np.nan
        df.loc[2, "meal_package"] = None
        df.loc[3, "room_category"] = np.nan
        df.loc[4, "booking_channel"] = pd.NA
        return df

    X_train = _build_features(n_train)
    signal = (
        0.015 * X_train["days_before_arrival"].fillna(60.0)
        - 0.9 * X_train["special_requests"].fillna(0.0)
        + 0.7 * (X_train["booking_channel"] == "Online").astype(float)
        + 0.5 * (X_train["prior_cancellations"] > 0).astype(float)
        - 1.2
    )
    prob = 1.0 / (1.0 + np.exp(-signal))
    y_train = pd.Series((rng.random(n_train) < prob).astype(int), name=config.TARGET_COL)

    # Inject duplicate and zero-guest rows in train (clean_train should drop them on outer train only)
    X_train.loc[n_train - 2] = X_train.loc[10]
    y_train.loc[n_train - 2] = y_train.loc[10]
    X_train.loc[n_train - 1, "num_adults"] = 0.0
    X_train.loc[n_train - 1, "num_children"] = 0.0

    X_test = _build_features(n_test)
    # Inject duplicate and zero-guest rows in test to verify transform never drops test rows
    X_test.loc[n_test - 2] = X_test.loc[5]
    X_test.loc[n_test - 1, "num_adults"] = 0.0
    X_test.loc[n_test - 1, "num_children"] = 0.0
    X_test.loc[6, "room_category"] = "UnseenCategory"

    test_ids = pd.Series([f"ID_{i:04d}" for i in range(n_test)], name=config.ID_COL)
    sample_sub = pd.DataFrame(
        {config.ID_COL: test_ids, config.TARGET_COL: np.zeros(n_test, dtype=int)}
    )
    return X_train, y_train, X_test, test_ids, sample_sub


def _patch_experiment_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    test_ids: pd.Series,
    sample_sub: pd.DataFrame,
) -> dict[str, Path]:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_marker = data_dir / "train.csv"
    test_marker = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    train_marker.write_text("placeholder\n", encoding="utf-8")
    test_marker.write_text("placeholder\n", encoding="utf-8")
    sample_sub.to_csv(sample_path, index=False)

    out_dir = tmp_path / "output"
    model_dir = out_dir / "models"
    figure_dir = out_dir / "figures"

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "TRAIN_PATH", train_marker)
    monkeypatch.setattr(config, "TEST_PATH", test_marker)
    monkeypatch.setattr(config, "SAMPLE_SUBMISSION_PATH", sample_path)
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(config, "MODEL_DIR", model_dir)
    monkeypatch.setattr(config, "FIGURE_DIR", figure_dir)

    monkeypatch.setattr(
        data_module, "load_train", lambda: (X_train.copy(deep=True), y_train.copy(deep=True))
    )
    monkeypatch.setattr(
        data_module, "load_test", lambda: (X_test.copy(deep=True), test_ids.copy(deep=True))
    )
    return {
        "data_dir": data_dir,
        "output_dir": out_dir,
        "model_dir": model_dir,
        "figure_dir": figure_dir,
    }


def test_import_runner_is_io_free(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TRAIN_PATH", tmp_path / "does_not_exist_train.csv")
    monkeypatch.setattr(config, "TEST_PATH", tmp_path / "does_not_exist_test.csv")
    monkeypatch.setattr(
        config, "SAMPLE_SUBMISSION_PATH", tmp_path / "does_not_exist_sample.csv"
    )
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "output")

    def _forbid_read_csv(*args, **kwargs):
        raise AssertionError("Module import must not perform CSV I/O")

    monkeypatch.setattr(pd, "read_csv", _forbid_read_csv)
    reloaded = importlib.reload(dtl_exp)
    assert reloaded is dtl_exp
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("search_mode", ["none", "grid"])
def test_missing_dataset_raises_clear_error_without_writing_artifacts(
    monkeypatch, tmp_path, search_mode
):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Only create train.csv; leave test.csv and sample_submission.csv missing
    (data_dir / "train.csv").write_text("id,is_canceled\n", encoding="utf-8")

    out_dir = tmp_path / "output"
    monkeypatch.setattr(config, "TRAIN_PATH", data_dir / "train.csv")
    monkeypatch.setattr(config, "TEST_PATH", data_dir / "test.csv")
    monkeypatch.setattr(
        config, "SAMPLE_SUBMISSION_PATH", data_dir / "sample_submission.csv"
    )
    monkeypatch.setattr(config, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(config, "MODEL_DIR", out_dir / "models")
    monkeypatch.setattr(config, "FIGURE_DIR", out_dir / "figures")

    with pytest.raises(FileNotFoundError, match=r"test\.csv.*sample_submission\.csv"):
        dtl_exp.run_experiment(search=search_mode)

    assert not out_dir.exists()


def test_infer_tree_schema_uses_unfitted_config_and_matches_fold_output():
    num_cols, cat_cols = data_module.get_feature_types()
    tree_factory = lambda: make_tree_pipeline(num_cols, cat_cols)

    columns, schema_cats, cat_indices = dtl_exp.infer_tree_schema(tree_factory)
    expected_columns = list(num_cols) + list(FeatureEngineer.ENGINEERED_COLS) + list(cat_cols)
    assert columns == expected_columns
    assert schema_cats == list(cat_cols)
    assert cat_indices == [expected_columns.index(c) for c in cat_cols]

    X_train, y_train, _, _, _ = _make_synthetic_hotel_data(n_train=90, n_test=10)
    checked = dtl_exp._SchemaCheckedPipeline(
        tree_factory(),
        expected_columns=columns,
        categorical_cols=schema_cats,
        expected_cat_indices=cat_indices,
    )
    Xt_tr = checked.fit_transform(X_train.iloc[:60], y_train.iloc[:60])
    Xt_va = checked.transform(X_train.iloc[60:])
    _, actual_indices = to_model_input(Xt_tr, schema_cats)
    assert list(Xt_tr.columns) == list(Xt_va.columns) == columns
    assert actual_indices == cat_indices


def test_schema_checked_pipeline_rejects_column_drift_and_dropped_rows():
    class _ReorderingTransformer(BaseTransformer):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X[list(reversed(X.columns))]

    class _DroppingTransformer(BaseTransformer):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X.iloc[:-1]

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "cat": ["x", "y", "x"]})
    drift_pipe = dtl_exp._SchemaCheckedPipeline(
        Pipeline([("flip", _ReorderingTransformer())]),
        expected_columns=["a", "cat"],
        categorical_cols=["cat"],
        expected_cat_indices=[1],
    )
    with pytest.raises(ValueError, match="schema mismatch"):
        drift_pipe.fit_transform(df)

    drop_pipe = dtl_exp._SchemaCheckedPipeline(
        Pipeline([("drop", _DroppingTransformer())]),
        expected_columns=["a", "cat"],
        categorical_cols=["cat"],
        expected_cat_indices=[1],
    )
    with pytest.raises(ValueError, match="row count"):
        drop_pipe.transform(df)


def test_roc_auc_uses_positive_class_1_column_even_when_classes_reordered():
    class _StubClassifier:
        def __init__(self, classes, proba):
            self.classes_ = np.asarray(classes)
            self._proba = np.asarray(proba, dtype=float)

        def predict_proba(self, X):
            return self._proba

        def predict(self, X):
            return self.classes_[np.argmax(self._proba, axis=1)]

        def get_depth(self):
            return 1

        def get_n_leaves(self):
            return 2

    y_val = np.array([0, 0, 1, 1])
    X_dummy = np.zeros((4, 2))
    # Standard order [0, 1]: column 1 has high prob for y=1
    standard = _StubClassifier(
        [0, 1], [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    )
    # Reversed order [1, 0]: column 0 has high prob for y=1
    reversed_cls = _StubClassifier(
        [1, 0], [[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]]
    )
    res_std = dtl_exp.evaluate_holdout(standard, X_dummy, y_val, threshold=0.50)
    res_rev = dtl_exp.evaluate_holdout(reversed_cls, X_dummy, y_val, threshold=0.50)
    assert res_std["roc_auc"] == pytest.approx(1.0)
    assert res_rev["roc_auc"] == pytest.approx(1.0)
    assert res_std["macro_f1"] == pytest.approx(1.0)
    assert res_rev["macro_f1"] == pytest.approx(1.0)


def test_threshold_selection_macro_f1_and_deterministic_tie_breaking():
    assert dtl_exp.THRESHOLD_CANDIDATES[0] == 0.05
    assert dtl_exp.THRESHOLD_CANDIDATES[-1] == 0.95
    assert 0.50 in dtl_exp.THRESHOLD_CANDIDATES
    assert len(dtl_exp.THRESHOLD_CANDIDATES) == 91

    # Optimal threshold clearly away from 0.50 (at 0.30, all positives >= 0.30 and negatives < 0.30)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.10, 0.20, 0.25, 0.32, 0.40, 0.45])
    res = dtl_exp.select_macro_f1_threshold(y_true, y_proba)
    # Perfect separation for every t in [0.26, 0.32]; closest to 0.50 is 0.32
    assert res["threshold"] == 0.32
    assert res["oof_macro_f1"] == pytest.approx(1.0)
    assert res["oof_macro_f1"] > res["oof_macro_f1_at_0_50"]

    # Equal-distance tie around 0.50 (0.42 vs 0.58 - tricky IEEE-754 pair): must pick smaller (0.42)
    tie_res = dtl_exp.select_macro_f1_threshold(
        y_true=np.array([0, 0, 1, 1]),
        y_proba=np.array([0.10, 0.20, 0.80, 0.90]),
        candidates=(0.42, 0.58),
    )
    assert tie_res["threshold"] == 0.42

    # Asymmetric tie: 0.46 is closer to 0.50 than 0.60
    closer_res = dtl_exp.select_macro_f1_threshold(
        y_true=np.array([0, 0, 1, 1]),
        y_proba=np.array([0.10, 0.20, 0.80, 0.90]),
        candidates=(0.40, 0.46, 0.60),
    )
    assert closer_res["threshold"] == 0.46


def test_fit_receives_only_training_rows_and_holdout_never_enters_search(
    monkeypatch, tmp_path
):
    X_train, y_train, X_test, test_ids, sample_sub = _make_synthetic_hotel_data(
        n_train=150, n_test=24, seed=42
    )
    _patch_experiment_env(
        monkeypatch, tmp_path, X_train, y_train, X_test, test_ids, sample_sub
    )

    # Record the exact holdout split so we can assert no fit ever sees holdout/test rows
    expected_tr, expected_va, expected_y_tr, _ = dtl_exp.sk_train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=config.SEED
    )
    holdout_signatures = set(expected_va["nightly_rate_idr"].dropna().round(6).tolist())
    test_signatures = set(X_test["nightly_rate_idr"].dropna().round(6).tolist())

    fit_row_counts: list[int] = []
    transform_row_counts: list[int] = []
    orig_fit_transform = Pipeline.fit_transform
    orig_transform = Pipeline.transform

    def _spy_fit_transform(self, X, y=None):
        seen = set(X["nightly_rate_idr"].dropna().round(6).tolist())
        assert seen.isdisjoint(holdout_signatures), "Holdout row leaked into fit_transform!"
        assert seen.isdisjoint(test_signatures), "Test row leaked into fit_transform!"
        fit_row_counts.append(len(X))
        out = orig_fit_transform(self, X, y)
        assert len(out) == len(X), "fit_transform dropped rows!"
        return out

    def _spy_transform(self, X):
        out = orig_transform(self, X)
        assert len(out) == len(X), "transform dropped rows!"
        transform_row_counts.append(len(X))
        return out

    monkeypatch.setattr(Pipeline, "fit_transform", _spy_fit_transform)
    monkeypatch.setattr(Pipeline, "transform", _spy_transform)

    completed_search = []
    holdout_order = []
    original_select = dtl_exp.select_macro_f1_threshold
    original_evaluate = dtl_exp.evaluate_holdout
    original_fit = DecisionTree.fit
    fitted_tree_sizes = []

    def _record_locked_threshold(*args, **kwargs):
        result = original_select(*args, **kwargs)
        completed_search.append(result["threshold"])
        holdout_order.append("threshold")
        return result

    def _record_holdout(*args, **kwargs):
        holdout_order.append("holdout")
        return original_evaluate(*args, **kwargs)

    def _record_fit(self, X, y):
        if self.max_depth is not None:
            fitted_tree_sizes.append(len(X))
        return original_fit(self, X, y)

    monkeypatch.setattr(dtl_exp, "select_macro_f1_threshold", _record_locked_threshold)
    monkeypatch.setattr(dtl_exp, "evaluate_holdout", _record_holdout)
    monkeypatch.setattr(DecisionTree, "fit", _record_fit)

    # Run 1: real holdout labels
    res1 = dtl_exp.run_experiment(
        search="grid",
        grid_param_space={"max_depth": [4, 6], "min_samples_leaf": [5, 10]},
        cv_grid_splits=3,
        cv_eval_splits=5,
        top_n=10,
    )
    assert len(fit_row_counts) > 0
    assert len(X_test) in transform_row_counts
    assert completed_search == [res1["threshold_selection"]["threshold"]]
    assert holdout_order[0] == "threshold"
    assert holdout_order.count("holdout") >= 4
    X_clean, y_clean = dtl_exp.clean_train(expected_tr, expected_y_tr)
    eval_cv = dtl_exp.StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    eval_fold_sizes = [len(train) for train, _ in eval_cv.split(X_clean, y_clean)]
    assert sorted(size for size in fitted_tree_sizes if size in eval_fold_sizes) == sorted(
        eval_fold_sizes
    )

    # Run 2: flip holdout y_val labels ONLY (outer train unchanged) -> search & threshold MUST match
    orig_split = dtl_exp.sk_train_test_split

    def _flipped_holdout_split(*args, **kwargs):
        X_tr, X_va, y_tr, y_va = orig_split(*args, **kwargs)
        return X_tr, X_va, y_tr, (1 - y_va)

    monkeypatch.setattr(dtl_exp, "sk_train_test_split", _flipped_holdout_split)
    res2 = dtl_exp.run_experiment(
        search="grid",
        grid_param_space={"max_depth": [4, 6], "min_samples_leaf": [5, 10]},
        cv_grid_splits=3,
        cv_eval_splits=5,
        top_n=10,
    )

    assert res1["grid_search"]["c45"]["best_params"] == res2["grid_search"]["c45"]["best_params"]
    assert res1["grid_search"]["c45"]["trials"] == res2["grid_search"]["c45"]["trials"]
    assert res1["threshold_selection"]["threshold"] == res2["threshold_selection"]["threshold"]
    assert (
        res1["threshold_selection"]["oof_macro_f1"]
        == res2["threshold_selection"]["oof_macro_f1"]
    )
    assert res1["tuned"]["c45_tuned"]["cv_5fold"] == res2["tuned"]["c45_tuned"]["cv_5fold"]
    # Holdout scores should differ because y_val was flipped
    assert (
        res1["tuned"]["c45_tuned"]["holdout"]["macro_f1"]
        != res2["tuned"]["c45_tuned"]["holdout"]["macro_f1"]
    )


@pytest.mark.parametrize("failure_point", ["render", "publish"])
def test_failed_grid_preserves_previous_artifacts(monkeypatch, tmp_path, failure_point):
    paths = _patch_experiment_env(monkeypatch, tmp_path, *_make_synthetic_hotel_data())
    output = paths["output_dir"]
    previous_files = [
        output / "decision_tree" / "comparison.csv",
        output / "decision_tree" / "run.json",
        output / "dtl_tuning_comparison.csv",
        paths["model_dir"] / "decision_tree_tuned.pkl",
        paths["model_dir"] / "decision_tree_tuned_pipeline.pkl",
        paths["figure_dir"] / "decision_tree_tuned.dot",
    ]
    for path in previous_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())

    def fail_render(*args, **kwargs):
        raise OSError("simulated artifact failure")

    original_replace = Path.replace

    def fail_publication(self, target):
        if Path(target) == output / "decision_tree" / "run.json":
            raise OSError("simulated artifact failure")
        return original_replace(self, target)

    if failure_point == "render":
        monkeypatch.setattr(dtl_exp, "plot_decision_tree", fail_render)
    else:
        monkeypatch.setattr(Path, "replace", fail_publication)

    with pytest.raises(OSError, match="simulated artifact failure"):
        dtl_exp.run_experiment(
            search="grid", grid_param_space={"max_depth": [4], "min_samples_leaf": [5]}
        )
    assert {p: p.read_bytes() for p in output.rglob("*") if p.is_file()} == {
        p: p.name.encode() for p in previous_files
    }


@pytest.mark.parametrize("saved_threshold", [None, -1.0])
def test_grid_rejects_invalid_reloaded_threshold(monkeypatch, tmp_path, saved_threshold):
    paths = _patch_experiment_env(monkeypatch, tmp_path, *_make_synthetic_hotel_data())
    original_load = DecisionTree.load

    def read_bad_model(path):
        model = original_load(path)
        if saved_threshold is None:
            del model.decision_threshold_
        else:
            model.decision_threshold_ = saved_threshold
        return model

    monkeypatch.setattr(DecisionTree, "load", staticmethod(read_bad_model))
    with pytest.raises(RuntimeError, match="Reloaded.*threshold"):
        dtl_exp.run_experiment(
            search="grid", grid_param_space={"max_depth": [4], "min_samples_leaf": [5]}
        )
    assert not any(p.is_file() for p in paths["output_dir"].rglob("*"))


def test_baseline_none_mode_writes_only_baseline_artifacts(monkeypatch, tmp_path):
    X_train, y_train, X_test, test_ids, sample_sub = _make_synthetic_hotel_data(
        n_train=150, n_test=20, seed=42
    )
    dirs = _patch_experiment_env(
        monkeypatch, tmp_path, X_train, y_train, X_test, test_ids, sample_sub
    )

    payload = dtl_exp.run_experiment(search="none", cv_eval_splits=3)

    baseline_dir = dirs["output_dir"] / "decision_tree" / "baseline"
    assert (baseline_dir / "comparison.csv").is_file()
    assert (baseline_dir / "run.json").is_file()
    # Tuned artifacts must NOT exist in --search none
    assert not (dirs["output_dir"] / "decision_tree" / "run.json").exists()
    assert not (dirs["output_dir"] / "dtl_tuning_comparison.csv").exists()
    assert not (dirs["model_dir"] / "decision_tree_tuned.pkl").exists()
    assert not (dirs["figure_dir"] / "decision_tree_tuned.png").exists()

    parsed = json.loads((baseline_dir / "run.json").read_text(encoding="utf-8"))
    assert parsed["search_mode"] == "none"
    assert parsed["bonus_status"]["optimization"] == "DEFERRED"
    assert payload["baselines"]["c45_default"]["holdout"]["depth"] >= 0


def test_grid_mode_artifacts_schema_reload_and_reproducibility(monkeypatch, tmp_path):
    X_train, y_train, X_test, test_ids, sample_sub = _make_synthetic_hotel_data(
        n_train=180, n_test=25, seed=42
    )
    dirs = _patch_experiment_env(
        monkeypatch, tmp_path, X_train, y_train, X_test, test_ids, sample_sub
    )

    run1 = dtl_exp.run_experiment(
        search="grid",
        grid_param_space={"max_depth": [4, 6], "min_samples_leaf": [5, 10]},
        cv_grid_splits=3,
        cv_eval_splits=5,
        top_n=12,
    )

    # Verify all expected artifacts exist
    comp_path = dirs["output_dir"] / "decision_tree" / "comparison.csv"
    json_path = dirs["output_dir"] / "decision_tree" / "run.json"
    team_csv_path = dirs["output_dir"] / "dtl_tuning_comparison.csv"
    model_path = dirs["model_dir"] / "decision_tree_tuned.pkl"
    pipe_path = dirs["model_dir"] / "decision_tree_tuned_pipeline.pkl"
    dot_path = dirs["figure_dir"] / "decision_tree_tuned.dot"
    png_path = dirs["figure_dir"] / "decision_tree_tuned.png"

    for p in (comp_path, json_path, team_csv_path, model_path, pipe_path, dot_path, png_path):
        assert p.is_file(), f"Missing artifact: {p}"

    # Validate JSON structure and strict compliance (no NaN/Infinity)
    raw_json = json_path.read_text(encoding="utf-8")
    assert "NaN" not in raw_json and "Infinity" not in raw_json
    parsed = json.loads(raw_json)
    assert parsed["bonus_status"]["optimization"] == "DEFERRED"
    assert "descriptive" in parsed["tuned"]["c45_tuned"]["cv_5fold_protocol"]
    assert parsed["persistence_verification"]["holdout_probabilities_exact"] is True
    assert parsed["persistence_verification"]["holdout_labels_exact"] is True
    assert parsed["kaggle_readiness"]["status"] == "READY"
    assert parsed["kaggle_readiness"]["submission_written"] is False
    assert parsed["kaggle_readiness"]["transformed_rows"] == len(X_test)

    # Best params must actually match best trial from 3-fold grid search
    trials = parsed["grid_search"]["c45"]["trials"]
    assert len(trials) == 4
    expected_best = max(trials, key=lambda t: t["mean_score"])["params"]
    assert parsed["grid_search"]["c45"]["best_params"] == expected_best
    assert parsed["tuned"]["c45_tuned"]["params"]["max_depth"] == expected_best["max_depth"]
    assert (
        parsed["tuned"]["c45_tuned"]["params"]["min_samples_leaf"]
        == expected_best["min_samples_leaf"]
    )

    # Validate comparison.csv schema & protocols
    with open(comp_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["model", "protocol", "fold", "metric", "value"]
        protocols = {row["protocol"] for row in reader}
    assert {
        "grid_3fold",
        "oof_threshold_sweep",
        "cv_5fold",
        "cv_5fold_descriptive",
        "holdout",
    } <= protocols

    # Validate team summary CSV format
    with open(team_csv_path, encoding="utf-8") as f:
        team_reader = csv.DictReader(f)
        assert team_reader.fieldnames == dtl_exp.TEAM_COMPARISON_COLUMNS
        team_rows = list(team_reader)
    assert len(team_rows) == 4
    by_key = {(r["model"], r["implementation"]): r for r in team_rows}
    assert by_key[("Decision Tree (default)", "from-scratch")]["threshold"] == ""
    assert by_key[("Decision Tree (default)", "scikit-learn")]["threshold"] == ""
    assert by_key[("Decision Tree (tuned)", "scikit-learn")]["threshold"] == ""
    assert float(by_key[("Decision Tree (tuned)", "from-scratch")]["threshold"]) == pytest.approx(
        parsed["threshold_selection"]["threshold"]
    )
    assert by_key[("Decision Tree (tuned)", "from-scratch")]["cv_mean_macro_f1_at_0_50"] != ""

    # Verify saved model + pipeline reload cleanly and produce valid binary test predictions
    reloaded_model = DecisionTree.load(model_path)
    with open(pipe_path, "rb") as f:
        reloaded_pipe = pickle.load(f)
    Xt_test = reloaded_pipe.transform(X_test)
    assert len(Xt_test) == len(X_test)
    X_test_arr, _ = to_model_input(Xt_test, data_module.CATEGORICAL_COLS)
    test_proba = reloaded_model.predict_proba(X_test_arr)[:, 1]
    assert test_proba.shape == (len(X_test),) and np.isfinite(test_proba).all()
    assert dot_path.read_text(encoding="utf-8").startswith("digraph DecisionTree {")
    assert png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    # Reproducibility check: second run yields identical best params, threshold, CV, and holdout
    run2 = dtl_exp.run_experiment(
        search="grid",
        grid_param_space={"max_depth": [4, 6], "min_samples_leaf": [5, 10]},
        cv_grid_splits=3,
        cv_eval_splits=5,
        top_n=12,
    )
    assert run1["grid_search"]["c45"]["best_params"] == run2["grid_search"]["c45"]["best_params"]
    assert run1["threshold_selection"]["threshold"] == run2["threshold_selection"]["threshold"]
    assert run1["tuned"]["c45_tuned"]["cv_5fold"] == run2["tuned"]["c45_tuned"]["cv_5fold"]
    assert run1["tuned"]["c45_tuned"]["holdout"] == run2["tuned"]["c45_tuned"]["holdout"]


def test_cli_main_none_mode(monkeypatch, tmp_path, capsys):
    X_train, y_train, X_test, test_ids, sample_sub = _make_synthetic_hotel_data(
        n_train=140, n_test=20, seed=42
    )
    _patch_experiment_env(
        monkeypatch, tmp_path, X_train, y_train, X_test, test_ids, sample_sub
    )
    exit_code = dtl_exp.main(["--search", "none"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "DTL Experiment complete (search=none" in out
    assert "C4.5 default" in out
