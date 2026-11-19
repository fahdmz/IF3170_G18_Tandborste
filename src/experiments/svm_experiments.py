"""Eksperimen SVM pada dataset asli: tuning Macro F1 dan perbandingan from scratch vs sklearn.

Jalankan dari root repo:
    python -m src.experiments.svm_experiments            # perbandingan + Pegasos, tulis ke output/svm_comparison/
    python -m src.experiments.svm_experiments --tune     # tambah grid tuning SVM (lama, ~1 jam di 10 core)

Fold, Macro F1, dan ROC AUC dihitung dengan NumPy di modul ini karena `src/evaluation` belum tersedia
di `main`. Setelah `cross_validate` milik tim di-merge, angka akhir untuk laporan sebaiknya diulang dengan itu.
sklearn hanya dipakai di `compare_with_sklearn` sebagai pembanding.
"""
import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src import config, data
from src.models.svm import SVM, PegasosSVM
from src.preprocessing.cleaning import clean_train
from src.preprocessing.presets import make_numeric_pipeline, to_model_input

OUTPUT_DIR = config.OUTPUT_DIR / "svm_comparison"

BEST_SVM_PARAMS = {"kernel": "rbf", "C": 3.0, "gamma": 0.1, "max_train_samples": None, "max_iter": -1}


def stratified_folds(y: np.ndarray, n_splits: int = 5, seed: int = config.SEED) -> list[np.ndarray]:
    """Indeks validasi per fold; tiap kelas diacak lalu dibagi rata ke semua fold."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        for k, part in enumerate(np.array_split(idx, n_splits)):
            folds[k].append(part)
    return [np.sort(np.concatenate(parts)) for parts in folds]


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for label in (0, 1):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0)
    return float(np.mean(scores))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC = peluang skor positif > skor negatif (statistik Mann-Whitney, ties dihitung setengah)."""
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score))
    start = 0
    while start < len(sorted_scores):
        end = start
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    n_pos = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    candidates = np.quantile(y_score, np.linspace(0.2, 0.9, 141))
    scores = [macro_f1(y_true, (y_score > t).astype(int)) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def prepare_folds(use_pca: bool = False, n_splits: int = 5) -> list[tuple[np.ndarray, ...]]:
    """(A_tr, y_tr, A_va, y_va) per fold; pipeline di-fit hanya pada train fold."""
    X, y = data.load_train()
    num, cat = data.get_feature_types()
    X, y = clean_train(X, y)
    y = y.to_numpy()
    folds = []
    all_idx = np.arange(len(y))
    for va in stratified_folds(y, n_splits):
        tr = np.setdiff1d(all_idx, va)
        pipeline = make_numeric_pipeline(num, cat, use_pca=use_pca)
        A_tr, _ = to_model_input(pipeline.fit_transform(X.iloc[tr], y[tr]), cat)
        A_va, _ = to_model_input(pipeline.transform(X.iloc[va]), cat)
        folds.append((A_tr, y[tr], A_va, y[va]))
    return folds


def cv_scores(model, folds) -> dict:
    """Macro F1 threshold 0, Macro F1 dengan threshold cross-fitted, dan AUC rata-rata per fold.

    Threshold untuk fold k dipilih dari skor out-of-fold fold lain, sehingga tidak bocor ke fold k.
    """
    oof, fit_sec = [], []
    for A_tr, y_tr, A_va, y_va in folds:
        start = time.time()
        fitted = model.clone().fit(A_tr, y_tr)
        fit_sec.append(time.time() - start)
        oof.append((y_va, fitted.decision_function(A_va)))
    tuned, thresholds = [], []
    for k, (y_va, score) in enumerate(oof):
        others = [o for j, o in enumerate(oof) if j != k]
        t = best_threshold(np.concatenate([o[0] for o in others]), np.concatenate([o[1] for o in others]))
        thresholds.append(t)
        tuned.append(macro_f1(y_va, (score > t).astype(int)))
    return {
        "macro_f1_thr0": float(np.mean([macro_f1(y, (s > 0).astype(int)) for y, s in oof])),
        "macro_f1_tuned_thr": float(np.mean(tuned)),
        "macro_f1_tuned_thr_std": float(np.std(tuned)),
        "roc_auc": float(np.mean([roc_auc(y, s) for y, s in oof])),
        "mean_threshold": float(np.mean(thresholds)),
        "fit_sec": float(np.mean(fit_sec)),
    }


def _evaluate(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(y),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "macro_f1": macro_f1(y, pred),
        "roc_auc": roc_auc(y, score),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def _sklearn_counterpart(model, n_train: int):
    from sklearn.svm import SVC, LinearSVC

    if isinstance(model, PegasosSVM):
        # LinearSVC (liblinear) juga meregularisasi intercept, sama dengan Pegasos dengan fitur konstan.
        return LinearSVC(C=1.0 / (model.lam * n_train), loss="hinge", max_iter=100_000,
                         class_weight=model.class_weight, random_state=config.SEED)
    return SVC(kernel=model.kernel, C=model.C, degree=model.degree, gamma=model.gamma, coef0=model.coef0,
               tol=model.tol, class_weight=model.class_weight)


def _compare_one(job):
    name, model, folds = job
    rows = []
    for k, (A_tr, y_tr, A_va, y_va) in enumerate(folds):
        own = model.clone()
        if isinstance(own, SVM):
            # Kedua implementasi dilatih pada baris yang persis sama (hasil subsampling yang sama).
            own.classes_ = np.unique(y_tr)
            A_tr, y_tr = own._subsample(A_tr, y_tr)
        start = time.time()
        own.fit(A_tr, y_tr)
        own_sec = time.time() - start
        ref = _sklearn_counterpart(own, len(y_tr))
        start = time.time()
        ref.fit(A_tr, y_tr)
        ref_sec = time.time() - start
        agreement = float(np.mean(own.predict(A_va) == ref.predict(A_va)))
        for impl, fitted, sec in (("scratch", own, own_sec), ("sklearn", ref, ref_sec)):
            n_sv = len(fitted.support_) if hasattr(fitted, "support_") else np.nan
            rows.append({"config": name, "impl": impl, "fold": k, "fit_sec": sec, "n_train": len(y_tr),
                         "n_sv": n_sv, "agreement": agreement,
                         **_evaluate(y_va, fitted.predict(A_va), fitted.decision_function(A_va))})
    return rows


def compare_with_sklearn(models: dict, folds, n_jobs: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bandingkan tiap model dengan padanan sklearn-nya di fold yang sama. Hasil: (per fold, ringkasan)."""
    jobs = [(name, model, folds) for name, model in models.items()]
    if n_jobs == 1:
        parts = map(_compare_one, jobs)
        per_fold = pd.DataFrame([r for part in parts for r in part])
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            per_fold = pd.DataFrame([r for part in ex.map(_compare_one, jobs) for r in part])
    summary = per_fold.groupby(["config", "impl"]).agg(
        n_train=("n_train", "mean"), n_sv=("n_sv", "mean"), fit_sec=("fit_sec", "mean"),
        accuracy=("accuracy", "mean"), precision=("precision", "mean"), recall=("recall", "mean"),
        f1=("f1", "mean"), macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        roc_auc=("roc_auc", "mean"), agreement=("agreement", "mean"),
        tn=("tn", "sum"), fp=("fp", "sum"), fn=("fn", "sum"), tp=("tp", "sum"),
    ).round(4)
    return per_fold, summary


def _tune_one(job):
    params, folds = job
    if params.get("solver") == "pegasos":
        model = PegasosSVM(**{k: v for k, v in params.items() if k != "solver"})
    else:
        model = SVM(**params)
    return {**params, **cv_scores(model, folds)}


def tune(param_grid: list[dict], folds, n_jobs: int = 1) -> pd.DataFrame:
    """Evaluasi daftar konfigurasi (dict parameter SVM; tambahkan solver='pegasos' untuk PegasosSVM)."""
    jobs = [(params, folds) for params in param_grid]
    if n_jobs == 1:
        rows = list(map(_tune_one, jobs))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            rows = list(ex.map(_tune_one, jobs))
    return pd.DataFrame(rows).sort_values("macro_f1_tuned_thr", ascending=False).round(4)


def svm_grid() -> list[dict]:
    """Grid yang dipakai untuk memilih BEST_SVM_PARAMS (lihat output/svm_comparison/svm_tuning_cv.csv)."""
    grid = []
    for C, gamma, cw in itertools.product([1, 3, 5, 10], ["scale", 0.05, 0.1, 0.2], [None, "balanced"]):
        grid.append({"kernel": "rbf", "C": C, "gamma": gamma, "class_weight": cw, "max_iter": -1})
    for C, degree in itertools.product([1, 3], [2, 3]):
        grid.append({"kernel": "poly", "C": C, "degree": degree, "max_iter": -1})
    return grid


def pegasos_grid() -> list[dict]:
    return [
        {"solver": "pegasos", "lam": lam, "class_weight": cw, "n_epochs": 30}
        for lam, cw in itertools.product([1e-5, 1e-4, 1e-3], [None, "balanced"])
    ]


def main(run_tuning: bool = False, n_jobs: int | None = None) -> None:
    n_jobs = n_jobs or max(1, min(8, (os.cpu_count() or 2) - 1))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = prepare_folds()

    pegasos = tune(pegasos_grid(), folds, n_jobs=n_jobs)
    pegasos.to_csv(OUTPUT_DIR / "pegasos_tuning_cv.csv", index=False)
    print(pegasos.to_string(index=False))
    best_peg = pegasos.iloc[0]
    best_peg_params = {"lam": float(best_peg["lam"]), "n_epochs": 30,
                       "class_weight": None if pd.isna(best_peg["class_weight"]) else best_peg["class_weight"]}

    if run_tuning:
        tuning = tune(svm_grid(), folds, n_jobs=n_jobs)
        tuning.to_csv(OUTPUT_DIR / "svm_tuning_cv_rerun.csv", index=False)
        print(tuning.head(10).to_string(index=False))

    models = {
        "rbf_best_C3_g0.1": SVM(**BEST_SVM_PARAMS),
        "rbf_default_C1_scale": SVM(kernel="rbf", C=1.0, gamma="scale", max_iter=-1),
        "poly_d3_C1": SVM(kernel="poly", C=1.0, degree=3, max_iter=-1),
        "linear_C0.1_n5000": SVM(kernel="linear", C=0.1, max_train_samples=5000, max_iter=-1),
        "pegasos_best": PegasosSVM(**best_peg_params),
    }
    per_fold, summary = compare_with_sklearn(models, folds, n_jobs=min(n_jobs, len(models)))
    per_fold.to_csv(OUTPUT_DIR / "svm_scratch_vs_sklearn_folds.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "svm_scratch_vs_sklearn_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune", action="store_true", help="jalankan ulang grid tuning SVM (lama)")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    main(run_tuning=args.tune, n_jobs=args.n_jobs)
