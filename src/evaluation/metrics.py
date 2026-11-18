"""Binary classification metrics accept numeric 0/1 labels, including object arrays."""

from numbers import Number

import numpy as np

from src import config


def _check_binary_labels(labels: np.ndarray, name: str) -> None:
    if labels.dtype.kind == "O":
        try:
            valid = all(
                isinstance(value, (Number, np.bool_))
                and not isinstance(value, np.timedelta64)
                and (value == 0 or value == 1)
                for value in labels
            )
        except (ArithmeticError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"{name} contains non-binary labels")
        return

    try:
        finite = np.all(np.isfinite(labels))
    except TypeError as exc:
        raise ValueError(f"{name} contains non-binary labels") from exc
    if not finite:
        raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{name} contains non-binary labels")


def _check_binary_inputs(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.dtype == bool:
        yt = yt.astype(int)
    if yp.dtype == bool:
        yp = yp.astype(int)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} != len(y_pred)={len(yp)}")
    if len(yt) == 0:
        raise ValueError("Inputs cannot be empty")
    _check_binary_labels(yt, "y_true")
    _check_binary_labels(yp, "y_pred")
    return yt, yp


def _check_auc_inputs(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if yt.dtype == bool:
        yt = yt.astype(int)
    if len(yt) != len(ys):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} != len(y_score)={len(ys)}")
    if len(yt) == 0:
        raise ValueError("Inputs cannot be empty")
    _check_binary_labels(yt, "y_true")
    if not np.all(np.isfinite(ys)):
        raise ValueError("y_score contains non-finite values (NaN or Inf)")
    return yt, ys

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt, yp = _check_binary_inputs(y_true, y_pred)
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _check_binary_inputs(y_true, y_pred)
    return float(np.mean(yt == yp))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _check_binary_inputs(y_true, y_pred)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    denom = tp + fp
    return float(tp / denom) if denom > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _check_binary_inputs(y_true, y_pred)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    denom = tp + fn
    return float(tp / denom) if denom > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    denom = p + r
    return float(2.0 * p * r / denom) if denom > 0.0 else 0.0


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    yt, ys = _check_auc_inputs(y_true, y_score)
    n_0 = int(np.sum(yt == 0))
    n_1 = int(np.sum(yt == 1))
    if n_0 == 0 or n_1 == 0:
        return float("nan")

    order = np.argsort(ys)
    sorted_scores = ys[order]
    sorted_y = yt[order]

    _, inverse, counts = np.unique(
        sorted_scores, return_inverse=True, return_counts=True
    )
    cumsum = np.cumsum(counts)
    start_idx = np.empty_like(counts)
    start_idx[0] = 0
    start_idx[1:] = cumsum[:-1]

    avg_ranks_unique = 1.0 + (start_idx + cumsum - 1) / 2.0
    ranks = avg_ranks_unique[inverse]
    sum_ranks_pos = float(np.sum(ranks[sorted_y == 1]))

    return float((sum_ranks_pos - n_1 * (n_1 + 1) / 2.0) / (n_0 * n_1))


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    support_0 = tn + fp
    support_1 = fn + tp
    total_support = support_0 + support_1

    p0 = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    r0 = float(tn / support_0) if support_0 > 0 else 0.0
    f0 = float(2.0 * p0 * r0 / (p0 + r0)) if (p0 + r0) > 0.0 else 0.0

    p1 = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    r1 = float(tp / support_1) if support_1 > 0 else 0.0
    f1 = float(2.0 * p1 * r1 / (p1 + r1)) if (p1 + r1) > 0.0 else 0.0

    acc = float((tn + tp) / total_support) if total_support > 0 else 0.0

    macro_p = float((p0 + p1) / 2.0)
    macro_r = float((r0 + r1) / 2.0)
    macro_f = float((f0 + f1) / 2.0)

    if total_support > 0:
        weighted_p = float((p0 * support_0 + p1 * support_1) / total_support)
        weighted_r = float((r0 * support_0 + r1 * support_1) / total_support)
        weighted_f = float((f0 * support_0 + f1 * support_1) / total_support)
    else:
        weighted_p, weighted_r, weighted_f = 0.0, 0.0, 0.0

    return {
        "0": {"precision": p0, "recall": r0, "f1-score": f0, "support": support_0},
        "1": {"precision": p1, "recall": r1, "f1-score": f1, "support": support_1},
        "accuracy": acc,
        "macro avg": {
            "precision": macro_p,
            "recall": macro_r,
            "f1-score": macro_f,
            "support": total_support,
        },
        "weighted avg": {
            "precision": weighted_p,
            "recall": weighted_r,
            "f1-score": weighted_f,
            "support": total_support,
        },
    }


def misclassification_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fp_cost: float = config.FP_COST,
    fn_cost: float = config.FN_COST,
) -> float:
    cm = confusion_matrix(y_true, y_pred)
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    return float(fp_cost * fp + fn_cost * fn)
