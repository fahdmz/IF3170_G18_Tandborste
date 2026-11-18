import numpy as np

from src import config


def tune_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fp_cost: float = config.FP_COST,
    fn_cost: float = config.FN_COST,
) -> float:
    if fp_cost < 0.0 or fn_cost < 0.0:
        raise ValueError("fp_cost and fn_cost must be non-negative")
    if fp_cost + fn_cost == 0.0:
        raise ValueError("fp_cost and fn_cost cannot both be zero")

    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()

    if yt.dtype == bool:
        yt = yt.astype(int)
    if len(yt) != len(ys):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} != len(y_score)={len(ys)}")
    if len(yt) == 0:
        raise ValueError("Inputs cannot be empty")
    if not np.all(np.isfinite(ys)):
        raise ValueError("y_score contains non-finite values (NaN or Inf)")

    unique_y = set(np.unique(yt))
    if not unique_y.issubset({0, 1}):
        raise ValueError("y_true contains non-binary labels")

    unique_scores, score_indices = np.unique(ys, return_inverse=True)
    negatives = np.bincount(score_indices[yt == 0], minlength=len(unique_scores))
    positives = np.bincount(score_indices[yt == 1], minlength=len(unique_scores))
    false_positives = np.append(np.cumsum(negatives[::-1])[::-1], 0)
    false_negatives = np.concatenate(([0], np.cumsum(positives)))
    costs = fp_cost * false_positives + fn_cost * false_negatives
    best_index = int(np.argmin(costs))
    if best_index == len(unique_scores):
        return float("inf")
    return float(unique_scores[best_index])


def bayes_optimal_threshold(
    fp_cost: float = config.FP_COST, fn_cost: float = config.FN_COST
) -> float:
    if fp_cost < 0.0 or fn_cost < 0.0:
        raise ValueError("fp_cost and fn_cost must be non-negative")
    if fp_cost + fn_cost == 0.0:
        raise ValueError("fp_cost and fn_cost cannot both be zero")
    return float(fp_cost / (fp_cost + fn_cost))


def cost_sensitive_class_weight(
    fp_cost: float = config.FP_COST, fn_cost: float = config.FN_COST
) -> dict[int, float]:
    if fp_cost < 0.0 or fn_cost < 0.0:
        raise ValueError("fp_cost and fn_cost must be non-negative")
    if fp_cost + fn_cost == 0.0:
        raise ValueError("fp_cost and fn_cost cannot both be zero")
    return {0: float(fp_cost), 1: float(fn_cost)}
