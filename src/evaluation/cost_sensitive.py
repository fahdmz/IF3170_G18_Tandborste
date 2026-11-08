import numpy as np


def tune_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fp_cost: float = 2.0,
    fn_cost: float = 1.0,
) -> float:
    raise NotImplementedError


def bayes_optimal_threshold(fp_cost: float = 2.0, fn_cost: float = 1.0) -> float:
    raise NotImplementedError


def cost_sensitive_class_weight(
    fp_cost: float = 2.0, fn_cost: float = 1.0
) -> dict[int, float]:
    raise NotImplementedError
