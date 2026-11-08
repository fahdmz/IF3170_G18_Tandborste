import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    raise NotImplementedError


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    raise NotImplementedError


def misclassification_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fp_cost: float = 2.0,
    fn_cost: float = 1.0,
) -> float:
    raise NotImplementedError
