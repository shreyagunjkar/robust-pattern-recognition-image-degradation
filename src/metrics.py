"""Classification metrics shared by every model and condition.

All scores are computed from integer label vectors over a fixed label set 0..n_classes-1, so a
class that is never predicted (or never present) still has a defined row/column and enters the
macro averages with precision/recall 0 (zero_division=0), never silently dropped.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def _check(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> None:
    if y_true.shape != y_pred.shape or y_true.ndim != 1 or len(y_true) == 0:
        raise ValueError(f"y_true/y_pred must be equal-length non-empty 1-D vectors, got "
                         f"{y_true.shape} and {y_pred.shape}")
    for name, y in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() >= n_classes:
            raise ValueError(f"{name} must hold integer labels in 0..{n_classes - 1}")


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class, shape (n_classes, n_classes)."""
    _check(y_true, y_pred, n_classes)
    return confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, float]:
    _check(y_true, y_pred, n_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n_classes), average="macro", zero_division=0)
    accuracy = float((y_true == y_pred).mean())
    return {"n": int(len(y_true)), "accuracy": accuracy, "error_rate": 1.0 - accuracy,
            "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1)}
