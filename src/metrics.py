from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float]:
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "attack_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc": float(auc),
        "fpr": float(fpr),
    }


def summarize_client_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float, np.floating))]
    return {k: float(np.nanmean([row[k] for row in rows])) for k in keys}

