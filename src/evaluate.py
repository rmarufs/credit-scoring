"""Метрики качества и операционные показатели модели кредитного скоринга."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
) -> dict[str, float]:
    """Считает ROC-AUC, PR-AUC, KS, Gini, Brier score и log loss по вероятностям дефолта."""
    roc_auc = float(roc_auc_score(y_true, proba))
    pr_auc = float(average_precision_score(y_true, proba))
    fpr, tpr, _ = roc_curve(y_true, proba)
    ks = float(np.max(tpr - fpr))
    gini = 2.0 * roc_auc - 1.0
    brier = float(brier_score_loss(y_true, proba))
    logloss = float(log_loss(y_true, proba))
    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "KS": ks,
        "Gini": gini,
        "Brier": brier,
        "log_loss": logloss,
    }


def compute_threshold_metrics(
    y_true: pd.Series | np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, int | float]:
    """Считает матрицу ошибок и операционные метрики при заданном пороге отказа."""
    y_arr = np.asarray(y_true)
    preds = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_arr, preds).ravel())
    total = tn + fp + fn + tp
    approved = tn + fn

    approval_rate = approved / total if total > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    bad_rate = fn / approved if approved > 0 else 0.0

    return {
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "approval_rate": float(approval_rate),
        "recall": float(recall),
        "precision": float(precision),
        "bad_rate": float(bad_rate),
    }
