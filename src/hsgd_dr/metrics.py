from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(fn, default: float = float("nan")) -> float:
    try:
        return float(fn())
    except Exception:
        return default


def classification_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "auroc": _safe_metric(lambda: roc_auc_score(y_true, y_score)),
        "aupr": _safe_metric(lambda: average_precision_score(y_true, y_score)),
        "precision": _safe_metric(lambda: precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe_metric(lambda: recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": _safe_metric(lambda: f1_score(y_true, y_pred, zero_division=0)),
        "mcc": _safe_metric(lambda: matthews_corrcoef(y_true, y_pred)),
    }


def compute_metrics(frame: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    return classification_metrics(frame["label"].tolist(), frame["score"].tolist(), threshold)

