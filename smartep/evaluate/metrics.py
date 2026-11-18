"""Discrimination, decision and calibration metrics per label.

AUC and PR-AUC are computed on raw scores (ranking quality); decisions and calibration use the calibrated
probabilities that would actually be served.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return float(ece)


def _f(x) -> float | None:
    return None if x is None else round(float(x), 4)


def label_metrics(Y: np.ndarray, P_raw: np.ndarray, P_cal: np.ndarray, thresholds: np.ndarray,
                  labels: list[str]) -> dict[str, dict]:
    out = {}
    for j, label in enumerate(labels):
        y = Y[:, j].astype(int)
        pred = (P_cal[:, j] >= thresholds[j]).astype(int)
        two_classes = len(np.unique(y)) > 1
        out[label] = {
            "prevalence": _f(y.mean()),
            "auc": _f(roc_auc_score(y, P_raw[:, j])) if two_classes else None,
            "pr_auc": _f(average_precision_score(y, P_raw[:, j])) if two_classes else None,
            "threshold": _f(thresholds[j]),
            "precision": _f(precision_score(y, pred, zero_division=0)),
            "recall": _f(recall_score(y, pred, zero_division=0)),
            "f1": _f(f1_score(y, pred, zero_division=0)),
            "positive_rate": _f(pred.mean()),
            "ece_raw": _f(expected_calibration_error(y, P_raw[:, j])),
            "ece_calibrated": _f(expected_calibration_error(y, P_cal[:, j])),
            "brier_calibrated": _f(brier_score_loss(y, P_cal[:, j])),
        }
    return out


def macro(per_label: dict[str, dict], key: str) -> float | None:
    values = [m[key] for m in per_label.values() if m.get(key) is not None]
    return round(float(np.mean(values)), 4) if values else None


def hamming_loss(Y: np.ndarray, P_cal: np.ndarray, thresholds: np.ndarray) -> float:
    return round(float(((P_cal >= thresholds).astype(int) != Y).mean()), 4)
