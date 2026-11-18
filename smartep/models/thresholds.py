"""Decision thresholds and severity bands.

The notebook routed educators by comparing a risk score with the AUC gates (0.85 and 0.70). An AUC is a
measure of model quality, not a cut-off, so thresholds are tuned per label on the validation split.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

DEFAULT_GRID = np.round(np.linspace(0.05, 0.95, 91), 4)


def tune_thresholds(P_val: np.ndarray, Y_val: np.ndarray, grid=DEFAULT_GRID, default: float = 0.5) -> np.ndarray:
    """Per-label threshold that maximises F1 on validation; ``default`` for a single-class label."""
    thresholds = np.full(P_val.shape[1], default, dtype=np.float64)
    for j in range(P_val.shape[1]):
        y = Y_val[:, j]
        if len(np.unique(y)) < 2:
            continue
        scores = [f1_score(y, P_val[:, j] >= t, zero_division=0) for t in grid]
        thresholds[j] = float(grid[int(np.argmax(scores))])
    return thresholds


def tune_severity_cuts(p: np.ndarray, true_severity: np.ndarray, threshold: float,
                       default: tuple[float, float] = (0.65, 0.8), min_rows: int = 15) -> tuple[float, float]:
    """Band cut-points (c1 <= c2) that best reproduce the true severity of real gaps above the threshold."""
    mask = (true_severity != "") & (p >= threshold)
    if mask.sum() < min_rows:
        return max(default[0], threshold), max(default[1], threshold)
    ps, truth = p[mask], true_severity[mask]
    grid = np.unique(np.round(np.quantile(ps, np.linspace(0, 1, 21)), 4))
    best, best_acc = (max(default[0], threshold), max(default[1], threshold)), -1.0
    for i, c1 in enumerate(grid):
        for c2 in grid[i:]:
            pred = np.where(ps >= c2, "high", np.where(ps >= c1, "medium", "low"))
            acc = float((pred == truth).mean())
            if acc > best_acc:
                best, best_acc = (float(c1), float(c2)), acc
    return best


def severity(prob: float, threshold: float, bands: tuple[float, float] = (0.65, 0.8)) -> str | None:
    """None below the threshold; otherwise low / medium / high by probability band."""
    if prob < threshold:
        return None
    if prob >= bands[1]:
        return "high"
    if prob >= bands[0]:
        return "medium"
    return "low"
