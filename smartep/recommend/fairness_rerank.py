"""Fairness-constrained top-1 recommendation (port of ``fairness_constrained_recommend``).

Fixes: the reported disparity is measured after the last re-assignment (the notebook measured it before),
and ``served_rankings`` gives the ranked list people actually see, so metrics are computed on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_arm_disparity(recs: np.ndarray, groups: np.ndarray, n_arms: int) -> tuple[float, int | None]:
    df = pd.DataFrame({"group": groups, "rec": recs})
    worst, worst_arm = 0.0, None
    for arm in range(n_arms):
        rates = (df["rec"] == arm).groupby(df["group"]).mean()
        gap = float(rates.max() - rates.min()) if len(rates) > 1 else 0.0
        if gap > worst:
            worst, worst_arm = gap, arm
    return worst, worst_arm


def fairness_constrained_recommend(raw_scores: np.ndarray, groups: np.ndarray, max_disparity: float,
                                   max_iter: int = 5) -> tuple[np.ndarray, float]:
    n_arms = raw_scores.shape[1]
    recs = raw_scores.argmax(axis=1)
    groups = np.asarray(groups)
    for _ in range(max_iter):
        disparity, arm = max_arm_disparity(recs, groups, n_arms)
        if disparity <= max_disparity or arm is None:
            break
        rates = pd.Series(recs == arm).groupby(groups).mean()
        over = rates.idxmax()
        for i in np.flatnonzero((groups == over) & (recs == arm)):
            order = np.argsort(-raw_scores[i])
            recs[i] = order[1] if order[0] == arm else order[0]
    final, _ = max_arm_disparity(recs, groups, n_arms)
    return recs, final


def served_rankings(raw_scores: np.ndarray, top1: np.ndarray) -> np.ndarray:
    """Full ranking per row: the served top-1 first, then the remaining arms by score."""
    order = np.argsort(-raw_scores, axis=1)
    out = np.empty_like(order)
    for i, first in enumerate(top1):
        rest = order[i][order[i] != first]
        out[i] = np.concatenate([[first], rest])
    return out
