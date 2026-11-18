"""Ranking metrics: MAP@k and NDCG@k (same definitions as the notebook)."""

from __future__ import annotations

import numpy as np


def map_ndcg_at_k(Y: np.ndarray, rankings: np.ndarray, k: int) -> tuple[float, float]:
    """``rankings`` holds arm indices per row, best first. Rows with no relevant arm score 0."""
    n_arms = Y.shape[1]
    k_eff = min(k, n_arms)
    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
    maps, ndcgs = [], []
    for y, rank in zip(Y, rankings, strict=True):
        rel = y[rank][:k_eff].astype(float)
        n_rel = y.sum()
        hits = np.cumsum(rel)
        precision_at_hits = (hits / np.arange(1, k_eff + 1)) * rel
        maps.append(precision_at_hits.sum() / min(k, n_rel) if n_rel > 0 else 0.0)
        ideal = np.sort(y)[::-1][:k_eff].astype(float)
        idcg = (ideal * discounts).sum()
        ndcgs.append((rel * discounts).sum() / idcg if idcg > 0 else 0.0)
    return round(float(np.mean(maps)), 4), round(float(np.mean(ndcgs)), 4)
