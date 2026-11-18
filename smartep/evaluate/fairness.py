"""Group fairness of gap predictions.

Two measures per label, both as the largest gap between any two groups:
- selection-rate disparity (demographic parity difference);
- equal-opportunity difference (true-positive rate among real gaps; groups with fewer than
  ``MIN_POSITIVES`` real gaps are left out).

Flags. With ``interval=True``, each pair of groups gets a 95% confidence interval for its difference
in rates (difference of two proportions, normal approximation). The flag is FAIL when, for some pair,
the whole interval lies beyond ``max_disparity`` (the disparity is reliably too large); WARN when the
largest point difference exceeds ``max_disparity`` but no interval does; PASS otherwise. Pairwise
differences are unbiased, unlike the max-minus-min spread, which grows with noise in small groups.
With ``interval=False`` the flag uses the point spread alone.

Numeric group attributes are bucketed with quantile edges learned on the training rows.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

MIN_POSITIVES = 5
Z95 = 1.96


def fit_bucket_edges(train_values, n_buckets: int) -> np.ndarray:
    edges = np.unique(np.quantile(np.asarray(train_values, dtype=float), np.linspace(0, 1, n_buckets + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def assign_groups(values, edges: np.ndarray | None = None) -> np.ndarray:
    if edges is None:
        return np.asarray(values).astype(str)
    return np.asarray(pd.cut(np.asarray(values, dtype=float), bins=edges, include_lowest=True).astype(str))


def _rates(hits: np.ndarray, n: np.ndarray, min_n: int) -> np.ndarray:
    return np.divide(hits, n, out=np.full(len(n), np.nan), where=n >= max(min_n, 1))


def _spread(v: np.ndarray) -> float:
    v = v[~np.isnan(v)]
    return float(v.max() - v.min()) if len(v) > 1 else 0.0


def _reliable_excess(rates: np.ndarray, n: np.ndarray, limit: float) -> tuple[float, list]:
    """Largest lower bound of |difference| - 1.96·SE over group pairs, and that pair's interval."""
    best, best_ci = -np.inf, None
    for a, b in combinations(np.flatnonzero(~np.isnan(rates)), 2):
        d = rates[a] - rates[b]
        se = np.sqrt(rates[a] * (1 - rates[a]) / n[a] + rates[b] * (1 - rates[b]) / n[b])
        lower = abs(d) - Z95 * se
        if lower > best:
            best, best_ci = lower, [round(float(abs(d) - Z95 * se), 4), round(float(abs(d) + Z95 * se), 4)]
    return best, best_ci


def fairness_audit(Y: np.ndarray, P_cal: np.ndarray | None, thresholds: np.ndarray | None, groups: np.ndarray,
                   labels: list[str], max_disparity: float, pred: np.ndarray | None = None,
                   interval: bool = False) -> pd.DataFrame:
    """``pred`` (0/1) overrides thresholding ``P_cal``, e.g. for out-of-fold predictions."""
    if pred is None:
        pred = (P_cal >= thresholds).astype(int)
    names, codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    k = len(names)
    size = np.bincount(codes, minlength=k).astype(float)
    rows = []
    for j, label in enumerate(labels):
        y, p = Y[:, j].astype(float), pred[:, j].astype(float)
        pos = np.bincount(codes, weights=y, minlength=k)
        sel = _rates(np.bincount(codes, weights=p, minlength=k), size, 1)
        tpr = _rates(np.bincount(codes, weights=y * p, minlength=k), pos, MIN_POSITIVES)
        sel_gap, eo_gap = _spread(sel), _spread(tpr)
        row = {"label": label, "selection_disparity": round(sel_gap, 4), "equal_opportunity_diff": round(eo_gap, 4)}
        worst = max(sel_gap, eo_gap)
        if interval:
            s_low, s_ci = _reliable_excess(sel, size, max_disparity)
            e_low, e_ci = _reliable_excess(tpr, pos, max_disparity)
            row.update({"selection_ci": s_ci, "equal_opportunity_ci": e_ci})
            row["flag"] = "FAIL" if max(s_low, e_low) > max_disparity else ("WARN" if worst > max_disparity else "PASS")
        else:
            row["flag"] = "FAIL" if worst > max_disparity else "PASS"
        row.update({f"selection_rate[{g}]": round(float(r), 4) for g, r in zip(names, sel, strict=True)})
        rows.append(row)
    return pd.DataFrame(rows)
