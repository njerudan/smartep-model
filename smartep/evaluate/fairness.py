"""Group fairness of gap predictions.

Two measures per label, both as the largest gap between any two groups:
- selection-rate disparity (demographic parity difference);
- equal-opportunity difference (true-positive rate among real gaps).

Numeric group attributes are bucketed with quantile edges learned on the training rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_POSITIVES = 5  # groups with fewer real gaps are left out of the equal-opportunity difference


def fit_bucket_edges(train_values, n_buckets: int) -> np.ndarray:
    edges = np.unique(np.quantile(np.asarray(train_values, dtype=float), np.linspace(0, 1, n_buckets + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def assign_groups(values, edges: np.ndarray | None = None) -> np.ndarray:
    if edges is None:
        return np.asarray(values).astype(str)
    return np.asarray(pd.cut(np.asarray(values, dtype=float), bins=edges, include_lowest=True).astype(str))


def _spread(series: pd.Series) -> float:
    series = series.dropna()
    return float(series.max() - series.min()) if len(series) > 1 else 0.0


def fairness_audit(Y: np.ndarray, P_cal: np.ndarray, thresholds: np.ndarray, groups: np.ndarray,
                   labels: list[str], max_disparity: float) -> pd.DataFrame:
    rows = []
    for j, label in enumerate(labels):
        df = pd.DataFrame({"group": groups, "y": Y[:, j], "pred": (P_cal[:, j] >= thresholds[j]).astype(int)})
        selection = df.groupby("group")["pred"].mean()
        positives = df[df["y"] == 1].groupby("group")["pred"].agg(["mean", "size"])
        tpr = positives.loc[positives["size"] >= MIN_POSITIVES, "mean"]
        sel_gap, eo_gap = _spread(selection), _spread(tpr)
        rows.append({
            "label": label,
            "selection_disparity": round(sel_gap, 4),
            "equal_opportunity_diff": round(eo_gap, 4),
            "flag": "FAIL" if max(sel_gap, eo_gap) > max_disparity else "PASS",
            **{f"selection_rate[{g}]": round(float(r), 4) for g, r in selection.items()},
        })
    return pd.DataFrame(rows)
