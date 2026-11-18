"""SHAP-based explanations for the XGBoost model.

Phase 2 adds evidence mapping: each feature will point back to the package records it came from
(plan items, lessons, videos with timestamps, evaluation questions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartep.models.xgb_multilabel import MultiLabelXGB


def global_importance(model: MultiLabelXGB, X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean absolute SHAP value per feature (rows) and label (columns)."""
    data = {label: np.abs(model.contributions(X, label)).mean(axis=0) for label in model.labels}
    return pd.DataFrame(data, index=feature_names).rename_axis("feature")


def explain_row(contribs: np.ndarray, feature_names: list[str], top_k: int = 3) -> list[dict]:
    order = np.argsort(-np.abs(contribs))[:top_k]
    return [
        {"feature": feature_names[i], "impact": float(contribs[i]),
         "direction": "increases" if contribs[i] > 0 else "decreases"}
        for i in order
    ]
