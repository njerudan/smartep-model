"""Per-label probability calibration fitted on the validation split."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class LabelCalibrator:
    def __init__(self, method: str = "isotonic"):
        if method not in {"isotonic", "none"}:
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self.calibrators: list[IsotonicRegression | None] = []

    def fit(self, P_val: np.ndarray, Y_val: np.ndarray) -> LabelCalibrator:
        self.calibrators = []
        for j in range(P_val.shape[1]):
            if self.method == "none" or len(np.unique(Y_val[:, j])) < 2:
                self.calibrators.append(None)  # identity
                continue
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.calibrators.append(iso.fit(P_val[:, j], Y_val[:, j]))
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        out = np.array(P, dtype=np.float64, copy=True)
        for j, cal in enumerate(self.calibrators):
            if cal is not None:
                out[:, j] = cal.predict(P[:, j])
        return out
