"""One XGBoost classifier per label, with early stopping on the validation split.

A label with a single class in the training rows gets a constant prior instead of a model.
Per-row SHAP values come from XGBoost's own TreeSHAP (``pred_contribs``), so no extra dependency is
needed and the values match the trees exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb


class MultiLabelXGB:
    def __init__(self, labels: list[str], params: dict | None = None, early_stopping_rounds: int = 30, seed: int = 0):
        self.labels = list(labels)
        self.params = dict(params or {})
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.models: dict[str, xgb.XGBClassifier] = {}
        self.constants: dict[str, float] = {}

    def _new_model(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            **self.params,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.seed,
            n_jobs=-1,
        )

    def fit(self, X_train, Y_train, X_val, Y_val) -> MultiLabelXGB:
        for j, label in enumerate(self.labels):
            y = Y_train[:, j]
            if len(np.unique(y)) < 2:
                self.constants[label] = float(y.mean())
                continue
            model = self._new_model()
            model.fit(X_train, y, eval_set=[(X_val, Y_val[:, j])], verbose=False)
            self.models[label] = model
        return self

    def predict_proba(self, X) -> np.ndarray:
        out = np.zeros((len(X), len(self.labels)), dtype=np.float64)
        for j, label in enumerate(self.labels):
            if label in self.models:
                out[:, j] = self.models[label].predict_proba(X)[:, 1]
            else:
                out[:, j] = self.constants[label]
        return out

    def contributions(self, X, label: str) -> np.ndarray:
        """SHAP values (log-odds) per row and feature for one label; zeros for a constant label."""
        if label not in self.models:
            return np.zeros(np.asarray(X).shape, dtype=np.float32)
        model = self.models[label]
        booster = model.get_booster()
        best = getattr(model, "best_iteration", None)
        iteration_range = (0, best + 1) if best is not None else (0, 0)
        contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True, iteration_range=iteration_range)
        return contribs[:, :-1]  # last column is the bias term

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for label, model in self.models.items():
            model.save_model(directory / f"xgb_{label}.json")
        meta = {"labels": self.labels, "constants": self.constants, "trained": list(self.models),
                "params": self.params, "early_stopping_rounds": self.early_stopping_rounds, "seed": self.seed}
        (directory / "xgb_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> MultiLabelXGB:
        meta = json.loads((directory / "xgb_meta.json").read_text(encoding="utf-8"))
        obj = cls(meta["labels"], meta["params"], meta["early_stopping_rounds"], meta["seed"])
        obj.constants = meta["constants"]
        for label in meta["trained"]:
            model = xgb.XGBClassifier()
            model.load_model(directory / f"xgb_{label}.json")
            obj.models[label] = model
        return obj
