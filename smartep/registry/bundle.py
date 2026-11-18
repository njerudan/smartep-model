"""Model bundles: everything needed to reproduce and serve one trained model, in one directory.

    <bundle>/
      manifest.json      version, created_at, config_hash, data_hash, git commit, library versions
      config.yaml        the exact config used
      metrics.json       run summary (metrics, fairness, gates)
      xgb/               one XGBoost model per label + meta
      preprocess.joblib  tabular featurizer, text encoder, calibrator, thresholds, feature names
      fusion.pt          fusion weights, if trained (the same weights that were evaluated)
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import yaml

from smartep.models.xgb_multilabel import MultiLabelXGB

LIBS = ("numpy", "pandas", "scikit-learn", "xgboost", "pydantic", "torch")


def git_commit(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=cwd, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=cwd, check=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for lib in LIBS:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = None
    return versions


def make_version(run_name: str, cfg_hash: str) -> str:
    return f"{run_name}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{cfg_hash[:6]}"


@dataclass
class Bundle:
    path: Path
    manifest: dict
    config: dict
    xgb: MultiLabelXGB
    preprocess: dict = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        P = self.xgb.predict_proba(X)
        return self.preprocess["calibrator"].transform(P) if calibrated else P


def save_bundle(path: Path, *, manifest: dict, config: dict, metrics: dict, xgb_model: MultiLabelXGB,
                preprocess: dict, fusion_state=None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    xgb_model.save(path / "xgb")
    joblib.dump(preprocess, path / "preprocess.joblib")
    if fusion_state is not None:
        import torch

        torch.save(fusion_state, path / "fusion.pt")
    return path


def load_bundle(path: Path) -> Bundle:
    return Bundle(
        path=path,
        manifest=json.loads((path / "manifest.json").read_text(encoding="utf-8")),
        config=yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8")),
        xgb=MultiLabelXGB.load(path / "xgb"),
        preprocess=joblib.load(path / "preprocess.joblib"),
    )
