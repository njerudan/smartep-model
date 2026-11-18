"""End-to-end runs on a small dataset."""

import importlib.util
import json
from pathlib import Path

import pytest

from pipelines.train import run
from smartep.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SMALL = ["data.n_educators=300", "models.xgb.n_estimators=60", "recommender.passes=1"]


@pytest.mark.slow
def test_xgb_pipeline_end_to_end(tmp_path):
    summary = run(load_config(CONFIGS / "train_xgb.yaml", SMALL), tmp_path)
    bundle = tmp_path / summary["version"]
    for name in ("manifest.json", "config.yaml", "metrics.json", "report.md", "preprocess.joblib", "xgb/xgb_meta.json"):
        assert (bundle / name).exists(), name
    assert summary["bundle_reload_check"] is True
    assert summary["split"] == {"train": 210, "val": 45, "test": 45}
    assert set(summary["gates"]) == {"macro_auc", "min_label_auc", "max_ece", "max_disparity"}
    history = (tmp_path / "registry" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(history[-1])["version"] == summary["version"]
    if summary["registry"]["promoted"]:
        champion = json.loads((tmp_path / "registry" / "champion.json").read_text(encoding="utf-8"))
        assert champion["version"] == summary["version"]


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs the research extra (torch)")
def test_fusion_pipeline_end_to_end(tmp_path):
    cfg = load_config(CONFIGS / "train_fusion.yaml", [*SMALL, "models.fusion.max_epochs=3"])
    summary = run(cfg, tmp_path)
    assert "fusion" in summary["models"] and (tmp_path / summary["version"] / "fusion.pt").exists()
    assert summary["models"]["fusion"]["epochs_run"] <= 3
