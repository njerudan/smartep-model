"""Phase 2: gap model, evidence mapping, catalogue matching, fairness intervals, end-to-end run."""

import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.generate_data import run as generate
from smartep.config import load_config
from smartep.contracts import AssessmentResult
from smartep.data.splits import grouped_holdout, grouped_kfold
from smartep.evaluate.fairness import fairness_audit
from smartep.explain.evidence_mapper import map_evidence
from smartep.models.thresholds import tune_severity_cuts
from smartep.recommend.catalogue_matcher import match_courses
from smartep.serve.predict import GapPredictor
from smartep.synth.generator import generate_dataset
from smartep.training.gap_model import run_gap_model

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def packages():
    return generate_dataset(load_config(CONFIGS / "generator.yaml", ["n_educators=40"])).packages


def test_grouped_kfold_and_holdout():
    groups = np.repeat(np.arange(30), 3)
    folds = grouped_kfold(groups, 5, seed=0)
    assert sorted(np.concatenate(folds).tolist()) == list(range(90))
    assert all(len(set(groups[f]) & set(groups[g])) == 0 for i, f in enumerate(folds) for g in folds[i + 1:])
    keep, held = grouped_holdout(groups, np.arange(90), 0.2, seed=0)
    assert not set(groups[keep]) & set(groups[held]) and len(set(groups[held])) == 6


def test_severity_cuts_recover_bands():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1.0, 600)
    truth = np.where(p >= 0.85, "high", np.where(p >= 0.7, "medium", "low")).astype(object)
    c1, c2 = tune_severity_cuts(p, truth, threshold=0.5)
    assert abs(c1 - 0.7) < 0.04 and abs(c2 - 0.85) < 0.04


def test_fairness_interval_separates_noise_from_real_disparity():
    rng = np.random.default_rng(1)
    groups = np.array(["a"] * 60 + ["b"] * 60)
    Y = np.ones((120, 1))
    noisy = (rng.random((120, 1)) < 0.6).astype(int)              # same true rate in both groups
    audit = fairness_audit(Y, None, None, groups, ["gap"], 0.10, pred=noisy, interval=True)
    assert audit.loc[0, "flag"] in ("PASS", "WARN")
    biased = np.concatenate([np.ones(60), np.zeros(60)])[:, None]   # group b never flagged
    audit = fairness_audit(Y, None, None, groups, ["gap"], 0.10, pred=biased, interval=True)
    assert audit.loc[0, "flag"] == "FAIL"


def test_evidence_prefers_records_relevant_to_the_area(packages):
    pkg = next(p for p in packages if p["plan"] and any(les["deliveredStatus"] == "not_delivered" for les in p["lessons"]))
    names = ["delivery_ratio", "plan_item_coverage"]
    # delivery_ratio is the larger driver, but a coverage gap is shown by plan items first.
    items, drivers = map_evidence(pkg, "content_coverage", np.array([0.9, 0.5]), names)
    assert drivers[0]["feature"] == "delivery_ratio"
    assert items[0]["source"] == "plan"
    items, _ = map_evidence(pkg, "session_delivery", np.array([0.9, 0.5]), names)
    assert items[0]["source"] == "delivery"


def test_evidence_always_present(packages):
    items, _ = map_evidence(packages[0], "pedagogy", np.zeros(2), ["enrolled_students", "n_previous_packages"])
    assert items  # relevant families tried directly, then the fallback


def test_catalogue_matcher_only_returns_snapshot_entries(packages):
    pkg = json.loads(json.dumps(packages[0]))
    pkg["reskillingPreferences"]["maxCostKes"] = 4000
    refs = {e["catalogueRef"] for e in pkg["reskillingCatalogue"]}
    for area in ("assessment_feedback", "video_delivery", "technical_skill"):
        courses = match_courses(pkg, area)
        assert courses and {c["catalogueRef"] for c in courses} <= refs
        assert all(c["cost"] == "free" or c["cost"]["amount"] <= 4000 for c in courses)
    assert match_courses(pkg, "assessment_feedback")[0]["catalogueRef"] == "cat_assess"


@pytest.mark.slow
def test_gap_model_end_to_end(tmp_path):
    data = generate(load_config(CONFIGS / "generator.yaml", ["n_educators=260", "classroom_test_packages=60"]),
                    tmp_path / "data")
    cfg = load_config(CONFIGS / "gap_model.yaml", [
        f"data.dir={Path(data['dir']).as_posix()}", "features.text.encoder=tfidf_svd", "cv.folds=3", "cv.seeds=[1]",
        "cv.evidence_sample_per_area=20", "models.xgb.n_estimators=150",
    ])
    summary = run_gap_model(cfg, tmp_path / "artifacts")
    bundle = tmp_path / "artifacts" / summary["version"]
    assert summary["bundle_reload_check"] and summary["holdout"]["results_valid_share"] == 1.0
    assert summary["holdout"]["evidence"]["coverage"] == 1.0
    assert summary["cv"]["macro_auc"]["mean"] > 0.6  # small data, but the signal is there
    assert set(summary["gates"]) == {"macro_auc", "min_label_auc", "max_ece", "evidence_coverage",
                                     "results_schema_valid", "fairness_fail_flags"}
    assert (bundle / "report.md").exists() and list((bundle / "examples").glob("*.result.json"))

    predictor = GapPredictor.from_bundle(bundle)
    pkg = json.loads(next((bundle / "examples").glob("*.result.json")).read_text())["packageId"]
    from smartep.synth.io import read_dataset

    ds = read_dataset(Path(data["dir"]))
    result = predictor.predict(next(p for p in ds.packages if p["packageId"] == pkg))
    assert isinstance(result, AssessmentResult) and result.model.validated is False
    snapshot = {e["catalogueRef"] for e in ds.packages[0]["reskillingCatalogue"]}
    for r in result.reskilling:
        assert {c.catalogueRef for c in r.courses} <= snapshot
