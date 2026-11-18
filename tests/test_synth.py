"""Generator v2: schema validity, label rules, evidence, and that traits reach the observable fields."""

from pathlib import Path

import numpy as np
import pytest

from pipelines.generate_data import run as generate
from smartep.config import config_hash, load_config
from smartep.contracts import EvidencePackage
from smartep.contracts.gap_areas import GAP_AREAS
from smartep.features.package import package_feature_frame
from smartep.synth.generator import generate_dataset
from smartep.synth.io import read_dataset
from smartep.synth.labels import assign_labels, weight_matrix
from smartep.synth.traits import TRAITS, correlation_matrix

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "generator.yaml"


@pytest.fixture(scope="module")
def small():
    cfg = load_config(CONFIG, ["n_educators=150", "classroom_test_packages=30"])
    return generate_dataset(cfg)


def test_every_package_validates(small):
    for pkg in small.packages:
        EvidencePackage.model_validate(pkg)
    assert len(small.packages) == 450


def test_generation_is_deterministic():
    cfg = load_config(CONFIG, ["n_educators=10"])
    a, b = generate_dataset(cfg), generate_dataset(cfg)
    assert a.packages == b.packages and a.labels.equals(b.labels)


def test_label_rules_preserve_prevalence():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(20_000, len(GAP_AREAS)))
    Y, severity, flipped = assign_labels(scores, dict.fromkeys(GAP_AREAS, 0.2), 0.1, rng)
    assert np.allclose(Y.mean(axis=0), 0.2, atol=0.01)
    assert np.allclose(flipped.mean(axis=0), 0.2 * 0.1 * 2, atol=0.01)
    assert set(np.unique(severity[Y == 1])) == {"low", "medium", "high"}
    assert (severity[Y == 0] == "").all()


def test_weights_are_unit_length_and_traits_valid():
    np.testing.assert_allclose(np.linalg.norm(weight_matrix(), axis=0), 1.0)
    assert correlation_matrix().shape == (len(TRAITS), len(TRAITS))


def test_every_gap_has_evidence(small):
    n_pos = int(small.labels[list(GAP_AREAS)].to_numpy().sum())
    gaps = [g for row in small.evidence for g in row["gaps"]]
    assert len(gaps) == n_pos and all(g["evidence"] for g in gaps)


def test_classroom_subset_holds_whole_educators(small):
    by_edu = small.index.groupby("educatorRef")["subset"].nunique()
    assert (by_edu == 1).all()
    assert (small.index["subset"] == "classroom_test").sum() >= 30


def test_traits_reach_observable_fields(small):
    f = package_feature_frame(small.packages)
    t = small.traits.set_index("packageId").loc[f.index]

    def corr(feature, trait):
        mask = f[feature].notna()
        return np.corrcoef(f.loc[mask, feature], t.loc[mask, trait])[0, 1]

    assert corr("delivery_ratio", "workload_pressure") < -0.3
    assert corr("grading_turnaround_hours", "responsiveness") < -0.3
    assert corr("questions_answered_72h_rate", "responsiveness") > 0.3
    assert corr("video_hosted_share", "digital_skill") > 0.2
    assert corr("likert_clarity_mean", "pedagogical_skill") > 0.3
    assert corr("document_age_years_mean", "currency") < -0.2
    assert corr("plan_item_coverage", "content_mastery") > 0.3


@pytest.mark.slow
def test_generate_pipeline_writes_dataset_and_checks(tmp_path):
    cfg = load_config(CONFIG, ["n_educators=200", "classroom_test_packages=60"])
    result = generate(cfg, tmp_path)
    out = Path(result["dir"])
    assert out.name == config_hash(cfg)
    for name in ("manifest.json", "packages.jsonl.gz", "labels.parquet", "evidence.jsonl.gz", "checks.json", "stats.md"):
        assert (out / name).exists(), name
    checks = result["checks"]
    assert checks["schema"]["pass"] and checks["evidence"]["pass"] and checks["prevalence"]["pass"]
    assert checks["oracle"]["macro_auc"] > checks["evidence_probe"]["macro_auc"] > checks["noise"]["macro_auc"]
    ds = read_dataset(out)
    assert len(ds.packages) == 600 and ds.labels.shape == (600, 1 + len(GAP_AREAS))
