import numpy as np
import pytest

from smartep.evaluate.fairness import assign_groups, fairness_audit, fit_bucket_edges
from smartep.evaluate.metrics import expected_calibration_error, label_metrics, macro
from smartep.evaluate.pilot import competency_gain, sus_score
from smartep.models.calibration import LabelCalibrator
from smartep.models.thresholds import severity, tune_thresholds
from smartep.models.xgb_multilabel import MultiLabelXGB


@pytest.fixture(scope="module")
def learnable():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5)).astype(np.float32)
    Y = np.stack([(X[:, 0] + 0.3 * rng.normal(size=600) > 0), np.zeros(600)], axis=1).astype(np.float32)
    return X, Y


def test_xgb_learns_signal_and_handles_constant_label(learnable, tmp_path):
    X, Y = learnable
    model = MultiLabelXGB(["signal", "never"], {"n_estimators": 200, "max_depth": 3}, 20, seed=0)
    model.fit(X[:400], Y[:400], X[400:500], Y[400:500])
    P = model.predict_proba(X[500:])
    per = label_metrics(Y[500:], P, P, np.array([0.5, 0.5]), ["signal", "never"])
    assert per["signal"]["auc"] > 0.9 and per["never"]["auc"] is None
    assert np.all(P[:, 1] == 0)

    contribs = model.contributions(X[500:], "signal")
    assert contribs.shape == (100, 5) and np.abs(contribs[:, 0]).mean() > np.abs(contribs[:, 1:]).mean()

    model.save(tmp_path)
    np.testing.assert_allclose(MultiLabelXGB.load(tmp_path).predict_proba(X[500:]), P, atol=1e-7)


def test_calibration_and_thresholds():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=(2000, 1))
    y = (rng.uniform(size=(2000, 1)) < p ** 2).astype(float)  # over-confident raw scores
    cal = LabelCalibrator("isotonic").fit(p, y)
    assert expected_calibration_error(y[:, 0], cal.transform(p)[:, 0]) < expected_calibration_error(y[:, 0], p[:, 0])
    t = tune_thresholds(p, y)
    assert 0.05 <= t[0] <= 0.95


def test_severity_bands():
    assert severity(0.4, 0.5) is None
    assert severity(0.55, 0.5) == "low"
    assert severity(0.7, 0.5) == "medium"
    assert severity(0.9, 0.5) == "high"


def test_ece_of_perfect_calibration_is_small():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=50_000)
    assert expected_calibration_error(rng.uniform(size=50_000) < p, p) < 0.01


def test_macro_ignores_missing():
    assert macro({"a": {"auc": 0.6}, "b": {"auc": None}, "c": {"auc": 0.8}}, "auc") == 0.7


def test_fairness_audit_detects_disparity():
    groups = np.array(["old"] * 50 + ["young"] * 50)
    Y = np.ones((100, 1))
    P = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])[:, None]
    audit = fairness_audit(Y, P, np.array([0.5]), groups, ["gap"], max_disparity=0.1)
    row = audit.iloc[0]
    assert row["selection_disparity"] == 1.0 and row["equal_opportunity_diff"] == 1.0 and row["flag"] == "FAIL"


def test_bucket_edges_come_from_training_values():
    edges = fit_bucket_edges(np.arange(1, 10), 3)
    assert np.isinf(edges[0]) and np.isinf(edges[-1])
    assert len(set(assign_groups([0, 5, 100], edges))) == 3


def test_sus_and_competency_gain():
    assert sus_score([3] * 10).tolist() == [50.0]
    assert sus_score([5, 1] * 5).tolist() == [100.0]
    with pytest.raises(ValueError):
        sus_score([6] * 10)
    gain = competency_gain([50, 60], [60, 66])
    assert gain["mean_gain_pct"] == 15.0 and gain["target_met"]
