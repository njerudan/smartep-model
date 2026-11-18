import json

import numpy as np

from smartep.recommend.bandit import LinUCB, train_bandit
from smartep.recommend.fairness_rerank import (
    fairness_constrained_recommend,
    max_arm_disparity,
    served_rankings,
)
from smartep.recommend.metrics import map_ndcg_at_k
from smartep.registry.champion import decide, promote_if_better


def test_linucb_sherman_morrison_matches_direct_inverse():
    rng = np.random.default_rng(0)
    bandit = LinUCB(n_arms=2, context_dim=3, alpha=0.5)
    A = np.eye(3)
    for _ in range(20):
        x = rng.normal(size=3)
        bandit.update(0, x, 1.0)
        A += np.outer(x, x)
    np.testing.assert_allclose(bandit.A_inv[0], np.linalg.inv(A), atol=1e-8)


def test_bandit_learns_rewarding_arm():
    rng = np.random.default_rng(0)
    X = np.ones((300, 2))
    Y = np.tile([0, 1, 0], (300, 1))
    bandit = train_bandit(LinUCB(3, 2, alpha=0.1), X, Y, np.zeros(300), passes=1, rng=rng)
    assert bandit.scores(X[:1]).argmax() == 1


def test_ranking_metrics_hand_computed():
    Y = np.array([[1, 0, 1], [0, 0, 0]])
    rankings = np.array([[0, 1, 2], [0, 1, 2]])
    m, n = map_ndcg_at_k(Y, rankings, k=3)
    # row 0: AP = (1/1 + 2/3) / 2; NDCG = (1 + 1/log2(4)) / (1 + 1/log2(3)); row 1 scores 0
    assert m == round(((1 + 2 / 3) / 2) / 2, 4)
    assert n == round(((1 + 0.5) / (1 + 1 / np.log2(3))) / 2, 4)


def test_fairness_rerank_reduces_disparity_and_reports_final_value():
    groups = np.array(["a"] * 50 + ["b"] * 50)
    scores = np.zeros((100, 3))
    scores[:50, 0], scores[50:, 1] = 1.0, 1.0
    scores[:, 2] = 0.5
    before, _ = max_arm_disparity(scores.argmax(axis=1), groups, 3)
    recs, after = fairness_constrained_recommend(scores, groups, max_disparity=0.1)
    assert after < before
    assert after == max_arm_disparity(recs, groups, 3)[0]


def test_served_rankings_put_served_item_first():
    scores = np.array([[0.9, 0.5, 0.1]])
    assert served_rankings(scores, np.array([2])).tolist() == [[2, 0, 1]]


def test_promotion_rules(tmp_path):
    cand = {"version": "v1", "bundle_path": "x", "macro_auc": 0.7, "fairness_ok": True, "data_hash": "d1"}
    assert decide(cand, None, False)[0]
    assert not decide({**cand, "fairness_ok": False}, None, False)[0]
    champ = {**cand, "macro_auc": 0.8}
    assert not decide(cand, champ, False)[0]
    assert decide({**cand, "macro_auc": 0.9}, champ, False)[0]
    assert not decide({**cand, "data_hash": "d2", "macro_auc": 0.9}, champ, False)[0]
    assert decide({**cand, "data_hash": "d2"}, champ, True)[0]
    promoted, reason = decide(cand, champ, False, override_reason="lighter model for deployment")
    assert promoted and reason.startswith("manual promotion")
    assert not decide({**cand, "fairness_ok": False}, champ, False, override_reason="x")[0]

    assert promote_if_better(tmp_path, cand)[0]
    assert not promote_if_better(tmp_path, {**cand, "version": "v2"})[0]
    history = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert [h["promoted"] for h in history] == [True, False]
    assert json.loads((tmp_path / "champion.json").read_text())["version"] == "v1"
