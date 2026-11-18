"""Pilot measures: System Usability Scale and pre/post competency gain (Phase 7 inputs)."""

from __future__ import annotations

import numpy as np


def sus_score(responses) -> np.ndarray:
    """SUS (0-100) per respondent from ten 1-5 answers in questionnaire order."""
    r = np.atleast_2d(np.asarray(responses, dtype=float))
    if r.shape[1] != 10 or ((r < 1) | (r > 5)).any():
        raise ValueError("each SUS response needs ten answers between 1 and 5")
    odd = r[:, 0::2] - 1   # items 1, 3, 5, 7, 9
    even = 5 - r[:, 1::2]  # items 2, 4, 6, 8, 10
    return (odd.sum(axis=1) + even.sum(axis=1)) * 2.5


def competency_gain(pre_scores, post_scores, target_gain_pct: float = 15.0) -> dict:
    pre, post = np.asarray(pre_scores, dtype=float), np.asarray(post_scores, dtype=float)
    gain = (post - pre) / np.clip(pre, 1e-6, None) * 100
    return {
        "per_educator_gain_pct": gain.round(2).tolist(),
        "mean_gain_pct": round(float(gain.mean()), 2),
        "target_met": bool(gain.mean() >= target_gain_pct),
    }
