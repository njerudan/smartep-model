"""Ground-truth gap labels from latent traits.

For each gap area, a deficiency score is a weighted sum of period traits (weights below, normalised to
unit length) plus Gaussian noise. The top ``prevalence`` share of packages are gaps. Assessor
disagreement is then simulated by flipping a share of labels in both directions, at rates that keep
prevalence unchanged. Severity comes from how far a score lies above the threshold.
"""

from __future__ import annotations

import numpy as np

from smartep.contracts.gap_areas import GAP_AREAS
from smartep.synth.traits import TRAITS

# Positive weight = more of this trait means MORE of a gap.
AREA_WEIGHTS: dict[str, dict[str, float]] = {
    "content_coverage":    {"content_mastery": -0.7, "workload_pressure": 0.5},
    "content_currency":    {"currency": -0.9, "digital_skill": -0.3},
    "resources":           {"pedagogical_skill": -0.5, "workload_pressure": 0.5, "digital_skill": -0.3},
    "video_delivery":      {"digital_skill": -0.9, "pedagogical_skill": -0.3},
    "session_delivery":    {"workload_pressure": 0.9, "responsiveness": -0.2},
    "assessment_feedback": {"responsiveness": -0.9, "workload_pressure": 0.3},
    "student_engagement":  {"responsiveness": -0.5, "pedagogical_skill": -0.6, "digital_skill": -0.2},
    "pedagogy":            {"pedagogical_skill": -1.0, "content_mastery": -0.2},
    "technical_skill":     {"content_mastery": -0.9, "digital_skill": -0.3},
}


def weight_matrix() -> np.ndarray:
    W = np.zeros((len(TRAITS), len(GAP_AREAS)))
    for j, area in enumerate(GAP_AREAS):
        for trait, w in AREA_WEIGHTS[area].items():
            W[TRAITS.index(trait), j] = w
    return W / np.linalg.norm(W, axis=0, keepdims=True)


def deficiency_scores(T: np.ndarray, rng: np.random.Generator, noise_sd: float) -> np.ndarray:
    return T @ weight_matrix() + rng.normal(0.0, noise_sd, size=(len(T), len(GAP_AREAS)))


def assign_labels(scores: np.ndarray, prevalence: dict[str, float], label_noise: float,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels (n, areas) in {0,1}, severity (n, areas) as str ('' for no gap), flipped mask."""
    n, k = scores.shape
    Y = np.zeros((n, k), dtype=np.int8)
    severity = np.full((n, k), "", dtype=object)
    flipped = np.zeros((n, k), dtype=bool)
    for j, area in enumerate(GAP_AREAS):
        p = prevalence[area]
        threshold = np.quantile(scores[:, j], 1 - p)
        clean = scores[:, j] >= threshold
        # Symmetric in count: positives lost = negatives gained, so prevalence is unchanged.
        to_neg = clean & (rng.random(n) < label_noise)
        to_pos = ~clean & (rng.random(n) < label_noise * p / (1 - p))
        y = (clean & ~to_neg) | to_pos
        Y[:, j] = y
        flipped[:, j] = to_neg | to_pos

        margin = scores[:, j] - threshold
        cuts = np.quantile(margin[clean], [1 / 3, 2 / 3]) if clean.any() else (0.0, 0.0)
        sev = np.where(margin >= cuts[1], "high", np.where(margin >= cuts[0], "medium", "low"))
        severity[:, j] = np.where(y, np.where(to_pos, "low", sev), "")
    return Y, severity, flipped
