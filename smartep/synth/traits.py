"""Latent educator traits.

Six standardised traits (mean 0, sd 1). Higher is better for every trait except workload_pressure.
Each educator has stable base traits; each period adds a small drift. Correlations are mild and
documented in the data sheet.
"""

from __future__ import annotations

import numpy as np

TRAITS = (
    "content_mastery",     # knows the subject (CK)
    "pedagogical_skill",   # explains and engages well (PK)
    "digital_skill",       # uses slides, video and platform tools well (TK)
    "workload_pressure",   # overloaded: cancels, posts late, skips (higher = worse)
    "responsiveness",      # grades and answers questions promptly
    "currency",            # keeps materials and examples up to date
)

DEFAULT_CORRELATIONS = {
    ("content_mastery", "pedagogical_skill"): 0.3,
    ("digital_skill", "currency"): 0.4,
    ("pedagogical_skill", "responsiveness"): 0.3,
    ("workload_pressure", "responsiveness"): -0.3,
}


def correlation_matrix(pairs: dict | None = None) -> np.ndarray:
    pairs = DEFAULT_CORRELATIONS if pairs is None else pairs
    idx = {t: i for i, t in enumerate(TRAITS)}
    corr = np.eye(len(TRAITS))
    for (a, b), r in pairs.items():
        corr[idx[a], idx[b]] = corr[idx[b], idx[a]] = r
    np.linalg.cholesky(corr)  # raises if the matrix is not positive definite
    return corr


def sample_base_traits(n: int, rng: np.random.Generator, pairs: dict | None = None) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(len(TRAITS)), correlation_matrix(pairs), size=n)


def period_traits(base: np.ndarray, rng: np.random.Generator, drift_sd: float) -> np.ndarray:
    return base + rng.normal(0.0, drift_sd, size=base.shape)


def as_dict(row: np.ndarray) -> dict[str, float]:
    return {t: float(v) for t, v in zip(TRAITS, row, strict=True)}
