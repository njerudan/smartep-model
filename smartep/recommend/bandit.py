"""LinUCB contextual bandit.

Phase 0 keeps the notebook's arms (the 7 legacy topics) and simulated reward. Phase 6 changes the arms to
catalogue courses or skill areas and the reward to enrolment and completion feedback.
"""

from __future__ import annotations

import numpy as np


class LinUCB:
    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0, ridge: float = 1.0):
        self.n_arms, self.alpha = n_arms, alpha
        # Inverse design matrices kept up to date with Sherman-Morrison (no inversion per call).
        self.A_inv = np.stack([np.eye(context_dim) / ridge for _ in range(n_arms)])
        self.b = np.zeros((n_arms, context_dim))

    def scores(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        theta = np.einsum("aij,aj->ai", self.A_inv, self.b)          # (arms, dim)
        mean = X @ theta.T                                            # (n, arms)
        var = np.einsum("ni,aij,nj->na", X, self.A_inv, X)
        return mean + self.alpha * np.sqrt(np.maximum(var, 0.0))

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        Ax = self.A_inv[arm] @ x
        self.A_inv[arm] -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[arm] += reward * x


def simulate_reward(true_gap_row: np.ndarray, arm: int, engagement: float, rng: np.random.Generator) -> float:
    base = 0.8 if true_gap_row[arm] else 0.1
    return float(np.clip(base + 0.2 * engagement + rng.normal(0, 0.05), 0, 1))


def train_bandit(bandit: LinUCB, X: np.ndarray, Y: np.ndarray, engagement: np.ndarray, passes: int,
                 rng: np.random.Generator) -> LinUCB:
    for _ in range(passes):
        for i in range(len(X)):
            arm = int(np.argmax(bandit.scores(X[i])[0]))
            bandit.update(arm, X[i], simulate_reward(Y[i], arm, engagement[i], rng))
    return bandit
