"""Educator personas (KMeans), fitted on training rows only. Research use, not served."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def fit_personas(X_train: np.ndarray, X_all: np.ndarray, n_clusters: int, seed: int):
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X_train)
    return km, km.predict(X_all), round(float(silhouette_score(X_train, km.labels_)), 4)
