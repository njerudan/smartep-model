"""One grouped train/validation/test split, shared by every model and modality.

Rows are split by group (educatorRef), so all packages of one educator land in the same part. This
matters once Phase 1 produces several packages per educator.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def grouped_split(groups, val_size: float, test_size: float, seed: int) -> Split:
    groups = np.asarray(groups)
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be between 0 and 1")
    unique = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(unique)
    n_test = max(1, round(len(unique) * test_size))
    n_val = max(1, round(len(unique) * val_size))
    test_g, val_g = perm[:n_test], perm[n_test:n_test + n_val]
    test = np.flatnonzero(np.isin(groups, test_g))
    val = np.flatnonzero(np.isin(groups, val_g))
    train = np.flatnonzero(~np.isin(groups, np.concatenate([test_g, val_g])))
    return Split(train=train, val=val, test=test)


def frame_hash(*frames: pd.DataFrame) -> str:
    """Stable fingerprint of the dataset a model was trained on."""
    h = hashlib.sha256()
    for df in frames:
        h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
        h.update(",".join(map(str, df.columns)).encode())
    return h.hexdigest()[:12]
