"""Weekly activity sequences for the sequence encoder.

The notebook defined this builder twice with different return values and normalised over all rows. Here
there is one builder, and normalisation statistics come from the training rows only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEGACY_SEQ_COLS = ("click_count", "forum_posts", "assignment_late_days")


def build_weekly_tensor(logs: pd.DataFrame, educator_ids, seq_len_weeks: int,
                        value_cols=LEGACY_SEQ_COLS, time_col: str = "timestamp") -> np.ndarray:
    """Return a (n_educators, seq_len_weeks, n_features) array of weekly means; missing weeks are 0."""
    logs = logs[["educator_id", time_col, *value_cols]].copy()
    logs[time_col] = pd.to_datetime(logs[time_col])
    start = logs.groupby("educator_id")[time_col].transform("min")
    logs["week"] = ((logs[time_col] - start).dt.days // 7).clip(0, seq_len_weeks - 1)
    weekly = logs.groupby(["educator_id", "week"])[list(value_cols)].mean().reset_index()

    position = {e: i for i, e in enumerate(educator_ids)}
    weekly = weekly[weekly["educator_id"].isin(position)]
    tensor = np.zeros((len(position), seq_len_weeks, len(value_cols)), dtype=np.float32)
    rows = weekly["educator_id"].map(position).to_numpy()
    weeks = weekly["week"].to_numpy(dtype=int)
    tensor[rows, weeks, :] = weekly[list(value_cols)].to_numpy(dtype=np.float32)
    return tensor


class SequenceNormalizer:
    def fit(self, tensor: np.ndarray) -> SequenceNormalizer:
        flat = tensor.reshape(-1, tensor.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0) + 1e-6
        return self

    def transform(self, tensor: np.ndarray) -> np.ndarray:
        return ((tensor - self.mean_) / self.std_).astype(np.float32)
