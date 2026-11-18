"""Watch-history neighbourhood features.

The notebook called this the "graph" module, but it holds no graph: each educator attends over their own
first ``max_neighbors`` watch rows (topic one-hot + completed flag). It is renamed here to say what it is.
A real educator-course-skill graph is planned for Phase 5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_watch_history(watch_logs: pd.DataFrame, video_catalog: pd.DataFrame, educator_ids, topics,
                        max_neighbors: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Return feats (n, max_neighbors, len(topics) + 1) and mask (n, max_neighbors)."""
    topic_idx = {t: i for i, t in enumerate(topics)}
    rows = watch_logs.merge(video_catalog[["video_id", "topic"]], on="video_id", how="left")
    rows = rows[rows["topic"].isin(topic_idx)]
    rows = rows.assign(slot=rows.groupby("educator_id").cumcount())
    rows = rows[rows["slot"] < max_neighbors]

    position = {e: i for i, e in enumerate(educator_ids)}
    rows = rows[rows["educator_id"].isin(position)]
    n = len(position)
    feats = np.zeros((n, max_neighbors, len(topics) + 1), dtype=np.float32)
    mask = np.zeros((n, max_neighbors), dtype=np.float32)
    r = rows["educator_id"].map(position).to_numpy()
    s = rows["slot"].to_numpy()
    feats[r, s, rows["topic"].map(topic_idx).to_numpy()] = 1.0
    feats[r, s, -1] = rows["completed"].to_numpy(dtype=np.float32)
    mask[r, s] = 1.0
    return feats, mask
