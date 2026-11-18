"""Tabular features.

``build_legacy_tabular`` merges the two feature cells of the notebook into one (LMS aggregates, survey,
evaluations, video viewing, lecture metadata). ``TabularFeaturizer`` learns category levels and scaling
from the training rows only, so validation and test rows never influence the transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from smartep.synth.legacy import LegacyTables


def build_legacy_tabular(tables: LegacyTables) -> pd.DataFrame:
    ids = pd.Index(tables.educator_ids, name="educator_id")
    surveys = tables.surveys.set_index("educator_id").reindex(ids)

    logs = tables.lms_logs.groupby("educator_id").agg(
        click_mean=("click_count", "mean"),
        forum_posts_sum=("forum_posts", "sum"),
        late_days_mean=("assignment_late_days", "mean"),
        num_courses=("course_id", "nunique"),
    )
    evals = tables.student_evaluations.groupby("educator_id").agg(
        avg_rating=("rating", "mean"), n_evaluations=("rating", "size")
    )
    watch = tables.video_watch_logs.merge(tables.video_catalog[["video_id", "topic"]], on="video_id", how="left")
    video = watch.groupby("educator_id").agg(
        total_videos_watched=("video_id", "count"),
        total_watch_seconds=("watch_duration_seconds", "sum"),
        avg_watch_seconds=("watch_duration_seconds", "mean"),
        completion_rate=("completed", "mean"),
        unique_topics=("topic", "nunique"),
    )
    lectures = tables.lectures
    level_share = pd.crosstab(lectures["educator_id"], lectures["academic_level"], normalize="index")
    level_share.columns = [f"level_share_{c}" for c in level_share.columns]
    slides = lectures.groupby("educator_id")["num_slides"].mean().rename("num_slides_mean")

    out = surveys.join([logs, evals, video, level_share, slides])
    numeric = out.columns.difference(["department"])
    out[numeric] = out[numeric].fillna(0).astype(float)
    return out


class TabularFeaturizer:
    """One-hot encodes categoricals with training levels and standardises with training statistics."""

    def __init__(self, categorical: list[str] | None = None):
        self.categorical = list(categorical or [])
        self.levels: dict[str, list[str]] = {}
        self.numeric: list[str] = []
        self.feature_names: list[str] = []
        self.scaler: StandardScaler | None = None

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [df[self.numeric].astype(float)]
        for col in self.categorical:
            values = df[col].astype(str)
            parts.append(pd.DataFrame(
                {f"{col}={lvl}": (values == lvl).astype(float) for lvl in self.levels[col]}, index=df.index
            ))
        return pd.concat(parts, axis=1)

    def fit(self, df: pd.DataFrame) -> TabularFeaturizer:
        self.numeric = [c for c in df.columns if c not in self.categorical]
        self.levels = {c: sorted(df[c].dropna().astype(str).unique()) for c in self.categorical}
        encoded = self._encode(df)
        self.feature_names = list(encoded.columns)
        self.scaler = StandardScaler().fit(encoded.to_numpy())
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("fit() first")
        encoded = self._encode(df)[self.feature_names]
        return self.scaler.transform(encoded.to_numpy()).astype(np.float32)
