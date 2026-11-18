"""Port of the Colab baseline's simulated tables (the SQLite "production database" cell).

Kept for Phase 0 only, so the refactored pipeline can be checked against the notebook. It reproduces the
notebook's known defect: skill gaps are drawn at random and independently of every other table, so no
model can learn them and test AUC should sit near 0.5. Phase 1 replaces this module with generator v2,
which emits evidence packages in schema v1.0 with a planted, documented signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

LEGACY_TOPICS = ("AI", "ML", "Networking", "Databases", "Programming", "Security", "Cloud")
COURSES = ("CS101", "CS102", "CS201", "CS301", "EE101", "MATH101", "PHY101")
DEPARTMENTS = ("CS", "Math", "Physics", "Engineering", "Business", "Education")
ACADEMIC_LEVELS = ("Undergraduate", "Postgraduate", "Diploma")
N_DAYS = 30


@dataclass
class LegacyTables:
    skill_gaps: pd.DataFrame
    lms_logs: pd.DataFrame
    surveys: pd.DataFrame
    lectures: pd.DataFrame
    student_evaluations: pd.DataFrame
    video_catalog: pd.DataFrame
    video_watch_logs: pd.DataFrame

    @property
    def educator_ids(self) -> list[str]:
        return self.surveys["educator_id"].tolist()

    def items(self):
        return vars(self).items()


def generate_legacy_tables(n_educators: int, seed: int) -> LegacyTables:
    rng = np.random.default_rng(seed)
    educators = [f"EDU{i:04d}" for i in range(n_educators)]
    dates = [(datetime(2026, 5, 1) + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(N_DAYS)]

    gaps = []
    for edu in educators:
        n_gaps = rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
        chosen = set(rng.choice(LEGACY_TOPICS, size=n_gaps, replace=False)) if n_gaps else set()
        for topic in LEGACY_TOPICS:
            is_gap = int(topic in chosen)
            score = rng.uniform(0, 60) if is_gap else rng.uniform(60, 100)
            gaps.append((edu, topic, is_gap, score))
    skill_gaps = pd.DataFrame(gaps, columns=["educator_id", "skill_domain", "is_skill_gap", "assessment_score"])

    logs = []
    for edu in educators:
        base_clicks = rng.poisson(30) + 20
        base_posts = rng.poisson(1)
        base_late = rng.exponential(0.3)
        clicks = np.maximum(0, (rng.poisson(base_clicks, N_DAYS) + rng.normal(0, 5, N_DAYS)).astype(int))
        posts = np.maximum(0, (rng.poisson(base_posts, N_DAYS) + rng.normal(0, 0.5, N_DAYS)).astype(int))
        late = np.maximum(0.0, base_late + rng.normal(0, 0.1, N_DAYS))
        courses = rng.choice(COURSES, N_DAYS)
        logs.extend(zip([edu] * N_DAYS, courses, dates, clicks, posts, late, strict=True))
    lms_logs = pd.DataFrame(
        logs, columns=["educator_id", "course_id", "timestamp", "click_count", "forum_posts", "assignment_late_days"]
    )

    surveys = pd.DataFrame({
        "educator_id": educators,
        "department": rng.choice(DEPARTMENTS, n_educators),
        "years_experience": rng.integers(1, 20, n_educators),
        "self_rated_skill": rng.choice([1, 2, 3, 4, 5], n_educators, p=[0.05, 0.10, 0.30, 0.35, 0.20]),
        "confidence_ai": rng.choice([1, 2, 3, 4, 5], n_educators, p=[0.05, 0.10, 0.25, 0.35, 0.25]),
        "hours_per_week_cpd": rng.uniform(0, 15, n_educators),
    })

    evals = []
    for edu in educators:
        for rating in rng.integers(1, 6, rng.integers(1, 10)):
            evals.append((edu, "Great educator!" if rating >= 4 else "Needs improvement.", int(rating)))
    student_evaluations = pd.DataFrame(evals, columns=["educator_id", "comment", "rating"])

    videos = []
    for topic in LEGACY_TOPICS:
        for j in range(5):
            videos.append((f"VID{topic[:3].upper()}{j:02d}", f"{topic} Lecture {j + 1}", topic,
                           int(rng.integers(300, 3600)), int(rng.integers(1, 6))))
    video_catalog = pd.DataFrame(videos, columns=["video_id", "title", "topic", "duration_seconds", "difficulty"])

    watches = []
    video_ids = video_catalog["video_id"].to_numpy()
    for edu in educators:
        for _ in range(rng.integers(5, 20)):
            duration = int(rng.integers(60, 1800))
            watches.append((edu, rng.choice(video_ids), rng.choice(dates), duration, int(duration > 1500)))
    video_watch_logs = pd.DataFrame(
        watches, columns=["educator_id", "video_id", "watch_date", "watch_duration_seconds", "completed"]
    )

    lectures = []
    for edu in educators:
        for _ in range(rng.integers(1, 5)):
            topic, level = rng.choice(LEGACY_TOPICS), rng.choice(ACADEMIC_LEVELS)
            lectures.append((edu, topic, level, f"This is a lecture on {topic} for {level} students.",
                             int(rng.integers(10, 50))))
    lectures_df = pd.DataFrame(lectures, columns=["educator_id", "topic", "academic_level", "lecture_text", "num_slides"])

    return LegacyTables(skill_gaps, lms_logs, surveys, lectures_df, student_evaluations, video_catalog, video_watch_logs)


def label_matrix(tables: LegacyTables, topics=LEGACY_TOPICS) -> pd.DataFrame:
    """One row per educator, one 0/1 column per topic, named ``gap_<topic>``."""
    wide = (
        tables.skill_gaps.pivot_table(index="educator_id", columns="skill_domain", values="is_skill_gap", aggfunc="max")
        .reindex(index=tables.educator_ids, columns=list(topics))
        .fillna(0)
        .astype(int)
    )
    wide.columns = [f"gap_{t}" for t in topics]
    wide.index.name = "educator_id"
    return wide
