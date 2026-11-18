"""The nine gap areas SMARTEP predicts (AssessmentResult.gaps[].area)."""

from typing import Literal

GapArea = Literal[
    "content_coverage",
    "content_currency",
    "resources",
    "video_delivery",
    "session_delivery",
    "assessment_feedback",
    "student_engagement",
    "pedagogy",
    "technical_skill",
]

GAP_AREAS: tuple[str, ...] = GapArea.__args__  # type: ignore[attr-defined]
