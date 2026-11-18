"""Structured features from one evidence package (schema v1.0).

Phase 1 version: numeric features read from package fields only, used by the evidence probe in the
G1 checks. Phase 2 adds text features (notes, slides, transcripts, comments) and evidence mapping.
Group attributes (seniority, discipline, institution type) are deliberately not features; they are kept
for the fairness audit.
"""

from __future__ import annotations

import re
from datetime import datetime

import numpy as np
import pandas as pd

YEAR = re.compile(r"\b(19|20)\d{2}\b")
LIKERT_DIMS = ("clarity", "organisation", "feedback", "engagement", "resources", "overall")


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


def _ratio(a, b, default=np.nan) -> float:
    return float(a) / float(b) if b else default


def _mean(values) -> float:
    values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(values)) if values else np.nan


def package_features(pkg: dict) -> dict[str, float]:
    own = [les for les in pkg["lessons"] if les["taughtByThisEducator"]] or pkg["lessons"]
    delivered = [les for les in own if les["deliveredStatus"] == "delivered"]
    by_ref = {les["lessonRef"]: les for les in pkg["lessons"]}
    plan = pkg["plan"]
    items = plan["items"] if plan else []
    period_year = int(pkg["period"]["startsAt"][:4])
    f: dict[str, float] = {}

    # Plan and coverage
    f["plan_present"] = float(plan is not None)
    f["plan_approved"] = float(bool(plan) and plan["status"] == "approved")
    f["plan_item_coverage"] = _mean([
        float(any(by_ref.get(r, {}).get("deliveredStatus") == "delivered" and by_ref[r].get("notes")
                  for r in it["lessonRefs"])) for it in items])
    f["required_notes_coverage"] = _mean([float(bool(by_ref.get(r, {}).get("notes")))
                                          for it in items if it["requiresNotes"] for r in it["lessonRefs"]])
    f["required_video_coverage"] = _mean([float(any(by_ref.get(r, {}).get("videos") for r in it["lessonRefs"]))
                                          for it in items if it["requiresVideo"]])

    # Delivery
    d = pkg["deliveryIndicators"]
    f["delivery_ratio"] = _ratio(d["sessionsDelivered"], d["sessionsPlanned"])
    f["contact_minutes_ratio"] = _ratio(d["contactMinutesDelivered"], d.get("contactMinutesPlanned"))
    f["unknown_status_share"] = _mean([float(les["deliveredStatus"] == "unknown") for les in own])
    notes = [les for les in own if les.get("notes")]
    f["notes_share"] = _ratio(len(notes), len(own))
    f["notes_late_share"] = _mean([float(_dt(les["notes"]["firstPostedAt"]) > _dt(les["scheduledAt"])) for les in notes])
    f["notes_mean_words"] = _mean([len(les["notes"]["text"].split()) for les in notes])
    f["slides_share"] = _ratio(sum(bool(les.get("slides")) for les in own), len(own))
    f["documents_per_lesson"] = _ratio(sum(len(les["documents"]) for les in own), len(own))
    years = [int(m.group(0)) for les in own for doc in les["documents"] for m in [YEAR.search(doc["title"])] if m]
    f["document_age_years_mean"] = _mean([period_year - y for y in years])
    attendance = [les["attendance"] for les in delivered if les.get("attendance")]
    f["attendance_rate"] = _mean([_ratio(a["present"], a["recorded"]) for a in attendance])

    # Video
    videos = [v for les in own for v in les["videos"]]
    f["videos_per_lesson"] = _ratio(len(videos), len(own))
    f["video_completion_mean"] = _mean([v["viewing"]["completionRate"] for v in videos if v["viewing"]["viewers"]])
    f["video_furthest_median_mean"] = _mean([v["viewing"]["medianFurthestPct"] for v in videos if v["viewing"]["viewers"]])
    f["video_viewer_share"] = _mean([_ratio(v["viewing"]["viewers"], pkg["course"]["enrolledStudents"]) for v in videos])
    f["video_transcript_share"] = _mean([float("transcript" in v) for v in videos])
    f["video_hosted_share"] = _mean([float(v["provider"] in ("upload", "youtube")) for v in videos])

    # Assessment and responsiveness
    f["assignments_graded"] = float(d["assignmentsGraded"])
    f["grading_turnaround_hours"] = np.nan if d.get("gradingTurnaroundMedianHours") is None else float(d["gradingTurnaroundMedianHours"])
    f["questions_answered_72h_rate"] = _ratio(d["questionsAnsweredWithin72h"], d["questionsAsked"])
    f["questions_per_student"] = _ratio(d["questionsAsked"], pkg["course"]["enrolledStudents"])

    # Evaluations
    rounds = pkg["evaluations"]
    f["evaluation_response_rate"] = _mean([_ratio(r["responses"], r["enrolled"]) for r in rounds])
    f["evaluation_below_threshold_share"] = _mean([float(r["belowThreshold"]) for r in rounds])
    for dim in LIKERT_DIMS:
        means = []
        for r in rounds:
            for q in r.get("questions") or []:
                if q.get("dimension") == dim and q.get("distribution") and sum(q["distribution"]):
                    dist = np.asarray(q["distribution"], dtype=float)
                    means.append(float((dist * np.arange(1, 6)).sum() / dist.sum()))
        f[f"likert_{dim}_mean"] = _mean(means)
    f["comments_per_response"] = _mean([
        _ratio(len(q.get("comments") or []), r["responses"]) for r in rounds for q in r.get("questions") or []
        if q["kind"] == "text"])

    # Context
    f["enrolled_students"] = float(pkg["course"]["enrolledStudents"])
    f["n_previous_packages"] = float(len(pkg["previousPackages"]))
    return f


def package_feature_frame(packages: list[dict]) -> pd.DataFrame:
    """One row per package, indexed by packageId. Missing values are left as NaN (XGBoost handles them)."""
    return pd.DataFrame([package_features(p) for p in packages], index=[p["packageId"] for p in packages])
