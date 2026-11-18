"""From model drivers to package evidence.

For a predicted gap, the features that pushed the prediction towards "gap" (positive SHAP values) are
taken in order of impact. Each feature knows which package records it was computed from, and returns
those records worst first as AssessmentResult evidence items. Text features rank their texts by
similarity to the gap area's description. The first ``max_items`` distinct records are kept; when no
driver yields a record, a fallback item points at the latest evaluation round.

Only schema fields and the neutral area descriptions are used, never the synthetic generator's rules.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from smartep.contracts.area_text import AREA_DESCRIPTIONS

MAX_ITEMS = 3
PER_FEATURE = 2


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


def _own(pkg):
    return [les for les in pkg["lessons"] if les["taughtByThisEducator"]] or pkg["lessons"]


def _by_ref(pkg):
    return {les["lessonRef"]: les for les in pkg["lessons"]}


def _items(pkg):
    return (pkg["plan"] or {}).get("items", [])


# --- record finders (each returns evidence items, worst first) ---------------------------------------------

def plan_records(pkg):
    if pkg["plan"] is None:
        return [{"source": "plan", "ref": "plan", "note": "No delivery plan was submitted"}]
    if pkg["plan"]["status"] != "approved":
        return [{"source": "plan", "ref": "plan", "note": f"Delivery plan status is {pkg['plan']['status']}"}]
    return []


def uncovered_items(pkg):
    lessons = _by_ref(pkg)
    out = [{"source": "plan", "ref": it["itemRef"],
            "note": f"No delivered lesson with notes for '{it['topic']}' (week {it['week']})"}
           for it in _items(pkg)
           if not any(lessons.get(r, {}).get("deliveredStatus") == "delivered" and lessons[r].get("notes")
                      for r in it["lessonRefs"])]
    return out or plan_records(pkg)


def missing_notes(pkg):
    lessons = _by_ref(pkg)
    required = [r for it in _items(pkg) if it["requiresNotes"] for r in it["lessonRefs"]]
    refs = required or [les["lessonRef"] for les in _own(pkg)]
    return [{"source": "notes", "ref": r, "note": "Notes were not posted"} for r in refs
            if r in lessons and not lessons[r].get("notes")]


def late_notes(pkg):
    out = []
    for les in _own(pkg):
        n = les.get("notes")
        if n and _dt(n["firstPostedAt"]) > _dt(les["scheduledAt"]):
            days = (_dt(n["firstPostedAt"]) - _dt(les["scheduledAt"])).days
            out.append((-days, {"source": "notes", "ref": les["lessonRef"], "note": f"Notes posted {days} days after the session"}))
    return [e for _, e in sorted(out, key=lambda x: x[0])]


def missing_materials(pkg):
    return [{"source": "slides", "ref": les["lessonRef"], "note": "No slides or documents shared"}
            for les in _own(pkg) if not les.get("slides") and not les["documents"]]


def missing_videos(pkg):
    lessons = _by_ref(pkg)
    return [{"source": "plan", "ref": it["itemRef"], "note": f"Required video for '{it['topic']}' not posted"}
            for it in _items(pkg) if it["requiresVideo"] and not any(lessons.get(r, {}).get("videos") for r in it["lessonRefs"])]


def weak_videos(pkg):
    out = []
    for les in pkg["lessons"]:
        for v in les["videos"]:
            view = v["viewing"]
            if view["viewers"]:
                at = (v.get("durationSeconds") or 0) * view["medianFurthestPct"] / 100
                out.append((view["completionRate"], {
                    "source": "viewing", "ref": v["videoRef"], "atSeconds": round(at, 1),
                    "note": f"Completion {view['completionRate']:.0%}; half of viewers stop by {view['medianFurthestPct']:.0f}%"}))
    return [e for _, e in sorted(out, key=lambda x: x[0])]


def videos_without_transcript(pkg):
    return [{"source": "viewing", "ref": v["videoRef"], "note": f"Video hosted on {v['provider']} without a transcript"}
            for les in pkg["lessons"] for v in les["videos"] if "transcript" not in v]


def undelivered(pkg):
    return [{"source": "delivery", "ref": les["lessonRef"], "note": "Scheduled session was not delivered"}
            for les in _own(pkg) if les["deliveredStatus"] == "not_delivered"]


def low_attendance(pkg):
    out = [(a["present"] / a["recorded"], {"source": "delivery", "ref": les["lessonRef"],
                                            "note": f"Attendance {a['present']} of {a['recorded']}"})
           for les in _own(pkg) for a in [les.get("attendance")] if a and a["recorded"]]
    return [e for _, e in sorted(out, key=lambda x: x[0])]


def grading(pkg):
    d = pkg["deliveryIndicators"]
    if d.get("gradingTurnaroundMedianHours") is None:
        return [{"source": "delivery", "ref": "deliveryIndicators.assignmentsGraded",
                 "note": f"{d['assignmentsGraded']} assignments graded"}]
    return [{"source": "delivery", "ref": "deliveryIndicators.gradingTurnaroundMedianHours",
             "note": f"Median grading turnaround {d['gradingTurnaroundMedianHours']:.0f} hours"}]


def questions(pkg):
    d = pkg["deliveryIndicators"]
    return [{"source": "delivery", "ref": "deliveryIndicators.questionsAnsweredWithin72h",
             "note": f"{d['questionsAnsweredWithin72h']} of {d['questionsAsked']} questions answered within 72 hours"}]


def response_rates(pkg):
    out = [(r["responses"] / r["enrolled"] if r["enrolled"] else 0, {
        "source": "evaluation", "ref": r["roundRef"], "note": f"{r['responses']} of {r['enrolled']} students responded"})
        for r in pkg["evaluations"]]
    return [e for _, e in sorted(out, key=lambda x: x[0])]


def likert(dim: str) -> Callable:
    def finder(pkg):
        out = []
        for r in pkg["evaluations"]:
            for q in r.get("questions") or []:
                if q.get("dimension") == dim and q.get("distribution") and sum(q["distribution"]):
                    dist = np.asarray(q["distribution"], dtype=float)
                    mean = float((dist * np.arange(1, 6)).sum() / dist.sum())
                    out.append((mean, {"source": "evaluation", "ref": f"{r['roundRef']}:{q['questionRef']}",
                                       "note": f"Mean {dim} rating {mean:.2f} of 5"}))
        return [e for _, e in sorted(out, key=lambda x: x[0])]
    return finder


def old_documents(pkg):
    import re

    year = int(pkg["period"]["startsAt"][:4])
    out = []
    for les in _own(pkg):
        for d in les["documents"]:
            m = re.search(r"\b(19|20)\d{2}\b", d["title"])
            if m and year - int(m.group(0)) >= 3:
                out.append((int(m.group(0)), {"source": "document", "ref": les["lessonRef"], "quote": d["title"]}))
    return [e for _, e in sorted(out, key=lambda x: x[0])]


def _rank_by_area(candidates: list[tuple[str, dict]], area: str, ranker: str = "tfidf") -> list[dict]:
    """Rank (text, evidence item) pairs by similarity of the text to the area description.

    ``ranker`` is ``minilm`` (sentence embeddings, semantic) or ``tfidf`` (word overlap, no model needed).
    """
    if not candidates:
        return []
    texts = [t for t, _ in candidates]
    if ranker == "minilm":
        from smartep.features.package_text import embed_units

        emb = embed_units(texts)
        sims = emb @ embed_units([AREA_DESCRIPTIONS[area]])[0]
        order = np.argsort(-sims, kind="stable")
        return [candidates[i][1] for i in order]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([*texts, AREA_DESCRIPTIONS[area]])
    sims = cosine_similarity(vec.transform(texts), vec.transform([AREA_DESCRIPTIONS[area]]))[:, 0]
    order = np.argsort(-sims, kind="stable")
    return [candidates[i][1] for i in order if sims[i] > 0]


def comment_records(pkg, area, ranker="tfidf"):
    cands = []
    for r in pkg["evaluations"]:
        for q in r.get("questions") or []:
            if q["kind"] == "text":
                cands += [(c, {"source": "evaluation", "ref": f"{r['roundRef']}:{q['questionRef']}", "quote": c})
                          for c in q.get("comments") or []]
    return _rank_by_area(cands, area, ranker)


def notes_records(pkg, area, ranker="tfidf"):
    cands = []
    for les in pkg["lessons"]:
        if les.get("notes"):
            cands.append((les["notes"]["text"], {"source": "notes", "ref": les["lessonRef"], "quote": les["notes"]["text"][:160]}))
        cands += [(s["text"], {"source": "slides", "ref": les["lessonRef"], "quote": s["text"][:160]}) for s in les.get("slides") or []]
        cands += [(d["extractedText"], {"source": "document", "ref": les["lessonRef"], "quote": d["extractedText"][:160]})
                  for d in les["documents"] if d.get("extractedText")]
    return _rank_by_area(cands, area, ranker)


def transcript_records(pkg, area, ranker="tfidf"):
    cands = [(s["text"], {"source": "transcript", "ref": v["videoRef"], "atSeconds": s["start"], "quote": s["text"][:160]})
             for les in pkg["lessons"] for v in les["videos"] for s in (v.get("transcript") or {}).get("segments", [])]
    return _rank_by_area(cands, area, ranker)


STRUCTURED: dict[str, Callable] = {
    "plan_present": plan_records, "plan_approved": plan_records,
    "plan_item_coverage": uncovered_items,
    "required_notes_coverage": missing_notes, "notes_share": missing_notes,
    "notes_late_share": late_notes, "notes_mean_words": missing_notes,
    "slides_share": missing_materials, "documents_per_lesson": missing_materials,
    "document_age_years_mean": old_documents,
    "required_video_coverage": missing_videos, "videos_per_lesson": missing_videos,
    "video_completion_mean": weak_videos, "video_furthest_median_mean": weak_videos, "video_viewer_share": weak_videos,
    "video_transcript_share": videos_without_transcript, "video_hosted_share": videos_without_transcript,
    "delivery_ratio": undelivered, "contact_minutes_ratio": undelivered, "unknown_status_share": undelivered,
    "attendance_rate": low_attendance,
    "assignments_graded": grading, "grading_turnaround_hours": grading,
    "questions_answered_72h_rate": questions, "questions_per_student": questions,
    "evaluation_response_rate": response_rates, "evaluation_below_threshold_share": response_rates,
    "comments_per_response": response_rates,
    **{f"likert_{d}_mean": likert(d) for d in ("clarity", "organisation", "feedback", "engagement", "resources", "overall")},
}
TEXT_GROUPS = {"comments": comment_records, "notes": notes_records, "transcripts": transcript_records}

# Feature families, and which families are evidence for each area. This follows from what each area means
# (spec §6 area list; contracts/area_text.py), e.g. a coverage gap is shown by plan items, a resources gap by
# notes and materials. Drivers from a relevant family are used first; others only after them.
FAMILY = {
    "plan_present": "plan", "plan_approved": "plan", "plan_item_coverage": "coverage",
    "required_notes_coverage": "notes", "notes_share": "notes", "notes_late_share": "notes", "notes_mean_words": "notes",
    "slides_share": "materials", "documents_per_lesson": "materials", "document_age_years_mean": "documents",
    "required_video_coverage": "video", "videos_per_lesson": "video", "video_completion_mean": "video",
    "video_furthest_median_mean": "video", "video_viewer_share": "video", "video_transcript_share": "video",
    "video_hosted_share": "video", "delivery_ratio": "delivery", "contact_minutes_ratio": "delivery",
    "unknown_status_share": "delivery", "attendance_rate": "attendance", "assignments_graded": "grading",
    "grading_turnaround_hours": "grading", "questions_answered_72h_rate": "questions", "questions_per_student": "questions",
    "evaluation_response_rate": "response", "evaluation_below_threshold_share": "response", "comments_per_response": "response",
    **{f"likert_{d}_mean": f"likert_{d}" for d in ("clarity", "organisation", "feedback", "engagement", "resources", "overall")},
}
RELEVANT = {
    "content_coverage": {"coverage", "plan", "notes"},
    "content_currency": {"documents", "text_notes", "likert_resources", "text_comments"},
    "resources": {"notes", "materials", "likert_resources", "text_comments"},
    "video_delivery": {"video"},
    "session_delivery": {"delivery", "likert_organisation", "text_comments"},
    "assessment_feedback": {"grading", "questions", "likert_feedback", "text_comments"},
    "student_engagement": {"attendance", "response", "likert_engagement", "text_comments"},
    "pedagogy": {"likert_clarity", "likert_engagement", "text_comments", "text_transcripts"},
    "technical_skill": {"text_transcripts", "text_comments", "likert_clarity", "coverage"},
}
def family(feature: str) -> str | None:
    if feature.startswith("text_"):
        return "text_" + feature.split("_")[1]
    return FAMILY.get(feature)


def records_for_feature(pkg: dict, feature: str, area: str, ranker: str = "tfidf") -> list[dict]:
    if feature in STRUCTURED:
        return STRUCTURED[feature](pkg)
    if feature.startswith("text_"):
        group = feature.split("_")[1]
        if group in TEXT_GROUPS:
            return TEXT_GROUPS[group](pkg, area, ranker)
    return []  # context features (class size, history) carry no evidence


def fallback_evidence(pkg: dict, area: str) -> dict:
    if pkg["evaluations"]:
        r = pkg["evaluations"][-1]
        return {"source": "evaluation", "ref": r["roundRef"], "note": f"Overall evaluation evidence for {area}"}
    return {"source": "delivery", "ref": "deliveryIndicators", "note": f"Delivery indicators for {area}"}


MAX_DRIVERS = 10


def map_evidence(pkg: dict, area: str, contributions: np.ndarray, feature_names: list[str],
                 max_items: int = MAX_ITEMS, ranker: str = "tfidf") -> tuple[list[dict], list[dict]]:
    """Return (evidence items, drivers).

    Drivers are the positive-SHAP features, largest first (reported as-is). Evidence is taken in this
    order: (1) positive drivers from a family relevant to the area, by SHAP; (2) the area's other relevant
    features; (3) the remaining positive drivers (at most ``MAX_DRIVERS``); then the fallback item.
    """
    order_all = np.argsort(-contributions)
    positive = [i for i in order_all if contributions[i] > 0]
    drivers = [{"feature": feature_names[i], "impact": round(float(contributions[i]), 4)} for i in positive[:5]]
    relevant = RELEVANT[area]
    is_rel = [family(f) in relevant for f in feature_names]
    order = [i for i in positive if is_rel[i]]
    order += [i for i in order_all if is_rel[i] and i not in order]
    order += [i for i in positive if not is_rel[i]][:MAX_DRIVERS]
    items, seen = [], set()
    for i in order:
        for rec in records_for_feature(pkg, feature_names[i], area, ranker)[:PER_FEATURE]:
            key = (rec["source"], rec["ref"])
            if key not in seen:
                seen.add(key)
                items.append(rec)
            if len(items) >= max_items:
                return items, drivers
    return items or [fallback_evidence(pkg, area)], drivers
