"""Ground-truth evidence for each gap, read back from the generated package.

The rules pick the records that best show the gap (worst first, at most ``max_items``). Every gap gets at
least one item: when no specific record qualifies (for example a label flipped by assessor noise), the
fallback points at the evaluation round or delivery indicators. Items match AssessmentResult.Evidence.
"""

from __future__ import annotations

from smartep.synth import phrases as ph
from smartep.synth.catalogue import COURSES

MAX_ITEMS = 3


def _own(pkg):
    return [les for les in pkg["lessons"] if les["taughtByThisEducator"]]


def _lesson_index(pkg):
    return {les["lessonRef"]: les for les in pkg["lessons"]}


def _rounds(pkg):
    return [r for r in pkg["evaluations"] if r.get("questions")]


def _question(round_, dim):
    return next((q for q in round_["questions"] if q.get("dimension") == dim), None)


def _mean_likert(q) -> float:
    dist = q["distribution"]
    n = sum(dist)
    return sum((i + 1) * c for i, c in enumerate(dist)) / n if n else 3.0


def _comment_quotes(pkg, dims):
    out = []
    for r in _rounds(pkg):
        q = next((q for q in r["questions"] if q["kind"] == "text"), None)
        for c in (q or {}).get("comments", []) or []:
            if any(c in ph.NEGATIVE_COMMENTS[d] for d in dims):
                out.append({"source": "evaluation", "ref": f"{r['roundRef']}:{q['questionRef']}", "quote": c})
    return out


def _likert_evidence(pkg, dim):
    out = []
    for r in _rounds(pkg):
        q = _question(r, dim)
        if q:
            mean = _mean_likert(q)
            out.append((mean, {"source": "evaluation", "ref": f"{r['roundRef']}:{q['questionRef']}",
                               "note": f"Mean {dim} rating {mean:.2f} of 5 ({r['responses']} responses)"}))
    return [e for _, e in sorted(out, key=lambda x: x[0])]


def _uncovered_items(pkg):
    if pkg["plan"] is None:
        return [{"source": "plan", "ref": "plan", "note": "No delivery plan was submitted"}]
    lessons = _lesson_index(pkg)
    out = []
    for item in pkg["plan"]["items"]:
        ok = any(lessons.get(ref, {}).get("deliveredStatus") == "delivered" and lessons[ref].get("notes")
                 for ref in item["lessonRefs"])
        if not ok:
            out.append({"source": "plan", "ref": item["itemRef"],
                        "note": f"No delivered lesson with notes for '{item['topic']}' (week {item['week']})"})
    return out


def _outdated(pkg):
    terms = COURSES[pkg["course"]["title"]]["outdated"]
    out = []
    for les in pkg["lessons"]:
        texts = [("notes", (les.get("notes") or {}).get("text", ""))]
        texts += [("slides", s["text"]) for s in les.get("slides") or []]
        texts += [("document", d.get("extractedText") or d["title"]) for d in les["documents"]]
        for source, text in texts:
            hit = next((term for term in terms if term in text), None)
            if hit:
                out.append({"source": source, "ref": les["lessonRef"], "quote": hit})
                break
    return out


def _missing_notes(pkg):
    if pkg["plan"] is None:
        return []
    lessons = _lesson_index(pkg)
    return [{"source": "notes", "ref": ref, "note": "Required notes were not posted"}
            for item in pkg["plan"]["items"] if item["requiresNotes"]
            for ref in item["lessonRefs"] if ref in lessons and not lessons[ref].get("notes")]


def _video_problems(pkg):
    out = []
    lessons = _lesson_index(pkg)
    for item in (pkg["plan"] or {}).get("items", []):
        if item["requiresVideo"] and not any(lessons.get(r, {}).get("videos") for r in item["lessonRefs"]):
            out.append((0.0, {"source": "plan", "ref": item["itemRef"], "note": "Required video not posted"}))
    for les in pkg["lessons"]:
        for v in les["videos"]:
            view = v["viewing"]
            if view["viewers"] and view["completionRate"] < 0.25:
                at = (v.get("durationSeconds") or 0) * view["medianFurthestPct"] / 100
                out.append((view["completionRate"], {
                    "source": "viewing", "ref": v["videoRef"], "atSeconds": round(at, 1),
                    "note": f"Completion {view['completionRate']:.0%}; half of viewers stop by {view['medianFurthestPct']:.0f}%"}))
    return [e for _, e in sorted(out, key=lambda x: x[0])]


def _undelivered(pkg):
    return [{"source": "delivery", "ref": les["lessonRef"], "note": "Scheduled session was not delivered"}
            for les in _own(pkg) if les["deliveredStatus"] == "not_delivered"]


def _feedback(pkg):
    d, out = pkg["deliveryIndicators"], []
    hours = d.get("gradingTurnaroundMedianHours")
    if hours is not None and hours > 72:
        out.append({"source": "delivery", "ref": "deliveryIndicators.gradingTurnaroundMedianHours",
                    "note": f"Median grading turnaround {hours:.0f} hours"})
    if d["questionsAsked"] and d["questionsAnsweredWithin72h"] / d["questionsAsked"] < 0.6:
        out.append({"source": "delivery", "ref": "deliveryIndicators.questionsAnsweredWithin72h",
                    "note": f"{d['questionsAnsweredWithin72h']} of {d['questionsAsked']} questions answered within 72 hours"})
    return out


def _engagement(pkg):
    out = []
    for les in _own(pkg):
        a = les.get("attendance")
        if a and a["recorded"] and a["present"] / a["recorded"] < 0.6:
            out.append((a["present"] / a["recorded"], {"source": "delivery", "ref": les["lessonRef"],
                                                        "note": f"Attendance {a['present']} of {a['recorded']}"}))
    for r in pkg["evaluations"]:
        if r["enrolled"] and r["responses"] / r["enrolled"] < 0.35:
            out.append((r["responses"] / r["enrolled"], {"source": "evaluation", "ref": r["roundRef"],
                                                          "note": f"{r['responses']} of {r['enrolled']} students responded"}))
    return [e for _, e in sorted(out, key=lambda x: x[0])]


def _hesitation(pkg):
    out = []
    for les in pkg["lessons"]:
        for v in les["videos"]:
            for seg in (v.get("transcript") or {}).get("segments", []):
                if any(m in seg["text"] for m in ph.HESITANT_MARKERS):
                    out.append({"source": "transcript", "ref": v["videoRef"], "atSeconds": seg["start"],
                                "quote": seg["text"]})
                    break
    return out


RULES = {
    "content_coverage": lambda p: _uncovered_items(p),
    "content_currency": lambda p: _outdated(p) + _comment_quotes(p, ["currency"]),
    "resources": lambda p: _missing_notes(p) + _comment_quotes(p, ["resources"]),
    "video_delivery": lambda p: _video_problems(p),
    "session_delivery": lambda p: _undelivered(p) + _comment_quotes(p, ["organisation"]),
    "assessment_feedback": lambda p: _feedback(p) + _comment_quotes(p, ["feedback"]),
    "student_engagement": lambda p: _engagement(p) + _comment_quotes(p, ["engagement"]),
    "pedagogy": lambda p: _comment_quotes(p, ["clarity", "engagement"]) + _likert_evidence(p, "clarity"),
    "technical_skill": lambda p: _hesitation(p) + _comment_quotes(p, ["mastery"]),
}


def _fallback(pkg, area):
    rounds = pkg["evaluations"]
    if rounds:
        r = rounds[-1]
        return {"source": "evaluation", "ref": r["roundRef"], "note": f"Overall evaluation evidence for {area}"}
    return {"source": "delivery", "ref": "deliveryIndicators", "note": f"Delivery indicators for {area}"}


def extract_evidence(pkg: dict, area: str, max_items: int = MAX_ITEMS) -> list[dict]:
    items = RULES[area](pkg)[:max_items]
    return items or [_fallback(pkg, area)]
