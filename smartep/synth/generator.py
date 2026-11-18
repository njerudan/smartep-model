"""Generator v2: synthetic evidence packages (schema v1.0) with a planted, documented signal.

Each educator has latent traits (``traits.py``). Every package field that depends on a trait does so
through a rule in ``build_package`` (all rules are listed in docs/data-sheet-synthetic-v2.md). Gap labels
are computed afterwards from the same period traits (``labels.py``), and ground-truth evidence is read
back from the generated package (``evidence.py``). The model only ever sees the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from smartep.contracts.gap_areas import GAP_AREAS
from smartep.synth import phrases as ph
from smartep.synth.catalogue import COURSES, RESKILLING_CATALOGUE
from smartep.synth.evidence import extract_evidence
from smartep.synth.labels import assign_labels, deficiency_scores
from smartep.synth.traits import TRAITS, as_dict, period_traits, sample_base_traits

GENERATOR_VERSION = "2.0.0"
SENIORITY = ("assistant_lecturer", "lecturer", "senior_lecturer", "professor")
SENIORITY_P = (0.3, 0.4, 0.2, 0.1)
INSTITUTION_TYPES = ("university", "tvet", "college")
INSTITUTION_TYPES_P = (0.6, 0.25, 0.15)
LEVELS = {"university": "Undergraduate", "tvet": "Diploma", "college": "Certificate"}
LIKERT_DIMS = ("clarity", "organisation", "feedback", "engagement", "resources")
# Student-perceived quality per dimension (higher = better).
QUALITY_WEIGHTS = {
    "clarity": {"pedagogical_skill": 0.6, "content_mastery": 0.5},
    "organisation": {"pedagogical_skill": 0.4, "workload_pressure": -0.6},
    "feedback": {"responsiveness": 0.8, "workload_pressure": -0.2},
    "engagement": {"pedagogical_skill": 0.7, "responsiveness": 0.2, "digital_skill": 0.2},
    "resources": {"digital_skill": 0.5, "currency": 0.3, "pedagogical_skill": 0.2},
    "currency": {"currency": 1.0},
    "mastery": {"content_mastery": 1.0},
}
LIKERT_CUTS = np.array([-1.8, -0.7, 0.3, 1.4])
HIST_BINS = 10


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _quality(t: dict, dim: str, rng) -> float:
    return sum(w * t[k] for k, w in QUALITY_WEIGHTS[dim].items()) + rng.normal(0, 0.2)


@dataclass
class SyntheticDataset:
    packages: list[dict]
    index: pd.DataFrame      # one row per package: ids, period, course, group attributes, subset
    labels: pd.DataFrame     # packageId + one 0/1 column per gap area
    severity: pd.DataFrame   # packageId + one column per gap area ('' when no gap)
    evidence: list[dict]     # per package: ground-truth evidence for each gap
    traits: pd.DataFrame     # packageId + period traits (for the oracle check only)
    config: dict


# ----------------------------------------------------------------------------------------------------------
# One package
# ----------------------------------------------------------------------------------------------------------

def _viewing(enrolled: int, t: dict, rng) -> dict:
    viewers = int(rng.binomial(enrolled, sig(0.3 + 0.5 * t["pedagogical_skill"])))
    if viewers == 0:
        return {"viewers": 0, "medianFurthestPct": 0.0, "completionRate": 0.0, "furthestPctHistogram": [0] * HIST_BINS}
    mean = sig(-0.2 + 0.7 * t["pedagogical_skill"] + 0.6 * t["digital_skill"] + rng.normal(0, 0.3))
    conc = 4.0
    furthest = rng.beta(mean * conc, (1 - mean) * conc, size=viewers)
    hist = np.histogram(furthest, bins=HIST_BINS, range=(0, 1))[0]
    return {"viewers": viewers, "medianFurthestPct": round(float(np.median(furthest) * 100), 1),
            "completionRate": round(float((furthest >= 0.9).mean()), 3), "furthestPctHistogram": hist.tolist()}


def _transcript(topic: str, t: dict, rng) -> dict:
    segments, start = [], 0.0
    for _ in range(int(rng.integers(4, 7))):
        if rng.random() < sig(-1.0 - 1.4 * t["content_mastery"]):
            text = rng.choice(ph.TRANSCRIPT_HESITANT)
        elif rng.random() < sig(-1.2 - 1.0 * t["pedagogical_skill"]):
            text = rng.choice(ph.TRANSCRIPT_SLIDE_READING)
        else:
            text = rng.choice(ph.TRANSCRIPT_CLEAR)
        length = float(rng.uniform(40, 160))
        segments.append({"start": round(start, 1), "end": round(start + length, 1),
                         "text": text.format(topic=topic, topic_l=topic.lower())})
        start += length + float(rng.uniform(5, 60))
    return {"language": "en", "segments": segments}


def _term(course: dict, t: dict, rng, base: float) -> tuple[str, bool]:
    """A technology term for examples: outdated with probability rising as currency falls."""
    outdated = rng.random() < sig(base - 1.6 * t["currency"])
    return str(rng.choice(course["outdated"] if outdated else course["current"])), outdated


def _lesson(item: dict | None, week: int, session: int, topic: str, difficulty: float, covered: bool,
            course: dict, t: dict, enrolled: int, scheduled: datetime, year: int, rng) -> dict:
    w, ped, dig = t["workload_pressure"], t["pedagogical_skill"], t["digital_skill"]
    if covered:
        delivered = rng.random() < sig(2.6 - 1.4 * w)
    else:
        delivered = rng.random() < 0.3
    status = "delivered" if delivered else "not_delivered"
    if rng.random() < 0.02:
        status = "unknown"
    lesson = {
        "lessonRef": f"les_{week:02d}{session}", "title": f"{topic} ({'part ' + str(session + 1)})",
        "taughtByThisEducator": bool(rng.random() > 0.04), "scheduledAt": _iso(scheduled),
        "durationMinutes": 120, "deliveredStatus": status, "documents": [], "videos": [],
    }
    fmt = {"topic": topic, "topic_l": topic.lower()}

    if covered and rng.random() < sig(1.2 + 0.9 * ped - 0.6 * w):
        strong = rng.random() < sig(0.3 + 1.0 * t["content_mastery"] + 0.6 * ped)
        bank = ph.NOTES_STRONG if strong else ph.NOTES_WEAK
        sentences = [s.format(**fmt) for s in rng.choice(bank, size=int(rng.integers(2, 4)), replace=False)]
        term, outdated = _term(course, t, rng, base=-1.2)
        sentences.append((ph.NOTES_OUTDATED if outdated else ph.NOTES_CURRENT).format(term=term))
        late = rng.random() < sig(-1.0 + 1.3 * w)
        posted = scheduled + timedelta(days=float(rng.uniform(1, 10))) if late else \
            scheduled - timedelta(days=float(rng.uniform(0.5, 3)))
        lesson["notes"] = {"text": " ".join(sentences), "firstPostedAt": _iso(posted),
                           "lastEditedAt": _iso(posted + timedelta(days=float(rng.uniform(0, 5))))}

    if covered and rng.random() < sig(0.3 + 1.0 * dig + 0.3 * ped):
        term, outdated = _term(course, t, rng, base=-1.4)
        text = str(rng.choice(ph.SLIDES)).format(**fmt) + f"; tools: {term}"
        lesson["slides"] = [{"text": text, "firstPostedAt": _iso(scheduled - timedelta(days=1))}]

    for _ in range(min(2, int(rng.poisson(1.2 * sig(0.2 + 0.8 * ped - 0.4 * w))))):
        term, outdated = _term(course, t, rng, base=-1.0)
        doc_year = int(rng.integers(2009, 2016)) if outdated else year
        doc = {"title": f"{topic} handout ({doc_year})", "mimeType": "application/pdf",
               "postedAt": _iso(scheduled - timedelta(days=float(rng.uniform(0, 4))))}
        if rng.random() < 0.7:
            doc["extractedText"] = f"{topic}. Reference material, {doc_year} edition. Worked examples with {term}."
        lesson["documents"].append(doc)

    wants_video = (item is not None and item["requiresVideo"] and session == 0)
    extra_video = rng.random() < sig(-2.0 + 0.8 * dig)
    if covered and (wants_video and rng.random() < sig(0.4 + 1.3 * dig) or extra_video):
        good_host = rng.random() < sig(0.5 + 1.0 * dig)
        provider = str(rng.choice(["upload", "youtube"] if good_host else ["vimeo", "google_drive", "generic"]))
        video = {"videoRef": f"vid_{week:02d}{session}", "title": f"{topic} walkthrough", "provider": provider,
                 "durationSeconds": float(rng.integers(600, 3000)),
                 "postedAt": _iso(scheduled + timedelta(hours=float(rng.uniform(2, 72)))),
                 "viewing": _viewing(enrolled, t, rng)}
        if provider in ("upload", "youtube") and rng.random() < 0.9:
            video["transcript"] = _transcript(topic, t, rng)
        lesson["videos"].append(video)

    if status == "delivered":
        recorded = int(max(1, enrolled - rng.integers(0, 4)))
        rate = sig(1.0 + 0.6 * ped + 0.3 * t["responsiveness"] - 0.3 * w + rng.normal(0, 0.3))
        lesson["attendance"] = {"recorded": recorded, "present": int(rng.binomial(recorded, rate))}
    return lesson


def _evaluation_round(ref: str, closed: date, enrolled: int, t: dict, rng) -> dict:
    rate = sig(-0.4 + 0.5 * t["pedagogical_skill"] + 0.3 * t["responsiveness"] + rng.normal(0, 0.2))
    responses = int(rng.binomial(enrolled, rate))
    round_ = {"roundRef": ref, "closedAt": closed.isoformat(), "responses": responses, "enrolled": enrolled,
              "belowThreshold": responses < 5}
    if round_["belowThreshold"]:
        return round_  # questions withheld to protect anonymity
    questions, quality = [], {d: _quality(t, d, rng) for d in QUALITY_WEIGHTS}
    for i, dim in enumerate(LIKERT_DIMS, start=1):
        latent = 0.6 + quality[dim] + rng.logistic(0, 1, size=responses)
        counts = np.bincount(np.searchsorted(LIKERT_CUTS, latent), minlength=5)
        questions.append({"questionRef": f"q{i}", "prompt": f"Rate the lecturer on {dim}", "dimension": dim,
                          "kind": "likert5", "distribution": counts.tolist()})
    overall = 0.6 + np.mean([quality[d] for d in LIKERT_DIMS]) + rng.logistic(0, 1, size=responses)
    questions.append({"questionRef": "q6", "prompt": "Overall rating", "dimension": "overall", "kind": "likert5",
                      "distribution": np.bincount(np.searchsorted(LIKERT_CUTS, overall), minlength=5).tolist()})
    comments = []
    dims = list(ph.COMMENTS)
    for _ in range(int(rng.binomial(responses, 0.3))):
        dim = dims[int(rng.integers(len(dims)))]
        positive, negative = ph.COMMENTS[dim]
        bank = negative if rng.random() < sig(-0.7 - 1.2 * quality[dim]) else positive
        comments.append(str(rng.choice(bank)))
    questions.append({"questionRef": "q7", "prompt": "Any other comments?", "kind": "text", "comments": comments})
    round_["questions"] = questions
    return round_


def build_package(*, rng, t: dict, package_id: str, educator: dict, institution: dict, course_name: str,
                  period: dict, previous: list[str], cfg: dict) -> dict:
    course = COURSES[course_name]
    w = t["workload_pressure"]
    start = datetime.fromisoformat(period["start"])
    end = start + timedelta(weeks=cfg["weeks_per_period"] + 2)
    year = start.year
    enrolled = int(rng.integers(20, 121))
    n_weeks, n_sessions = cfg["weeks_per_period"], cfg["sessions_per_week"]

    has_plan = rng.random() > sig(-3.0 + 1.5 * w)
    status = str(rng.choice(["approved", "submitted", "draft"], p=[0.8, 0.12, 0.08])) if has_plan else None
    items, lessons = [], []
    for week in range(1, n_weeks + 1):
        topic = course["topics"][(week - 1) % len(course["topics"])]
        difficulty = -1.0 + 2.0 * (week - 1) / max(1, n_weeks - 1)
        covered = rng.random() < sig(2.0 + 1.5 * (t["content_mastery"] - difficulty) - 0.8 * w)
        item = None
        if has_plan:
            item = {"itemRef": f"pi_{week:02d}", "week": week, "topic": topic,
                    "learningOutcomes": [f"Explain {topic.lower()}", f"Apply {topic.lower()} in a practical task"],
                    "skills": list(course["skills"]), "plannedSessions": n_sessions, "requiresNotes": True,
                    "requiresVideo": bool(rng.random() < 0.5),
                    "lessonRefs": [f"les_{week:02d}{s}" for s in range(n_sessions)]}
            items.append(item)
        for s in range(n_sessions):
            scheduled = start + timedelta(weeks=week - 1, days=2 * s, hours=8)
            lessons.append(_lesson(item, week, s, topic, difficulty, covered, course, t, enrolled, scheduled, year, rng))

    own = [les for les in lessons if les["taughtByThisEducator"]]
    delivered = [les for les in own if les["deliveredStatus"] == "delivered"]
    graded = int(rng.poisson(1 + 4 * sig(t["responsiveness"] - 0.3 * w)))
    asked = int(rng.poisson(enrolled * 0.25 * sig(0.3 + 0.4 * t["pedagogical_skill"])) + 1)
    turnaround = float(np.exp(rng.normal(np.log(60) - 0.45 * t["responsiveness"] + 0.35 * w, 0.3)))
    indicators = {
        "sessionsPlanned": len(own), "sessionsDelivered": len(delivered),
        "contactMinutesPlanned": float(120 * len(own)),
        "contactMinutesDelivered": float(sum(120 * float(np.clip(rng.normal(0.92 - 0.05 * w, 0.05), 0.5, 1.0))
                                             for _ in delivered)),
        "assignmentsGraded": graded,
        "gradingTurnaroundMedianHours": round(turnaround, 1) if graded else None,
        "gradingTurnaroundApproximate": bool(rng.random() < 0.5),
        "questionsAsked": asked,
        "questionsAnsweredWithin72h": int(rng.binomial(asked, sig(0.6 + 1.3 * t["responsiveness"] - 0.5 * w))),
    }
    evaluations = [
        _evaluation_round("ev_mid", (start + timedelta(weeks=n_weeks // 2)).date(), enrolled, t, rng),
        _evaluation_round("ev_end", (start + timedelta(weeks=n_weeks)).date(), enrolled, t, rng),
    ]
    return {
        "schemaVersion": "1.0", "packageId": package_id,
        "trigger": str(rng.choice(["period_end", "evaluation_round_closed"], p=[0.7, 0.3])),
        "createdAt": _iso(end + timedelta(days=1)),
        "classroom": {"orgRef": institution["orgRef"], "baseUrl": "https://app.example.test",
                      "resultsCallbackUrl": "https://app.example.test/v1/assessment/results"},
        "institution": {"name": institution["name"], "country": "KE", "type": institution["type"]},
        "period": {"name": period["name"], "startsAt": start.date().isoformat(), "endsAt": end.date().isoformat()},
        "educator": {"educatorRef": educator["educatorRef"], "discipline": course["discipline"],
                     "seniority": educator["seniority"]},
        "course": {"courseRef": f"crs_{course_name.lower().replace(' ', '_')[:20]}", "title": course_name,
                   "level": LEVELS[institution["type"]], "description": f"{course_name}: " + ", ".join(course["topics"][:4]) + ".",
                   "enrolledStudents": enrolled},
        "plan": {"status": status, "plannedContactHours": float(2 * n_sessions * n_weeks), "items": items} if has_plan else None,
        "lessons": lessons,
        "deliveryIndicators": indicators,
        "evaluations": evaluations,
        "reskillingCatalogue": RESKILLING_CATALOGUE,
        "reskillingPreferences": {"trustedWebProviders": [{"name": "Example Academy", "domain": "academy.example.test"}],
                                  "preferredModes": ["online", "hybrid"], "maxCostKes": 20000.0,
                                  "location": "Nairobi", "webResultsPerSkill": 3},
        "previousPackages": list(previous),
    }


# ----------------------------------------------------------------------------------------------------------
# Whole dataset
# ----------------------------------------------------------------------------------------------------------

def _apply_bias(T: np.ndarray, seniority: str, bias: dict) -> np.ndarray:
    """Optional, off by default: shift digital_skill by seniority to test that the fairness audit notices."""
    effect = float(bias.get("seniority_digital_effect", 0.0))
    if effect:
        T = T.copy()
        T[TRAITS.index("digital_skill")] += effect * (SENIORITY.index(seniority) - 1.5) / 1.5
    return T


def generate_dataset(cfg: dict) -> SyntheticDataset:
    rng = np.random.default_rng(int(cfg["seed"]))
    n_edu, periods = int(cfg["n_educators"]), cfg["periods"]
    institutions = [
        {"orgRef": f"org_{i:02d}", "name": f"Synthetic Institution {i:02d}",
         "type": str(rng.choice(INSTITUTION_TYPES, p=INSTITUTION_TYPES_P))}
        for i in range(int(cfg["n_institutions"]))
    ]
    pairs = cfg.get("trait_correlations")
    if pairs is not None:  # YAML form: [{a: trait, b: trait, r: 0.3}, ...]
        pairs = {(p["a"], p["b"]): float(p["r"]) for p in pairs}
    base = sample_base_traits(n_edu, rng, pairs)
    course_names = list(COURSES)

    packages, index_rows, trait_rows = [], [], []
    for e in range(n_edu):
        educator = {"educatorRef": f"edu_{e:05d}", "seniority": str(rng.choice(SENIORITY, p=SENIORITY_P))}
        institution = institutions[int(rng.integers(len(institutions)))]
        primary = course_names[int(rng.integers(len(course_names)))]
        previous: list[str] = []
        for p, period in enumerate(periods):
            T = _apply_bias(period_traits(base[e], rng, cfg["period_drift_sd"]), educator["seniority"], cfg.get("bias", {}))
            course_name = primary if rng.random() < 0.6 else course_names[int(rng.integers(len(course_names)))]
            pid = f"pkg_{e:05d}_{p}"
            # What the package shows is the period traits plus observation noise (a hard semester, a
            # large class, a co-teacher). Labels use the period traits without this noise.
            T_obs = T + rng.normal(0.0, cfg.get("observation_noise_sd", 0.0), size=T.shape)
            pkg = build_package(rng=rng, t=as_dict(T_obs), package_id=pid, educator=educator, institution=institution,
                                course_name=course_name, period=period, previous=previous, cfg=cfg)
            packages.append(pkg)
            previous = [*previous, pid]
            index_rows.append({"packageId": pid, "educatorRef": educator["educatorRef"], "period": period["name"],
                               "course": course_name, "seniority": educator["seniority"],
                               "discipline": COURSES[course_name]["discipline"],
                               "institution": institution["orgRef"], "institution_type": institution["type"]})
            trait_rows.append(T)

    T_all = np.vstack(trait_rows)
    scores = deficiency_scores(T_all, rng, cfg["label_noise_sd"])
    Y, severity, _ = assign_labels(scores, cfg["prevalence"], cfg["label_flip_rate"], rng)

    index = pd.DataFrame(index_rows)
    index["subset"] = _classroom_subset(index, int(cfg["classroom_test_packages"]), rng)
    ids = index["packageId"]
    labels = pd.DataFrame(Y, columns=list(GAP_AREAS)).assign(packageId=ids)[["packageId", *GAP_AREAS]]
    sev = pd.DataFrame(severity, columns=list(GAP_AREAS)).assign(packageId=ids)[["packageId", *GAP_AREAS]]
    traits = pd.DataFrame(T_all, columns=list(TRAITS)).assign(packageId=ids)[["packageId", *TRAITS]]
    evidence = []
    for i, pkg in enumerate(packages):
        gaps = [{"area": area, "severity": severity[i, j], "evidence": extract_evidence(pkg, area)}
                for j, area in enumerate(GAP_AREAS) if Y[i, j]]
        evidence.append({"packageId": pkg["packageId"], "gaps": gaps})
    return SyntheticDataset(packages, index, labels, sev, evidence, traits, cfg)


def _classroom_subset(index: pd.DataFrame, n_packages: int, rng) -> np.ndarray:
    """Hold out whole educators until at least ``n_packages`` packages are set aside for The Classroom."""
    educators = rng.permutation(index["educatorRef"].unique())
    per = index.groupby("educatorRef").size()
    chosen, total = [], 0
    for e in educators:
        if total >= n_packages:
            break
        chosen.append(e)
        total += int(per[e])
    return np.where(index["educatorRef"].isin(chosen), "classroom_test", "train_pool")
