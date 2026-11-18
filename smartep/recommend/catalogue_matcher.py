"""Match predicted gaps to courses in the package's reskilling catalogue snapshot (MVP recommender).

Score per course = 0.7 × share of the gap's skill tags the course covers + 0.3 × TF-IDF similarity
between the gap description and the course title + description. For a subject-knowledge gap the course's
own skills (from the plan or package course) are added to the tags. Courses over the preferred budget or
outside the preferred modes are dropped. Only catalogueRefs from the snapshot can be returned.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from smartep.contracts.area_text import AREA_DESCRIPTIONS, AREA_SKILLS

SUBJECT_AREAS = ("technical_skill", "content_currency")
TOP_K = 3


def gap_skills(pkg: dict, area: str) -> list[str]:
    skills = list(AREA_SKILLS[area])
    if area in SUBJECT_AREAS:
        plan_skills = {s for it in (pkg["plan"] or {}).get("items", []) for s in it["skills"]}
        skills += sorted(plan_skills) or [pkg["course"]["title"].lower()]
    return skills


def _affordable(entry: dict, prefs: dict) -> bool:
    if prefs.get("preferredModes") and entry["mode"] not in prefs["preferredModes"]:
        return False
    cost, cap = entry["cost"], prefs.get("maxCostKes")
    return cost == "free" or cap is None or cost["amount"] <= cap


def match_courses(pkg: dict, area: str, top_k: int = TOP_K) -> list[dict]:
    catalogue = [e for e in pkg["reskillingCatalogue"] if _affordable(e, pkg["reskillingPreferences"])]
    if not catalogue:
        return []
    skills = set(gap_skills(pkg, area))
    docs = [f"{e['title']}. {e['description']} {' '.join(e['skills'])}" for e in catalogue]
    query = AREA_DESCRIPTIONS[area] + " " + " ".join(sorted(skills))
    vec = TfidfVectorizer(sublinear_tf=True).fit([*docs, query])
    sims = cosine_similarity(vec.transform(docs), vec.transform([query]))[:, 0]
    scored = []
    for entry, sim in zip(catalogue, sims, strict=True):
        overlap = len(skills & set(entry["skills"])) / len(skills)
        score = 0.7 * overlap + 0.3 * float(sim)
        if overlap > 0:
            scored.append((score, entry))
    scored.sort(key=lambda x: -x[0])
    return [{
        "source": e["source"], "catalogueRef": e["catalogueRef"], "title": e["title"], "provider": e["provider"],
        "url": e["url"], "mode": e["mode"], "durationText": e.get("durationText"),
        "cost": e["cost"] if e["cost"] == "free" else {"amount": e["cost"]["amount"], "currency": e["cost"]["currency"]},
        "nextStartDate": e.get("nextStartDate"), "certification": e.get("certification"),
        "matchScore": round(score, 4),
    } for score, e in scored[:top_k]]
