"""Evidence package in, AssessmentResult out, using a saved gap-model bundle.

The same feature code as training is used (``package_feature_frame`` + the bundle's fitted text
featurizer), so there is no training/serving skew. Every result is validated against the contract
before it is returned.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from smartep.contracts import AssessmentResult
from smartep.contracts.area_text import AREA_SKILLS, AREA_TITLES
from smartep.explain.evidence_mapper import map_evidence
from smartep.features.package import package_feature_frame
from smartep.recommend.catalogue_matcher import match_courses
from smartep.registry.bundle import load_bundle
from smartep.training.gap_model import FittedGapModel

PRIORITY = {"high": "high", "medium": "medium", "low": "low"}


def _readable(feature: str) -> str:
    return feature.replace("text_", "text: ").replace("likert_", "rating: ").replace("_", " ")


class GapPredictor:
    def __init__(self, model: FittedGapModel, model_name: str, version: str, training_data: str, validated: bool):
        self.model, self.name, self.version = model, model_name, version
        self.training_data, self.validated = training_data, validated

    @classmethod
    def from_bundle(cls, path: Path) -> GapPredictor:
        b = load_bundle(Path(path))
        p = b.preprocess
        if p.get("kind") != "gap_model_v0":
            raise ValueError(f"{path} is not a gap-model bundle")
        model = FittedGapModel(p["labels"], p["struct_names"], p["text"], b.xgb, p["calibrator"],
                               p["thresholds"], p["severity_cuts"], p["feature_names"])
        m = b.manifest
        return cls(model, m["model_name"], m["version"], m["training_data"], m["validated"])

    def _features(self, packages: list[dict]) -> np.ndarray:
        struct = package_feature_frame(packages).reindex(columns=self.model.struct_names)
        return self.model.features(struct.to_numpy(dtype=np.float32), packages)

    def predict_packages(self, packages: list[dict]) -> list[AssessmentResult]:
        X = self._features(packages)
        _, P_cal = self.model.predict(X)
        pred, sev = self.model.decide(P_cal)
        contribs = {lab: self.model.xgb.contributions(X, lab) for lab in self.model.labels}
        return [self._result(pkg, P_cal[i], pred[i], sev[i], {k: v[i] for k, v in contribs.items()})
                for i, pkg in enumerate(packages)]

    def predict(self, package: dict) -> AssessmentResult:
        return self.predict_packages([package])[0]

    def _result(self, pkg, probs, pred, sev, contribs) -> AssessmentResult:
        plan_refs = {it["itemRef"]: set(it["lessonRefs"]) for it in (pkg["plan"] or {}).get("items", [])}
        gaps, reskilling = [], []
        for j, area in enumerate(self.model.labels):
            if not pred[j]:
                continue
            evidence, drivers = map_evidence(pkg, area, contribs[area], self.model.feature_names, ranker=self.model.ranker)
            refs = {e["ref"] for e in evidence}
            related = sorted(item for item, lessons in plan_refs.items() if item in refs or lessons & refs)
            signals = ", ".join(_readable(d["feature"]) for d in drivers[:3]) or "overall pattern of evidence"
            gaps.append({
                "gapRef": f"gap_{area}", "area": area, "title": AREA_TITLES[area],
                "description": f"Predicted from: {signals}.", "severity": sev[j],
                "confidence": round(float(probs[j]), 3), "relatedPlanItems": related, "evidence": evidence,
            })
            courses = match_courses(pkg, area)
            if courses:
                reskilling.append({
                    "recommendationRef": f"rec_{area}", "skillArea": AREA_SKILLS[area][0].title(),
                    "rationale": f"Addresses '{AREA_TITLES[area]}' ({sev[j]} severity).",
                    "addressesGaps": [f"gap_{area}"], "priority": PRIORITY[sev[j]], "courses": courses,
                })
        gaps.sort(key=lambda g: -g["confidence"])
        if gaps:
            summary = f"{len(gaps)} possible gap(s): " + "; ".join(f"{g['title']} ({g['severity']})" for g in gaps) + "."
            overall = float(np.mean([g["confidence"] for g in gaps]))
        else:
            summary = "No gaps predicted above the model's thresholds."
            overall = float(1 - probs.max())
        result = {
            "schemaVersion": "1.0", "packageId": pkg["packageId"],
            "model": {"name": self.name, "version": self.version, "trainingData": self.training_data,
                      "validated": self.validated},
            "generatedAt": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "summary": summary, "overallConfidence": round(overall, 3), "gaps": gaps, "reskilling": reskilling,
        }
        try:
            return AssessmentResult.model_validate(result)
        except ValidationError as exc:  # never send an invalid result
            raise RuntimeError(f"invalid AssessmentResult for {pkg['packageId']}: {exc}") from exc
