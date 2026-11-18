import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.export_schemas import SCHEMA_DIR, SCHEMAS, render
from smartep.contracts import GAP_AREAS, AssessmentResult, EvidencePackage
from smartep.contracts.ids import pseudonymize

FIXTURE = Path(__file__).parent / "fixtures" / "evidence_package_example.json"


def _result(**gap_overrides) -> dict:
    gap = {
        "gapRef": "gap_1", "area": "assessment_feedback", "title": "Slow grading",
        "description": "Median grading turnaround is 96 hours.", "severity": "medium", "confidence": 0.72,
        "relatedPlanItems": ["pi_1"], "evidence": [{"source": "delivery", "ref": "gradingTurnaroundMedianHours"}],
    }
    gap.update(gap_overrides)
    return {
        "schemaVersion": "1.0", "packageId": "pkg_0001",
        "model": {"name": "smartep-xgb", "version": "0.0.1", "trainingData": "synthetic", "validated": False},
        "generatedAt": "2026-11-18T10:00:00Z", "summary": "One gap found.", "overallConfidence": 0.7,
        "gaps": [gap], "reskilling": [],
    }


def test_example_package_validates():
    pkg = EvidencePackage.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    assert pkg.lessons[0].videos[0].viewing.completionRate == 0.4


def test_package_rejects_unknown_fields():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["unexpected"] = 1
    with pytest.raises(ValidationError):
        EvidencePackage.model_validate(data)


def test_result_requires_evidence_and_known_area():
    AssessmentResult.model_validate(_result())
    with pytest.raises(ValidationError):
        AssessmentResult.model_validate(_result(evidence=[]))
    with pytest.raises(ValidationError):
        AssessmentResult.model_validate(_result(area="AI"))


def test_nine_gap_areas():
    assert len(GAP_AREAS) == 9 and "pedagogy" in GAP_AREAS


@pytest.mark.parametrize("name", list(SCHEMAS))
def test_committed_schemas_match_models(name):
    committed = (SCHEMA_DIR / name).read_text(encoding="utf-8")
    assert committed == render(SCHEMAS[name]), "run: python scripts/export_schemas.py"


def test_pseudonymize_needs_a_salt(monkeypatch):
    monkeypatch.delenv("SMARTEP_ID_SALT", raising=False)
    with pytest.raises(RuntimeError):
        pseudonymize("EDU0001")
    monkeypatch.setenv("SMARTEP_ID_SALT", "s")
    assert pseudonymize("EDU0001") == pseudonymize("EDU0001", salt="s") != pseudonymize("EDU0001", salt="t")
