"""Data contracts shared with The Classroom (educator-assessment spec v3, sections 3.2 and 6)."""

from smartep.contracts.assessment_result import AssessmentResult
from smartep.contracts.evidence_package import EvidencePackage
from smartep.contracts.gap_areas import GAP_AREAS

__all__ = ["AssessmentResult", "EvidencePackage", "GAP_AREAS"]
