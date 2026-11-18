"""AssessmentResult v1.0, posted back to The Classroom (spec v3, section 6)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat

from smartep.contracts.gap_areas import GapArea

Probability = Annotated[float, Field(ge=0, le=1)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelInfo(_Model):
    name: str
    version: str
    trainingData: Literal["synthetic", "synthetic+real"]
    validated: bool


class Evidence(_Model):
    source: Literal["plan", "notes", "slides", "document", "transcript", "viewing", "delivery", "evaluation"]
    ref: str
    quote: str | None = None
    atSeconds: NonNegativeFloat | None = None
    note: str | None = None


class Gap(_Model):
    gapRef: str
    area: GapArea
    title: str
    description: str
    severity: Literal["low", "medium", "high"]
    confidence: Probability
    relatedPlanItems: list[str]
    # The Classroom rejects a gap without evidence (HTTP 422).
    evidence: list[Evidence] = Field(min_length=1)


class SkillFramework(_Model):
    name: str
    code: str


class ResultCost(_Model):
    amount: NonNegativeFloat
    currency: str


class RecommendedCourse(_Model):
    source: Literal["classroom", "institution", "web"]
    catalogueRef: str | None = None
    title: str
    provider: str
    url: str
    isTrustedWebProvider: bool | None = None
    mode: Literal["online", "in_person", "hybrid"]
    location: str | None = None
    durationText: str | None = None
    cost: ResultCost | Literal["free", "unknown"] | None = None
    nextStartDate: str | None = None
    certification: str | None = None
    retrievedAt: str | None = None
    matchScore: float


class Reskilling(_Model):
    recommendationRef: str
    skillArea: str
    skillFramework: SkillFramework | None = None
    rationale: str
    addressesGaps: list[str]
    priority: Literal["low", "medium", "high"]
    courses: list[RecommendedCourse]


class AssessmentResult(_Model):
    schemaVersion: Literal["1.0"]
    packageId: str
    model: ModelInfo
    generatedAt: str
    summary: str
    overallConfidence: Probability
    gaps: list[Gap]
    reskilling: list[Reskilling]
