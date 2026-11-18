"""Evidence package v1.0, as The Classroom sends it (spec v3, section 3.2).

Field names keep the camelCase of the wire format so a package validates as-is.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassroomInfo(_Model):
    orgRef: str
    baseUrl: str
    resultsCallbackUrl: str


class Institution(_Model):
    name: str
    country: Literal["KE"]
    type: Literal["university", "tvet", "college"]


class Period(_Model):
    name: str
    startsAt: str
    endsAt: str


class Educator(_Model):
    educatorRef: str
    discipline: str | None = None
    seniority: Literal["assistant_lecturer", "lecturer", "senior_lecturer", "professor"] | None = None


class Course(_Model):
    courseRef: str
    title: str
    level: str | None = None
    description: str
    enrolledStudents: NonNegativeInt


class PlanItem(_Model):
    itemRef: str
    week: int
    topic: str
    learningOutcomes: list[str]
    skills: list[str]
    plannedSessions: NonNegativeInt
    requiresNotes: bool
    requiresVideo: bool
    lessonRefs: list[str]


class Plan(_Model):
    status: Literal["approved", "submitted", "draft"]
    plannedContactHours: NonNegativeFloat | None = None
    items: list[PlanItem]


class Notes(_Model):
    text: str
    firstPostedAt: str
    lastEditedAt: str


class Slides(_Model):
    text: str
    firstPostedAt: str


class Document(_Model):
    title: str
    mimeType: str
    extractedText: str | None = None
    postedAt: str


class TranscriptSegment(_Model):
    start: NonNegativeFloat
    end: NonNegativeFloat
    text: str


class Transcript(_Model):
    language: str
    segments: list[TranscriptSegment]


class Viewing(_Model):
    viewers: NonNegativeInt
    medianFurthestPct: float = Field(ge=0, le=100)
    completionRate: float = Field(ge=0, le=1)
    furthestPctHistogram: list[NonNegativeInt]


class Video(_Model):
    videoRef: str
    title: str
    provider: Literal["upload", "youtube", "vimeo", "google_drive", "generic"]
    durationSeconds: NonNegativeFloat | None = None
    postedAt: str
    transcript: Transcript | None = None
    viewing: Viewing


class Attendance(_Model):
    recorded: NonNegativeInt
    present: NonNegativeInt


class Lesson(_Model):
    lessonRef: str
    title: str
    taughtByThisEducator: bool
    scheduledAt: str | None = None
    durationMinutes: NonNegativeFloat | None = None
    deliveredStatus: Literal["delivered", "not_delivered", "unknown"]
    notes: Notes | None = None
    slides: list[Slides] | None = None
    documents: list[Document]
    videos: list[Video]
    attendance: Attendance | None = None


class DeliveryIndicators(_Model):
    sessionsPlanned: NonNegativeInt
    sessionsDelivered: NonNegativeInt
    contactMinutesPlanned: NonNegativeFloat | None = None
    contactMinutesDelivered: NonNegativeFloat
    assignmentsGraded: NonNegativeInt
    gradingTurnaroundMedianHours: NonNegativeFloat | None = None
    gradingTurnaroundApproximate: bool
    questionsAsked: NonNegativeInt
    questionsAnsweredWithin72h: NonNegativeInt


class EvaluationQuestion(_Model):
    questionRef: str
    prompt: str
    dimension: str | None = None
    kind: Literal["likert5", "yes_no", "text"]
    distribution: list[NonNegativeInt] | None = None
    comments: list[str] | None = None


class EvaluationRound(_Model):
    roundRef: str
    closedAt: str
    responses: NonNegativeInt
    enrolled: NonNegativeInt
    belowThreshold: bool
    questions: list[EvaluationQuestion] | None = None


class Cost(_Model):
    amount: NonNegativeFloat
    currency: Literal["KES"]


class CatalogueEntry(_Model):
    catalogueRef: str
    source: Literal["classroom", "institution"]
    title: str
    provider: str
    description: str
    skills: list[str]
    level: Literal["beginner", "intermediate", "advanced"] | None = None
    mode: Literal["online", "in_person", "hybrid"]
    durationText: str | None = None
    cost: Cost | Literal["free"]
    nextStartDate: str | None = None
    certification: str | None = None
    url: str


class TrustedWebProvider(_Model):
    name: str
    domain: str


class ReskillingPreferences(_Model):
    trustedWebProviders: list[TrustedWebProvider]
    preferredModes: list[Literal["online", "in_person", "hybrid"]]
    maxCostKes: NonNegativeFloat | None = None
    location: str | None = None
    webResultsPerSkill: NonNegativeInt = 3


class EvidencePackage(_Model):
    schemaVersion: Literal["1.0"]
    packageId: str
    trigger: Literal["evaluation_round_closed", "period_end", "manual"]
    createdAt: str
    classroom: ClassroomInfo
    institution: Institution
    period: Period
    educator: Educator
    course: Course
    plan: Plan | None
    lessons: list[Lesson]
    deliveryIndicators: DeliveryIndicators
    evaluations: list[EvaluationRound]
    reskillingCatalogue: list[CatalogueEntry]
    reskillingPreferences: ReskillingPreferences
    previousPackages: list[str]
