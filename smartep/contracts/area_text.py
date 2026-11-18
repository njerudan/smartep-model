"""Plain-language text for each gap area: titles for results, and a description used to rank text
evidence and catalogue courses by similarity. Written from the spec's area meanings, independently of
the synthetic generator's phrase banks.
"""

AREA_TITLES = {
    "content_coverage": "Planned topics not covered",
    "content_currency": "Outdated course content",
    "resources": "Missing learning resources",
    "video_delivery": "Weak video delivery",
    "session_delivery": "Sessions not delivered as planned",
    "assessment_feedback": "Slow assessment and feedback",
    "student_engagement": "Low student engagement",
    "pedagogy": "Teaching approach needs work",
    "technical_skill": "Subject knowledge needs strengthening",
}

# Skill tags that address each gap area (catalogue entries carry the same tags).
AREA_SKILLS: dict[str, list[str]] = {
    "content_coverage": ["curriculum planning", "course design"],
    "content_currency": ["industry practice", "course design"],
    "resources": ["learning materials", "digital tools"],
    "video_delivery": ["video production", "digital tools"],
    "session_delivery": ["time management", "curriculum planning"],
    "assessment_feedback": ["assessment", "feedback"],
    "student_engagement": ["active learning", "student engagement"],
    "pedagogy": ["pedagogy", "active learning"],
    "technical_skill": ["subject expertise"],
}

AREA_DESCRIPTIONS = {
    "content_coverage": "planned topics skipped or not taught; syllabus behind the course outline; topics missing",
    "content_currency": "old or outdated tools, examples and materials; not current with industry practice",
    "resources": "notes, slides or handouts missing or not posted; few learning materials for students",
    "video_delivery": "videos missing, not watched to the end, poorly hosted or without transcripts",
    "session_delivery": "classes cancelled, missed or moved; sessions not delivered on schedule",
    "assessment_feedback": "assignments marked late; no comments on work; student questions not answered",
    "student_engagement": "students absent, bored or not taking part; few responses and little discussion",
    "pedagogy": "explanations unclear; lecturer reads from slides; no examples or active learning",
    "technical_skill": "lecturer unsure of the subject; errors in explanations; weak command of the topic",
}
