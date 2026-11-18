"""Phrase banks for notes, slides, transcripts and student comments.

Banks are split by quality so that text carries the latent traits: strong vs weak explanations (content
mastery, pedagogy), hesitation and slide-reading in transcripts (mastery, pedagogy), and comment tone per
evaluation dimension.
"""

from __future__ import annotations

NOTES_STRONG = [
    "We define {topic_l} precisely and then work through a worked example step by step.",
    "The key idea in {topic_l} is shown with a real case from a Kenyan company, followed by an exercise.",
    "Common mistakes in {topic_l} are listed with how to recognise and fix each one.",
    "By the end of this session you should be able to apply {topic_l} to the lab task in section 3.",
    "We compare two approaches to {topic_l} and discuss when each one is the better choice.",
]
NOTES_WEAK = [
    "{topic} is an important topic. Students should read more about it.",
    "This lesson covers {topic_l}. See the textbook for details.",
    "{topic} will be examined. Revise it.",
    "Notes on {topic_l} to be added.",
]
NOTES_CURRENT = "Examples use {term}, which is current practice."
NOTES_OUTDATED = "Examples use {term}."

SLIDES = [
    "{topic}: definition, example, exercise",
    "{topic} in practice: case study and discussion questions",
    "{topic}: key terms and a short quiz",
]

TRANSCRIPT_CLEAR = [
    "Today we look at {topic_l}. Let us start with a simple example on the screen.",
    "Notice what happens when we change this value. That is the core of {topic_l}.",
    "Pause the video and try the next step yourself before I show the answer.",
    "Here is a common mistake students make, and here is how to avoid it.",
    "To summarise, {topic_l} lets us solve the problem we started with.",
]
TRANSCRIPT_HESITANT = [
    "Um, I think this is how {topic_l} works, let me check my notes.",
    "I am not very sure about this part, we will come back to it later.",
    "So, uh, this one is a bit complicated, just read about it on your own.",
]
TRANSCRIPT_SLIDE_READING = [
    "Slide four says {topic_l} is defined as follows. Slide five says the next point.",
    "As written on the slide, number one, number two, number three.",
]

# Student comments by evaluation dimension: (positive, negative).
COMMENTS: dict[str, tuple[list[str], list[str]]] = {
    "clarity": (
        ["Explanations were clear and the examples helped a lot.", "The lecturer breaks hard topics into simple steps."],
        ["Hard to follow; the lecturer reads from the slides.", "Concepts were explained too quickly and without examples."],
    ),
    "organisation": (
        ["Classes started on time and followed the course outline.", "The course was well organised week by week."],
        ["Many classes were cancelled or moved without notice.", "We fell behind the course outline and rushed at the end."],
    ),
    "feedback": (
        ["Assignments were marked quickly with useful comments.", "Questions on the forum were answered the same day."],
        ["We waited weeks for marks and got no comments.", "Questions on the forum were never answered."],
    ),
    "engagement": (
        ["Group activities made the class interesting.", "The lecturer encourages everyone to take part."],
        ["Classes were boring and attendance dropped.", "There was no chance to ask questions or discuss."],
    ),
    "resources": (
        ["Notes and videos were posted before every class.", "The recorded videos were very helpful for revision."],
        ["Notes were missing for many topics.", "No videos or slides were shared after class."],
    ),
    "currency": (
        ["Examples matched what industry uses today.", "The lecturer uses current tools in the labs."],
        ["The examples and tools are outdated.", "Materials look like they were written many years ago."],
    ),
    "mastery": (
        ["The lecturer clearly knows the subject very well.", "Difficult questions were answered with confidence."],
        ["The lecturer seemed unsure of the topic.", "Some explanations were wrong and had to be corrected later."],
    ),
}

NEGATIVE_COMMENTS = {dim: set(neg) for dim, (_, neg) in COMMENTS.items()}
HESITANT_MARKERS = ("I think this is how", "I am not very sure", "just read about it on your own")
