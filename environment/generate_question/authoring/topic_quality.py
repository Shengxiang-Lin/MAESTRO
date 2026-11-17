"""Pre-authoring quality gates for MMQA-derived topic labels.

No source fact or original benchmark answer is reused. Unusable fallback
labels are skipped *before* assigning a new Goal or spending LLM calls.
"""
from __future__ import annotations

import re

_PLACEHOLDER_TOPIC = re.compile(
    r"\b(?:reference|document|source)\s+collection\s+"
    r"[0-9a-f]{5,}\b", re.I,
)
_AMBIGUOUS_TOPIC = {
    "the man", "the woman", "this person", "that person",
    "unknown entity", "unknown subject", "a person",
    "a place", "a city", "a movie", "an event",
}


def topic_seed_issues(seed: dict) -> list[str]:
    topic = str(seed.get("topic", "")).strip()
    problems = []
    if (seed.get("scene_topic_policy")
            == "original_mmqa_source_bundle_fallback"
            or _PLACEHOLDER_TOPIC.search(topic)):
        problems.append("topic_generated_placeholder")
    if len(topic) < 3 or topic.casefold() in _AMBIGUOUS_TOPIC:
        problems.append("topic_ambiguous_subject")
    if len(topic) > 140:
        problems.append("topic_excessive_length")
    return problems
