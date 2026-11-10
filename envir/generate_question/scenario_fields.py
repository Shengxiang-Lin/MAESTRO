"""Deterministic, auditable repairs of Scenario *presentation fields*.

Only surface-shape failures are repaired. Semantic claims, source rules and
the frozen Goal remain subject to the original independent guards/Critic.
"""
from __future__ import annotations

import copy
import re

_FIELDS = ("setting", "user_role", "purpose", "motivation", "entity_kind")
_MAX_LENGTH = 250
_FORBIDDEN = re.compile(
    r"\b(?:TaskGraph|WorkProduct|synthetic_attribute|original media)\b", re.I
)


def _compact(text: str, limit: int = _MAX_LENGTH) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    # Prefer a self-contained first sentence; never introduce new facts.
    for delimiter in (". ", "; ", ", "):
        first = normalized.split(delimiter, 1)[0].strip()
        if 12 <= len(first) <= limit - 1:
            return first.rstrip(".;,") + "."
    clipped = normalized[: limit - 1].rsplit(" ", 1)[0].rstrip(" .;,")
    return clipped + "." if clipped else normalized[:limit]


def repair_scenario_fields(spec: dict, candidate: dict) -> tuple[dict, list[dict]]:
    """Return repaired proposal and private audit; never edit source metadata."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("scenario"), dict):
        return candidate, []
    revised = copy.deepcopy(candidate)
    narrative = revised["scenario"]
    topic = spec["scenario"]["topic"]
    goal = spec.get("macro_plan", {}).get("goal", {}).get(
        "public_label", "requested result"
    )
    default = {
        "setting": f"Review the supplied {topic} working materials",
        "user_role": "research analyst",
        "purpose": f"Determine the confirmed {goal}",
        "motivation": f"Compare independent evidence to confirm the {goal} for the matching item.",
        "entity_kind": "records and events",
    }
    repairs = []
    for field in _FIELDS:
        before = narrative.get(field)
        if not isinstance(before, str):
            narrative[field] = default[field]
            repairs.append({"field": field, "reason": "invalid_type_or_missing",
                            "before_type": type(before).__name__,
                            "after_length": len(narrative[field])})
            continue
        value = " ".join(before.split())
        if _FORBIDDEN.search(value):
            narrative[field] = default[field]
            repairs.append({"field": field, "reason": "internal_jargon",
                            "before_length": len(before),
                            "after_length": len(narrative[field])})
        elif len(value) < 3:
            narrative[field] = default[field]
            repairs.append({"field": field, "reason": "too_short",
                            "before_length": len(before),
                            "after_length": len(narrative[field])})
        elif len(value) > _MAX_LENGTH:
            narrative[field] = _compact(value)
            repairs.append({"field": field, "reason": "too_long",
                            "before_length": len(before),
                            "after_length": len(narrative[field])})
        elif value != before:
            narrative[field] = value
            repairs.append({"field": field, "reason": "whitespace",
                            "before_length": len(before),
                            "after_length": len(value)})
    # Topic presence is a presentation rule; do not invent source content.
    from .surface_quality import topic_mentioned
    setting = narrative["setting"]
    if not topic_mentioned(setting, topic):
        prefix = f"Regarding {topic}: "
        narrative["setting"] = _compact(prefix + setting)
        repairs.append({"field": "setting", "reason": "topic_prefix",
                        "before_length": len(setting),
                        "after_length": len(narrative["setting"])})
    return revised, repairs
