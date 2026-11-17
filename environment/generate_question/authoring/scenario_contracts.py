"""Versioned Scenario presentation contracts: titles, fields and topic guards.

Only surface fields may be repaired; no Goal, World, source affordance,
gold answer or information-dependency condition is changed.
"""
from __future__ import annotations

# --- canonical source-title and domain contracts ---
from copy import deepcopy

SCENE_REALISM_VERSION = "maestro.scene_realism.v1"

DOMAIN_FRAMES = {
    "sports": ("match and event review", "game or competition records"),
    "film": ("screening and episode identification", "production materials"),
    "music": ("performance and recording investigation", "performance materials"),
    "culture": ("exhibition and performance investigation", "exhibition materials"),
    "education": ("course and campus investigation", "academic materials"),
    "logistics": ("shipment and arrival investigation", "transport materials"),
    "civic": ("site and permit investigation", "civic working materials"),
    "environment": ("field survey and species investigation", "field observations"),
    "science": ("experiment and review investigation", "research records"),
    "business": ("filing and resolution investigation", "business records"),
    "book": ("edition and publication investigation", "edition records"),
    "history": ("artifact and historical collection investigation", "collection records"),
    "geography": ("site and geographical survey investigation", "survey records"),
    "general": ("source comparison and record investigation", "working materials"),
}

SOURCE_KIND = {
    "visual": {
        "image": "still-image annotations",
        "video": "video action and frame-time annotations",
    },
    "caption": {
        "audio": "spoken-commentary cue annotations",
        "text": "written caption annotations",
    },
    "footage": {"video": "video action timeline"},
    "commentary": {"audio": "spoken commentary timeline"},
    "context": {"text": "independent event log"},
    "report_a": {"text": "first reported-year account"},
    "report_b": {"table": "independent reported-year table"},
    "archive": {"table": "verified result table"},
    "stat_a": {"table": "first independently recorded count"},
    "stat_b": {"table": "second independently recorded count"},
}
 
def has_scene_realism(spec: dict) -> bool:
    return spec.get("scene_realism_version") == SCENE_REALISM_VERSION


def _domain_frame(spec: dict) -> tuple[str, str]:
    return DOMAIN_FRAMES.get(
        spec.get("macro_plan", {}).get("domain", "general"),
        DOMAIN_FRAMES["general"],
    )


def expected_source_titles(spec: dict) -> dict[str, str]:
    """Return source-ID-keyed titles determined by modality and source role."""
    topic = spec["scenario"]["topic"]
    frame, _ = _domain_frame(spec)
    titles = {}
    for source in spec["sources"]:
        sid, modality = source["source_id"], source["modality"]
        if sid.startswith("index_"):
            if modality != "text":
                raise ValueError("index_requires_text_source")
            desc = "keyed cross-reference index"
        else:
            desc = SOURCE_KIND.get(sid, {}).get(modality)
            if not desc:
                raise ValueError(f"invalid_source_modality_pair:{sid}:{modality}")
        titles[sid] = f"{topic} — {frame}: {desc}"
    return titles


def canonicalize_scenario_proposal(spec: dict, proposal: dict) -> dict:
    """Accept only the LLM's narrative; do not trust authored source titles.

    Deterministically overwrite every proposed title from the frozen schema.
    Unknown or partial titles cannot authorize new media types.
    """
    if not has_scene_realism(spec) or not isinstance(proposal, dict):
        return proposal
    chosen = deepcopy(proposal)
    titles = expected_source_titles(spec)
    chosen["source_titles"] = [
        {"source_id": sid, "title": titles[sid]}
        for sid in [row["source_id"] for row in spec["sources"]]
    ]
    return chosen


def scenario_domain_context(spec: dict) -> dict:
    frame, materials = _domain_frame(spec)
    return {
        "domain": spec["macro_plan"]["domain"],
        "activity": frame,
        "material_kind": materials,
        "goal_label": spec["macro_plan"]["goal"]["public_label"],
        "source_titles": expected_source_titles(spec),
        "source_modalities": {
            row["source_id"]: row["modality"] for row in spec["sources"]
        },
        "restriction": (
            "Use only these source types; do not invent new media, "
            "extra output targets, real historical events or clearance rules."
        ),
    }

# --- presentation-field repair ---
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

# --- source-topic quality gate ---
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
