"""Field-level repair for versioned source-locked Scenario drafts.

Repairs only presentation fields and references; never changes Goal,
source scopes, modalities, or the frozen Gold Solver. Repaired drafts
still pass the complete deterministic guard and the independent Critic.
"""
from __future__ import annotations

import re
from copy import deepcopy

SCENARIO_FIELDS = ("setting", "user_role", "purpose", "motivation", "entity_kind")
_FORBIDDEN = re.compile(
    r"\b(?:TaskGraph|WorkProduct|synthetic_attribute|original media)\b", re.I
)


def diagnose_fields(spec: dict, proposal: dict) -> dict[str, list[str]]:
    narrative = proposal.get("scenario") if isinstance(proposal, dict) else None
    if not isinstance(narrative, dict):
        return {"scenario": ["not_an_object"]}
    flags = {}
    for key in SCENARIO_FIELDS:
        value = narrative.get(key)
        reasons = []
        if not isinstance(value, str):
            reasons.append("not_a_string")
        else:
            length = len(value.strip())
            if length < 3:
                reasons.append("too_short")
            if length > 250:
                reasons.append("too_long")
            if _FORBIDDEN.search(value):
                reasons.append("implementation_term")
        if reasons:
            flags[key] = reasons
    from .surface_quality import topic_mentioned
    setting = narrative.get("setting")
    if isinstance(setting, str) and not topic_mentioned(
        setting, spec["scenario"]["topic"]
    ):
        flags.setdefault("setting", []).append("topic_missing")
    return flags


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    # Prefer a complete sentence; otherwise cut at a word boundary.
    prefix = text[:limit]
    endings = list(re.finditer(r"[.!?](?:\s|$)", prefix))
    if endings and endings[-1].end() >= max(40, limit // 3):
        return prefix[:endings[-1].end()].strip()
    cropped = prefix.rsplit(" ", 1)[0].strip()
    return cropped if len(cropped) >= 3 else prefix.strip()


def repair_draft(spec: dict, proposal: dict) -> tuple[dict, dict]:
    """Return a minimally repaired draft and a PRIVATE trace of changes."""
    draft = deepcopy(proposal)
    diagnostics = diagnose_fields(spec, draft)
    if not isinstance(draft, dict) or not isinstance(
        draft.get("scenario"), dict
    ):
        return draft, {"before": diagnostics, "changes": {}}
    narrative = draft["scenario"]
    changes = {}
    frozen = spec["scenario"]
    for key, reasons in diagnostics.items():
        if key not in SCENARIO_FIELDS:
            continue
        original = narrative.get(key)
        if not isinstance(original, str):
            replacement = frozen[key]
        else:
            replacement = _FORBIDDEN.sub("working material", original).strip()
            if len(replacement) < 3:
                replacement = frozen[key]
            if key == "setting" and "topic_missing" in reasons:
                topic = frozen["topic"]
                replacement = f"{topic}: {replacement}"
            replacement = _shorten(replacement, 250)
            if len(replacement) < 3:
                replacement = frozen[key]
        if replacement != original:
            narrative[key] = replacement
            changes[key] = reasons
    return draft, {"before": diagnostics, "changes": changes}
