"""Candidate-only, question-answer probes of PUBLIC Stage-2 semantics.

The compiler derives expected answers from the frozen public ScopedPlan, but
the probing model sees ONLY the natural candidate plus short questions.
Probes are conservative targeted diagnostics, not an English equivalence proof.
Unsupported/unlabelled natural phrasings fall back to the ordinary reviewer.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .scoped_surface import ScopedPlan

_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth",
             "seventh", "eighth")
_ORDINAL = "|".join(_ORDINALS)
_ALLOWED = frozenset(("AND", "OR", "SELF", "MEMBER",
                      "ALL_MUST", "NOT_ALL", "UNCLEAR"))
_PROBE_SYSTEM = """You are a candidate-only semantic reader.
Read ONLY the candidate question below. You are NOT given the reference,
gold answer, expected probe answers or private world. For each requested
probe, determine what the CANDIDATE really says, regardless of what seems
logical or likely to have been intended. Distinguish membership of the
result entity from membership of one of its members. 'Either/or' permits
either branch; 'both' on the SAME entity requires the intersection.
If the candidate does not clearly specify the requested distinction, use
UNCLEAR instead of guessing. Return JSON only with exactly one answer
per probe ID, as {"answers":[{"id":"P1","answer":"AND"}]}.
Do not give explanations, intermediate reasoning or extra fields.
"""
_PROBE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "answer": {"enum": sorted(_ALLOWED)},
    },
    "required": ["id", "answer"],
    "additionalProperties": False,
}
_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": _PROBE_ITEM_SCHEMA}},
    "required": ["answers"],
    "additionalProperties": False,
}


class CandidateProbeFormatError(ValueError):
    """A candidate-only probe response is missing or malformed."""

    def __init__(self, message: str, *, expected=(), received=()):
        super().__init__(message)
        self.diagnostic = {
            "expected_ids": list(expected),
            "received_ids": list(received),
            "received_count": len(received),
        }


def _candidate_defines_group(candidate: str, ordinal: str) -> bool:
    idx = _ORDINALS.index(ordinal) + 1
    label = rf"(?:the\s+)?(?:{ordinal}\s+group|group\s*{idx})"
    return bool(re.search(
        rf"\b{label}\s+(?:consists?|contains?|includes?|comprises?|"
        rf"is\s+defined|to\s+include|to\s+consist)\b",
        candidate, flags=re.I,
    ))


def _surface_join(node: Mapping[str, Any]) -> str | None:
    """Peel unary public filters, never traverse a relation to guess a join."""
    current = node
    for _ in range(5):
        op = str(current.get("definition", ""))
        if op in ("INTERSECT", "UNION"):
            return "AND" if op == "INTERSECT" else "OR"
        if not op.startswith("FILTER_"):
            return None
        inputs = current.get("inputs") or []
        if len(inputs) != 1 or not isinstance(inputs[0], Mapping):
            return None
        current = inputs[0]
    return None


def compile_candidate_probes(
    plan: ScopedPlan, reference: str, candidate: str,
) -> list[dict[str, str]]:
    """Generate a bounded number of *public* contrast questions.

    Probe questions refer only to named surface groups the draft still
    defines. This avoids forcing free paraphrases to reproduce group labels.
    Expected answers NEVER enter a model request.
    """
    if not isinstance(plan, ScopedPlan):
        raise TypeError("candidate_probes_need_public_scoped_plan")
    candidates = []
    for group, meaning in zip(plan, plan.meanings, strict=True):
        if not group or not isinstance(meaning, Mapping):
            continue
        excerpt = str(group[0])
        found = re.match(
            rf"Let the ({_ORDINAL}) group consist\b", excerpt, flags=re.I,
        )
        if not found:
            continue
        ordinal = found.group(1).lower()
        if not _candidate_defines_group(candidate, ordinal):
            continue

        join = _surface_join(meaning)
        if join:
            candidates.append({
                "kind": "group_join", "group": ordinal,
                "expected": join,
                "question": (
                    f"When the candidate defines the {ordinal} group, does "
                    "the SAME entity have to satisfy BOTH branch conditions "
                    "(AND), or is satisfying EITHER branch enough (OR)? "
                    "Reply AND, OR or UNCLEAR."
                ),
            })

        # Guard a specific scoping contrast: the entity ITSELF belongs
        # to the second intersected group vs one of its members belongs.
        m = re.search(
            rf"\bentities in both entities with at least one member in the "
            rf"({_ORDINAL}) group and the ({_ORDINAL}) group\b",
            excerpt, flags=re.I,
        )
        if m:
            member_group, own_group = m.group(1).lower(), m.group(2).lower()
            candidates.append({
                "kind": "member_owner", "group": ordinal,
                "expected": "SELF",
                "question": (
                    f"When defining the {ordinal} group, must the candidate "
                    f"entity ITSELF belong to the {own_group} group (SELF), "
                    f"or is it enough for one of its members to belong to "
                    f"the {own_group} group (MEMBER)? Reply SELF, MEMBER "
                    "or UNCLEAR."
                ),
            })

    # A union with one branch derived from Group N and another genuinely
    # independent branch: do not accidentally restrict the second branch
    # to Group N. This probe is intentionally limited to explicit syntax.
    final = re.search(
        rf"\bAmong\b[^\?]{{0,300}}?\bdrawn from the ({_ORDINAL}) "
        rf"group or entities in both\b",
        reference, flags=re.I,
    )
    if final:
        ordinal = final.group(1).lower()
        if re.search(rf"\b(?:the\s+)?{ordinal}\s+group\b", candidate, re.I):
            candidates.append({
                "kind": "final_union_scope", "group": ordinal,
                "expected": "NOT_ALL",
                "question": (
                    f"In the final selection, MUST EVERY eligible candidate "
                    f"belong to the {ordinal} group (ALL_MUST), or can the "
                    f"alternative OR branch admit entities that are not in "
                    f"that group (NOT_ALL)? Reply ALL_MUST, NOT_ALL or UNCLEAR."
                ),
            })

    # Prefer scope-sensitive probes over generic join questions when bounded.
    rank = {"member_owner": 0, "final_union_scope": 1, "group_join": 2}
    chosen = sorted(candidates, key=lambda item: (rank[item["kind"]], item["group"]))[:5]
    return [{"id": f"P{i}", **item} for i, item in enumerate(chosen, start=1)]


def validate_candidate_probe_records(probes: list[dict], records: Any) -> None:
    """Fail closed if the saved v17 probe certificate differs from public plan."""
    if not isinstance(records, list) or len(records) != len(probes):
        raise ValueError("candidate_probe_certificate_incomplete")
    for probe, record in zip(probes, records, strict=True):
        if (not isinstance(record, dict)
                or set(record) != {"id", "kind", "expected", "answer"}
                or record["id"] != probe["id"]
                or record["kind"] != probe["kind"]
                or record["expected"] != probe["expected"]
                or record["answer"] != probe["expected"]):
            raise ValueError("candidate_probe_certificate_mismatch")


def probe_candidate_semantics(
    backend: Any, reference: str, candidate: str,
    probes: list[dict[str, str]], *, use_schema: bool = True,
) -> list[dict[str, str]]:
    """Read candidate only; reference argument is deliberately not transmitted.

    The common review-retry wrapper accepts reference for all review roles.
    Expected answers are held here on the teacher side and never in the
    model request. Return ordered observed answers for audit/certification.
    """
    del reference
    if not callable(getattr(backend, "complete_json", None)):
        raise CandidateProbeFormatError("candidate_probe_backend_unavailable")
    public = [{"id": p["id"], "question": p["question"]}
              for p in probes]
    raw = backend.complete_json(
        _PROBE_SYSTEM,
        json.dumps({"candidate_question": candidate, "probes": public},
                   ensure_ascii=False),
        response_schema=_PROBE_SCHEMA if use_schema else None,
    )
    answers = raw.get("answers") if isinstance(raw, dict) else None
    ids = [p["id"] for p in probes]
    got = ([x.get("id") for x in answers if isinstance(x, dict)]
           if isinstance(answers, list) else [])
    if (not isinstance(answers, list) or len(answers) != len(probes)
            or len(set(got)) != len(ids) or set(got) != set(ids)):
        raise CandidateProbeFormatError(
            "candidate_probe_answers_incomplete", expected=ids, received=got,
        )
    by_id = {a["id"]: a for a in answers}
    for item in answers:
        if (not isinstance(item, dict) or set(item) != {"id", "answer"}
                or item["answer"] not in _ALLOWED):
            raise CandidateProbeFormatError(
                "candidate_probe_answer_invalid", expected=ids, received=got,
            )
    return [{"id": p["id"], "kind": p["kind"],
             "expected": p["expected"], "answer": by_id[p["id"]]["answer"]}
            for p in probes]


__all__ = [
    "CandidateProbeFormatError", "compile_candidate_probes",
    "probe_candidate_semantics", "validate_candidate_probe_records",
]
