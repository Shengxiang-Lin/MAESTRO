"""Coverage-aware PUBLIC semantic obligations for Stage-2 naturalization.

The model sees candidate text and scoped contrast questions ONLY: it never
receives the reference question, the intended answer token, hidden World, or
Gold execution results. These checks are *fallible independent assessments*,
not a formal natural-language equivalence proof.

Unlike v19, the required semantic questions are compiled from the frozen
PUBLIC ScopedPlan rather than triggered by candidate group labels. Every
relation and set-operator occurrence gets its own obligation; an output-kind
probe is always required. The full obligation manifest is replayed by Stage 3.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .scoped_surface import ScopedPlan
from .semantic_probes import compile_candidate_probes, validate_candidate_probe_records

MAX_COVERAGE_PROBES = 24
PROBE_BATCH_SIZE = 4
_OUTPUT_TYPES = {
    "COUNT": "COUNT", "QUERY_NAME": "ENTITIES",
    "QUERY_ATTR": "ATTRIBUTE", "QUERY_RELATION": "RELATIONS",
    "SELECT_AMONG": "RANK", "SELECT_BETWEEN": "RANK",
}


class UnverifiableCoverage(ValueError):
    """The compiler cannot provide complete bounded review coverage."""


def _output_label(definition: str) -> str:
    if definition.startswith("VERIFY_"):
        return "VERIFY"
    if definition not in _OUTPUT_TYPES:
        raise UnverifiableCoverage("unsupported_public_final_answer_kind")
    return _OUTPUT_TYPES[definition]


def _meaning_hint(node: Mapping[str, Any], depth: int = 0) -> str:
    """A short PUBLIC anchor, only to locate a clause in candidate prose."""
    if depth > 4:
        return ""
    attrs = []
    for cond in node.get("conditions") or []:
        if not isinstance(cond, Mapping):
            continue
        if cond.get("entity_type"):
            attrs.append(str(cond["entity_type"]))
        elif cond.get("attribute") and isinstance(cond.get("value"), Mapping):
            value = cond["value"].get("value")
            if value is not None:
                attrs.append(f"{cond['attribute']} {value}")
        if len(attrs) == 2:
            break
    if attrs:
        return ", ".join(attrs)[:105]
    for key in ("anchor", "type_anchor", "anchor_type"):
        if node.get(key):
            return str(node[key])[:105]
    for child in node.get("inputs") or []:
        if isinstance(child, Mapping):
            found = _meaning_hint(child, depth + 1)
            if found:
                return found
    return ""


def _walk_public(plan: ScopedPlan):
    """Yield unique semantic nodes; repeated uses of one group are deduped."""
    seen = set()

    def walk(node, group, level):
        if not isinstance(node, Mapping) or level > 15:
            return
        key = json.dumps(node, ensure_ascii=False, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        yield node, group
        for child in node.get("inputs") or ():
            yield from walk(child, group, level + 1)

    for index, meaning in enumerate(plan.meanings):
        label = f"group {index + 1}" if index < 8 else "the question"
        yield from walk(meaning, label, 0)
    yield from walk(plan.query, "the final question", 0)


def compile_coverage_probes(
    plan: ScopedPlan, reference: str, candidate: str,
) -> list[dict[str, str]]:
    """Compile ALL bounded high-risk obligations independently of paraphrase.

    Extends v19's narrow owner/conditional probes; those remain valuable
    but are no longer the only reason a question gets checked.
    """
    if not isinstance(plan, ScopedPlan):
        raise TypeError("coverage_requires_public_scoped_plan")
    preliminary = list(compile_candidate_probes(plan, reference, candidate))
    obligations = []
    for node, group in _walk_public(plan):
        op = str(node.get("definition") or "")
        if op == "RELATE":
            relation = str(node.get("relation") or "").strip()
            direction = str(node.get("direction") or "").strip().lower()
            if not relation or direction not in ("forward", "backward"):
                raise UnverifiableCoverage("relation_missing_public_direction")
            source = (node.get("inputs") or [{}])[0]
            hint = _meaning_hint(source) if isinstance(source, Mapping) else ""
            # Select a disambiguating public lexical anchor, not a private ID.
            suffix = f" near the clue '{hint}'" if hint else ""
            obligations.append({
                "kind": "relation_path",
                "group": group,
                "relation": relation,
                "direction": direction,
                "anchor": hint,
                "expected": direction.upper(),
                "question": (
                    f"Read the candidate's clause concerning the recorded "
                    f"'{relation}' relationship{suffix}. Which structure "
                    "does the candidate actually express? FORWARD: the "
                    "selected entities are targets of this relationship "
                    "from other entities; BACKWARD: the selected entities "
                    "have this relationship pointing to other entities; "
                    "DIRECT: the relation is replaced by a direct property "
                    "or shared name; OTHER: a weaker/different relationship; "
                    "UNCLEAR: direction/endpoint is not recoverable. "
                    "Reply FORWARD, BACKWARD, DIRECT, OTHER or UNCLEAR."
                ),
            })
        elif op in ("INTERSECT", "UNION"):
            source_hints = [
                _meaning_hint(child)
                for child in (node.get("inputs") or [])
                if isinstance(child, Mapping)
            ]
            hint = " / ".join(h for h in source_hints[:2] if h)[:125]
            suffix = f" near the clues '{hint}'" if hint else ""
            obligations.append({
                "kind": "set_scope",
                "group": group,
                "expected": "AND" if op == "INTERSECT" else "OR",
                "anchor": hint,
                "question": (
                    f"For the candidate's set-building clause{suffix}, "
                    "is each selected entity required to satisfy BOTH "
                    "branches itself (AND), or is qualifying for EITHER "
                    "branch sufficient (OR)? If the clause is missing, "
                    "has moved to a different owner, or cannot be "
                    "identified, reply UNCLEAR. Reply AND, OR or UNCLEAR."
                ),
            })
    # An independent reader always checks the requested answer type.
    out = _output_label(str(plan.query.get("definition") or ""))
    obligations.append({
        "kind": "answer_type", "group": "final", "expected": out,
        "question": (
            "What does the final candidate question ask the solver to "
            "return? COUNT (a number of distinct entities), ENTITIES "
            "(entity names), ATTRIBUTE (recorded attribute values), "
            "RELATIONS (relation labels), RANK (a best/worst result "
            "with ties), VERIFY (yes/no about any recorded value), or "
            "UNCLEAR? Return exactly one token."
        ),
    })
    # Preserve the targeted v19 checks; no duplicate final-answer question.
    obligations.extend(preliminary)
    if len(obligations) > MAX_COVERAGE_PROBES:
        raise UnverifiableCoverage(
            f"coverage_probe_budget_exceeded:{len(obligations)}"
        )
    if not obligations:
        raise UnverifiableCoverage("missing_coverage_obligations")
    return [
        {"id": f"P{i}", **{k: v for k, v in probe.items() if k != "id"}}
        for i, probe in enumerate(obligations, start=1)
    ]


def coverage_manifest(probes: list[dict], records: list[dict]) -> dict:
    """A *reported model-check coverage* metric, not a correctness proof."""
    by_id = {row["id"]: row for row in records}
    counted = Counter(p["kind"] for p in probes)
    preserved = Counter(
        p["kind"] for p in probes
        if p["id"] in by_id and by_id[p["id"]].get("answer") == p["expected"]
    )
    return {
        "required": len(probes),
        "verified": sum(preserved.values()),
        "required_by_kind": dict(sorted(counted.items())),
        "verified_by_kind": dict(sorted(preserved.items())),
        "status": ("complete" if sum(preserved.values()) == len(probes)
                   and len(records) == len(probes) and probes else "incomplete"),
        "method": "candidate_only_model_assessment_not_formal_proof",
    }


def validate_coverage_manifest(
    probes: list[dict], records: list[dict], manifest: Any,
) -> None:
    """Stage-3 replay of the full required set and exact model decisions."""
    validate_candidate_probe_records(probes, records)
    if (not probes or not any(p["kind"] == "answer_type" for p in probes)
            or (any(p["kind"] == "relation_path" for p in probes)
                and not any(r["kind"] == "relation_path" for r in records))):
        raise ValueError("stage2_coverage_required_obligations_missing")
    expected = coverage_manifest(probes, records)
    if expected["status"] != "complete" or manifest != expected:
        raise ValueError("stage2_semantic_coverage_incomplete")


__all__ = [
    "MAX_COVERAGE_PROBES", "PROBE_BATCH_SIZE", "UnverifiableCoverage",
    "compile_coverage_probes", "coverage_manifest",
    "validate_coverage_manifest",
]
