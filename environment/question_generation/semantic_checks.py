"""Small, reference-grounded semantic checks for editorial reviewers.

This is NOT ProgramIR readback. The validator sends a compact set of PUBLIC
answer-defining checks to the Critic and Final Judge; each verdict must quote
where the candidate expresses the requirement. The model remains responsible
for natural-language meaning, not for reproducing an execution tree.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from datetime import datetime
from decimal import Decimal
from .contracts import typed_values
from typing import Any

from .scoped_surface import ScopedPlan


class CheckFormatError(ValueError):
    """Review did not substantiate every requested public semantic check."""


def semantic_checks(plan: ScopedPlan) -> list[dict[str, str]]:
    """Create reference-only checkpoints from public scoped semantics.

    No World facts, hidden FIND identifiers, answer values or source IDs are
    available here. Deduplicating nested nodes prevents repeated obligations
    when a group is referred to by a later sentence.
    """
    if not isinstance(plan, ScopedPlan):
        raise TypeError("reference_checks_need_scoped_plan")
    checks: list[dict[str, str]] = []
    seen = set()

    def append(kind: str, expected: str, excerpt: str, signature: Any):
        key = (kind, json.dumps(signature, ensure_ascii=False, sort_keys=True))
        if key in seen:
            return
        seen.add(key)
        checks.append({
            "id": f"C{len(checks) + 1}",
            "kind": kind,
            "expected": expected,
            "reference_excerpt": excerpt[:550],
        })

    def walk(node, excerpt):
        if not isinstance(node, dict):
            raise ValueError("invalid_scoped_semantics")
        op = node.get("definition", "")
        for condition in node.get("conditions", ()):
            if condition.get("entity_type"):
                text = f"Entity type restriction: {condition['entity_type']}; keep its owner."
            else:
                val = condition.get("value", {})
                text = (
                    f"Filter on the SAME owning entity set: "
                    f"{condition['attribute']} {condition['comparison']} "
                    f"{val.get('value')}{val.get('unit', '') if isinstance(val, dict) else ''}. "
                    "Do not move this filter to an entity reached through a relation."
                )
            append("condition_owner", text, excerpt, (op, node.get("relation"),
                   node.get("direction"), condition))
        if op in ("INTERSECT", "UNION"):
            text = (
                "INTERSECTION: the same result entity must satisfy ALL branches; "
                "not the union of separate branch results."
                if op == "INTERSECT" else
                "UNION: result entities satisfying ANY branch are included; "
                "do not require all branches simultaneously."
            )
            append("set_logic", text, excerpt, node)
        if op == "RELATE":
            text = (
                f"Preserve the complete '{node.get('relation')}' relation chain: "
                f"{node.get('direction')} traversal, including which entity "
                "is the subject and which is the object."
            )
            append("relation_scope", text, excerpt, node)
        if op in ("QUERY_NAME", "COUNT", "QUERY_ATTR", "QUERY_RELATION",
                  "SELECT_AMONG", "SELECT_BETWEEN") or op.startswith("VERIFY_"):
            meaning = {
                "QUERY_NAME": "Return matching entity names (not just a count).",
                "COUNT": "Count distinct matching entities (not list names).",
                "QUERY_ATTR": "Return recorded attribute values.",
                "QUERY_RELATION": "Return relation names connecting the source and target groups.",
                "SELECT_AMONG": "Select ranked entities and retain ties.",
                "SELECT_BETWEEN": "Select ranked entities across BOTH candidate pools and retain ties.",
            }.get(op, "Verify whether ANY recorded value satisfies the specified comparison.")
            append("final_answer", meaning, excerpt, node.get("definition"))
        for child in node.get("inputs", ()):
            walk(child, excerpt)

    for options, meaning in zip(plan, plan.meanings, strict=True):
        walk(meaning, options[0])
    # Reviewers inspect a compact checklist, not an expected nested AST.
    return checks


def _compact(value: str) -> str:
    return " ".join(value.casefold().split())


def check_evidence(checks: list[dict], outcomes: Any, candidate: str) -> None:
    """Validate shape, check coverage and exact candidate evidence witnesses.

    A quote is *necessary*, but not proof that the quoted text fully supports
    the claimed requirement; semantic interpretation stays with the reviewer.
    """
    if not isinstance(outcomes, list):
        raise CheckFormatError("review_check_results_missing")
    required = {c["id"] for c in checks}
    actual = set()
    for item in outcomes:
        if not isinstance(item, dict):
            raise CheckFormatError("review_check_not_object")
        ident = item.get("id")
        if ident not in required or ident in actual:
            raise CheckFormatError("review_check_id_invalid_or_duplicate")
        actual.add(ident)
        if item.get("status") not in ("preserved", "changed", "uncertain"):
            raise CheckFormatError("review_check_status_invalid")
        if not isinstance(item.get("explanation"), str) or not item["explanation"].strip():
            raise CheckFormatError("review_check_explanation_missing")
        evidence = item.get("candidate_quote")
        if item["status"] == "preserved":
            if (not isinstance(evidence, str)
                    or not 3 <= len(evidence) <= 550
                    or _compact(evidence) not in _compact(candidate)):
                raise CheckFormatError("review_check_candidate_evidence_invalid")
        elif evidence not in (None, ""):
            if (not isinstance(evidence, str)
                    or _compact(evidence) not in _compact(candidate)):
                raise CheckFormatError("review_check_failure_quote_invalid")
    if actual != required:
        raise CheckFormatError("review_check_results_incomplete")


def check_failures(outcomes: list[dict]) -> bool:
    return any(item["status"] != "preserved" for item in outcomes)


_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
_GROUP = re.compile(
    r"\b(?:the\s+)?(?:(first|second|third|fourth|fifth|sixth|seventh|eighth)"
    r"\s+group|group\s+([1-8]))\b", re.I,
)
_AND = re.compile(r"^(?:\s*(?:consist of|as|is|comprises|includes|contains|"
                  r"is defined as|defined as)\s+)*(?:the\s+)?"
                  r"(?:entities?\s+(?:in\s+)?both|intersection\b|intersect\b)", re.I)
_OR = re.compile(r"^(?:\s*(?:consist of|as|is|comprises|includes|contains|"
                 r"is defined as|defined as)\s+)*(?:the\s+)?"
                 r"(?:entities?\s+in\s+at\s+least\s+one|union\b|either\b)", re.I)


def _explicit_group_joins(text: str) -> dict[int, str]:
    """Detect only *explicit* group-level AND/OR definitions.

    Absence of a marker is not a contradiction: free English need not
    reproduce the template's group labels or exact syntax.
    """
    results = {}
    matches = list(_GROUP.finditer(text))
    for m in matches:
        ordinal, number = m.group(1), m.group(2)
        idx = _ORDINALS.index(ordinal.casefold()) + 1 if ordinal else int(number)
        after = text[m.end():m.end()+100]
        # Avoid matching an unrelated group reference; only act if the
        # immediately following clause *defines* the set operator.
        conjunction = _AND.match(after)
        disjunction = _OR.match(after)
        if conjunction:
            results.setdefault(idx, "intersection")
        elif disjunction:
            results.setdefault(idx, "union")
    return results


def explicit_join_contradictions(reference: str, candidate: str) -> list[str]:
    reference_joins = _explicit_group_joins(reference)
    candidate_joins = _explicit_group_joins(candidate)
    return [
        f"group_{idx}_changed_{expected}_to_{candidate_joins[idx]}"
        for idx, expected in reference_joins.items()
        if idx in candidate_joins and candidate_joins[idx] != expected
    ]


_PROSE_DATE_DMY = re.compile(
    r"\b(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})\b", re.I,
)


def missing_typed_literals(plan: ScopedPlan, candidate: str) -> list[str]:
    """Check only high-confidence public numbers, dates and string codes.

    This never compares raw canonical lexical forms. It reads typed values
    from the checked PUBLIC contract; comma notation, short English numbers,
    ISO/prose dates and +prefixed codes can be equivalent. It deliberately
    leaves names and non-code strings to natural-language reviewers.
    """
    found = set(typed_values(candidate, words=True))
    for m in _PROSE_DATE_DMY.finditer(candidate):
        try:
            parsed = datetime.strptime(
                f"{m[1]} {m[2]} {m[3]}", "%d %B %Y"
            ).date().isoformat()
            found.add(("date", parsed))
        except ValueError:
            pass
    required = set()

    def visit(tree):
        values = [c.get("value") for c in tree.get("conditions", ())
                  if isinstance(c, dict) and c.get("value") is not None]
        if tree.get("definition", "").startswith("VERIFY_"):
            values.append(tree.get("value"))
        for value in values:
            if not isinstance(value, dict):
                continue
            kind, raw = value.get("type"), str(value.get("value", ""))
            if kind == "date":
                required.add(("date", raw))
            elif kind in ("number", "year"):
                try:
                    required.add(("number", Decimal(raw.replace(",", ""))))
                except ArithmeticError:
                    pass
            elif kind == "string":
                if raw.startswith("+") and raw[1:].isdigit():
                    required.add(("string", raw))
                elif re.fullmatch(r"\d+", raw):
                    required.add(("number", Decimal(raw)))
                elif re.fullmatch(r"[A-Z]{2,5}", raw):
                    required.add(("public_code", raw))
        for child in tree.get("inputs", ()):
            visit(child)

    visit(plan.query)
    missing = []
    for kind, value in sorted(required, key=lambda x: (x[0], str(x[1]))):
        if kind == "public_code":
            if not re.search(r"\b" + re.escape(value) + r"\b", candidate):
                missing.append(f"code:{value}")
        elif (kind, value) not in found:
            missing.append(f"{kind}:{value}")
    return missing


def naturalness_gate(reference: str, candidate: str) -> dict[str, Any]:
    """Reject obvious cosmetic edits, not all conservatively worded questions."""
    a = _compact(reference)
    b = _compact(candidate)
    similarity = SequenceMatcher(None, a, b, autojunk=False).ratio()
    old_words = len(reference.split())
    new_words = len(candidate.split())
    # A major shortening or meaningful restructuring can be worthwhile even
    # if some public group identifiers are retained.
    # Adding a generic preamble to an unchanged reference can lower string
    # similarity without making the actual problem any more natural.
    wraps_original = a in b
    requires_real_rewrite = not wraps_original and (
        similarity < 0.90 or
        (new_words <= int(old_words * 0.80) and similarity < 0.965)
    )
    return {
        "improved_enough_to_review": bool(requires_real_rewrite),
        "similarity": round(similarity, 4),
        "reference_words": old_words,
        "candidate_words": new_words,
        "reason": "ok" if requires_real_rewrite else "cosmetic_rephrase",
    }


__all__ = [
    "CheckFormatError", "semantic_checks", "check_evidence",
    "check_failures", "explicit_join_contradictions", "naturalness_gate",
    "missing_typed_literals",
]
