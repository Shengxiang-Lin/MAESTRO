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
    """Reviewer omitted or malformed a requested semantic risk group."""


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

    def public_owner_path(node: dict) -> str:
        """Name the relation path of a filter using PUBLIC scoped semantics."""
        steps = []
        cursor = node
        for _ in range(4):
            op = cursor.get("definition")
            if op == "RELATE":
                relation = str(cursor.get("relation") or "unspecified")
                direction = str(cursor.get("direction") or "unspecified")
                steps.append(f"{direction} traversal of '{relation}'")
            elif op in ("INTERSECT", "UNION"):
                break  # Multiple owners cannot be represented by one path.
            children = cursor.get("inputs") or []
            if not children or not isinstance(children[0], dict):
                break
            cursor = children[0]
        return (" from ".join(steps) if steps
                else "the entity set identified by this reference clause")

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
                text = (
                    f"Entity type restriction: {condition['entity_type']}. "
                    f"Owning scope: {public_owner_path(node)}; "
                    "do not transfer the type restriction to a related entity."
                )
            else:
                val = condition.get("value", {})
                text = (
                    f"Filter on the SAME owning entity set: "
                    f"{condition['attribute']} {condition['comparison']} "
                    f"{val.get('value')}{val.get('unit', '') if isinstance(val, dict) else ''}. "
                    f"Owning scope: {public_owner_path(node)}. "
                    "Do not move this filter to a different relation endpoint."
                )
            # The full owning expression is part of identity: the same
            # attribute/value restriction on two DIFFERENT related entity
            # sets must remain two separately checked obligations.
            append("condition_owner", text, excerpt, (node, condition))
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
                "is the subject and which is the object. Retain the actual "
                "relation meaning (e.g., 'named after' is NOT the same as "
                "'has the same name')."
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



_RISK_KINDS = ("set_logic", "condition_owner", "relation_scope", "final_answer")


def grouped_semantic_checks(plan: ScopedPlan) -> list[dict]:
    """Collapse 9-20 atomic facts into at most FOUR output judgments.

    Retain every distinct public requirement in the INPUT; only the model's
    output format is shorter. A category passes only if ALL its obligations
    are preserved. This is not a formal semantic equivalence oracle.
    """
    atomic = semantic_checks(plan)
    groups = []
    for kind in _RISK_KINDS:
        members = [c for c in atomic if c["kind"] == kind]
        if not members:
            continue
        entries = []
        seen = set()
        for item in members:
            requirement = {
                "id": item["id"],
                "expected": item["expected"],
                "reference_excerpt": item["reference_excerpt"][:190],
            }
            key = json.dumps(requirement, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                entries.append(requirement)
        groups.append({
            "id": f"R{len(groups) + 1}",
            "kind": kind,
            "requirements": entries,
        })
    return groups


_AFFIRMATIVE_EXPLANATION = re.compile(
    r"\b(?:is|are|was|were|remain|remains)\s+(?:fully\s+|correctly\s+)?"
    r"(?:preserved|retained|correct|unchanged)\b|"
    r"\bcorrectly\s+(?:assigned|maps|mapped|preserves|preserved|retains|"
    r"represents|matches|reflects)\b|"
    r"\bmatching the reference (?:logic|meaning)\b", re.I,
)
_CONTRAST_EXPLANATION = re.compile(
    r"\b(?:not|never|except|but|however|although|instead|incorrectly|"
    r"different|differs|changed|missing|omitted|lost|ambiguous|unclear|"
    r"mismatch|unverifiable|cannot|doesn't|fails?)\b", re.I,
)


def validate_group_decisions(groups: list[dict], outcomes: Any) -> None:
    """Fail closed on missing groups, but never require byte-exact quotations.

    v14 required a separate quote for up to 20 checks and failed in 14/26
    Critic calls on text-witness formatting alone. Per-category explanations
    preserve auditability without making surface quotation the task.
    """
    if not isinstance(outcomes, list) or len(outcomes) != len(groups):
        raise CheckFormatError("review_group_results_incomplete")
    expected = {group["id"] for group in groups}
    by_id = {group["id"]: group for group in groups}
    seen = set()
    for item in outcomes:
        if not isinstance(item, dict):
            raise CheckFormatError("review_group_not_object")
        ident = item.get("id")
        if ident not in expected or ident in seen:
            raise CheckFormatError("review_group_id_invalid")
        seen.add(ident)
        if item.get("status") not in ("preserved", "changed", "uncertain"):
            raise CheckFormatError("review_group_status_invalid")
        explanation = item.get("explanation")
        if not isinstance(explanation, str) or len(explanation.strip()) < 8:
            raise CheckFormatError("review_group_reason_missing")
        # Changed/uncertain judgments must identify WHICH public atomic
        # obligation failed. A generic group-level negative is not actionable
        # and cannot be trusted as a semantic difference.
        problem_ids = item.get("problem_ids")
        allowed = {str(req["id"]) for req in by_id[ident]["requirements"]
                   if "id" in req}
        if (not isinstance(problem_ids, list)
                or len(problem_ids) != len(set(map(str, problem_ids)))
                or any(not isinstance(pid, str) or pid not in allowed
                       for pid in problem_ids)):
            raise CheckFormatError("review_problem_ids_invalid")
        if item["status"] == "preserved" and problem_ids:
            raise CheckFormatError("preserved_review_has_problem_ids")
        if item["status"] != "preserved" and not problem_ids:
            raise CheckFormatError("negative_review_missing_problem_ids")
        if (item["status"] != "preserved"
                and _AFFIRMATIVE_EXPLANATION.search(explanation)
                and not _CONTRAST_EXPLANATION.search(explanation)):
            raise CheckFormatError("negative_review_explanation_affirms_preservation")


def _compact(value: str) -> str:
    return " ".join(value.casefold().split())


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


_NAMED_AFTER = re.compile(r"\b(?:named after|named for|named in honou?r of|namesake of)\b", re.I)
_NAME_EQUALITY = re.compile(
    r"\b(?:names? (?:that |which )?match(?:es|ing|ed)?|matching names?|share(?:s|d)? "
    r"(?:the )?same name|(?:have|has|had) (?:the )?same name|"
    r"(?:their|whose) names? (?:are|is) (?:the )?same)\b", re.I,
)


def relation_semantic_contradictions(reference: str, candidate: str) -> list[str]:
    """Catch narrow, unambiguous meaning substitutions, not missing phrases.

    This public-text-only guard is deliberately conservative: it does not
    demand the original wording, and it is NOT an exhaustive relation parser.
    """
    if (_NAMED_AFTER.search(reference) and _NAME_EQUALITY.search(candidate)
            and not _NAME_EQUALITY.search(reference)):
        return ["named_after_replaced_by_name_equality"]
    return []


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
    "CheckFormatError", "semantic_checks",
    "check_failures", "explicit_join_contradictions", "naturalness_gate",
    "missing_typed_literals", "relation_semantic_contradictions",
    "grouped_semantic_checks", "validate_group_decisions",
]
