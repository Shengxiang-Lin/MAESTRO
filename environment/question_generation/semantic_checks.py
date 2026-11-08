"""Only high-confidence, deterministic public-text guards for naturalization.

No grouped risk checklists, parse trees, model-owned C-IDs, semantic
probes, or coverage certificates. Free natural-language meaning is checked
by the single Critic.
"""
from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from .contracts import typed_values
from .scoped_surface import ScopedPlan


_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth",
             "seventh", "eighth")
_GROUP = re.compile(
    r"\b(?:the\s+)?(?:(first|second|third|fourth|fifth|sixth|seventh|eighth)"
    r"\s+group|group\s+([1-8]))\b", re.I,
)
_AND = re.compile(
    r"^(?:\s*(?:consists? of|as|is|comprises|includes|contains|"
    r"is defined as|defined as)\s+)*(?:the\s+)?"
    r"(?:entities?\s+(?:in\s+)?both|intersection\b|intersect\b)", re.I,
)
_OR = re.compile(
    r"^(?:\s*(?:consists? of|as|is|comprises|includes|contains|"
    r"is defined as|defined as)\s+)*(?:the\s+)?"
    r"(?:entities?\s+in\s+(?:at\s+least\s+one|either)|union\b|either\b)", re.I,
)


def _explicit_group_joins(text: str) -> dict[int, str]:
    """Only catch overt AND ↔ OR substitutions for the SAME named group."""
    results = {}
    for m in _GROUP.finditer(text):
        ordinal, number = m.group(1), m.group(2)
        idx = _ORDINALS.index(ordinal.casefold()) + 1 if ordinal else int(number)
        after = text[m.end():m.end() + 100]
        if _AND.match(after):
            results.setdefault(idx, "intersection")
        elif _OR.match(after):
            results.setdefault(idx, "union")
    return results


def explicit_join_contradictions(reference: str, candidate: str) -> list[str]:
    orig = _explicit_group_joins(reference)
    edited = _explicit_group_joins(candidate)
    return [
        f"group_{idx}_changed_{expected}_to_{edited[idx]}"
        for idx, expected in orig.items()
        if idx in edited and expected != edited[idx]
    ]


def explicit_global_join_contradictions(plan, candidate: str) -> list[str]:
    """Reject only a clear OR substitution in an otherwise AND-only query.

    A mixed INTERSECT/UNION DAG may legitimately contain 'either ... or'
    within a branch; leave those cases to independent semantic review.
    This is a high-precision safety net, NOT a general natural-language
    parser or full proof of Boolean equivalence.
    """
    query = getattr(plan, "query", None)
    if not isinstance(query, dict):
        return []
    join_ops = set()
    def scan(node):
        if not isinstance(node, dict):
            return
        kind = node.get("definition")
        if kind in ("INTERSECT", "UNION"):
            join_ops.add(kind)
        for child in node.get("inputs", ()):
            scan(child)
    scan(query)
    if join_ops != {"INTERSECT"}:
        return []
    # This explicit template changes the required all-branch membership
    # into an alternative. A plain "or" is too broad to reject safely.
    if re.search(r"\beither\b[^.!?;]{0,160}\bor\b", candidate, re.I):
        return ["all_branch_intersection_rewritten_as_either_or"]
    # In the 30-task anchored pilot a critic approved "at least one of the
    # following criteria ... mass != X OR height > Y" even though the only
    # join was INTERSECT. This is an explicit substitution, not a mere OR
    # appearing in a source title or a legitimate alternative in a mixed DAG.
    if re.search(
        r"\b(?:at least one|any one|one or more)\s+of\s+"
        r"(?:the\s+)?(?:following\s+)?"
        r"(?:criteria|conditions|requirements|filters)\b"
        r"[^?;]{0,220}\bor\b", candidate, re.I,
    ):
        return ["all_branch_intersection_rewritten_as_any_criterion"]
    return []


_NAMED_AFTER = re.compile(
    r"\b(?:named after|named for|named in honou?r of|namesake of)\b", re.I,
)
_NAME_EQUALITY = re.compile(
    r"\b(?:names? (?:that |which )?match(?:es|ing|ed)?|matching names?|"
    r"share(?:s|d)? (?:the )?same name|(?:have|has|had) (?:the )?same name|"
    r"(?:their|whose) names? (?:are|is) (?:the )?same)\b", re.I,
)


def relation_semantic_contradictions(reference: str, candidate: str) -> list[str]:
    """Catch a clear relation substitution, not free paraphrase."""
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
    """Guard public numeric/date/code clues, allowing equivalent formatting.

    This is deliberately NOT full semantics. It does not infer attribute
    ownership, relations, or semantics of natural-language conjunctions.
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
        values = [
            c.get("value") for c in tree.get("conditions", ())
            if isinstance(c, dict) and c.get("value") is not None
        ]
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
                except (ArithmeticError, ValueError):
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


__all__ = [
    "explicit_join_contradictions", "explicit_global_join_contradictions",
    "relation_semantic_contradictions", "missing_typed_literals",
]
