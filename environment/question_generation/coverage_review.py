"""Coverage-aware PUBLIC semantic obligations for Stage-2 naturalization.

The model sees candidate text and scoped contrast questions ONLY: it never
receives the reference question, the intended answer token, hidden World, or
Gold execution results. These checks are *fallible independent assessments*,
not a formal natural-language equivalence proof.

Required questions are compiled from the frozen PUBLIC ScopedPlan. Every
relation and set operator gets an obligation, with all relationship endpoint
attributes checked separately. Output type and ranking are independent.
Stage 3 replays both the manifest and grounding in the exact candidate text.
"""
from __future__ import annotations

import json
import re
from hashlib import sha256
from datetime import date
from decimal import Decimal, InvalidOperation
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .scoped_surface import ScopedPlan
from .semantic_probes import compile_candidate_probes, validate_candidate_probe_records

MAX_COVERAGE_PROBES = 64
PROBE_BATCH_SIZE = 4
_OUTPUT_TYPES = {
    "COUNT": "COUNT", "QUERY_NAME": "ENTITIES",
    "QUERY_ATTR": "ATTRIBUTE", "QUERY_RELATION": "RELATIONS",
    "SELECT_AMONG": "ENTITIES", "SELECT_BETWEEN": "ENTITIES",
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
                attrs.append(f"{cond['attribute']} {_prose_value(cond['value'])}")
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


def _prose_value(value: Mapping) -> str:
    raw = str(value.get("value", ""))
    if value.get("type") in ("number", "year"):
        try:
            number = format(Decimal(raw.replace(",", "")),
                            "f" if value.get("type") == "year" else ",f")
            return number + str(value.get("unit", ""))
        except InvalidOperation:
            pass
    if value.get("type") == "date":
        try:
            parsed = date.fromisoformat(raw)
            return f"{parsed:%B} {parsed.day}, {parsed.year}"
        except ValueError:
            pass
    return raw


def _endpoint_conditions(node: Mapping) -> list[dict]:
    """Collect restrictions on this endpoint; NEVER cross a relation edge."""
    found = [dict(c) for c in node.get("conditions", ())
             if c.get("attribute") and isinstance(c.get("value"), Mapping)]
    op = str(node.get("definition", ""))
    if op.startswith("FILTER_") or op in ("INTERSECT", "UNION"):
        for child in node.get("inputs", ()):
            found.extend(_endpoint_conditions(child))
    return found


def _relation_contexts(plan: ScopedPlan):
    """Carry same-entity restrictions through filters/intersections only.

    In particular, a sibling filter in INTERSECT belongs to the relation's
    result endpoint, while filters below RELATE belong to its input endpoint.
    Dedup by edge AND owner restrictions, not just relation name.
    """
    seen = set()
    def walk(node, group, inherited=(), depth=0):
        if depth > 15:
            raise UnverifiableCoverage("public_semantic_tree_too_deep")
        op = str(node.get("definition", ""))
        own = _endpoint_conditions(node)
        combined = [*inherited, *own]
        if op == "RELATE":
            key = json.dumps([node, combined], sort_keys=True)
            if key not in seen:
                seen.add(key)
                yield node, group, combined
            for child in node.get("inputs", ()):
                yield from walk(child, group, (), depth + 1)
        else:
            carry = (combined if op.startswith("FILTER_") or op == "INTERSECT"
                     else inherited if op == "UNION" else ())
            for child in node.get("inputs", ()):
                yield from walk(child, group, carry, depth + 1)
    for node, group in [(m, f"group {i+1}") for i, m in enumerate(plan.meanings)]:
        yield from walk(node, group)
    yield from walk(plan.query, "the final question")


def _endpoint_hint(node, plan):
    for i, meaning in enumerate(plan.meanings):
        if node == meaning and not str(node.get("definition", "")).startswith(("QUERY", "SELECT", "COUNT", "VERIFY")):
            match = re.match(r"Let (.+?) consist\b", plan[i][0])
            label = match[1] if match else f"group {i+1}"
            return f"{label} ({_meaning_hint(node) or 'the described pool'})"
    hint = _meaning_hint(node) or "the described entity pool"
    if node.get("definition") == "RELATE":
        hint += f"; intermediate entities from the '{node['relation']}' clause"
    return hint


def _pool_identity(node, plan):
    """Endpoint locator without attribute/value answers for owner questions."""
    for i, meaning in enumerate(plan.meanings):
        if node == meaning and i < len(plan):
            match = re.match(r"Let (.+?) consist\b", plan[i][0])
            if match:
                return match[1]
    op = str(node.get("definition", ""))
    if op.startswith("FILTER_") and node.get("inputs"):
        return "a subset of " + _pool_identity(node["inputs"][0], plan)
    if op == "RELATE":
        return f"the intermediate pool produced by the '{node['relation']}' clause"
    return str(node.get("anchor") or node.get("type_anchor")
               or node.get("anchor_type") or "the described input entity pool")


def _walk_public(plan: ScopedPlan):
    """Yield unique semantic nodes; repeated uses of one group are deduped."""
    seen = set()

    def walk(node, group, level):
        if level > 15:
            raise UnverifiableCoverage("public_semantic_tree_too_deep")
        if not isinstance(node, Mapping):
            raise UnverifiableCoverage("malformed_public_semantic_tree")
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
            # Compiled separately with local endpoint identities and owners.
            continue
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
    seen_paths, seen_owners = set(), set()
    for node, group, result_conditions in _relation_contexts(plan):
        relation = str(node.get("relation") or "").strip()
        direction = str(node.get("direction") or "").strip().lower()
        if not relation or direction not in ("forward", "backward"):
            raise UnverifiableCoverage("relation_missing_public_direction")
        inputs = node.get("inputs") or []
        if len(inputs) != 1 or not isinstance(inputs[0], Mapping):
            raise UnverifiableCoverage("relation_missing_public_input_endpoint")
        source = inputs[0]
        hint = _endpoint_hint(source, plan)
        result_hint = "; ".join(dict.fromkeys(
            f"{c['attribute']} {c.get('comparison', '')} {_prose_value(c['value'])}"
            for c in result_conditions
        )) or "entities produced by this particular relationship clause"
        context = (
            f"Locate the '{relation}' clause in {group}. "
            f"Local endpoint A is the input pool: {hint}. "
            f"Local endpoint B is the entities obtained in this clause: {result_hint}. "
            "These labels refer ONLY to this hop, never automatically to the final answer. "
        )
        owner_context = (
            f"Locate the '{relation}' clause in {group}. "
            f"Local endpoint A is its input pool: {_pool_identity(source, plan)}. "
            "Local endpoint B is its result entities. "
            "A/B are local to this hop. Determine ownership from the candidate's "
            "grammar, not from the locator or an intended condition. "
        )
        edge_key = json.dumps(node, sort_keys=True)
        edge_id = sha256(edge_key.encode("utf-8")).hexdigest()[:16]
        path_probe = {
            "kind": "relation_path",
            "edge_id": edge_id,
            "group": group,
            "relation": relation,
            "direction": direction,
            "anchor": hint,
            "expected": "A_TO_B" if direction == "forward" else "B_TO_A",
            "question": (
                context + "Extract the predicate's actual subject (the entity "
                "the relation points FROM) and object (the entity it points TO). "
                "Quote their noun phrases verbatim as subject and object, quote "
                "the full local clause as evidence, and assign subject_role and "
                "object_role independently to A or B. For 'X is a target of r "
                "from Y', subject is Y and object is X. Return A_TO_B or B_TO_A "
                "consistent with these roles; use UNCLEAR with empty roles if "
                "the specific hop or either endpoint cannot be identified."
            ),
        }
        if edge_key not in seen_paths:
            seen_paths.add(edge_key)
            obligations.append(path_probe)
        for role, conditions in (("A", _endpoint_conditions(source)),
                                 ("B", result_conditions)):
            for cond in conditions:
                key = (edge_key, role, json.dumps(cond, sort_keys=True))
                if key in seen_owners:
                    continue
                seen_owners.add(key)
                value = _prose_value(cond["value"])
                obligations.append({
                    "kind": "condition_owner", "group": group,
                    "edge_id": edge_id,
                    "relation": relation, "direction": direction,
                    "attribute": cond["attribute"],
                    "comparison": cond.get("comparison", ""),
                    "value_type": cond["value"].get("type"),
                    "canonical_value": str(cond["value"].get("value", "")),
                    "value": value, "expected": role,
                    "question": owner_context + (
                        f"For the condition '{cond['attribute']} "
                        f"{cond.get('comparison', '')} {value}', which endpoint "
                        "actually owns this attribute in the CANDIDATE, A or B? "
                        "Quote its noun phrase as owner and the full condition "
                        "with its attachment as evidence. A property of a member "
                        "reached by the relation is not a property of the result "
                        "entity. If attachment is ambiguous or absent, return UNCLEAR."
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
            "RELATIONS (relation labels), VERIFY (yes/no about any recorded value), or "
            "UNCLEAR? Return exactly one token."
        ),
    })
    # Preserve the targeted v19 checks; no duplicate final-answer question.
    operation = str(plan.query.get("definition") or "")
    if operation in ("SELECT_AMONG", "SELECT_BETWEEN"):
        ranking = str(plan.query.get("ranking") or "").lower()
        expected = {"largest": "ARGMAX", "smallest": "ARGMIN",
                    "maximum": "ARGMAX", "minimum": "ARGMIN"}.get(ranking)
        if expected is None:
            raise UnverifiableCoverage("unsupported_public_ranking")
        obligations.extend([
            {"kind": "selection_operation", "group": "final", "expected": expected,
             "question": "Which selection operation does the final question request? "
                         "ARGMAX (greatest value), ARGMIN (least value), ALL (no "
                         "ranking), THRESHOLD (only a fixed cutoff), or UNCLEAR? "
                         "This is independent of whether the answer contains entity names."},
            {"kind": "ranking_attribute", "group": "final",
             "expected": str(plan.query["rank_attribute"]),
             "question": "What exact recorded attribute is used to rank the final "
                         "entities? Return its label as answer and quote the request "
                         "as evidence; use UNCLEAR if missing."},
            {"kind": "ranking_ties", "group": "final", "expected": "INCLUDE",
             "question": "Does the question explicitly include all tied best/worst "
                         "entities (INCLUDE), exclude ties (EXCLUDE), or leave tie "
                         "handling unstated (UNCLEAR)? Quote the tie instruction."},
        ])
    obligations.extend(p for p in preliminary if p["kind"] != "condition_owner")
    if len(obligations) > MAX_COVERAGE_PROBES:
        raise UnverifiableCoverage(
            f"coverage_probe_budget_exceeded:{len(obligations)}"
        )
    if not obligations:
        raise UnverifiableCoverage("missing_coverage_obligations")
    return [
        {"id": f"P{i}", "evidence_required": True,
         **{k: v for k, v in probe.items() if k != "id"}}
        for i, probe in enumerate(obligations, start=1)
    ]



def _matches_endpoint(owner: str, phrase: str) -> bool:
    """Conservative noun-phrase match, used ONLY to find explicit conflicts.

    Generic/same-type endpoints remain ambiguous: the check never infers a
    positive owner role from matching, and thus does not prove semantics.
    """
    if not isinstance(owner, str) or not isinstance(phrase, str):
        return False
    target = " ".join(owner.casefold().strip().split())
    clause = " ".join(phrase.casefold().strip().split())
    if len(target) < 4 or len(clause) < 4:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", clause))


def conflicting_endpoint_evidence(
    probes: list[dict], records: list[dict],
) -> dict[str, str]:
    """Catch a self-contradictory owner label without using model expectations.

    For a RELATE edge, the independent path reader supplies candidate-local
    subject/object quotes with A/B roles. An owner quote matching ONLY the
    opposite role is evidence against its own label. If both endpoints use
    the same noun or no quote is available, do not invent an inference.
    """
    by_probe = {p["id"]: p for p in probes}
    paths = {}
    for record in records:
        p = by_probe.get(record.get("id"))
        if (p and p["kind"] == "relation_path"
                and p.get("edge_id") and record.get("answer") in ("A_TO_B", "B_TO_A")
                and record.get("evidence_status", "verified") == "verified"):
            paths[p["edge_id"]] = record
    conflicts = {}
    for record in records:
        p = by_probe.get(record.get("id"))
        if (not p or p["kind"] != "condition_owner"
                or not p.get("edge_id") or record.get("answer") not in ("A", "B")
                or record.get("evidence_status", "verified") != "verified"):
            continue
        path = paths.get(p["edge_id"])
        if not path:
            continue
        owner = record.get("owner", "")
        by_role = {
            path.get("subject_role"): path.get("subject", ""),
            path.get("object_role"): path.get("object", ""),
        }
        claimed = record["answer"]
        other = "B" if claimed == "A" else "A"
        # A lexical match to ONLY the other endpoint is a candidate-side
        # contradiction; don't interpret one expected label as evidence.
        if (_matches_endpoint(owner, by_role.get(other, ""))
                and not _matches_endpoint(owner, by_role.get(claimed, ""))):
            conflicts[record["id"]] = "owner_quote_matches_opposite_relation_endpoint"
    return conflicts


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
    probes: list[dict], records: list[dict], manifest: Any, *, candidate: str | None = None,
) -> None:
    """Stage-3 replay of the full required set and exact model decisions."""
    validate_candidate_probe_records(probes, records, candidate=candidate)
    if conflicting_endpoint_evidence(probes, records):
        raise ValueError("stage2_owner_relation_evidence_conflict")
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
    "validate_coverage_manifest", "conflicting_endpoint_evidence",
]
