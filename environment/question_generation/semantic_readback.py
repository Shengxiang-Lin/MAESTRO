"""Blind, text-only semantic readback for free-form Stage-2 questions.

Unlike the closed Scoped Grammar's round-trip, this reader never receives
ProgramIR, reference scopes, expected answers, or a previous model response.
It reconstructs a typed expression from the QUESTION TEXT ALONE. Two isolated
passes are required; a deterministic comparator then checks it against the
public Question Contract. This is conservative model-assisted verification,
not a formal natural-language equivalence proof.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import typed_values
from .scoped_surface import ScopedPlan

READBACK_VERSION = "blind_semantic_readback.v1"

_KINDS = {
    "FIND", "FIND_ALL", "FILTER_TYPE", "FILTER_STRING", "FILTER_NUMBER",
    "FILTER_YEAR", "FILTER_DATE", "RELATE", "INTERSECT", "UNION",
    "COUNT", "QUERY_NAME", "QUERY_ATTR", "QUERY_RELATION",
    "SELECT_AMONG", "SELECT_BETWEEN", "VERIFY_STRING", "VERIFY_NUMBER",
    "VERIFY_YEAR", "VERIFY_DATE",
}
_FIELDS = {
    "definition", "conditions", "inputs", "relation", "direction",
    "ranking", "rank_attribute", "output_attribute", "anchor_type", "anchor",
    "type_anchor", "range", "comparison", "value", "quantifier",
    "include_ties", "evidence",
}
_CONDITION_FIELDS = {"entity_type", "attribute", "comparison", "value", "evidence"}
_TOKENS = {
    "equal to": "equal to", "=": "equal to", "equals": "equal to",
    "greater than": "greater than", ">": "greater than",
    "less than": "less than", "<": "less than",
    "at least": "at least", ">=": "at least",
    "at most": "at most", "<=": "at most",
    "different from": "different from", "!=": "different from",
}


# A schema steers the 9B reader away from inventing keys and emitting untyped
# values. OpenAI-compatible servers without recursive JSON-schema support
# already fall back to ordinary JSON via OpenAICompatibleBackend.
_READBACK_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"$ref": "#/$defs/node"},
        "uncertain": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["query", "uncertain", "reason"],
    "additionalProperties": False,
    "$defs": {
        "value": {
            "type": "object",
            "properties": {
                "type": {"enum": ["date", "year", "number", "string"]},
                "value": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["type", "value"],
            "additionalProperties": False,
        },
        "condition": {
            "type": "object",
            "properties": {
                "entity_type": {"type": "string"},
                "attribute": {"type": "string"},
                "comparison": {"type": "string"},
                "value": {"$ref": "#/$defs/value"},
                "evidence": {"type": "string"},
            },
            "required": ["evidence"],
            "additionalProperties": False,
        },
        "node": {
            "type": "object",
            "properties": {
                "definition": {"enum": sorted(_KINDS)},
                "inputs": {"type": "array", "items": {"$ref": "#/$defs/node"}},
                "conditions": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/condition"},
                },
                "evidence": {"type": "string"},
                "relation": {"type": "string"},
                "direction": {"enum": ["forward", "backward"]},
                "ranking": {"type": "string"},
                "rank_attribute": {"type": "string"},
                "output_attribute": {"type": "string"},
                "anchor_type": {"type": "string"},
                "anchor": {"type": "string"},
                "type_anchor": {"type": "string"},
                "range": {"type": "string"},
                "comparison": {"type": "string"},
                "value": {"$ref": "#/$defs/value"},
                "quantifier": {"type": "string"},
                "include_ties": {"type": "boolean"},
            },
            "required": ["definition", "inputs", "conditions", "evidence"],
            "additionalProperties": False,
        },
    },
}

_SYSTEM = """You are a semantic parser, NOT a reviewer and NOT a solver.
Read the supplied English question only. No ProgramIR, reference contract,
ground-truth answer, generated grammar or previous parse is available.
Reconstruct EXACTLY the explicit information demand as a nested typed query
expression. Do not infer missing filters or facts from outside knowledge.
If a reference, relation direction, filter owner, set join, or requested
output is ambiguous, set uncertain=true. Never guess the omitted detail.

Supported definition operators: FIND, FIND_ALL, RELATE, FILTER_TYPE,
FILTER_STRING, FILTER_NUMBER, FILTER_YEAR, FILTER_DATE, INTERSECT, UNION,
QUERY_NAME, COUNT, QUERY_ATTR, QUERY_RELATION, SELECT_AMONG,
SELECT_BETWEEN, VERIFY_STRING, VERIFY_NUMBER, VERIFY_YEAR, VERIFY_DATE.
Represent chained unary filters as "conditions" on the entity group they
modify, not as invented restrictions on other entities. Every nested node
has "definition", "inputs" (ordered child expressions), "conditions"
(list), and an EXACT short "evidence" quotation copied from the question.
Conditions have either "entity_type", or "attribute"/"comparison"/"value",
and their own short exact "evidence" quotations. Do not insert operation
numbers, program node IDs or supposed gold answers.

RELATE means: for forward, the child is the subjects x and result is the
objects y of x --relation--> y; for backward, the child is the objects y
and result is subjects x. Always identify endpoint direction, not merely
that two entities are connected.
FIND is an explicitly identified anchor or hidden anchor described by
an identifying clue ("anchor", "type_anchor", or "anchor_type").
FIND_ALL means a task-evidence candidate domain, not a supplied roster;
use range="bounded_public_task_evidence". UNION means ANY branch,
INTERSECT means ALL branches for the SAME result entity.
COUNT means the number of distinct entities; QUERY_NAME returns names;
VERIFY checks whether any recorded attribute meets the claim.
Preserve codes including leading + or zero as STRING; dates as YYYY-MM-DD;
numbers as number-valued objects. Use exactly:
{"type":"date","value":"YYYY-MM-DD"}, {"type":"number","value":"1203","unit":""},
{"type":"year","value":"1997","unit":""},
or {"type":"string","value":"+650"}.
Normalize comparisons to: equal to, different from, greater than,
less than, at least, at most.
Use precisely the public relation and attribute labels as ordinary
English terms, not invented YAGO identifiers. Omit optional fields unless
the text explicitly supports them. Set uncertain=true rather than forcing
an interpretation of ambiguous pronouns, ownership, or joins.
Output one JSON object only: {"query":{...}, "uncertain":false, "reason":""}.
"""

_SECOND_SYSTEM = _SYSTEM + """
Second-pass instructions: work bottom-up from the final information demand.
Resolve every pronoun and named group from the actual words. Do not preserve
a first-pass interpretation: none is supplied. Treat omitted intermediate
links, swapped source/target roles, and missing constraints as substantial.
"""

def _clean_text(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def _comparison(value: Any) -> str:
    result = _TOKENS.get(_clean_text(value))
    if result is None:
        raise ValueError("unsupported_comparison")
    return result


def _normalized_value(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("readback_value_must_be_typed")
    if set(value) - {"type", "value", "unit"}:
        raise ValueError("readback_value_has_unknown_fields")
    kind = value.get("type")
    if kind not in ("date", "year", "number", "string"):
        raise ValueError("readback_value_invalid_type")
    raw = str(value.get("value") or "")
    if not raw:
        raise ValueError("readback_value_missing")
    if kind == "date":
        from datetime import date
        result = date.fromisoformat(raw).isoformat()
        return {"type": kind, "value": result}
    if kind in ("year", "number"):
        from decimal import Decimal, InvalidOperation
        try:
            numeric = Decimal(raw.replace(",", ""))
        except InvalidOperation as exc:
            raise ValueError("readback_value_invalid_number") from exc
        unit = str(value.get("unit") or "")
        return {"type": kind, "value": str(numeric.normalize()), "unit": unit}
    return {"type": kind, "value": raw}


def normalize_query(tree: Any) -> dict:
    """Exact typed graph normalization; no inferred missing nodes/operands."""
    if not isinstance(tree, Mapping):
        raise ValueError("readback_query_must_be_object")
    if set(tree) - _FIELDS:
        raise ValueError("readback_query_unknown_fields")
    op = tree.get("definition")
    if op not in _KINDS:
        raise ValueError("readback_query_invalid_operation")
    children = tree.get("inputs", ())
    conditions = tree.get("conditions", ())
    if not isinstance(children, list) or not isinstance(conditions, list):
        raise ValueError("readback_query_invalid_children_or_conditions")
    normalized = {"definition": op, "inputs": [normalize_query(c) for c in children]}
    normalized_conditions = []
    for c in conditions:
        if not isinstance(c, Mapping) or set(c) - _CONDITION_FIELDS:
            raise ValueError("readback_condition_invalid")
        if "entity_type" in c:
            if set(c) - {"entity_type", "evidence"}:
                raise ValueError("mixed_type_and_attribute_condition")
            normalized_conditions.append({"entity_type": _clean_text(c["entity_type"])})
        else:
            if not all(k in c for k in ("attribute", "comparison", "value")):
                raise ValueError("incomplete_readback_condition")
            normalized_conditions.append({
                "attribute": _clean_text(c["attribute"]),
                "comparison": _comparison(c["comparison"]),
                "value": _normalized_value(c["value"]),
            })
    normalized["conditions"] = sorted(
        normalized_conditions, key=lambda c: json.dumps(c, sort_keys=True),
    )
    for key in (
        "relation", "ranking", "rank_attribute", "output_attribute",
        "anchor_type", "anchor", "type_anchor",
    ):
        if key in tree and tree[key] is not None:
            if not isinstance(tree[key], str) or not tree[key].strip():
                raise ValueError("readback_invalid_string_field")
            normalized[key] = _clean_text(tree[key])
    for key in ("direction", "range", "quantifier"):
        if key in tree and tree[key] is not None:
            normalized[key] = str(tree[key]).strip()
    if "comparison" in tree and tree["comparison"] is not None:
        normalized["comparison"] = _comparison(tree["comparison"])
    if "value" in tree and tree["value"] is not None:
        normalized["value"] = _normalized_value(tree["value"])
    if "include_ties" in tree and tree["include_ties"] is not None:
        if type(tree["include_ties"]) is not bool:
            raise ValueError("readback_include_ties_not_boolean")
        normalized["include_ties"] = tree["include_ties"]
    if op in ("INTERSECT", "UNION"):
        if len(normalized["inputs"]) < 2:
            raise ValueError("readback_join_requires_branches")
        normalized["inputs"].sort(key=lambda c: json.dumps(c, sort_keys=True))
    return normalized


def _quote_is_real(quote: Any, question: str) -> bool:
    if not isinstance(quote, str) or not 3 <= len(quote.strip()) <= 240:
        return False
    return _clean_text(quote) in _clean_text(question)


def _claim_has_value(quote: str, expected: dict) -> bool:
    kind = expected["type"]
    if kind == "date":
        return ("date", expected["value"]) in typed_values(quote)
    if kind in ("year", "number"):
        from decimal import Decimal
        return ("number", Decimal(expected["value"])) in typed_values(quote)
    return expected["value"].casefold() in quote.casefold()


def verify_text_grounding(parsed_tree: Any, question: str) -> None:
    """Require actual textual witnesses for EVERY expressed logical atom."""
    def walk(tree):
        if not isinstance(tree, Mapping):
            raise ValueError("readback_unstructured_tree")
        evidence = tree.get("evidence")
        if not _quote_is_real(evidence, question):
            raise ValueError("readback_node_evidence_not_in_question")
        for condition in tree.get("conditions", ()):
            quote = condition.get("evidence") if isinstance(condition, Mapping) else None
            if not _quote_is_real(quote, question):
                raise ValueError("readback_condition_evidence_not_in_question")
            if "value" in condition and not _claim_has_value(
                quote, _normalized_value(condition["value"]),
            ):
                raise ValueError("readback_condition_value_not_in_quote")
        if "value" in tree and tree["value"] is not None and not _claim_has_value(
            evidence, _normalized_value(tree["value"]),
        ):
            raise ValueError("readback_terminal_value_not_in_quote")
        for child in tree.get("inputs", ()):
            walk(child)
    walk(parsed_tree)


@dataclass(frozen=True)
class ReadbackResult:
    eligible: bool
    reason: str
    parsed_query: dict | None
    calls: int
    audits: tuple[dict, ...]
    raw_responses: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {
            "eligible": self.eligible, "reason": self.reason,
            "parsed_query": self.parsed_query, "calls": self.calls,
            "audits": list(self.audits),
            "raw_responses": list(self.raw_responses),
            "version": READBACK_VERSION,
        }


def validate_readback_response(response: Any, question: str, expected: Any) -> dict:
    """Fail closed on unknown text, uncertain meaning, or structural mismatch."""
    if not isinstance(response, Mapping) or response.get("uncertain") is not False:
        raise ValueError("readback_uncertain_or_missing")
    parsed = response.get("query")
    verify_text_grounding(parsed, question)
    actual = normalize_query(parsed)
    target = normalize_query(expected)
    if actual != target:
        raise ValueError("readback_typed_semantics_mismatch")
    return actual


def read_free_question(question: str, reference: ScopedPlan, backend: Any,
                       *, passes: int = 2) -> ReadbackResult:
    """Run isolated text-only parses, then compare against the held-out tree.

    The reference expression is NEVER sent to the backend. It is consulted
    only by the deterministic comparison after the model returns each parse.
    """
    if not isinstance(reference, ScopedPlan):
        raise TypeError("independent readback requires a compiled ScopedPlan")
    if passes != 2:
        raise ValueError("two independent parse passes are mandatory")
    audits: list[dict] = []
    raw: list[dict] = []
    first = None
    for index, system in enumerate((_SYSTEM, _SECOND_SYSTEM), start=1):
        try:
            # No program, canonical, scope labels, answer or previous parse.
            response = backend.complete_json(
                system, json.dumps({"question": question}, ensure_ascii=False),
                response_schema=_READBACK_RESPONSE_SCHEMA,
            )
            # Retain failed parses in the private Stage-2 diagnostic audit.
            # Previously every first-pass rejection erased the raw response,
            # so protocol mistakes could not be distinguished from drift.
            raw.append(response)
            actual = validate_readback_response(response, question, reference.query)
            if first is not None and actual != first:
                raise ValueError("independent_readback_passes_disagree")
            first = actual
            audits.append({"pass": index, "status": "verified"})
        except Exception as exc:
            # The actual candidate is not approved on parser failures.
            from .backend import BackendError
            if not isinstance(exc, (BackendError, ValueError, TypeError, KeyError)):
                raise
            diagnostic = {
                "pass": index, "status": "failed",
                "reason": str(exc)[:200],
            }
            if raw and isinstance(raw[-1], dict):
                diagnostic["uncertain"] = raw[-1].get("uncertain")
                diagnostic["root_keys"] = sorted(map(str, raw[-1].keys()))
                query = raw[-1].get("query")
                if isinstance(query, dict):
                    diagnostic["query_keys"] = sorted(map(str, query.keys()))
            audits.append(diagnostic)
            return ReadbackResult(
                False, str(exc)[:200], None, index, tuple(audits), tuple(raw),
            )
    return ReadbackResult(True, "independent_typed_readback_matched",
                          first, 2, tuple(audits), tuple(raw))


__all__ = [
    "READBACK_VERSION", "ReadbackResult", "normalize_query",
    "read_free_question", "validate_readback_response",
    "verify_text_grounding",
]
