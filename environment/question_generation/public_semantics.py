"""Public-only semantic scaffold projected from the frozen ProgramIR.

This is NOT a serialization of the private ProgramIR. It contains only
operation topology and values already disclosed in the Controlled Renderer
question. Hidden FIND identities, World facts, private answers and source IDs
are never copied to an Author/Critic prompt.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from environment.task_construction.rendering.naming import (
    comparison_phrase, predicate_label, select_mode_phrase, type_label,
)

VERSION = "public_semantic_scaffold.v1"
_ALLOWED = {
    "FIND", "FIND_ALL", "FILTER_TYPE", "FILTER_STRING", "FILTER_NUMBER",
    "FILTER_YEAR", "FILTER_DATE", "RELATE", "INTERSECT", "UNION",
    "COUNT", "QUERY_NAME", "QUERY_ATTR", "QUERY_RELATION",
    "SELECT_AMONG", "SELECT_BETWEEN", "VERIFY_STRING", "VERIFY_NUMBER",
    "VERIFY_YEAR", "VERIFY_DATE",
}


def _public_match(value, canonical):
    """Use an exact public span, or disclose NO private argument."""
    word = str(value).strip()
    if not word or len(word) > 200:
        return None
    alternatives = [word]
    if re.fullmatch(r"\d{4,}", word):
        alternatives.append(f"{int(word):,}")
    for token in alternatives:
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(token) +
                     r"(?![A-Za-z0-9])", canonical, flags=re.I):
            return token
    return None


def _predicate(raw, canonical):
    return _public_match(predicate_label(str(raw)), canonical)


def _value(raw, canonical):
    return _public_match(raw, canonical)


def public_semantic_scaffold(world: Mapping, canonical: str) -> dict:
    """Give Author a safe step graph, without exposing hidden raw operands.

    The complete question remains the textual source for all public names and
    values. If a raw ProgramIR argument was not printed by that question, do
    not insert it into this scaffold.
    """
    if not isinstance(canonical, str) or not canonical.strip():
        raise ValueError("missing_controlled_renderer_question")
    nodes = world["program"]["nodes"]
    known_ids = {int(n["node_id"]) for n in nodes}
    if len(known_ids) != len(nodes):
        raise ValueError("duplicate_program_node")
    steps = []
    for n in nodes:
        op = str(n["op"])
        if op not in _ALLOWED:
            raise ValueError("unsupported_public_semantic_operation")
        inputs = n.get("inputs", [])
        dependencies = [int(v) for v in n.get("dependencies", [])]
        if any(d not in known_ids for d in dependencies):
            raise ValueError("invalid_public_semantic_dependency")
        step = {
            "id": f"S{int(n['node_id'])}",
            "inputs_from": [f"S{d}" for d in dependencies],
            "operation": op,
        }
        if op == "FIND":
            # NEVER emit raw FIND inputs: they can be concealed entity IDs.
            step["goal"] = "Identify the starting entity using the clue in the canonical question."
        elif op == "FIND_ALL":
            step["goal"] = "Discover candidates from the bounded task evidence; no pre-supplied complete roster."
        elif op == "FILTER_TYPE":
            label = type_label(str(inputs[0]))
            step["entity_type"] = _public_match(label, canonical)
            step["goal"] = "Keep matching entities of this type; do not transfer constraints between entities."
        elif op.startswith("FILTER_"):
            step.update(
                attribute=_predicate(inputs[0], canonical),
                comparison=comparison_phrase(str(inputs[1])),
                public_value=_value(inputs[2], canonical),
                goal="Filter the entities in inputs_from by THEIR OWN attribute.",
            )
        elif op == "RELATE":
            direction = str(inputs[1])
            if direction not in ("forward", "backward"):
                raise ValueError("invalid_relation_direction")
            step.update(
                relation=_predicate(inputs[0], canonical),
                direction=direction,
                goal=(
                    "Find objects y of x --relation--> y, where x is from inputs_from."
                    if direction == "forward" else
                    "Find subjects x of x --relation--> y, where y is from inputs_from."
                ),
            )
        elif op in ("INTERSECT", "UNION"):
            step["goal"] = (
                "The same entities must satisfy ALL input branches."
                if op == "INTERSECT" else
                "Entities satisfying ANY input branch are included."
            )
        elif op in ("SELECT_AMONG", "SELECT_BETWEEN"):
            step.update(
                rank_attribute=_predicate(inputs[0], canonical),
                ranking=select_mode_phrase(inputs[1]),
                goal="Rank entities across the candidate pool, including ties; do not turn this into a threshold filter.",
            )
        elif op == "QUERY_ATTR":
            step.update(
                output_attribute=_predicate(inputs[0], canonical),
                goal="Return the recorded attribute VALUES, not entity names.",
            )
        elif op == "QUERY_NAME":
            step["goal"] = "Return all matching entity NAMES."
        elif op == "COUNT":
            step["goal"] = "Count distinct matching entities."
        elif op.startswith("VERIFY_"):
            step.update(
                comparison=comparison_phrase(str(inputs[0])),
                public_value=_value(inputs[1], canonical),
                goal="Check whether ANY recorded value meets the comparison.",
            )
        elif op == "QUERY_RELATION":
            step["goal"] = "Return relation names between the two input entity groups."
        steps.append(step)
    return {
        "version": VERSION,
        "scope": "ONLY public constraints already expressed in canonical_question",
        "output": steps[-1]["id"] if steps else None,
        "steps": steps,
        "guidance": (
            "Follow all dependencies; preserve predicate direction, attribute "
            "ownership, logical joins, exact public values and requested output. "
            "A null label/value means consult the canonical question; do not guess."
        ),
    }


def naturalization_metrics(canonical: str, candidate: str) -> dict:
    """Readability proxy only; NEVER a semantic-quality certificate."""
    from difflib import SequenceMatcher
    boilerplate = (
        "entities drawn from", "candidates in", "entities for which",
        "after requiring", "entities connected from", "entities connected into",
        "entities appearing in at least one of", "entities meeting both sides of",
        "what entity or entities are selected",
    )
    def count(text):
        lower = text.lower()
        return sum(lower.count(term) for term in boilerplate)
    original_count = count(canonical)
    candidate_count = count(candidate)
    similarity = round(SequenceMatcher(
        None, canonical, candidate, autojunk=False,
    ).ratio(), 4)
    # Do not confuse superficial additions with improved natural phrasing.
    meaningful = (
        candidate != canonical
        and (original_count - candidate_count >= 2
             or (similarity < .85 and len(candidate.split()) <= len(canonical.split()) * 1.2))
    )
    return {
        "character_similarity": similarity,
        "canonical_words": len(canonical.split()),
        "surface_words": len(candidate.split()),
        "canonical_template_phrases": original_count,
        "surface_template_phrases": candidate_count,
        "meaningful_rewrite_proxy": meaningful,
    }


__all__ = ["VERSION", "public_semantic_scaffold", "naturalization_metrics"]
