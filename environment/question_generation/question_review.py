"""Benchmark-style question authoring with one local-difference Critic.

The immutable public delegation blueprint supplies goals and a compositional
semantic contract. The mechanical reference is a literal-spelling fallback,
NOT the target style. No parser-like probes or independent final judge.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class ReviewFormatError(ValueError):
    """A model response violates the editorial protocol."""


_AUTHOR_SYSTEM = """Write ONE realistic, natural benchmark question for an
autonomous information-seeking/multimodal agent. The immutable public task
contract states what the question must mean. A delegation blueprint's Goals
are deliberately SHORT INTENT SUMMARIES and may omit essential numeric
thresholds, dates, filters and relation hops. Never treat those omissions
as permission to drop constraints from the final Question. The immutable
public_steps and precise reference have priority over Goal summaries.
The blueprint describes how independent Sub-Agents could work, but must
NOT be reproduced as a numbered list. The mechanical reference is ONLY a fallback for
public names, exact values and ambiguous relation wording.

Start with the user's final information need, then integrate the clues
naturally. Prefer a compact investigation, comparison, identification or
calculation question, as appropriate. Avoid 'Let the first group consist
of...' and numbered operator descriptions. Clear multi-sentence requests
are acceptable if the logic is complicated. Naturalness is meaningful
rewriting, not inserting 'that', 'relation' or punctuation into the original.

Preserve ALL answer-defining semantics: directed relation hops; attributes
on their original entities; shared-entity AND vs branch-wise OR; independent
candidate pools; dates, numeric comparisons, negations, ranking, ties,
output type and conditional empty results. 'both A and B' means one
candidate satisfies BOTH, not 'two types of' candidates. Do not make
an entity into a group containing that entity. Do not erase a relation
hop simply because its endpoints have the right types.

Only mention media if the public modalities actually include it. A
discoverable video/document is not a user-attached file. Never invent
the media's contents, a real-world story, source title, unknown entities,
intermediate results, answer, or additional evidence requirements.
Do not expose internal graph IDs, task splits or model tools.

When given a previous_draft plus reviewer_feedback, edit the previous
question by making the smallest repairs to its specified semantic problems.
If only the style is weak, make a more user-facing rephrase WITHOUT changing
any task semantics. Return JSON {"question":"..."} only.
"""

_AUTHOR_BRANCH_SYSTEM = """Write ONE natural, answer-preserving user Question
from the provided PUBLIC branch-level Question Brief.

Start from answer_scope and follow its inputs. Each scope owns its own
conditions. A RELATE edge reads subject --relation--> object, regardless
of which endpoint is an input. Express every required edge, including
intermediate links. An identity clue identifies one entity by its
conditions. Keep all exact public values and their comparisons.

INTERSECT requires the SAME entity to satisfy ALL input branches. UNION
includes entities in ANY branch. SELECT_BETWEEN ranks over the union of
its inputs. Keep answer type, ranking direction, ties, and distinct counts.
The subtask intents are optional context. The scopes define the question.

Combine conditions on the same entity in one clause when this preserves
the meaning. Use clear antecedents or multiple sentences for longer paths.
Do not expose scope IDs, the execution graph, internal groups, or tools.
Use the supplied public modalities to distinguish attached evidence from
discoverable evidence. Never invent a source or a concealed identity.

When given previous_draft and reviewer_feedback, make the smallest
correction that preserves every required constraint.
Return exactly one JSON object: {"question":"..."}.
"""


_CRITIC_SYSTEM = """Act as a semantic reviewer of a natural benchmark question.
The public semantic contract is authoritative; the delegation blueprint
contains only HIGH-LEVEL subtask intents that may deliberately omit values,
comparisons, filters, relation hops and other exact constraints. Never
use missing details from a Goal to justify omissions in the full Question.
Review whether a user solving the candidate would seek EXACTLY the same
answer under the same complete public contract and filters,
not whether it resembles the mechanical reference.

Before PASS, compare: (1) every required relation hop and its direction;
(2) which entity owns each number/date/attribute; (3) intersection (SAME
entity must satisfy every branch) vs union, including independent candidate
pools; (4) entity types, named singleton anchors and evidence discovery
requirements; (5) final answer type, rank/ties and empty/negative cases.
Check that the question's goal can be fulfilled by the blueprint's terminal
subtask and that it does not demand an unrelated extra step.

A candidate can be equivalent without listing every internal Sub-Agent
task, naming tools or duplicating source acquisition steps. Do not require
the reader to follow the hidden graph. A concise user-facing question is
better than a copied mechanical query. A 'group containing X' is not
equivalent to the singleton X unless expressly defined as that one entity.
Do not add unsupported real-world facts or claim an unprovided source is
attached. Do not assume a final answer matches merely because all words
or numeric values appear.

For a clear semantic discrepancy, REVISE with at most THREE local
differences and minimal repair instructions, quoting short ACTUAL spans
from the reference (if present) and candidate. For a missing clause,
candidate_excerpt may be an adjacent actual phrase. For a genuine
unresolved meaning, UNCERTAIN; never invent a reason for failure.
If correct but mechanical/only cosmetic, REVISE with naturalness
'needs_polish' and give a concrete style request. Do not force rewrites
of already natural, faithful questions.

PASS only when the meaning is preserved AND the phrasing is substantially
more like a user request. No requirement for large edit distance.
Return JSON only:
{"verdict":"pass|revise|uncertain",
 "naturalness":"acceptable|needs_polish",
 "differences":[{"reference_excerpt":"...", "candidate_excerpt":"...",
                 "problem":"...", "minimal_fix":"..."}],
 "summary":"..."}
A PASS requires differences=[]; no other keys.
"""

_AUTHOR_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"], "additionalProperties": False,
}
_DIFF_SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in (
        "reference_excerpt", "candidate_excerpt", "problem", "minimal_fix"
    )},
    "required": [
        "reference_excerpt", "candidate_excerpt", "problem", "minimal_fix",
    ],
    "additionalProperties": False,
}
_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"enum": ["pass", "revise", "uncertain"]},
        "naturalness": {"enum": ["acceptable", "needs_polish"]},
        "differences": {"type": "array", "items": _DIFF_SCHEMA},
        "summary": {"type": "string"},
    },
    "required": ["verdict", "naturalness", "differences", "summary"],
    "additionalProperties": False,
}


def _invoke(backend: Any, system: str, payload: dict, *, schema: dict | None):
    if not callable(getattr(backend, "complete_json", None)):
        raise ReviewFormatError("review_backend_unavailable")
    result = backend.complete_json(
        system, json.dumps(payload, ensure_ascii=False), response_schema=schema,
    )
    if not isinstance(result, dict):
        raise ReviewFormatError("review_json_not_object")
    return result


def _brief(blueprint: Mapping | None):
    if blueprint is None:
        return None
    outline = blueprint["outline"]
    return {
        "task_semantics": outline["task_semantics"],
        "public_modalities": outline["public_modalities"],
        "subtask_goals": blueprint["goals"],
        "subtask_dependencies": [{
            "fragment_id": task["fragment_id"],
            "depends_on": task["depends_on"],
            "public_steps": task["public_steps"],
            "output_kind": task["output_kind"],
        } for task in outline["tasks"]],
    }


def author_question(
    backend: Any, reference: str, *, previous: str | None = None,
    feedback: dict | None = None, blueprint: Mapping | None = None,
    question_brief: Mapping | None = None,
) -> str:
    # Branch mode must NOT accidentally reintroduce the full reference or
    # nested AST to the Author. The independent Critic still receives those.
    if question_brief is not None:
        payload = {"question_brief": dict(question_brief)}
        system = _AUTHOR_BRANCH_SYSTEM
    else:
        payload = {"reference_question": reference}
        system = _AUTHOR_SYSTEM
        if blueprint is not None:
            payload["benchmark_task_brief"] = _brief(blueprint)
            payload["instruction"] = (
                "Write the final user question; do not reproduce the subtask plan."
            )
    if previous is not None:
        payload["previous_draft"] = previous
        payload["reviewer_feedback"] = feedback or {}
        payload["instruction"] = (
            "Patch only verified semantic issues or improve naturalness, "
            "while keeping every exact public constraint."
        )
    response = _invoke(backend, system, payload, schema=_AUTHOR_SCHEMA)
    question = response.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ReviewFormatError("author_question_missing")
    return " ".join(question.split())


def critique_question(
    backend: Any, reference: str, candidate: str,
    *, use_schema: bool = True, blueprint: Mapping | None = None,
) -> dict:
    payload = {
        "reference_question": reference,
        "candidate_question": candidate,
    }
    if blueprint is not None:
        payload["benchmark_task_brief"] = _brief(blueprint)
    if not use_schema:
        payload["format_repair_instruction"] = (
            "Return the four requested JSON fields. If truly unclear, set "
            "verdict to uncertain; do not invent missing clauses."
        )
    answer = _invoke(
        backend, _CRITIC_SYSTEM, payload,
        schema=_CRITIC_SCHEMA if use_schema else None,
    )
    verdict = answer.get("verdict")
    naturalness = answer.get("naturalness")
    diffs = answer.get("differences")
    summary = answer.get("summary")
    if (verdict not in ("pass", "revise", "uncertain")
            or naturalness not in ("acceptable", "needs_polish")
            or not isinstance(diffs, list) or len(diffs) > 3
            or not isinstance(summary, str)):
        raise ReviewFormatError("critic_review_invalid")
    for d in diffs:
        if (not isinstance(d, dict)
                or not all(isinstance(d.get(k), str) for k in (
                    "reference_excerpt", "candidate_excerpt",
                    "problem", "minimal_fix",
                ))):
            raise ReviewFormatError("critic_difference_invalid")
        if not d["problem"].strip() or not d["minimal_fix"].strip():
            raise ReviewFormatError("critic_difference_empty")
    if verdict == "pass" and (diffs or naturalness != "acceptable"):
        raise ReviewFormatError("critic_pass_contradiction")
    if verdict == "revise" and not diffs and naturalness != "needs_polish":
        raise ReviewFormatError("critic_revision_without_local_issue")
    return {
        "verdict": verdict, "naturalness": naturalness,
        "differences": diffs, "summary": summary,
    }


def passing_judgment(review: Mapping[str, Any]) -> bool:
    return (review.get("verdict") == "pass"
            and review.get("naturalness") == "acceptable"
            and review.get("differences") == [])


__all__ = [
    "ReviewFormatError", "author_question", "critique_question",
    "passing_judgment",
]
