"""Minimal Author → Critic → Author local-repair review.

The reference is a compiler-checked PUBLIC question. The Critic is an
imperfect LLM reviewer, not a semantic theorem prover. It must identify
specific textual differences before asking Author to change a sentence.
No risk-group trees, A/B roles, candidate-only probes or final judge.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class ReviewFormatError(ValueError):
    """A model response violates this small editing protocol."""


_AUTHOR_SYSTEM = """You are an editor naturalizing a mechanically written question.
Rewrite the reference so that a real person can read it comfortably.
Make CONSERVATIVE, LOCAL language edits. A clear multi-sentence question
is better than a compact but ambiguous one. You may retain group labels,
technical relation terms, and intermediate steps when they clarify scope.
Do not add assumptions, delete conditions, move attribute filters between
entities, reverse relation directions, swap AND/OR, or change the answer.
Keep exact dates, values, codes, requested answer kind and tie handling.

A modest improvement in readability is enough. Do not optimize for a
large wording difference or for maximum shortening.

If given a previous draft and reviewer feedback, PATCH THE PREVIOUS DRAFT.
Make ONLY the localized corrections indicated in the feedback; preserve
correct wording elsewhere. Do not rewrite the entire question from scratch.
Return JSON {"question": "the complete edited question"}.
"""

_CRITIC_SYSTEM = """You are a conservative semantic copy editor.
You see a correct mechanical reference and one rewritten candidate.
Your primary job is to find REAL CHANGES IN MEANING, not to reproduce
reference phrasing. A reader can express AND using 'that ... and ...',
and use two sentences or group labels without harming naturalness.

Check whether the rewrite still selects the SAME entities and answer:
relation direction, which entity owns each filter, AND vs OR, numbers,
dates/codes, negative/conditional cases, final answer kind, sorting/ties.
Do not assume that a named relation is the same as name equality.
Do not demand identical sentences or exact reference wording.
Prefer PASS when the meaning is preserved and the wording is at least
somewhat more readable. A conservative edit is acceptable.

If a clear semantic error exists, return REVISE and at most THREE
specific local differences. For each, identify a short excerpt from
BOTH texts (copy verbatim where possible), explain exactly which
constraint or entity changed, and give the SMALLEST repair instruction
for the previous candidate. Do not invent missing requirements.
If meaning is genuinely ambiguous, return UNCERTAIN instead of guessing.
If the only issue is that the candidate is literally unchanged or
unreadable, use REVISE with naturalness='needs_polish'.

Return SHORT JSON only:
{"verdict":"pass|revise|uncertain",
 "naturalness":"acceptable|needs_polish",
 "differences":[{"reference_excerpt":"...", "candidate_excerpt":"...",
                 "problem":"...", "minimal_fix":"..."}],
 "summary":"..."}
For PASS, differences must be empty. Do not issue a REVISE without a
concrete difference or a genuine readability problem.
"""


_AUTHOR_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"], "additionalProperties": False,
}
_DIFF_SCHEMA = {
    "type": "object",
    "properties": {
        k: {"type": "string"} for k in (
            "reference_excerpt", "candidate_excerpt", "problem", "minimal_fix"
        )
    },
    "required": ["reference_excerpt", "candidate_excerpt", "problem", "minimal_fix"],
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
    response = backend.complete_json(
        system, json.dumps(payload, ensure_ascii=False),
        response_schema=schema,
    )
    if not isinstance(response, dict):
        raise ReviewFormatError("review_json_not_object")
    return response


def author_question(
    backend: Any, reference: str, *, previous: str | None = None,
    feedback: dict | None = None,
) -> str:
    payload = {"reference_question": reference}
    if previous is not None:
        payload["previous_draft"] = previous
        payload["reviewer_feedback"] = feedback or {}
        payload["instruction"] = (
            "Patch only the localized issues in the previous draft."
        )
    response = _invoke(backend, _AUTHOR_SYSTEM, payload, schema=_AUTHOR_SCHEMA)
    question = response.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ReviewFormatError("author_question_missing")
    return " ".join(question.split())


def critique_question(
    backend: Any, reference: str, candidate: str,
    *, use_schema: bool = True,
) -> dict:
    payload = {
        "reference_question": reference,
        "candidate_question": candidate,
    }
    if not use_schema:
        payload["format_repair_instruction"] = (
            "Respond with the four JSON fields verbatim. If unclear, set "
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
                or not all(isinstance(d.get(k), str) for k in
                           ("reference_excerpt", "candidate_excerpt",
                            "problem", "minimal_fix"))):
            raise ReviewFormatError("critic_difference_invalid")
        if not d["problem"].strip() or not d["minimal_fix"].strip():
            raise ReviewFormatError("critic_difference_empty")
    if verdict == "pass" and (diffs or naturalness != "acceptable"):
        # Contradictory reviews must not authorize publication.
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
