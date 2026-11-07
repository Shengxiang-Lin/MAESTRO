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


_AUTHOR_SYSTEM = """You are naturalizing a compiler-checked mechanical
question. Your only job is to make its English easier to read WITHOUT
changing which entities qualify or which answer is requested.

Use conservative, local wording edits. Keep clear numbered/grouped steps
when necessary; a readable multi-sentence question is better than a short
but ambiguous one. Do not shorten just for its own sake.

Before writing, carefully preserve these semantic commitments:
- "in both A and B" = SAME entity must satisfy A AND B. Never reword as
  "either", "one of", "two types of entities", or "A and B entities" when
  that describes a union. "At least one of A or B" is an OR.
- Every branch belongs to its original candidate pool. A filter applied
  later to a subset must NOT be inserted into the definition of an earlier
  group. An alternative branch from all relevant records must NOT be
  restricted to the first/second group.
- For "entities whose recorded RELATION points to X", keep the chosen
  entity, relationship, direction, and X distinct. For "entities listed
  as targets of RELATION from Y", keep the target/source direction.
- Keep filters on the SAME entity as in the reference. In particular,
  "organizations targeted by a relation from EVENTS WITH A LOGO" does
  not allow moving the logo onto the organizations. "Locations of taxa"
  are not the taxa themselves.
- Preserve every date, numeric comparison, code, negation, conditional
  empty case, return type, ranking direction and tie rule.

Prefer retaining precise recorded relation words in quotes and using
clearer transitions between steps. Naturalization does not require new
logical phrasing. A modest improvement is sufficient.

If previous_draft and reviewer_feedback are supplied, PATCH ONLY the
local difference described by the Critic. Keep the remaining clauses
unchanged. Do not start again from the reference and do not make
unrequested logical changes.

Return JSON only: {"question": "the complete edited question"}.
"""

_CRITIC_SYSTEM = """You are a semantic copy editor for question naturalization.
The reference is correct; the candidate should be MORE READABLE and
have EXACTLY the SAME answer-defining conditions. You must actively
look for errors before voting PASS. Natural language fluency is NOT
evidence of equivalence.

Make a short, careful reference-to-candidate comparison. Specifically:
1. SAME GROUPS / LOGIC? "in both" means one entity in A AND B;
   "at least one" means A OR B. Check each group independently, even
   when the candidate puts 'both' in one clause and 'either' in another.
   "two types", "A and B", "includes X and Y" can denote a UNION rather
   than the required INTERSECTION. Never approve merely because all
   condition words are present.
2. SAME FILTER OWNERS? For each number, date, logo, population etc.,
   identify which entity it modifies in BOTH versions. Watch for filters
   moving from event to organization, from subject to related place, or
   from an answer entity to a relation target.
3. SAME RELATION PATH AND ENTITY TYPE? Check the subject, target and
   direction of each "points to" or "targets of ... from" relation.
   Locations of taxa are not taxa. Do not turn a recorded relation
   into simple shared names or a direct attribute.
4. SAME POOLS? A reference "A drawn from group G OR B from all records"
   does not allow the candidate to require B to belong to G. Check
   earlier group definitions are not narrowed by downstream filters.
5. SAME ANSWER? Preserve required values, comparisons, negation,
   conditional false/empty cases, ranking/max/min/ties and answer type.

Calibration examples (these are NOT the user's question):
- Reference: "entities in BOTH entities with height H and entities
  with width W." Candidate: "entities with EITHER height H OR width W."
  This is CHANGED (AND became OR), even though both values remain.
- Reference: "organizations targeted by relation R FROM EVENTS with
  logo L." Candidate: "organizations targeted by R from events, with
  logo L." Do not silently PASS if the logo is now on organizations;
  if its attachment is genuinely unresolved, choose UNCERTAIN.
- Reference: "A from group G OR B in the relevant records."
  Candidate: "A or B from group G." This is CHANGED: B was restricted.
- Conversely, "entities that have H and have W" faithfully expresses
  intersection; merely combining clauses with AND is NOT an error.

A clear semantic difference -> REVISE and at most THREE localized edits.
For each, quote a short ACTUAL excerpt from reference AND candidate,
explain which logical operator, entity owner, direction, or candidate
pool changed, and give the SMALLEST correction to the previous draft.
Do not invent missing requirements or demand identical wording.
An ambiguous but consequential attachment -> UNCERTAIN; do not guess.
A modest improvement in English is acceptable; do not force extensive
rewriting. A merely unchanged or unreadable candidate can be REVISE
with naturalness='needs_polish'.

PASS ONLY AFTER the above comparisons do not reveal a discrepancy.
For PASS, differences=[] and the summary should mention a concrete
preserved constraint, NOT a generic "all semantics preserved" claim.
A false PASS is worse than requesting one localized correction.
Still, do not reject a faithful reordering or faithful AND expression.

Return SHORT JSON only:
{"verdict":"pass|revise|uncertain",
 "naturalness":"acceptable|needs_polish",
 "differences":[{"reference_excerpt":"...", "candidate_excerpt":"...",
                 "problem":"...", "minimal_fix":"..."}],
 "summary":"..."}
For PASS differences must be empty. REVISE requires a real localized
difference or genuine readability issue. Do not add any other keys.
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
