"""Free-form rewrite on top of a checked scoped question, with no gold data."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_REWRITE_SYSTEM = """You are writing a natural benchmark question for a real person.
The supplied scoped question is already a complete and carefully validated
public information need. Rewrite it into natural, clear English. You may
merge clauses, reorder the exposition, and replace artificial 'first group /
second group' language with explicit referents such as 'those places' or
'the people identified above'.

You MUST preserve the exact answer semantics, including:
* every public date, code, attribute restriction and named starting clue;
* which person, location or intermediate result owns each restriction;
* the direction and full chain of every relation;
* whether groups intersect (same entities satisfy ALL) or union (ANY);
* whether the output asks for names, count, values, or verification.
Do NOT add 'other', 'unique', 'only', extra existence assumptions or
unobserved facts. Do NOT collapse a multi-hop path into a direct link.
Do NOT reveal internal program steps, task IDs, resource IDs, source IDs,
or answers. Never replace a +prefixed identifier with a numeric value.
Avoid "Let the first group consist of", "Define the second group", "entities
drawn from", and "whose recorded relation points to" when natural English
can communicate the SAME exact relation.

Favor one focused question, preceded by at most one short clarifying
sentence if nested dependencies would otherwise be ambiguous. An unclear
reference is worse than a slightly longer faithful question.
Return JSON only: {"question":"..."}.
"""

def free_rewrite_prompt(scoped_question: str, *, feedback: str | None = None) -> str:
    """Only a certified public text is sent; no program or hidden gold."""
    data = {"scoped_question": scoped_question}
    if feedback:
        data["retry_guidance"] = feedback
    return json.dumps(data, ensure_ascii=False)


def call_free_rewriter(author: Any, scoped_question: str,
                       *, feedback: str | None = None) -> str:
    """Ask the Author for open text, not a selection among grammar options."""
    from .backend import BackendError, LLMQuestionAuthor
    if not isinstance(author, LLMQuestionAuthor):
        raise BackendError("free rewrite needs a configured LLMQuestionAuthor")
    response = author.backend.complete_json(
        _REWRITE_SYSTEM, free_rewrite_prompt(scoped_question, feedback=feedback),
    )
    if not isinstance(response, dict) or not isinstance(response.get("question"), str):
        raise BackendError("free rewrite did not return a question string")
    return " ".join(response["question"].split())


_NEW_SCOPE_RESTRICTIONS = re.compile(
    r"\b(?:other|another|except|excluding|unless|only|exactly|unique|"
    r"uniquely|besides|rather\s+than)\b", re.I,
)


def public_rewrite_guard(scoped: str, candidate: str) -> list[str]:
    """Conservative literal/new-restriction check before expensive readback.

    Scope ownership, relations and joins remain the BLIND parser's job.
    Dates are normalized to calendar days; +codes are string tokens, not
    integers. This does not claim to recognize every English paraphrase.
    """
    from .contracts import typed_values
    original = typed_values(scoped, words=True)
    observed = typed_values(candidate, words=False)
    issues = []
    if observed - original:
        issues.append("free_rewrite_invents_or_changes_public_literal")
    original_limits = {
        match.group().casefold()
        for match in _NEW_SCOPE_RESTRICTIONS.finditer(scoped)
    }
    candidate_limits = {
        match.group().casefold()
        for match in _NEW_SCOPE_RESTRICTIONS.finditer(candidate)
    }
    if candidate_limits - original_limits:
        issues.append("free_rewrite_adds_exclusion_or_uniqueness")
    return issues


_GROUP_LANGUAGE = re.compile(
    r"\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth)\s+group\b"
    r"|\b(?:let|define)\s+the\s+(?:reference|first|second|third|fourth)\s+"
    r"|\bentities\s+(?:in\s+both|drawn\s+from|in\s+at\s+least\s+one)\b",
    re.I,
)


@dataclass(frozen=True)
class StyleAssessment:
    improved: bool
    scoped_words: int
    candidate_words: int
    scoped_program_phrases: int
    candidate_program_phrases: int
    reason: str

    def to_dict(self) -> dict:
        return {
            "improved": self.improved,
            "scoped_words": self.scoped_words,
            "candidate_words": self.candidate_words,
            "scoped_program_phrases": self.scoped_program_phrases,
            "candidate_program_phrases": self.candidate_program_phrases,
            "reason": self.reason,
        }


def assess_free_style(scoped: str, candidate: str) -> StyleAssessment:
    """Cheap *quality* gate; independent semantic verification is still required."""
    from difflib import SequenceMatcher
    old = len(_GROUP_LANGUAGE.findall(scoped))
    new = len(_GROUP_LANGUAGE.findall(candidate))
    old_words, new_words = len(scoped.split()), len(candidate.split())
    # Significant gains without rewarding a one-word cosmetic edit.
    less_programmatic = old - new >= 1 and new <= old // 2
    much_shorter = new_words <= max(10, int(old_words * 0.80))
    similarity = SequenceMatcher(None, scoped, candidate, autojunk=False).ratio()
    improved = (
        candidate != scoped and similarity < 0.98
        and new_words <= int(old_words * 1.25)
        and (less_programmatic or much_shorter)
    )
    reason = (
        "natural_language_gain" if improved
        else "rewrite_not_clearly_more_natural"
    )
    return StyleAssessment(improved, old_words, new_words, old, new, reason)


__all__ = [
    "StyleAssessment", "assess_free_style", "call_free_rewriter",
    "free_rewrite_prompt", "public_rewrite_guard",
]
