"""Main-visible semantic checks for Stage-4C reasoning.

These guards use only the *public* evidence requests generated from already
returned Sub-Agent feedback. No private fault type, Gold or future result can
justify an asserted discrepancy. They deliberately target high-precision
contradictions, leaving ordinary narrative evaluation to the LLM Critic.
"""
from __future__ import annotations

import re


_UNCERTAIN = re.compile(
    r"\b(?:conflict(?:ing)?|contradic(?:tion|tory)|disagree(?:ment)?|"
    r"inconsisten(?:t|cy)|discrepan(?:t|cy)|mismatch(?:ed)?|"
    r"unresolved|uncertain|ambiguous|verify|verification|recheck|"
    r"cross-check|reconcile|validate|resolve|re-examin\w*|"
    r"re-validat\w*|compar\w*|disjoint|unaligned|not\s+(?:yet\s+)?confirmed|"
    r"needs?\s+(?:another\s+)?check)\b", re.I,
)
_FALSE_FINALITY = (
    re.compile(
        r"\b(?:previous|prior|earlier|last)\s+(?:round|step|attempt|"
        r"investigation)\s+(?:has\s+)?successfully\s+"
        r"(?:resolved|reconciled|verified)\b", re.I,
    ),
    re.compile(
        r"\b(?:the|this)\s+(?:conflict|discrepancy|mismatch|"
        r"inconsistency|contradiction)\s+"
        r"(?:has\s+been|was|is)\s+(?:fully\s+)?"
        r"(?:resolved|reconciled|verified)\b", re.I,
    ),
    re.compile(
        r"\b(?:no|without\s+any|no\s+remaining)\s+"
        r"(?:conflict|discrepancy|mismatch|inconsistency)\b", re.I,
    ),
    re.compile(
        r"\b(?:this|the)\s+(?:joined\s+)?(?:result|join|"
        r"candidate\s+match|binding)\s+(?:is|was|has\s+been)\s+"
        r"(?:fully\s+|already\s+|complete\s+and\s+)?"
        r"(?:verified|confirmed|reconciled)\b", re.I,
    ),
)


def reasoning_evidence_issues(
    reasoning: str, evidence_requests: list[dict]
) -> list[str]:
    """Reject only clearly false certainty about an unresolved VERIFY.

    A statement such as "the report *claims* verification, but its
    identifiers disagree" is not a claim that Main completed verification.
    Avoid deciding whether a particular reported value is true from Gold.
    """
    if not isinstance(reasoning, str):
        return ["reasoning_not_text"]
    verify = [
        item for item in evidence_requests
        if item.get("operation") == "verify_conflicting_evidence"
    ]
    if not verify:
        return []
    # A pending verification cannot justify dropping that verification.
    # This is distinct from a harmless future conditional such as
    # "once independently verified, we can finish".
    if re.search(
        r"\b(?:no\s+(?:need|reason)\s+to|need\s+not)\s+"
        r"(?:verify|recheck|cross-check|reconcile|validate)\b"
        r"|\b(?:we|I)\s+(?:can|should|may)\s+"
        r"(?:now\s+)?(?:complete|finalize|finish)\s+"
        r"(?:the\s+)?(?:task|answer|question)\b",
        reasoning, re.I,
    ):
        return ["premature_completion_without_required_verification"]
    # The current action must motivate the very check being delegated.
    if not _UNCERTAIN.search(reasoning):
        return ["missing_current_conflict_or_verification_motivation"]
    for phrase in re.split(r"(?<=[.!?])\s+", reasoning):
        if not any(pattern.search(phrase) for pattern in _FALSE_FINALITY):
            continue
        # Future conditional ("once this conflict is resolved") is not
        # a past-tense claim that verification already happened.
        if re.search(
            r"\b(?:once|until|if|when|unless|after)\b.{0,90}"
            r"\b(?:conflict|discrepancy|mismatch|inconsistency)\b"
            r".{0,90}\b(?:resolved|reconciled|verified)\b",
            phrase, re.I,
        ):
            continue
        # Quoting a previous Sub-Agent's *claim* is allowed if Main explicitly
        # questions it in the same sentence.
        if (re.search(r"\b(?:claims?|asserts?|reported?|allegedly)\b", phrase, re.I)
                and _UNCERTAIN.search(phrase)):
            continue
        if re.search(
            r"\b(?:not|never|cannot|hasn't|hadn't|wasn't|isn't)\s+"
            r"(?:already\s+|yet\s+)?(?:resolved|verified|confirmed|reconciled)\b",
            phrase, re.I,
        ):
            continue
        return ["prematurely_claims_conflict_verified_or_resolved"]
    return []
