"""Typed stage-local knowledge boundary for simulated Sub-Agent reports.

Stage 2 reports must only describe properties established by THIS Macro's
executed operations and upstream evidence. The complete question is context,
not evidence of a fact already verified. These deterministic rules catch
high-confidence unsupported claims while preserving conditional language.
"""
from __future__ import annotations

import re

CLAIM_CONTRACT_VERSION = "maestro.stage_local_claims.v1"
_REPORT_CONFLICT = re.compile(
    r"\b(?:disagree(?:s|d|ment)?|discrepanc(?:y|ies)|conflict(?:s|ing)?|"
    r"contradict(?:s|ory|ing)?|differ(?:s|ed|ent|ing)?|mismatch(?:es|ed)?)\b",
    re.I,
)
_REPORT_CONTEXT = re.compile(
    r"\b(?:report(?:s|ed)?|year(?:s)?|account(?:s)?|"
    r"source(?:s)?|records?)\b", re.I,
)
_ASSERTION = re.compile(
    r"\b(?:found|identified|selected|retained|matched|verified|"
    r"confirmed|determined|showed|indicated|contained|contains|"
    r"returned|captured|established|revealed|"
    r"consists?\s+of|are|were)\b", re.I,
)
_DEFERRED = re.compile(
    r"\b(?:will|would|could|to|for|pending|requires?|"
    r"need(?:s)?\s+to|subsequent|later|next)\s+"
    r"(?:\w+\s+){0,3}(?:check|verify|compare|reconcile|"
    r"determine|assess|whether)\b", re.I,
)
_UNKNOWN_DISAGREEMENT = re.compile(
    r"\b(?:no|not|never|without|cannot|can't|hasn't|haven't|isn't|aren't)\b"
    r"[^.!?]{0,100}\b(?:disagreement|conflict|discrepancy|difference|"
    r"discrepant|contradiction|differ|disagree|mismatch|verified|"
    r"confirmed|established|checked|compared)\b"
    r"|\b(?:disagreement|conflict|discrepancy|difference|"
    r"contradiction)\b[^.!?]{0,80}"
    r"\b(?:not\s+yet|unverified|unchecked|pending|unknown)\b"
    # A source-disagreement CHECK reported as not completed is not an
    # established cross-source disagreement.
    r"|\b(?:source|report)[-\s]+(?:disagreement|conflict|discrepancy)"
    r"\s+checks?\s+(?:was|were|has\s+been|have\s+been)?\s*"
    r"(?:not|never)\s+(?:yet\s+)?"
    r"(?:completed|performed|conducted|established)\b"
    r"|\b(?:yet\s+to|still\s+need(?:s)?\s+to)\s+"
    r"(?:compare|verify|check|establish|confirm)\b",
    re.I,
)
_CLAIMED_DISAGREEMENT_CONDITION_MET = re.compile(
    r"\b(?:year|report(?:ed)?)\s+"
    r"(?:disagreement|discrepancy|conflict|mismatch)\s+"
    r"(?:condition|criterion|criteria)\s+"
    r"(?:was|were|is|has\s+been|have\s+been)\s+"
    r"(?:met|satisfied|verified|established)\b",
    re.I,
)

_REFUTED_DISAGREEMENT = re.compile(
    r"\b(?:the\s+)?(?:two\s+)?(?:reports?|years?|sources?)\b"
    r"[^.!?]{0,65}\b(?:agree|match|coincide|are\s+equal)\b",
    re.I,
)


def classify_disagreement_claim(sentence: str) -> str:
    """Distinguish asserted fact from explicit uncertainty or negation.

    Conservative: a mixed contradictory sentence remains an assertion
    unless its complete disagreement claim is explicitly scoped as unknown.
    """
    pieces = re.split(
        r"\s*[;:]\s*|\bbut\b|\bhowever\b", sentence, flags=re.I,
    )
    if len(pieces) > 1:
        labels = [classify_disagreement_claim(piece) for piece in pieces]
        return next(
            (label for label in ("established", "refuted", "unknown", "pending")
             if label in labels), "none",
        )
    if not (_REPORT_CONFLICT.search(sentence)
            and _REPORT_CONTEXT.search(sentence)):
        return "none"
    if _DEFERRED.search(sentence):
        return "pending"
    if _UNKNOWN_DISAGREEMENT.search(sentence):
        # A later separate clause can still assert actual disagreement.
        pieces = re.split(r"\s*[;:]\s*|\bbut\b|\bhowever\b", sentence, flags=re.I)
        if len(pieces) > 1 and any(
            _REPORT_CONFLICT.search(piece) and _ASSERTION.search(piece)
            and not _UNKNOWN_DISAGREEMENT.search(piece)
            for piece in pieces
        ):
            return "established"
        return "unknown"
    if _REFUTED_DISAGREEMENT.search(sentence):
        return "refuted"
    if _ASSERTION.search(sentence) or re.search(
        r"\b(?:reports?|years?|accounts?|sources?)\b"
        r"[^.!?]{0,65}\b(?:disagree|conflict|differ|mismatch)\b",
        sentence, re.I,
    ):
        return "established"
    return "none"


_JARGON_REPLACEMENTS = (
    (re.compile(r"\boracle\s+(?:answer|output|result)\b", re.I),
     "verified source result"),
    (re.compile(r"\bgold\s+program\b", re.I), "recorded checks"),
    (re.compile(r"\bteacher\s+world\b", re.I), "working evidence"),
    (re.compile(r"\bsynthetic\s+ground\s+truth\b", re.I),
     "supplied evidence"),
    (re.compile(r"\bmacro\s+taskgraph\b", re.I), "work sequence"),
)


def compile_claim_contract(record: dict) -> dict:
    steps = record["trace_summary"]["steps"]
    sources = set(record["trace_summary"]["available_evidence_source_ids"])
    comparisons = [
        step["action"] for step in steps
        if step.get("operation") in {
            "compare_fields", "claim_relation_filter",
        }
    ]
    local_year_disagreement = (
        {"report_a", "report_b"}.issubset(sources)
        and any(
            "reported_year_a" in action
            and "reported_year_b" in action
            and "differs from" in action
            for action in comparisons
        )
    )
    inherited = set(record["trace_summary"].get("verified_stage_claims", []))
    year_disagreement = (
        local_year_disagreement
        or "independent_report_year_disagreement" in inherited
    )
    return {
        "version": CLAIM_CONTRACT_VERSION,
        "available_source_ids": sorted(sources),
        "executed_operations": [step["operation"] for step in steps],
        "reported_year_disagreement_established": year_disagreement,
        "reported_year_disagreement_checked_locally": local_year_disagreement,
        "verified_inherited_claims": sorted(inherited),
        "comparison_descriptions": comparisons,
        "output_row_count": len(record["answer"]),
        "source_scope_rule": (
            "Question intent is not a completed Macro observation; "
            "do not claim independent report disagreement before both "
            "sources are compared."
        ),
    }


def _asserts_report_disagreement(sentence: str) -> bool:
    return classify_disagreement_claim(sentence) in {
        "established", "refuted",
    }


def stage_local_claim_issues(text: str, contract: dict) -> list[str]:
    if not isinstance(text, str):
        return []
    if contract.get("reported_year_disagreement_established"):
        return []
    if (_CLAIMED_DISAGREEMENT_CONDITION_MET.search(text)
            or any(_asserts_report_disagreement(sentence) for sentence in
                   re.split(r"(?<=[.!?])\s+", text))):
        return ["claims_unverified_independent_report_disagreement"]
    if any(_asserts_report_disagreement(sentence) for sentence in
           re.split(r"(?<=[.!?])\s+", text)):
        return ["claims_unverified_independent_report_disagreement"]
    return []


def normalize_report_language(text: str) -> tuple[str, list[str]]:
    """Only replace internal vocabulary, never change claimed facts."""
    if not isinstance(text, str):
        return text, []
    revised, replacements = text, []
    for regex, phrase in _JARGON_REPLACEMENTS:
        revised, n = regex.subn(phrase, revised)
        if n:
            replacements.append(regex.pattern)
    return revised, replacements
