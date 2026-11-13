"""High-precision detection of internal generation/reviewer dialogue.

Only evidence actually visible to the Main may appear in published Main
reasoning or task prose. The critic/reviewer evaluating an Author draft is
NOT an O1 Sub-Agent and is not part of user-visible execution history.
Shared by the Stage-3/4 Author guard and Stage-4 publication guard to
avoid drift between generation-time and release-time checks.
"""
from __future__ import annotations

import re


GENERATION_PROCESS_RE = re.compile(
    r"\b(?:the\s+)?(?:critic|reviewer|judge|evaluator)\s+"
    r"(?:(?:correctly|previously|also|rightly|accurately)\s+)?"
    r"(?:said|says|noted?|notes?|identified|identifies|indicated?|"
    r"indicates|found|finds|confirmed?|confirms|observed?|"
    r"requested|suggested|pointed\s+out|rejected|flagged|"
    r"concluded|recommended|asked|highlighted|explained)\b"
    r"|\b(?:the\s+)?(?:critic|reviewer|judge|evaluator)(?:'s)?\s+"
    r"(?:feedback|critique|assessment|evaluation|review)\b"
    r"|\b(?:previous|earlier|last)\s+(?:draft|revision|generation)\b"
    r"|\b(?:per|according\s+to|as\s+(?:the|my))\s+"
    r"(?:review\s+feedback|critic\s+feedback|reviewer\s+feedback)\b"
    r"|\b(?:in|from|per)\s+(?:the|this)\s+prompt\b"
    r"|\b(?:my|the)\s+(?:prompt|system\s+instruction)\s+"
    r"(?:says|instructs|requested)\b"
    r"|\b(?:my|the)\s+(?:tool|action)\s+(?:choice|selection)\s+"
    r"(?:was|is)\s+(?:wrong|invalid)\b"
    r"|\brevision\s+(?:attempt|feedback)\b"
    r"|\bevidence\s+planner\b"
    r"|\b(?:delegate_task|requested_not_executed)\b"
    r"|\b(?:locked|frozen|preselected|pre-selected|predetermined)\s+"
    r"(?:main\s+)?(?:decision|facts?|plan|assignment|delegation|contract)\b"
    r"|\b(?:internal|deterministic)\s+"
    r"(?:planner|contract|compiler|teacher)\b"
    r"|\b(?:the|this)\s+(?:plan|action|task)\s+"
    r"(?:was|has\s+been|is)\s+"
    r"(?:already\s+)?(?:selected|fixed|chosen)\s+"
    r"by\s+(?:the\s+)?(?:planner|teacher|compiler)\b"
    r"|\b(?:code-owned|planner-owned)\b",
    re.I,
)


def generation_process_leak(text: str) -> bool:
    """Whether prose describes private generation/review machinery."""
    return isinstance(text, str) and bool(GENERATION_PROCESS_RE.search(text))
