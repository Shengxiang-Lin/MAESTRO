"""Shared semantic orchestration modes used by MAESTRO.

These modes are internal Main-Agent decisions.  They never replace the public
Orchestra runtime actions, which remain ``delegate_task`` and ``complete``.
"""
from __future__ import annotations

from typing import Dict, Tuple

ORCHESTRATION_MODES: Tuple[str, ...] = (
    "advance",
    "verify",
    "retry",
    "reformulate",
    "switch",
    "rollback",
    "complete",
)

ORCHESTRATION_MODE_DESCRIPTIONS: Dict[str, str] = {
    "advance": "Use sufficiently supported current information to progress to the next dependency.",
    "verify": "Acquire evidence for a load-bearing claim whose current support is insufficient.",
    "retry": "Repeat essentially the same method because another sample still has information value.",
    "reformulate": "Change the task wording, decomposition, crop/query, or method after the previous formulation failed.",
    "switch": "Change model, tool, provider, or execution route because the current route is unlikely to recover.",
    "rollback": "Stop relying on a previously promoted premise and repair downstream work that depended on it.",
    "complete": "Stop orchestration because the information needed for the final answer is sufficient.",
}

MODE_TO_INDEX = {name: index for index, name in enumerate(ORCHESTRATION_MODES)}
INDEX_TO_MODE = {index: name for name, index in MODE_TO_INDEX.items()}

__all__ = [
    "ORCHESTRATION_MODES",
    "ORCHESTRATION_MODE_DESCRIPTIONS",
    "MODE_TO_INDEX",
    "INDEX_TO_MODE",
]
