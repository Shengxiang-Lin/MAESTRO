"""Strictly surface-only removal of synthetic execution identifiers.

A task's immutable WorkProduct, status, source IDs, execution binding and
provenance remain untouched. Internal Macro/primitive *labels* in natural
language do not belong in the Main-visible O1 trace.
"""
from __future__ import annotations

from copy import deepcopy
import re


_INTERNAL = re.compile(r"\b(?:work_M\d+|m\d+_op\d+)\b", re.I)
_SUBTASK = re.compile(r"\bwork_M\d+\b", re.I)
_PRIMITIVE = re.compile(r"\bm\d+_op\d+\b", re.I)


def sanitize_public_prose(value: str) -> str:
    """Replace private labels, not evidence identities or measurements."""
    return _PRIMITIVE.sub(
        "the recorded execution step",
        _SUBTASK.sub("an upstream Sub-Agent result", value),
    )


def sanitize_public_report(report: dict) -> dict:
    """Return a copy with only Main-facing prose fields normalized."""
    result = deepcopy(report)
    for field in ("trace_summary", "task_instruction"):
        if isinstance(result.get(field), str):
            result[field] = sanitize_public_prose(result[field])
    finish = result.get("finish_result")
    if isinstance(finish, dict):
        for field in ("result", "summary"):
            if isinstance(finish.get(field), str):
                finish[field] = sanitize_public_prose(finish[field])
    return result


def internal_labels_in_text(value: str) -> list[str]:
    return sorted(set(_INTERNAL.findall(value))) if isinstance(value, str) else []
