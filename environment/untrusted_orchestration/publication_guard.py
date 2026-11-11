"""Independent final publication checks for Main review contamination.

Unlike an LLM critic, this runs after all Stage 4A/B/C processing.
Only the final public turn surface is checked. Hidden review dialogue must
never be presented as a Main-observed event; a failed branch is excluded
from positive turns but retained with private diagnostic provenance.
"""
from __future__ import annotations

import re


_GENERATION_PROCESS = re.compile(
    r"\b(?:the\s+)?(?:critic|reviewer|judge|evaluator)\s+"
    r"(?:(?:correctly|previously|also)\s+)?"
    r"(?:noted|said|observed|requested|suggested|pointed\s+out|"
    r"rejected|flagged|concluded|recommended|asked|indicated)\b|"
    r"\b(?:previous|earlier|last)\s+(?:draft|revision|generation)\b|"
    r"\b(?:per|according\s+to|as\s+(?:the|my))\s+"
    r"(?:review\s+feedback|critic\s+feedback|reviewer\s+feedback)\b|"
    r"\b(?:my|the)\s+(?:prompt|system\s+instruction)\s+"
    r"(?:says|instructs|requested)\b",
    re.I,
)


def review_turn_surface(turn: dict) -> list[str]:
    output = turn.get("output") or {}
    if not isinstance(output, dict):
        return ["invalid_action"]
    candidates = [("reasoning", output.get("reasoning"))]
    params = output.get("params") or {}
    for index, task in enumerate(params.get("tasks", []) or []):
        if isinstance(task, dict):
            candidates += [
                (f"task_{index}_instruction", task.get("task_instruction")),
                (f"task_{index}_context", task.get("context")),
            ]
    return [
        "generation_process_leak:" + name
        for name, value in candidates
        if isinstance(value, str) and _GENERATION_PROCESS.search(value)
    ]
