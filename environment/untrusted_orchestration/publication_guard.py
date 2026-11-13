"""Independent final publication checks for Main review contamination.

Unlike an LLM critic, this runs after all Stage 4A/B/C processing.
Only the final public turn surface is checked. Hidden review dialogue must
never be presented as a Main-observed event; a failed branch is excluded
from positive turns but retained with private diagnostic provenance.
"""
from __future__ import annotations

from environment.base_orchestration.generation_meta import (
    GENERATION_PROCESS_RE,
)
from .public_language import internal_labels_in_text
from .visible_history import visible_history_reasoning_issues


# Kept as an alias for existing callers; canonical rules live in one module.
_GENERATION_PROCESS = GENERATION_PROCESS_RE


def review_turn_surface(turn: dict) -> list[str]:
    output = turn.get("output") or {}
    if not isinstance(output, dict):
        return ["invalid_action"]
    candidates = [("reasoning", output.get("reasoning"))]
    # The input history is as important as Main's action text: a
    # seemingly clean SFT output can still expose internal work_M* IDs
    # through earlier trusted Stage-2 reports.
    inputs = turn.get("input") or {}
    if isinstance(inputs, dict):
        candidates.append(("subtask_history", inputs.get("subtask_history")))
    params = output.get("params") or {}
    for index, task in enumerate(params.get("tasks", []) or []):
        if isinstance(task, dict):
            candidates += [
                (f"task_{index}_instruction", task.get("task_instruction")),
                (f"task_{index}_context", task.get("context")),
            ]
    problems = [
        "generation_process_leak:" + name
        for name, value in candidates
        if isinstance(value, str) and _GENERATION_PROCESS.search(value)
    ]
    # Only Main's authored reasoning is banned from repeating synthetic
    # workspace handles / reserved fixture URLs. Public instructions and
    # source history can legitimately carry known synthetic URLs needed for
    # the frozen tool-acquisition contract.
    problems += [
        issue + ":reasoning"
        for issue in visible_history_reasoning_issues(
            str(output.get("reasoning") or ""), []
        )
    ]
    problems += [
        "synthetic_execution_identifier_leak:" + name
        for name, value in candidates
        if isinstance(value, str) and internal_labels_in_text(value)
    ]
    return problems
