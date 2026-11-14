"""Conservative deterministic binding of O1 prose actions to evidence goals.

No similarity scoring and no teacher-action comparison. Tasks are grounded by
their model/tool contract, explicitly named authorized source IDs and exact
requested evidence fields. If two eligible goals cannot be distinguished
without private Macro IDs, the binding is AMBIGUOUS and not guessed.
"""
from __future__ import annotations

import re


def _mentions(text: str, literal: str) -> bool:
    if not literal:
        return False
    # Field markers may be emitted with spaces instead of underscores.
    for name in {literal.casefold(), literal.replace("_", " ").casefold()}:
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])"
        if re.search(pattern, text.casefold()):
            return True
    return False


# Lightweight, auditable operation-intent evidence, not LLM semantics.
# Missing an explicit verb reduces Priority credit but does not invalidate
# a scoped, otherwise legal request. More synonyms can be added with tests.
_OPERATION_CUES = {
    "verify_conflicting_evidence": r"\b(?:verify|reconcile|cross.check|validate|conflict|contradict|independent)\b",
    "retry_missing_evidence": r"\b(?:retry|repeat|re.run|missing|recover|again)\b",
    "complete_partial_evidence": r"\b(?:partial|remaining|incomplete|missing|complete)\b",
    "refresh_dependent_evidence": r"\b(?:refresh|recompute|recalculate|superseded|changed|update)\b",
    "derive_from_returned_evidence": r"\b(?:derive|combine|compute|calculate|synthesize|aggregate)\b",
}


def _operation_alignment(operation: str, instruction: str) -> float:
    pattern = _OPERATION_CUES.get(operation)
    return 1.0 if pattern is None or re.search(
        pattern, instruction, re.IGNORECASE
    ) else 0.5


def bind_tasks(tasks: list[dict], state: dict) -> dict:
    """Unique eligible task ID per candidate slot, or a structured failure.

    The private IDs are never required in generated text and never become
    model inputs. No cosine/LLM/string similarity fallback is used.
    """
    options = state["actionable"]
    fields = sorted({
        field for opt in options for field in opt["target_fields"]
    })
    candidates = []
    focus = []
    operation_focus = []
    if not isinstance(tasks, list) or not tasks:
        return {"valid": False, "reason": "empty_task_batch"}
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            return {"valid": False, "reason": f"slot_{i}_not_object"}
        instruction = task.get("task_instruction")
        context = task.get("context")
        model = task.get("model")
        tools = task.get("tools")
        if (not isinstance(instruction, str) or not instruction.strip()
                or not isinstance(context, str) or not context.strip()
                or not isinstance(model, str)
                or not isinstance(tools, list)
                or any(not isinstance(t, str) for t in tools)):
            return {"valid": False, "reason": f"slot_{i}_invalid_o1_contract"}
        matched = []
        for option in options:
            if model != option["model"] or tools != option["tools"]:
                continue
            if not all(_mentions(instruction, sid)
                       for sid in option["source_ids"]):
                continue
            if not all(_mentions(instruction, field)
                       for field in option["target_fields"]):
                continue
            # Source-free derivations must not claim an unrelated authorized
            # source. Source-bearing tasks must name the exact scope.
            extra_sources = [
                sid for other in options for sid in other["source_ids"]
                if sid not in option["source_ids"] and _mentions(instruction, sid)
            ]
            if extra_sources:
                continue
            matched.append(option)
        if not matched:
            return {"valid": False, "reason": f"slot_{i}_unbound_evidence_goal"}
        if len(matched) > 1:
            combined = instruction + " " + context
            specificity = [
                sum(_mentions(combined, identifier)
                    for identifier in opt.get("known_identifiers", []))
                for opt in matched
            ]
            best = max(specificity)
            winners = [opt for opt, points in zip(matched, specificity)
                       if points == best]
            if best == 0 or len(winners) != 1:
                return {"valid": False, "reason": f"slot_{i}_ambiguous_evidence_goal"}
            chosen = winners[0]
        else:
            chosen = matched[0]
        # Penalize requesting unrelated additional typed fields, not merely
        # additional explanation/provenance in free text.
        mentioned = {field for field in fields if _mentions(instruction, field)}
        needed = set(chosen["target_fields"])
        precision = len(needed) / max(len(needed), len(mentioned))
        focus.append(min(1.0, precision))
        operation_focus.append(
            _operation_alignment(chosen["operation"], instruction)
        )
        candidates.append(chosen)
    ids = [opt["task_id"] for opt in candidates]
    if len(ids) != len(set(ids)):
        return {"valid": False, "reason": "duplicate_evidence_dispatch"}
    descendants = {opt["task_id"]: set(opt["descendants"]) for opt in options}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if b in descendants[a] or a in descendants[b]:
                return {"valid": False, "reason": "same_batch_dependency_violation"}
    return {
        "valid": True, "reason": None,
        "task_ids": ids,
        "selected": candidates,
        "scope_precision": sum(focus) / len(focus),
        "operation_alignment": operation_focus,
    }
