"""Evidence-centric dynamic requests for the Stage-4 offline planner.

The frozen Macro DAG constrains *executability*; it is not the Main-Agent
delegation language. Each dispatch targets current evidence gaps and retains
only fields and identifiers that were already Main-visible at this round.
"""
from __future__ import annotations

from copy import deepcopy
import re

from environment.base_orchestration.action_author import _assignment


_IDENT = re.compile(r"\b(?:KEY[12]-\d+|MK-\d+|RK-\d+|Case-\d+|Entry-\d+)\b", re.I)


def _visible_identifiers(state, task: dict) -> list[str]:
    """Identifiers only from previously delivered natural-language reports."""
    found = []
    for parent in task["depends_on"]:
        report = state.public_reports.get(parent) or {}
        finish = report.get("finish_result") or {}
        text = " ".join(str(finish.get(k) or "") for k in ("result", "summary"))
        found.extend(_IDENT.findall(text))
    return list(dict.fromkeys(found))[:8]


def _required_fields(state, task: dict) -> dict:
    return {
        parent: sorted(state.required_parent_fields(task, parent))
        for parent in task["depends_on"]
    }


def evidence_request(state, task_id: str, *, round_number: int,
                     retry: bool) -> dict:
    """Derive an executable evidence goal from the current *visible* state.

    No hidden Gold, fault label, future operation results, or unexecuted
    alternative sources are read. Source access is frozen by the task IR.
    """
    task = state.tasks[task_id]
    previous = state.current.get(task_id)
    issue = state.visible_discrepancy(task_id) if previous else None
    target = list(task["outputs"]["typed_relation"])
    required = _required_fields(state, task)
    missing = []
    if previous is not None:
        missing = list(previous.runtime.get("missing_fields", []))
        if previous.product is None:
            missing = target
        if previous.status == "stale":
            missing = target
    if issue and previous and previous.status not in {"missing", "partial", "stale"}:
        operation = "verify_conflicting_evidence"
    elif previous and previous.status == "missing":
        operation = "retry_missing_evidence"
    elif previous and previous.status == "partial":
        operation = "complete_partial_evidence"
    elif previous and previous.status == "stale":
        operation = "refresh_dependent_evidence"
    elif retry:
        operation = "recheck_prior_evidence"
    elif task["source_ids"]:
        operation = "retrieve_required_evidence"
    else:
        operation = "derive_from_returned_evidence"
    missing = list(dict.fromkeys(missing))
    focus = missing or target
    identifiers = _visible_identifiers(state, task)
    if issue:
        justification = issue
    elif previous and previous.status == "missing":
        justification = "The earlier assignment returned no usable Finish Result."
    elif previous and previous.status == "partial":
        justification = (
            "The previous delivery lacks required evidence: "
            + ", ".join(missing or target) + "."
        )
    elif previous and previous.status == "stale":
        justification = (
            "Previously returned dependent evidence must be refreshed "
            "because an upstream result was superseded."
        )
    elif required:
        justification = (
            "The prerequisite Sub-Agent reports are available; the remaining "
            "goal requires " + ", ".join(focus) + "."
        )
    else:
        justification = (
            "The user question still requires the evidence fields "
            + ", ".join(focus) + "."
        )
    primitives = [{
        "operation": node["op"],
        "source_id": node.get("source_id"),
        "acquisition": (node.get("evidence_acquisition") or {}).get("strategy"),
    } for node in task["operator_chain"]]
    return {
        "request_id": f"R{round_number:02d}-E{task_id}",
        "operation": operation,
        "target_fields": focus,
        "required_parent_fields": required,
        "known_identifiers": identifiers,
        "unresolved_fields": missing,
        "authorized_source_ids": list(task["source_ids"]),
        "primitive_plan": primitives,
        "main_visible_justification": justification,
        "acceptance_criteria": [
            "Return only the requested evidence with its source provenance",
            "Preserve previously established identifiers and source scope",
            "Report missing or inconsistent fields explicitly",
        ],
        # Internal execution binding, NOT a public Main-Agent identifier.
        "execution_binding": {"macro_task_id": task_id},
    }


def evidence_assignment(state, task_id: str, *, catalog: dict,
                        model_alias: str, request: dict) -> dict:
    """Ground an O1 assignment in the current information gap.

    A source/tool contract may not be modified merely to make prose look
    better. One assignment can cover multiple field-level evidence needs.
    """
    task = state.tasks[task_id]
    base = _assignment(task, catalog, state.public_reports,
                       model_alias=model_alias)
    fields = ", ".join(request["target_fields"])
    ids = ", ".join(request["known_identifiers"])
    operation = request["operation"]
    if operation == "verify_conflicting_evidence":
        objective = (
            "Reconcile the visible conflicting identifiers or values. "
            "Inspect the authorized observations, report which claims "
            "remain supported, and do not silently assume either is correct."
        )
    elif operation == "retry_missing_evidence":
        objective = (
            "The previous evidence acquisition returned no usable finding. "
            "Repeat only the unresolved evidence retrieval and preserve "
            "any independent previously returned evidence."
        )
    elif operation == "complete_partial_evidence":
        objective = (
            "The preceding handoff was partial. Obtain specifically the "
            "unavailable fields or remaining rows without redoing "
            "unrelated completed investigations."
        )
    elif operation == "refresh_dependent_evidence":
        objective = (
            "An upstream result has changed. Recompute only the evidence "
            "that depends on its current identifier or measurement; "
            "do not reuse the superseded dependent values."
        )
    elif operation == "derive_from_returned_evidence":
        objective = (
            "Use the already returned prerequisite observations to derive "
            "the still-required fields; do not acquire or invent "
            "an unrelated new source."
        )
    else:
        objective = (
            "Acquire the next missing evidence required by the question "
            "using the authorized source and prerequisite observations."
        )
    context = base["context"]
    if ids:
        context += " Previously reported identifiers: " + ids + "."
    # Retain Stage-3 source/workflow specification and O1 tool routing;
    # narrow the actual evidence goal and constraints dynamically.
    base["target_instruction"] = (
        objective + " Focus on " + fields + ". "
        + base["target_instruction"]
    )[:1500]
    base["context"] = (
        context + " Current evidence need: "
        + request["main_visible_justification"]
    )[:3500]
    base["evidence_request_id"] = request["request_id"]
    return base


def plan_evidence_batch(state, batch: list[str], *, round_number: int,
                        repairs: list[str], catalog: dict,
                        model_alias: str) -> tuple[list[dict], list[dict]]:
    requests = []
    assignments = []
    for tid in batch:
        request = evidence_request(
            state, tid, round_number=round_number, retry=tid in repairs,
        )
        assignment = evidence_assignment(
            state, tid, catalog=catalog, model_alias=model_alias,
            request=request,
        )
        requests.append(request)
        assignments.append(assignment)
    return requests, assignments
