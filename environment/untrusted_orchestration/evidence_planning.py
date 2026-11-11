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
        identities = list(dict.fromkeys(_IDENT.findall(text)))
        # A long candidate set does not identify the particular record
        # needed next. Do not require Main to reiterate all candidates.
        # Only uniquely established upstream identifiers constrain a new
        # evidence request; conflict-specific IDs are handled separately.
        if len(identities) == 1:
            found.extend(identities)
    return list(dict.fromkeys(found))[:4]


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
    identifiers = (
        list(dict.fromkeys(_IDENT.findall(issue)))[:4]
        if issue else _visible_identifiers(state, task)
    )
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
    old_public = state.public_reports.get(task_id) or {}
    old_finish = old_public.get("finish_result") or {}
    observed_ids = list(dict.fromkeys(_IDENT.findall(
        str(old_finish.get("result") or "") + " "
        + str(old_finish.get("summary") or "")
    )))[:5]
    if observed_ids and operation in {
        "retry_missing_evidence", "complete_partial_evidence",
        "verify_conflicting_evidence", "recheck_prior_evidence",
    }:
        # Only name identifiers already sent back to Main, never hidden
        # missing rows inferred from Gold or the operator's future output.
        context += (
            " Identifiers visible in the earlier returned handoff: "
            + ", ".join(observed_ids) + "."
        )
    if operation == "complete_partial_evidence":
        if "remaining_rows" in request["target_fields"]:
            objective += (
                " Retrieve the complete applicable record set rather "
                "than copying the truncated prior subset. Preserve both "
                "previously seen and newly retrieved identifiers; "
                "report which rows remain unavailable. "
                "Do not infer the identity of any unseen row."
            )
        else:
            objective += (
                " Distinguish delivered field values from unavailable "
                "fields; do not replace missing fields with intermediate "
                "values that were not finally delivered."
            )
    elif operation == "retry_missing_evidence":
        objective += (
            " State the precise requested output fields and return a "
            "usable current handoff, without claiming the previous "
            "unsuccessful acquisition already established the answer."
        )
    elif operation == "verify_conflicting_evidence":
        objective += (
            " Preserve the competing previously reported identifiers "
            "and report any remaining disagreement rather than "
            "silently picking an unsupported candidate."
        )
    # Retain Stage-3 source/workflow specification and O1 tool routing;
    # narrow the actual evidence goal and constraints dynamically.
    focus = (
        "Recover the complete missing row set"
        if fields == "remaining_rows" else
        "Return precisely the requested evidence field(s): " + fields
    )
    base["target_instruction"] = (
        objective + " " + focus + ". "
        + base["target_instruction"]
    )[:1500]
    base["context"] = (
        context + " Current evidence need: "
        + request["main_visible_justification"]
    )[:3500]
    return base


def plan_evidence_batch(state, batch: list[str], *, round_number: int,
                        repairs: list[str], catalog: dict,
                        model_alias: str) -> tuple[list[dict], list[dict]]:
    # Plan at the field/evidence level first. A single executable operator
    # chain may satisfy several requests, which are grouped into one legal
    # dispatch. This avoids pretending that every field needs a separate
    # Agent call (or that a Macro ID is itself an evidence requirement).
    requests = []
    assignments = []
    for tid in batch:
        group = evidence_request(
            state, tid, round_number=round_number, retry=tid in repairs,
        )
        fields = list(dict.fromkeys(group["target_fields"]))
        group_ids = []
        for index, field in enumerate(fields):
            request = deepcopy(group)
            request["request_id"] = group["request_id"] + f"-F{index + 1:02d}"
            request["target_fields"] = [field]
            request["evidence_unit"] = {
                "field": field,
                "entity_identifiers": list(group["known_identifiers"]),
                "source_scope": list(group["authorized_source_ids"]),
                "needed_for": list(group["required_parent_fields"]),
            }
            requests.append(request)
            group_ids.append(request["request_id"])
        assignment = evidence_assignment(
            state, tid, catalog=catalog, model_alias=model_alias,
            request=group,
        )
        # This is private dispatch metadata: NEVER pass binding to Main LLM.
        assignment["grouped_request_ids"] = group_ids
        assignments.append(assignment)
    return requests, assignments


def schedule_evidence_requests(state, *, repairs: list[str],
                               ready: list[str], round_number: int,
                               max_parallel_tasks: int, catalog: dict,
                               model_alias: str) -> tuple[list[str], list[dict], list[dict]]:
    """Global evidence-gap-first scheduling over every eligible operation.

    The candidate set derives from the whole DAG and current WorkProducts.
    An evidence goal, not a Macro identifier, determines urgency; only then
    are requests mapped into parallel-safe executable Macro adapters.
    """
    candidates = list(dict.fromkeys(repairs + ready))
    if not candidates:
        return [], [], []
    goals = {
        tid: evidence_request(
            state, tid, round_number=round_number,
            retry=tid in repairs,
        ) for tid in candidates
    }
    priority = {
        "verify_conflicting_evidence": 0,
        "retry_missing_evidence": 1,
        "complete_partial_evidence": 2,
        "refresh_dependent_evidence": 3,
        "recheck_prior_evidence": 4,
        "retrieve_required_evidence": 5,
        "derive_from_returned_evidence": 6,
    }
    order = {tid: i for i, tid in enumerate(candidates)}
    # More affected descendants => more global information gain, but never
    # overturn an earlier correctness-critical verification/repair class.
    ranked = sorted(candidates, key=lambda tid: (
        priority.get(goals[tid]["operation"], 99),
        -len(state._dependents(tid)),
        -len(goals[tid]["target_fields"]),
        order[tid],
    ))
    selected = []
    repair_set = set(repairs)
    for tid in ranked:
        if len(selected) >= max_parallel_tasks:
            break
        if not state.ready(tid):
            continue
        # Two tasks can share a round only when neither has a causal
        # dependency on the other's *new* evidence.
        if any(tid in state._dependents(rep) for rep in repair_set
               if rep != tid):
            continue
        if any(
            tid in state._dependents(other)
            or other in state._dependents(tid)
            for other in selected
        ):
            continue
        selected.append(tid)
    requests, assignments = plan_evidence_batch(
        state, selected, round_number=round_number,
        repairs=repairs, catalog=catalog, model_alias=model_alias,
    )
    return selected, requests, assignments
