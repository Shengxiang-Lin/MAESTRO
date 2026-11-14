"""Private, decision-time scoring contracts for Stage-4 Main orchestration.

The returned state is for the REWARD EVALUATOR, never for the Main prompt.
It captures every currently eligible evidence goal, not just the Teacher's
chosen batch. No hidden future answer or injected-fault identity is used.

No LLM calls are made; the snapshot is taken before the Main action is
authored or its Sub-Agent results become available.
"""
from __future__ import annotations

from .evidence_planning import evidence_assignment, evidence_request

SCHEMA = "maestro.decision_reward_state.v1"

# Empirical urgency defaults, not unique correct-action labels.
URGENCY = {
    "verify_conflicting_evidence": 1.0,
    "retry_missing_evidence": 0.95,
    "complete_partial_evidence": 0.88,
    "refresh_dependent_evidence": 0.85,
    "recheck_prior_evidence": 0.70,
    "retrieve_required_evidence": 0.67,
    "derive_from_returned_evidence": 0.63,
}


def capture_decision_state(
    state, *, round_number: int, repairs: list[str],
    ready: list[str], blocked: list[str],
    max_parallel_tasks: int, catalog: dict, model_alias: str,
    event_pending: bool, can_complete: bool,
) -> dict:
    """Snapshot admissible actions from the same state as the actual planner.

    A private-only repair (not supported by any Main-observable discrepancy)
    makes the reward state UNSCORABLE; it must not force the policy to divine
    a hidden fault. Held future events are not assumed to be observed.
    """
    if round_number < 1 or max_parallel_tasks < 1:
        raise ValueError("invalid decision reward round/parallel cap")
    repair_set = set(repairs)
    unobservable = []
    for tid in repairs:
        observed_status = state.public_status(tid)
        current = state.current.get(tid)
        issue = state.visible_discrepancy(tid)
        # The current state_machine also uses the private WorkProduct to
        # detect an absent result. If Main was actually sent a DONE report,
        # absence of the private product is NOT itself visible evidence.
        if (observed_status == "done" and current is not None
                and current.product is None):
            issue = None
        # A stale dependent result follows a changed earlier report; this
        # can be inferred from versioned deliveries and the public DAG.
        stale = current is not None and current.status == "stale"
        if not issue and observed_status == "done" and not stale:
            unobservable.append(tid)

    # A newly repaired upstream product cannot be used by a descendant in
    # the very same parallel batch, even if an old partial value looks ready.
    feasible = []
    for tid in dict.fromkeys(repairs + ready):
        if not state.ready(tid):
            continue
        if any(tid in state._dependents(parent) for parent in repair_set
               if parent != tid):
            continue
        feasible.append(tid)
    # Future-event holds are private to the simulator. If a task is
    # otherwise ready but secretly held, Main has no public way to know it
    # cannot be dispatched yet. Do not grade that decision state.
    released = set(repairs) | set(ready)
    privately_held_ready = sorted(
        tid for tid in state.tasks
        if tid not in state.current and tid not in released
        and state.ready(tid)
    )
    actions = []
    for tid in feasible:
        task = state.tasks[tid]
        goal = evidence_request(
            state, tid, round_number=round_number, retry=tid in repair_set,
        )
        contract = evidence_assignment(
            state, tid, catalog=catalog, model_alias=model_alias, request=goal,
        )
        operation = goal["operation"]
        if operation not in URGENCY:
            raise ValueError("unrecognized evidence operation")
        # Downstream leverage is a small, bounded tie-breaker, not an excuse
        # to ignore an observed contradiction.
        descendants = len(state._dependents(tid))
        priority = round(
            URGENCY[operation] * (1.0 + min(0.20, 0.04 * descendants)), 6
        )
        fields = list(dict.fromkeys(goal["target_fields"]))
        if not fields:
            raise ValueError("scorable goal needs an explicit evidence field")
        actions.append({
            "task_id": tid,
            "operation": operation,
            "kind": "repair" if tid in repair_set else "new",
            "target_fields": fields,
            "source_ids": list(goal["authorized_source_ids"]),
            "known_identifiers": list(goal["known_identifiers"]),
            "required_parent_fields": goal["required_parent_fields"],
            "depends_on": list(task["depends_on"]),
            "descendants": sorted(state._dependents(tid)),
            "tools": list(contract["tools"]),
            "model": contract["model"],
            "priority": priority,
            "evidence_units": [f"{tid}::{field}" for field in fields],
        })

    return {
        "schema": SCHEMA,
        "decision_round": round_number,
        "max_parallel_tasks": max_parallel_tasks,
        "can_complete": bool(can_complete),
        "event_pending": bool(event_pending),
        "scorable": not unobservable and not privately_held_ready,
        "unobservable_private_repairs": sorted(unobservable),
        "unobservable_event_held_ready": privately_held_ready,
        "actionable": actions,
        "blocked_task_ids": list(blocked),
        "completed_task_ids": sorted(tid for tid in state.current
                                   if tid not in repair_set),
        "eligible_task_count": len(actions),
        "source": "pre_action_execution_state",
        "private_only": True,
    }
