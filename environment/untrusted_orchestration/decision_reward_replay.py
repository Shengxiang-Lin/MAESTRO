"""Rebuild decision-time reward snapshots from received one-turn reports.

This does NOT invoke any LLM or rerun faulty Tool/Sub-Agent generations.
It replays *logged* attempt versions and actually delivered public reports.
Every reconstructed action batch is checked against the recorded Main
decision audit. On any mismatch it fails closed; no teacher-only shortcut.

Inputs are private journal files and original Stage-1 scenes. Output is the
same private decision_reward_states.jsonl as the new Stage-4 publisher.
"""
from __future__ import annotations

from copy import deepcopy

from environment.generate_question.execution.program_ir import Relation
from environment.untrusted_orchestration.runtime.faults import relation_rows
from environment.untrusted_orchestration.state_machine import Attempt, ExecutionState
from environment.untrusted_orchestration.trajectory_builder import (
    _main_local_terminal, source_catalog,
)
from environment.untrusted_orchestration.rewards.decision_reward_state import (
    capture_decision_state,
)


def _rehydrate(state, task_id, report, round_number, audit, positions):
    """Load one already executed Attempt; never call run_attempt/fault generator."""
    history = audit["runtime"]["attempt_versions"].get(task_id, [])
    pos = positions.get(task_id, 0)
    if pos >= len(history):
        raise ValueError("journal missing audited task attempt " + task_id)
    item = history[pos]
    if item["version"] != pos + 1:
        raise ValueError("audited attempt version is not monotone")
    runtime = deepcopy(item["runtime"])
    # Recompute visible diagnostics only from the reports Main received
    # before this one pending decision.
    runtime.pop("visible_issue", None)
    runtime.pop("cross_source_recheck_requested", None)
    runtime.pop("history_conflict_recheck_requested", None)
    state._public_conflict_cache = None
    original_status = runtime.get("status")
    if original_status not in {"done", "missing", "partial", "disputed"}:
        raise ValueError("audited original status missing/unknown")
    product = None
    if original_status != "missing":
        rows = item["rows"]
        fields = state.tasks[task_id]["outputs"]["typed_relation"]
        product = Relation(
            tuple(fields),
            tuple({
                "values": deepcopy(row["values"]),
                "support": tuple(row["source_fact_refs"]),
            } for row in rows),
        )
    before = state.current.get(task_id)
    previous_delivered = (
        state.communicated_product(task_id) if before is not None else None
    )
    attempt = Attempt(
        task_id=task_id, version=item["version"], status=original_status,
        product=product, available_fields=tuple(runtime["available_fields"]),
        used_parent_versions=dict(item["used_parent_versions"]),
        runtime=runtime, cost=item["cost"],
    )
    state.current[task_id] = attempt
    state.attempts[task_id].append(attempt)
    state.total_cost += attempt.cost
    positions[task_id] = pos + 1
    if before is not None:
        after_delivered = state.communicated_product(task_id)
        if relation_rows(previous_delivered) != relation_rows(after_delivered):
            for child in state._dependents(task_id):
                if child in state.current and state.current[child].status != "stale":
                    state.current[child].status = "stale"
        # Equivalent retry leaves already computed descendants untouched.
    if report is not None:
        state.set_public_report(task_id, report, round_number=round_number)
    return attempt


def replay_reward_states(scene, episode, audit, *,
                         max_parallel_tasks=4, max_retries=1,
                         model_alias="model_1"):
    """Recreate the one decision-time state; never execute a future batch."""
    if episode["task_id"] != scene["task_id"]:
        raise ValueError("scene/episode mismatch")
    if episode.get("format") != "maestro_untrusted_single_turn.v1":
        raise ValueError("reward_replay_requires_single_turn_episode")
    fork = episode["branch_round"]
    if audit["branch_round"] != fork or audit.get("decision_unit") != "single_turn":
        raise ValueError("private audit single-turn fork mismatch")
    decisions = [
        d for d in audit["decision_audit"]
        if d.get("decision_kind") == "delegate"
    ]
    if len(decisions) != 1 or decisions[0].get("round") != fork:
        raise ValueError("single_turn_requires_one_teacher_dispatch")
    graph = scene["macro_graph"]
    state = ExecutionState(scene, reviewed_reports={})
    positions = {}
    catalog = source_catalog(scene)
    prior_round = 0
    state.cross_source_checks_enabled = True
    for step in episode["trajectory"]:
        rnd = step.get("attempt")
        if type(rnd) is not int or rnd <= prior_round:
            raise ValueError("nonmonotone replay timeline")
        prior_round = rnd
        if step.get("action") != "delegate_task":
            raise ValueError("single_turn_replay_disallows_terminal_action")
        result = step.get("result") or {}
        returned = result.get("subtask_results")
        if rnd < fork:
            if rnd > len(graph["reference_waves"]):
                raise ValueError("prefix round beyond reference waves")
            task_ids = graph["reference_waves"][rnd - 1]
            if not isinstance(returned, list) or len(returned) != len(task_ids):
                raise ValueError("replay report/decision cardinality mismatch")
            for tid, report in zip(task_ids, returned):
                if not isinstance(report, dict):
                    raise ValueError("replay report is not an object")
                _rehydrate(state, tid, report, rnd, audit, positions)
            continue

        if rnd != fork or step is not episode["trajectory"][-1]:
            raise ValueError("single_turn_has_future_decision")
        if result.get("pending") is not True or returned != []:
            raise ValueError("single_turn_pending_batch_has_future_result")

        scheduling = state.scheduling_state()
        repairs = [
            tid for tid in scheduling["repair"]
            if len(state.attempts[tid]) <= max_retries and state.ready(tid)
        ]
        ready = [tid for tid in scheduling["ready"] if tid not in repairs]
        terminal = graph["terminal_task_id"]
        if terminal in ready and _main_local_terminal(graph) and not repairs:
            _rehydrate(state, terminal, None, rnd, audit, positions)
            ready.remove(terminal)
        can_complete = (
            not repairs and state.effective_answer() is not None and not ready
        )
        snapshot = capture_decision_state(
            state, round_number=rnd, repairs=repairs,
            ready=ready, blocked=scheduling["blocked"],
            max_parallel_tasks=max_parallel_tasks, catalog=catalog,
            model_alias=model_alias, event_pending=False,
            can_complete=can_complete,
        )
        chosen = decisions[0]
        allowed = {x["task_id"] for x in snapshot["actionable"]}
        if not set(chosen["task_ids"]) <= allowed:
            raise ValueError(
                "teacher batch unavailable in replay state: " +
                repr(sorted(set(chosen["task_ids"]) - allowed))
            )
        return [snapshot]
    raise ValueError("single_turn_missing_pending_decision")
