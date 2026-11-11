"""Round-local untrusted branch construction with dynamic O1 delegation.

A branch has one injection *window*: immediately before the Main decision
in round t>=2, alter one or more Sub-Agents from the batch returned in
round t-1. The whole earlier history stays byte-for-byte trusted. After
branching, each Main decision uses the actual new return values; it is
never forced to replay an obsolete clean Teacher wave.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from typing import Any

from environment.base_orchestration.action_author import (
    _assignment, action_issues, author_delegate_action,
    author_complete_action,
)
from environment.base_orchestration.o1_compat import (
    o1_history, o1_prompt, source_catalog,
)
from environment.base_orchestration.trajectory_builder import (
    _answer_text, _strict_reports, _main_local_terminal,
)
from .faults import Fault, FaultUnsupported, choose_targets
from .state_machine import (
    ExecutionState, deterministic_report, public_history_item,
)
from .report_author import naturalize_report


def _as_public(sub_result: dict) -> dict:
    return {k: deepcopy(v) for k, v in sub_result.items()
            if not k.startswith("_")}


def _history_batch(number: int, returned: list[dict]) -> dict:
    return {
        "attempt": number,
        "is_parallel_batch": True,
        "subtask_results": [
            public_history_item(result, i)
            for i, result in enumerate(returned)
        ],
        "total_steps": sum(result["steps_taken"] for result in returned),
        "total_cost": 0.0,
    }


def _clean_prefix(
    scene: dict, reviewed: dict, base_episode: dict, *,
    branch_round: int, faults: list[Fault],
    report_author=None, report_critic=None,
    report_revisions: int = 1,
    model_alias: str = "model_1",
) -> tuple[ExecutionState, list[dict], list[dict], list[dict]]:
    waves = scene["macro_graph"]["reference_waves"]
    actions = [
        step for step in base_episode["trajectory"]
        if step["action"] == "delegate_task"
    ]
    if not 2 <= branch_round <= len(actions) + 1:
        raise ValueError("branch_round_must_follow_actual_delegate_batch")
    if any(len(waves[index]) !=
           len(actions[index]["result"]["subtask_results"])
           for index in range(branch_round - 1)):
        raise ValueError("clean_base_alignment_invalid")
    targets = {fault.task_id: fault for fault in faults}
    if not targets or not set(targets) <= set(waves[branch_round - 2]):
        raise ValueError("fault_targets_must_be_in_last_completed_batch")

    state = ExecutionState(scene, reviewed)
    history, episode_prefix, fault_log = [], [], []
    for index in range(branch_round - 1):
        source_action = actions[index]
        previous_batch = source_action["result"]["subtask_results"]
        visible_results = []
        for slot, tid in enumerate(waves[index]):
            clean_result = previous_batch[slot]
            fault = targets.get(tid) if index == branch_round - 2 else None
            attempt = state.register(tid, fault=fault)
            if fault is None:
                # Preserve exactly what the trusted Main saw, including the
                # natural-language Stage-2 report and O1 task instruction.
                public = _as_public(clean_result)
                report_mode = "reviewed_clean"
            else:
                public, status = naturalize_report(
                    state.tasks[tid], attempt,
                    instruction=clean_result["task_instruction"],
                    model_alias=clean_result["model"],
                    reviewed=reviewed.get(tid),
                    author=report_author, critic=report_critic,
                    max_revisions=report_revisions,
                )
                public["subtask_index"] = slot
                report_mode = status["mode"]
                fault_log.append({
                    "task_id": tid, "mode": fault.mode,
                    "persistence": fault.persistence, "seed": fault.seed,
                    "injection_primitive": attempt.runtime.get("primitive"),
                    "execution": deepcopy(attempt.runtime),
                    "report_mode": status,
                })
            state.set_public_report(tid, public)
            visible_results.append(public)
        history.append(_history_batch(index + 1, visible_results))
        modified = deepcopy(source_action)
        modified["result"] = {
            **deepcopy(source_action["result"]),
            "subtask_results": visible_results,
            "total_subtasks": len(visible_results),
            "all_done": all(
                (item.get("finish_result") or {}).get("status") == "done"
                for item in visible_results
            ),
        }
        episode_prefix.append(modified)
    return state, history, episode_prefix, fault_log


def counterfactual_continuation(
    state: ExecutionState, *, start_wave_index: int,
) -> dict:
    """Execute the *old, no-repair* schedule on ACTUAL poisoned WorkProducts.

    This is private comparison evidence. It is not an action policy fed to
    Main and cannot make the branch Teacher omniscient.
    """
    clone = deepcopy(state)
    waves = state.scene["macro_graph"]["reference_waves"]
    events = []
    for wave in waves[start_wave_index:]:
        for tid in wave:
            if tid in clone.current:
                continue
            if not clone.ready(tid):
                return {
                    "status": "blocked", "events": events,
                    "answer": None, "correct": False,
                }
            try:
                attempt = clone.register(tid)
            except (ValueError, KeyError, TypeError) as exc:
                return {
                    "status": "execution_error",
                    "events": events, "error": str(exc)[:200],
                    "answer": None, "correct": False,
                }
            events.append({
                "task_id": tid, "version": attempt.version,
                "rows": clone.audit()["attempt_versions"][tid][-1]["rows"],
            })
    return {
        "status": "completed" if clone.effective_answer() is not None
                  else "blocked",
        "answer": clone.effective_answer(),
        "correct": clone.answer_matches_gold(),
        "events": events,
    }


def _repair_assignment(state: ExecutionState, tid: str,
                       *, catalog: dict, model_alias: str) -> dict:
    task = state.tasks[tid]
    assigned = _assignment(
        task, catalog, state.public_reports, model_alias=model_alias,
    )
    # The newly visible result is named as a reason for the repair; no
    # hidden attack kind or Gold value appears in the instruction.
    issue = state.visible_discrepancy(tid)
    if not issue:
        issue = (
            "The earlier handoff no longer has the necessary usable "
            "fields or depends on a superseded upstream result."
        )
    assigned["target_instruction"] = (
        "Recheck only the disputed or missing part of this investigation. "
        "Use the already returned supporting records where sufficient; "
        "retrieve the authorized source only when the evidence is absent. "
        + assigned["target_instruction"]
    )
    assigned["context"] = (
        assigned["context"] + " Earlier Sub-Agent feedback: " + issue
    )[:3500]
    return assigned


def _completion_template(state: ExecutionState, *, answer: str) -> dict:
    final = state.scene["macro_graph"]["terminal_task_id"]
    terminal = state.tasks[final]
    references = (
        [final] if final in state.public_reports
        else terminal["depends_on"]
    )
    quotes = []
    for tid in references[-2:]:
        finish = state.public_reports.get(tid, {}).get("finish_result") or {}
        result = str(finish.get("result", "")).strip().replace("\n", " ")
        if result:
            quotes.append(result[:360].rstrip())
    if not quotes:
        raise ValueError("no_visible_evidence_for_completion")
    reason = " ".join('The returned finding states: "' + q + '".'
                      for q in quotes)
    reason += (
        " These completed investigations now provide the necessary "
        "evidence and no unresolved premise remains for the answer."
    )
    return {
        "action": "complete", "reasoning": reason,
        "params": {"answer": answer},
    }


def dependency_safe_batch(state: ExecutionState,
                          repairs: list[str], ready: list[str],
                          max_parallel_tasks: int) -> list[str]:
    """Select a batch against ONE pre-dispatch execution snapshot.

    Even if an old partial product exposes enough fields to make a
    descendant technically ready, that descendant must not be dispatched
    in parallel with a parent being repaired. O1 parallel delegation has
    no intra-batch result handoff; sequential Python register() calls
    must never be used to justify an otherwise illegal parallel wave.
    """
    candidates = list(dict.fromkeys(repairs + ready))
    priority = set(repairs)
    chosen = []
    for tid in candidates:
        if len(chosen) >= max_parallel_tasks:
            break
        if not state.ready(tid):
            continue
        # Repairing any ancestor can change the input version even if the
        # currently available partial fields happen to suffice.
        if any(tid in state._dependents(repair) for repair in priority
               if repair != tid):
            continue
        if any((tid in state._dependents(other)
                or other in state._dependents(tid))
               for other in chosen):
            continue
        chosen.append(tid)
    if not batch_has_independent_dependencies(state, chosen):
        raise ValueError("untrusted_parallel_batch_has_dependencies")
    return chosen


def batch_has_independent_dependencies(state: ExecutionState,
                                       batch: list[str]) -> bool:
    if len(batch) != len(set(batch)):
        return False
    return all(
        b not in state._dependents(a)
        for a in batch for b in batch if a != b
    )


def build_branch(
    scene: dict, stage2_rows: list[dict], base_episode: dict, *,
    branch_round: int, faults: list[Fault],
    author=None, critic=None, report_author=None, report_critic=None,
    max_rounds: int = 16, max_cost: int = 80, max_retries: int = 1,
    max_parallel_tasks: int = 4, max_revisions: int = 2,
    sub_model: str = "qwen3-8b", model_alias: str = "model_1",
) -> tuple[dict, list[dict], dict]:
    """Returns (public episode, accepted *post-branch* turn labels, audit)."""
    if max_rounds < branch_round or max_parallel_tasks < 1:
        raise ValueError("invalid_branch_round_or_budget")
    if base_episode.get("task_id") != scene["task_id"]:
        raise ValueError("base_episode_wrong_scene")
    reviewed = _strict_reports(scene, stage2_rows)
    state, task_entries, trajectory, injected = _clean_prefix(
        scene, reviewed, base_episode,
        branch_round=branch_round, faults=faults,
        report_author=report_author, report_critic=report_critic,
        report_revisions=max_revisions, model_alias=model_alias,
    )
    before = counterfactual_continuation(
        state, start_wave_index=branch_round - 1,
    )
    catalog = source_catalog(scene)
    retry_counts = Counter()
    turns = []
    decisions = []
    stopped = None
    final_mode = None
    graph = scene["macro_graph"]
    terminal = graph["terminal_task_id"]

    def emit(turn_number: int, action: dict, mode: str,
             *, kind: str, task_ids: list[str] | None = None) -> None:
        history = o1_history(task_entries)
        turns.append({
            "sample_id": (
                scene["task_id"] + f"-branch-{branch_round:02d}" +
                f"-round-{turn_number:02d}"
            ),
            "task_id": scene["task_id"],
            "round": turn_number,
            "input": {"subtask_history": history},
            "output": deepcopy(action),
            "generation_mode": mode,
        })
        decisions.append({
            "round": turn_number, "decision_kind": kind,
            "task_ids": list(task_ids or []),
            "mode": mode, "cost_before": state.total_cost,
            "visible_history": history,
        })

    for round_number in range(branch_round, max_rounds + 1):
        if state.total_cost > max_cost:
            stopped = "budget_exhausted"
            break
        scheduling = state.scheduling_state()
        repairs = [
            tid for tid in scheduling["repair"]
            if len(state.attempts[tid]) <= max_retries and state.ready(tid)
        ]
        ready = [
            tid for tid in scheduling["ready"]
            if tid not in repairs
        ]
        if (terminal in ready and _main_local_terminal(graph)
                and not repairs):
            # Deterministic local computation on *current* parent
            # WorkProducts. No future Oracle/clean report can leak.
            state.register(terminal)
            ready.remove(terminal)
        if (not repairs and state.effective_answer() is not None
                and not ready):
            if not state.answer_matches_gold():
                stopped = "incorrect_result_after_actual_propagation"
                break
            history = o1_history(task_entries)
            prompt = o1_prompt(
                scene, task_entries, attempt=round_number,
                max_attempts=max_rounds, sub_model=sub_model,
                model_alias=model_alias,
            )
            template = _completion_template(
                state, answer=_answer_text(scene),
            )
            action, status = author_complete_action(
                template=template, prompt=prompt,
                delivered={
                    k: v for k, v in state.public_reports.items()
                    if isinstance(v.get("finish_result"), dict)
                },
                author=author, critic=critic, max_revisions=max_revisions,
            )
            emit(round_number, action, status["mode"], kind="complete")
            final_mode = status["mode"]
            trajectory.append({
                "attempt": round_number, "action": "complete",
                "reasoning": action["reasoning"],
                "params": deepcopy(action["params"]),
                "result": {"success": True, "done": True},
            })
            stopped = "recovered"
            break

        batch = dependency_safe_batch(
            state, repairs, ready, max_parallel_tasks,
        )
        if not batch:
            stopped = "unresolved_dependencies"
            break
        assignments = [
            (_repair_assignment(state, tid, catalog=catalog,
                                model_alias=model_alias)
             if tid in repairs else
             _assignment(state.tasks[tid], catalog, state.public_reports,
                         model_alias=model_alias))
            for tid in batch
        ]
        history = o1_history(task_entries)
        prompt = o1_prompt(
            scene, task_entries, attempt=round_number,
            max_attempts=max_rounds, sub_model=sub_model,
            model_alias=model_alias,
        )
        action, status = author_delegate_action(
            assignments=assignments, prompt=prompt,
            author=author, critic=critic,
            max_revisions=max_revisions,
        )
        if action is None:
            stopped = "main_action_not_accepted"
            decisions.append({
                "round": round_number,
                "teacher_batch": batch,
                "failed_authoring": status,
            })
            break
        if action_issues(action, assignments, prompt=prompt):
            raise ValueError("invalid_dynamic_o1_main_action")
        if not batch_has_independent_dependencies(state, batch):
            raise ValueError("invalid_parallel_delegation")
        emit(
            round_number, action, status["mode"],
            kind="delegate", task_ids=batch,
        )
        # A parallel batch is atomic from Main's perspective: all tasks
        # consume the snapshot that existed BEFORE the batch was launched.
        # No task can secretly use an earlier sibling's new WorkProduct.
        parent_snapshot = {
            task_id: item.version for task_id, item in state.current.items()
        }
        returned = []
        for slot, tid in enumerate(batch):
            if not state.ready(tid):
                raise ValueError("parallel_task_became_unready:" + tid)
            if any(
                parent_snapshot.get(p) != state.current[p].version
                for p in state.tasks[tid]["depends_on"]
            ):
                raise ValueError("parallel_batch_consumed_new_parent:" + tid)
            dispatched = action["params"]["tasks"][slot]
            fault = next((
                f for f in faults if f.task_id == tid
                and f.persistence == "same_path"
                and tid in repairs
            ), None)
            attempt = state.register(tid, fault=fault)
            if tid in repairs:
                retry_counts[tid] += 1
            report, report_status = naturalize_report(
                state.tasks[tid], attempt,
                instruction=dispatched["task_instruction"],
                model_alias=dispatched["model"],
                reviewed=reviewed.get(tid),
                author=report_author, critic=report_critic,
                max_revisions=max_revisions,
            )
            report["subtask_index"] = slot
            public = _as_public(report)
            state.set_public_report(tid, public)
            returned.append(public)
        task_entries.append(_history_batch(round_number, returned))
        trajectory.append({
            "attempt": round_number,
            "action": "delegate_task",
            "reasoning": action["reasoning"],
            "params": deepcopy(action["params"]),
            "result": {
                "subtask_results": returned,
                "total_subtasks": len(returned),
                "all_done": all(
                    (item.get("finish_result") or {}).get("status") == "done"
                    for item in returned
                ),
                "total_cost": 0.0,
                "total_steps": sum(x["steps_taken"] for x in returned),
            },
        })
    if stopped is None:
        stopped = "max_decision_rounds"
    success = stopped == "recovered"
    episode = {
        "task_id": scene["task_id"], "question": scene["question"],
        "format": "maestro_untrusted_round_branch.v1",
        "branch_round": branch_round,
        "trajectory": trajectory,
        "outcome": stopped,
        "recovered": success,
        "main_decision_rounds": len(trajectory),
    }
    audit = {
        "task_id": scene["task_id"], "branch_round": branch_round,
        "faults": injected,
        "counterfactual_unrepaired": before,
        "runtime": state.audit(),
        "decision_audit": decisions,
        "completion_generation_mode": final_mode,
        "parallel_batches_verified": all(
            batch_has_independent_dependencies(
                state, d.get("task_ids", []),
            ) for d in decisions if d.get("decision_kind") == "delegate"
        ),
        "outcome": stopped,
        "accepted_turns": len(turns) if success else 0,
        "private_only": True,
    }
    # No incorrect or unresolved teacher actions are exported as
    # positive supervised labels. They remain inspectable as Episodes.
    return episode, turns if success else [], audit
