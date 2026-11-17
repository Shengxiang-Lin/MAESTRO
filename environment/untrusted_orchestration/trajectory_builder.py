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
    author_complete_action, _AUTHOR_SYSTEM, _CRITIC_SYSTEM,
)
from environment.base_orchestration.o1_compat import (
    o1_history, o1_prompt, source_catalog,
)
from environment.base_orchestration.trajectory_builder import (
    _answer_text, _strict_reports, _main_local_terminal,
    clean_reference_identity,
)
from .faults import Fault, FaultUnsupported, choose_targets
from .atomic_faults import AtomicFault
from .event_scheduler import EventScheduler
from .state_machine import (
    ExecutionState, deterministic_report, public_history_item,
)
from .report_author import naturalize_report
from .evidence_planning import schedule_evidence_requests


_MAIN_AUTHOR_SYSTEM = _AUTHOR_SYSTEM + """
UNTRUSTED ROUND-LOCAL RECOVERY:
This round has a real prior Sub-Agent feedback history, including results
that may be missing, partial, contradictory, or replaced. The independent
planner already selected the exact set of LEGAL tasks and parallelism from
only visible evidence. You do NOT choose the action, change batch size,
replace tools, or add verification that was not selected. For Stage 4C, the evidence-bound task instructions and handoff context
are IMMUTABLE: output only natural Main reasoning; the compiler restores the
exact authorized assignments from the planner. Explain the specific
observable missing field, conflicting key, or dependency motivating the
assigned repair. When other tasks are independent and ready, explain
why they can proceed concurrently. NEVER claim you know which report was
attacked or quote a private correct answer. Explicitly retain usable
previous findings; do not imply an unperformed verification succeeded.
"""


_MAIN_DECISION_AUTHOR_SYSTEM = """You are the Main-Agent coordinating a
multi-source investigation. Write a concise, natural next-step explanation
based on the user's question and the Sub-Agent feedback already received.

Some returned results may be missing, incomplete or contradictory. Only
assert observations actually visible in the supplied history. When Result
and Summary explicitly disagree, treat both as reported statements, not
as independently established truth. Explain what remains to be checked,
which evidence needs obtaining, and why independent tasks can proceed
together when relevant.

Your next investigations are listed under work_to_explain. Focus on the
substantive reason for those investigations and do not claim they have
already completed. Do not announce a final answer before the needed
evidence is available, invent an extra result, or describe the process
used to write or assess your text.

Return JSON containing only:
{"reasoning": "a natural explanation of the next investigation"}
"""


_MAIN_CRITIC_SYSTEM = _CRITIC_SYSTEM + """
UNTRUSTED RECOVERY REVIEW:
Judge whether the produced prose faithfully conveys the planner's
fixed targeted repair, still-usable prior results, and independent
parallel tasks. The planner has ALREADY proved dependency legality.
A Sub-Agent done flag is execution termination, not proof of factual
correctness. Rejected text must be repaired as prose ONLY; do not
invent a new task, future evidence, attack origin, or hidden Gold.
"""


_MAIN_REASONING_CRITIC_SYSTEM = """Review ONLY the Main-Agent's reasoning for
the current delegated evidence requests. The assignment contract was
deterministically validated in the evidence simulator and is IMMUTABLE.
The review input deliberately contains no editable task_instruction/context,
tool names or model routing. Do not demand any changes to these fields.

Judge whether the reasoning:
(1) refers only to information already present in the current Main-visible
history, not hidden fault origins, Gold, future returns, or invented sources;
(2) gives a coherent reason to collect the currently requested evidence,
without claiming a missing or partial result was verified;
(3) does not claim a dependency was satisfied by a not-yet-returned result,
and does not falsely claim independent tasks are sequentially dependent.

The attached locked Main decision facts, if present, were compiled
from the actual Main-visible Sub-Agent Result and Summary in the latest
returned batch, and distinguish observed statements, unresolved evidence,
and planned actions. A confirmed discrepancy is a factual observation, not
proof that Result (or Summary) is correct. Judge the Author's EXTRA claims,
not the validity of this directly observed discrepancy. Do NOT demand a
complete action or veto the frozen delegate plan just because a structured
Result exists. If the Author invents completed actions, causal explanations,
sources, or a previously verified answer, FAIL with a precise prose-only
repair hint. Return JSON:
{"verdict":"pass"|"fail","grounded":true|false,
"assignment_fidelity":true|false,"o1_style":true|false,
"reason":"brief explanation","repair_hint":"reasoning-only fix"}
"""


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
    on_report_review=None,
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
                if on_report_review is not None:
                    on_report_review(status)
                fault_log.append({
                    "task_id": tid, "mode": fault.mode,
                    "persistence": fault.persistence, "seed": fault.seed,
                    "injection_primitive": attempt.runtime.get("primitive"),
                    "execution": deepcopy(attempt.runtime),
                    "report_mode": status,
                })
            state.set_public_report(
                tid, public, round_number=index + 1,
            )
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
    scenario_plan: dict | None = None,
    author=None, critic=None, report_author=None, report_critic=None,
    max_rounds: int = 16, max_cost: int = 80, max_retries: int = 1,
    max_parallel_tasks: int = 4, max_revisions: int = 2,
    report_max_revisions: int | None = None,
    main_max_revisions: int | None = None,
    sub_model: str = "qwen3-8b", model_alias: str = "model_1",
    on_report_review=None, on_main_review=None,
) -> tuple[dict, list[dict], dict]:
    """Returns (public episode, accepted *post-branch* turn labels, audit)."""
    if max_rounds < branch_round or max_parallel_tasks < 1:
        raise ValueError("invalid_branch_round_or_budget")
    if report_max_revisions is None:
        report_max_revisions = max_revisions
    if main_max_revisions is None:
        main_max_revisions = max_revisions
    if base_episode.get("task_id") != scene["task_id"]:
        raise ValueError("base_episode_wrong_scene")
    reviewed = _strict_reports(scene, stage2_rows)
    clean_identity = clean_reference_identity(scene, reviewed)
    declared_reference = base_episode.get("clean_reference_id")
    if (declared_reference is not None and
            declared_reference != clean_identity["clean_reference_id"]):
        raise ValueError("stage4_clean_reference_fingerprint_mismatch")
    state, task_entries, trajectory, injected = _clean_prefix(
        scene, reviewed, base_episode,
        branch_round=branch_round, faults=faults,
        report_author=report_author, report_critic=report_critic,
        report_revisions=report_max_revisions, model_alias=model_alias,
        on_report_review=on_report_review,
    )
    before = counterfactual_continuation(
        state, start_wave_index=branch_round - 1,
    )
    # All error origins are causal events, not historical rewrites.
    # An old round_fork is simply one atom on the most recent batch.
    scenario_plan = deepcopy(scenario_plan) if scenario_plan else {
        "scenario_id": scene["task_id"] + f"-fork-{branch_round}",
        "mode": "round_fork", "blueprint": "immediate_feedback_fault",
        "fault_atoms": [{
            "kind": fault.atom_kind or (
                "legacy_wrong_evidence" if fault.mode == "wrong" else
                "no_result" if fault.mode == "missing" else
                "incomplete_retrieval"
            ),
            "task_id": fault.task_id, "origin_round": branch_round - 1,
            "seed": fault.seed, "persistence": fault.persistence,
        } for fault in faults],
        "evidence_events": [],
    }
    scheduler = EventScheduler(
        scenario_plan.get("evidence_events", []),
        state=state, fork_round=branch_round,
    )
    state.cross_source_checks_enabled = (
        scenario_plan.get("mode") == "event_driven"
    )
    future_atoms = {}
    for atom in scenario_plan.get("fault_atoms", []):
        if atom["origin_round"] >= branch_round:
            if atom["task_id"] in state.current:
                raise ValueError("future_fault_targets_historical_attempt")
            if atom["task_id"] in future_atoms:
                raise ValueError("duplicate_future_fault_target")
            future_atoms[atom["task_id"]] = AtomicFault(
                kind=atom["kind"], task_id=atom["task_id"],
                origin_round=atom["origin_round"],
                persistence=atom.get("persistence", "once"),
                seed=atom.get("seed", 0), atom_id=atom.get("atom_id", "F"),
            )
    applied_future_atoms = set()
    applied_fault_events = [{
        "task_id": fault.task_id,
        "kind": fault.atom_kind or fault.mode,
        "actual_execution_round": branch_round - 1,
        "where": "incoming_feedback",
    } for fault in faults]
    catalog = source_catalog(scene)
    retry_counts = Counter()
    turns = []
    decisions = []
    report_reviews = [{
        "task_id": item["task_id"], "attempt_version": 1,
        "round": branch_round - 1, "authoring": deepcopy(item["report_mode"]),
    } for item in injected]
    main_reviews = []
    evidence_plan = []
    decision_reward_states = []
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
        ready = scheduler.available(ready, round_number)
        event_pending = any(
            row["status"] == "scheduled"
            for row in scheduler.audit()
        )
        if (terminal in ready and _main_local_terminal(graph)
                and not repairs):
            # Deterministic local computation on *current* parent
            # WorkProducts. No future Oracle/clean report can leak.
            state.register(terminal)
            ready.remove(terminal)
        can_complete = (
            not repairs and state.effective_answer() is not None
            and not ready and not event_pending
        )
        # Record the FULL decision choice set before the Author or any new
        # Sub-Agent observations. The record is private training metadata.
        from .decision_reward_state import capture_decision_state
        decision_reward_states.append(capture_decision_state(
            state, round_number=round_number, repairs=repairs,
            ready=ready, blocked=scheduling["blocked"],
            max_parallel_tasks=max_parallel_tasks, catalog=catalog,
            model_alias=model_alias, event_pending=event_pending,
            can_complete=can_complete,
        ))
        if can_complete:
            # A fabricated result can be unseen as wrong when Main has
            # no independent evidence. Do not publish a "successful"
            # recovery merely because the PRIVATE executor still holds
            # a clean product that was never delivered to Main.
            if any(
                item.runtime.get("report_fault_kind") == "fabricated_claim"
                for item in state.current.values()
            ):
                stopped = "unverified_report_claim"
                break
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
                author=author, critic=critic,
                max_revisions=main_max_revisions,
            )
            main_reviews.append({
                "round": round_number, "kind": "complete",
                "authoring": deepcopy(status),
            })
            if on_main_review is not None:
                on_main_review(status)
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

        # Evidence gaps drive prioritization globally. Macro IR is only
        # used after selection to enforce executable dependency and source
        # constraints for each grouped Sub-Agent assignment.
        batch, requests, assignments = schedule_evidence_requests(
            state, repairs=repairs, ready=ready,
            round_number=round_number,
            max_parallel_tasks=max_parallel_tasks,
            catalog=catalog, model_alias=model_alias,
        )
        if not batch:
            stopped = (
                "event_wait_without_legal_action"
                if event_pending else "unresolved_dependencies"
            )
            break
        # One dispatch can fulfill several atomic evidence requests.
        # Request IDs and Macro bindings are PRIVATE simulation metadata,
        # not fields permitted in the Main LLM's assignment prompt.
        grouped_ids = [
            item.pop("grouped_request_ids") for item in assignments
        ]
        evidence_plan.append({
            "round": round_number,
            "evidence_requests": deepcopy(requests),
            "dispatch": [{
                "assignment_id": f"R{round_number:02d}-A{i:02d}",
                "evidence_request_ids": grouped_ids[i],
                "execution_binding": {"macro_task_id": tid},
                "assignment_contract": deepcopy(assignment),
            } for i, (tid, assignment) in
                enumerate(zip(batch, assignments))],
            "parallel": len(batch) > 1,
        })
        history = o1_history(task_entries)
        prompt = o1_prompt(
            scene, task_entries, attempt=round_number,
            max_attempts=max_rounds, sub_model=sub_model,
            model_alias=model_alias,
        )
        action, status = author_delegate_action(
            assignments=assignments, prompt=prompt,
            author=author, critic=critic,
            max_revisions=main_max_revisions,
            author_system=_MAIN_AUTHOR_SYSTEM,
            critic_system=_MAIN_CRITIC_SYSTEM,
        )
        main_reviews.append({
            "round": round_number, "kind": "delegate",
            "authoring": deepcopy(status),
        })
        if on_main_review is not None:
            on_main_review(status)
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
            future = future_atoms.get(tid)
            if (future is not None
                    and round_number >= future.origin_round
                    and (tid not in applied_future_atoms
                         or future.persistence == "same_path")):
                fault = future.as_legacy_fault()
                applied_future_atoms.add(tid)
                applied_fault_events.append({
                    "task_id": tid, "kind": future.kind,
                    "actual_execution_round": round_number,
                    "where": "future_execution",
                })
            attempt = state.register(tid, fault=fault)
            scheduler.note_execution(tid, round_number)
            if tid in repairs:
                retry_counts[tid] += 1
            report, report_status = naturalize_report(
                state.tasks[tid], attempt,
                instruction=dispatched["task_instruction"],
                model_alias=dispatched["model"],
                reviewed=reviewed.get(tid),
                author=report_author, critic=report_critic,
                max_revisions=report_max_revisions,
            )
            report_reviews.append({
                "task_id": tid, "attempt_version": attempt.version,
                "round": round_number,
                "authoring": deepcopy(report_status),
            })
            if on_report_review is not None:
                on_report_review(report_status)
            report["subtask_index"] = slot
            public = _as_public(report)
            state.set_public_report(
                tid, public, round_number=round_number,
            )
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
    if stopped == "recovered" and (
        set(future_atoms) - applied_future_atoms
        or any(x["status"] != "released" for x in scheduler.audit())
    ):
        stopped = "scenario_events_not_realized"
    # Never advertise a delayed *recovery* if the supposed late evidence
    # did not expose an actual visible contradiction followed by a real
    # upstream replacement and downstream invalidation.
    if (stopped == "recovered"
            and scenario_plan.get("blueprint")
                == "delayed_independent_evidence"
            and (not state.detected_conflicts or not state.invalidations)):
        stopped = "delayed_blueprint_not_demonstrated"
    # Positive reviewed-turn in LLM mode must be language-reviewed on *both*
    # surfaces. The existing deterministic report remains inspectable
    # as a diagnostic when a review fails, not secretly accepted as an
    # LLM-written trajectory.
    execution_outcome = stopped
    if stopped == "recovered" and (
        (report_author is not None and any(
            x["authoring"]["mode"] == "grounded_template_fallback"
            for x in report_reviews
        )) or (author is not None and any(
            x["authoring"]["mode"] ==
            "teacher_grounded_completion_fallback"
            for x in main_reviews
        ))
    ):
        stopped = "language_review_unaccepted"
    success = stopped == "recovered"
    # Teacher-side state is computed from the ACTUAL public report text in
    # each returned batch, including injected faults and retry replacements.
    # It is never passed to Main (which still sees only O1 formatted history).
    from .visible_claim_state import audit_visible_claim_states
    decision_bindings = {
        d["round"]: d["task_ids"] for d in decisions
        if d.get("decision_kind") == "delegate"
    }
    report_bindings = {}
    for action in trajectory:
        if action.get("action") != "delegate_task":
            continue
        number = action["attempt"]
        report_bindings[number] = (
            graph["reference_waves"][number - 1]
            if number < branch_round
            else decision_bindings[number]
        )
    visible_claim_state = audit_visible_claim_states(
        scene, trajectory, task_ids_by_round=report_bindings,
    )
    episode = {
        "task_id": scene["task_id"], "question": scene["question"],
        "format": "maestro_untrusted_round_branch.v1",
        "clean_reference_id": clean_identity["clean_reference_id"],
        "branch_round": branch_round,
        "trajectory": trajectory,
        "outcome": stopped,
        "recovered": success,
        "main_decision_rounds": len(trajectory),
    }
    audit = {
        "task_id": scene["task_id"], "branch_round": branch_round,
        "clean_reference_identity": clean_identity,
        "faults": injected,
        "counterfactual_unrepaired": before,
        "runtime": state.audit(),
        "visible_claim_state": visible_claim_state,
        "event_scenario": scenario_plan,
        "event_timeline": scheduler.audit(),
        "fault_events_applied": applied_fault_events,
        "unrealized_future_fault_tasks": sorted(
            set(future_atoms) - applied_future_atoms
        ),
        "decision_audit": decisions,
        "decision_reward_settings": {
            "max_parallel_tasks": max_parallel_tasks,
            "max_retries": max_retries,
            "model_alias": model_alias,
        },
        "evidence_plan": evidence_plan,
        "decision_reward_states": decision_reward_states,
        "subagent_report_reviews": report_reviews,
        "main_action_reviews": main_reviews,
        "completion_generation_mode": final_mode,
        "execution_outcome": execution_outcome,
        "execution_recovered": execution_outcome == "recovered",
        "language_review_accepted": stopped != "language_review_unaccepted",
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
