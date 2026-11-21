"""Construct one Main decision from a frozen Stage-3 feedback fork.

Stage 4A replays only the already-executed Stage-3 prefix, changing exactly
one of the latest returned Sub-Agent reports. It never replays future agents,
inserts historical witnesses, rewrites past Main dispatches, or tests eventual
recovery against Gold. Stage 4B/4C review only the resulting public reports
and one pending Main delegation.
"""
from __future__ import annotations

from copy import deepcopy

from environment.base_orchestration.action_author import (
    _assignment, action_issues, author_delegate_action,
)
from environment.base_orchestration.o1_compat import (
    o1_history, o1_prompt, source_catalog,
)
from environment.base_orchestration.trajectory_builder import (
    _strict_reports, clean_reference_identity, _main_local_terminal,
)
from environment.untrusted_orchestration.runtime.faults import Fault
from environment.untrusted_orchestration.runtime.event_scheduler import EventScheduler
from environment.untrusted_orchestration.runtime.evidence_planning import (
    schedule_evidence_requests,
)
from environment.untrusted_orchestration.rewards.decision_reward_state import (
    capture_decision_state,
)
from environment.untrusted_orchestration.inspection.visible_claim_state import (
    audit_visible_claim_states,
)

from .state_machine import ExecutionState, public_history_item
from .reporting.author import naturalize_report


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
already completed. A source-free synthesis/derivation remains a legitimate
delegated task even when all external evidence has already arrived. Explain
that existing observations still need to be combined; do not confuse
"no further retrieval" with "ready to complete". Do not announce a final
answer before the needed
evidence is available, invent an extra result, or describe the process
used to write or assess your text.

Follow the named calculation inputs and operations in work_to_explain.
A date pair is not the earliest/latest observation-log date unless that
operation is explicitly requested. Intermediate identifiers support the
investigation; do not say the question explicitly requests those fields.
For source-free work, explain reuse of returned findings and recomputation;
do not promise new source inspection. Explain only the current evidence
step and avoid repeatedly calling intermediate retrievals the final step.

Copy opaque record keys literally from received evidence. Prefer referring
to the previously returned case identifier instead of repeating a long hash key
when its exact spelling is not needed to explain the decision. When comparing
conflicting keys, quote only the exact already returned strings.

Use two to four sentences focused on this round's missing fields or visible
discrepancy. Name concrete returned findings and the next requested output.
For a final derivation, explain which already delivered inputs will be combined.
Describe an earlier filter or verification as completed only when that operation
is explicitly described in a returned report. A citation or returned entity
identifier alone does not establish that a particular event condition was checked.
Use the perspective of the investigator. Omit implementation vocabulary such as
work_M identifiers, synthetic, simulator, Oracle, Gold, teacher, and Critic.

Return JSON containing only:
{"reasoning": "a natural explanation of the next investigation"}
"""


_MAIN_REASONING_CRITIC_SYSTEM = """Review ONLY the Main-Agent's reasoning for
the current delegated evidence requests. The assignment contract was
deterministically validated in the evidence simulator and is IMMUTABLE.
The review input deliberately contains no editable task_instruction/context,
tool names or model routing. Do not demand any changes to these fields.

Check consistency with the named inputs and operations in the requested work.
Do not substitute date extrema for a specified date pair, claim an intermediate
identifier is a user-requested final output, or promise new external acquisition
when the assignment only combines already returned findings.

Judge whether the reasoning:
(1) refers only to information already present in the current Main-visible
history, not hidden fault origins, Gold, future returns, or invented sources;
(2) gives a coherent reason to collect the currently requested evidence,
without claiming a missing or partial result was verified;
(3) does not claim a dependency was satisfied by a not-yet-returned result,
and does not falsely claim independent tasks are sequentially dependent.

IMPORTANT: A pending derive_from_returned_evidence request may need no
external web/source retrieval but still needs a real synthesis computation.
Existing source data being sufficient does NOT mean the planned derivation
already ran or that Main should Complete now. Judge the authored reasoning
for grounding, not the frozen delegate-versus-complete choice.

CRITICAL CLAIM-SCOPE RULE: values of DIFFERENT attributes (for example,
metric_a=87 and metric_b=55 for the same entity) are not contradictory
merely because their values differ. Only explicit rival values for the SAME
entity, SAME attribute and SAME scope, or a Result/Summary mismatch on
the SAME field, establish an observed discrepancy. Do not invent a
cross-source conflict where the visible history provides only complementary
measurements. If the inputs have arrived but a comparison, join or synthesis
has not yet run, it is correct to delegate that pending computation; do not
reject a grounded description of available inputs as a premature conclusion.

A value explicitly present in an earlier multi-row Result remains a returned
observation. A pending matching or aggregation task may select those records;
it does not require a new external retrieval merely because no prior task
returned the selected row in isolation. Distinguish this available observation
from an independently verified conclusion.

The confirmed_repeated_query_conflicts, when present, compare actual
received reports for the identical delegated query and identical prerequisite
input versions. Distinct values there are an unresolved public disagreement.
An already computed downstream answer is provisional while its input is
disputed. Do not infer that one report self-corrected, that the inputs differ,
or that completion is justified merely because a later calculation finished.
Judge any additional claims of truth or verification independently.
For a pending verification, re-inspecting the assigned source to determine
which reported value it supports IS a valid way to reconcile the disagreement.
For an explicit public arithmetic mismatch, directly recomputing the named
operands is also a valid verification motivation: a correct statement such as
"recompute 319 / 80" does not need to include the literal word "verify".
Still check the operands, operation and result against PUBLIC returned values,
and do not mistake an unperformed recalculation for a completed check.
Do not reject faithful reasoning merely for saying verify rather than reconcile,
or speculate that an inconsistent source makes the assigned check impossible.
The contract validates delegation legality, not the factual correctness of
Sub-Agent feedback. A report saying corrected or done is itself a claim,
and a newer report does not automatically settle an older same-query conflict.
Reject assertions of successful verification before the check actually returns.

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
    return {key: deepcopy(value) for key, value in sub_result.items()
            if not key.startswith("_")}


def _history_batch(number: int, returned: list[dict]) -> dict:
    return {
        "attempt": number,
        "is_parallel_batch": True,
        "subtask_results": [
            public_history_item(result, i) for i, result in enumerate(returned)
        ],
        "total_steps": sum(result["steps_taken"] for result in returned),
        "total_cost": 0.0,
    }


def batch_has_independent_dependencies(state: ExecutionState,
                                       batch: list[str]) -> bool:
    if len(batch) != len(set(batch)):
        return False
    return all(
        b not in state._dependents(a)
        for a in batch for b in batch if a != b
    )


def _replay_latest_feedback(
    scene: dict, reviewed: dict, base_episode: dict, *,
    branch_round: int, fault: Fault,
    report_author=None, report_critic=None,
    report_revisions: int = 2, on_report_review=None,
) -> tuple[ExecutionState, list[dict], list[dict], list[dict]]:
    """Replay actual prior batches; fault only the most recent returned wave."""
    waves = scene["macro_graph"]["reference_waves"]
    actions = [
        step for step in base_episode["trajectory"]
        if step["action"] == "delegate_task"
    ]
    if not 2 <= branch_round <= len(actions) + 1:
        raise ValueError("branch_round_must_follow_actual_delegate_batch")
    if fault.task_id not in waves[branch_round - 2]:
        raise ValueError("fault_must_target_latest_returned_batch")
    if any(
        len(waves[i]) != len(actions[i]["result"]["subtask_results"])
        for i in range(branch_round - 1)
    ):
        raise ValueError("clean_base_alignment_invalid")

    state = ExecutionState(scene, reviewed)
    history, trajectory, injected = [], [], []
    for index in range(branch_round - 1):
        action = deepcopy(actions[index])
        returned = []
        for slot, tid in enumerate(waves[index]):
            assignment = action["params"]["tasks"][slot]
            local_fault = fault if (index == branch_round - 2
                                    and tid == fault.task_id) else None
            attempt = state.register(tid, fault=local_fault)
            if local_fault is not None:
                public, status = naturalize_report(
                    state.tasks[tid], attempt,
                    instruction=assignment["task_instruction"],
                    model_alias=assignment["model"],
                    reviewed=reviewed.get(tid),
                    author=report_author, critic=report_critic,
                    max_revisions=report_revisions,
                )
                public["subtask_index"] = slot
                injected.append({
                    "task_id": tid, "mode": fault.mode,
                    "persistence": fault.persistence,
                    "seed": fault.seed,
                    "injection_primitive": attempt.runtime.get("primitive"),
                    "execution": deepcopy(attempt.runtime),
                    "report_mode": status,
                    "origin_round": branch_round - 1,
                })
                if on_report_review is not None:
                    on_report_review(status)
            else:
                public = _as_public(
                    actions[index]["result"]["subtask_results"][slot]
                )
            public["task_instruction"] = assignment["task_instruction"]
            public["model"] = assignment["model"]
            state.set_public_report(tid, public, round_number=index + 1)
            returned.append(public)
        history.append(_history_batch(index + 1, returned))
        action["result"] = {
            **action["result"],
            "subtask_results": returned,
            "total_subtasks": len(returned),
            "all_done": all(
                (x.get("finish_result") or {}).get("status") == "done"
                for x in returned
            ),
        }
        trajectory.append(action)

    # Earlier Main assignments and returned Finish Results are immutable.
    trusted = actions[:branch_round - 2]
    for original, current in zip(trusted, trajectory):
        if original["params"] != current["params"]:
            raise ValueError("single_turn_modified_prior_main_dispatch")
        if [x.get("finish_result") for x in
                original["result"]["subtask_results"]] != [
                x.get("finish_result") for x in
                current["result"]["subtask_results"]]:
            raise ValueError("single_turn_modified_prior_feedback")
    return state, history, trajectory, injected


def build_branch(
    scene: dict, stage2_rows: list[dict], base_episode: dict, *,
    branch_round: int, faults: list[Fault],
    scenario_plan: dict | None = None,
    author=None, critic=None, report_author=None, report_critic=None,
    max_rounds: int = 16, max_cost: int = 80, max_retries: int = 1,
    max_parallel_tasks: int = 4,
    max_revisions: int = 2,
    report_max_revisions: int | None = None,
    main_max_revisions: int | None = None,
    sub_model: str = "qwen3-8b", model_alias: str = "model_1",
    on_report_review=None, on_main_review=None,
    validated_reference: tuple[dict, dict] | None = None,
) -> tuple[dict, list[dict], dict]:
    """Return public single-turn episode, one label (if accepted), and audit."""
    if max_rounds < branch_round or max_parallel_tasks < 1:
        raise ValueError("invalid_branch_round_or_budget")
    if len(faults) != 1:
        raise ValueError("single_turn_requires_exactly_one_fault")
    if any(f.persistence != "once" for f in faults):
        raise ValueError("single_turn_fault_must_not_persist")
    if base_episode.get("task_id") != scene["task_id"]:
        raise ValueError("base_episode_wrong_scene")
    report_max_revisions = (max_revisions if report_max_revisions is None
                            else report_max_revisions)
    main_max_revisions = (max_revisions if main_max_revisions is None
                          else main_max_revisions)

    if validated_reference is None:
        reviewed = _strict_reports(scene, stage2_rows)
        clean_identity = clean_reference_identity(scene, reviewed)
    else:
        reviewed, clean_identity = validated_reference
    declared = base_episode.get("clean_reference_id")
    if declared is not None and declared != clean_identity["clean_reference_id"]:
        raise ValueError("stage4_clean_reference_fingerprint_mismatch")

    scenario_plan = deepcopy(scenario_plan or {})
    if (scenario_plan.get("history_first_plan")
            or scenario_plan.get("same_source_recheck_plan")
            or scenario_plan.get("evidence_events")
            or any(atom.get("origin_round") != branch_round - 1
                   for atom in scenario_plan.get("fault_atoms", []))):
        raise ValueError("single_turn_cannot_rewrite_history_or_future")
    state, task_entries, trajectory, injected = _replay_latest_feedback(
        scene, reviewed, base_episode, branch_round=branch_round,
        fault=faults[0], report_author=report_author,
        report_critic=report_critic,
        report_revisions=report_max_revisions,
        on_report_review=on_report_review,
    )
    state.cross_source_checks_enabled = True
    scheduler = EventScheduler([], state=state, fork_round=branch_round)
    catalog = source_catalog(scene)
    graph = scene["macro_graph"]
    round_number = branch_round
    visible_issues = state.observable_issues()
    observable_issue_snapshots = [{
        "decision_round": round_number,
        "issues": visible_issues, "private_only": True,
    }]
    recent_issue = any(
        item.get("reported_round") == round_number - 1
        for item in visible_issues
    ) or any(
        event.get("latest_round") == round_number - 1
        for event in state.active_public_conflicts()
    )

    evidence_plan, decisions, turns, main_reviews, reward_states = [], [], [], [], []
    stopped = None
    if not recent_issue:
        stopped = "no_current_visible_issue"
    else:
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
        terminal = graph["terminal_task_id"]
        if terminal in ready and _main_local_terminal(graph) and not repairs:
            state.register(terminal)
            ready.remove(terminal)
        can_complete = (not repairs and state.effective_answer() is not None
                        and not ready)
        reward_states.append(capture_decision_state(
            state, round_number=round_number, repairs=repairs,
            ready=ready, blocked=scheduling["blocked"],
            max_parallel_tasks=max_parallel_tasks, catalog=catalog,
            model_alias=model_alias, event_pending=False,
            can_complete=can_complete,
        ))
        if not repairs and not ready:
            stopped = "no_legal_current_action"
        elif state.total_cost > max_cost:
            stopped = "budget_exhausted"
        else:
            batch, requests, assignments = schedule_evidence_requests(
                state, repairs=repairs, ready=ready,
                round_number=round_number, max_parallel_tasks=max_parallel_tasks,
                catalog=catalog, model_alias=model_alias,
            )
            if not batch:
                stopped = "unresolved_dependencies"
            else:
                request_ids = [
                    item.pop("grouped_request_ids") for item in assignments
                ]
                evidence_plan.append({
                    "round": round_number,
                    "evidence_requests": deepcopy(requests),
                    "dispatch": [{
                        "assignment_id": f"R{round_number:02d}-A{i:02d}",
                        "evidence_request_ids": request_ids[i],
                        "execution_binding": {"macro_task_id": tid},
                        "assignment_contract": deepcopy(assignment),
                    } for i, (tid, assignment) in
                    enumerate(zip(batch, assignments))],
                    "parallel": len(batch) > 1,
                })
                prompt = o1_prompt(
                    scene, task_entries, attempt=round_number,
                    max_attempts=max_rounds, sub_model=sub_model,
                    model_alias=model_alias,
                )
                action, status = author_delegate_action(
                    assignments=assignments, prompt=prompt,
                    author=author, critic=critic,
                    max_revisions=main_max_revisions,
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
                        "round": round_number, "teacher_batch": batch,
                        "failed_authoring": status,
                    })
                else:
                    if action_issues(action, assignments, prompt=prompt):
                        raise ValueError("invalid_dynamic_o1_main_action")
                    if not batch_has_independent_dependencies(state, batch):
                        raise ValueError("invalid_parallel_delegation")
                    history = o1_history(task_entries)
                    turns.append({
                        "sample_id": (scene["task_id"] +
                                      f"-branch-{branch_round:02d}" +
                                      f"-round-{round_number:02d}"),
                        "task_id": scene["task_id"],
                        "round": round_number,
                        "input": {"subtask_history": history},
                        "output": deepcopy(action),
                        "generation_mode": status["mode"],
                    })
                    decisions.append({
                        "round": round_number, "decision_kind": "delegate",
                        "task_ids": list(batch),
                        "mode": status["mode"],
                        "cost_before": state.total_cost,
                        "visible_history": history,
                    })
                    trajectory.append({
                        "attempt": round_number,
                        "action": "delegate_task",
                        "reasoning": action["reasoning"],
                        "params": deepcopy(action["params"]),
                        "result": {
                            "subtask_results": [], "total_subtasks": 0,
                            "all_done": False, "pending": True,
                        },
                    })
                    stopped = "single_turn_planned"

    decision_bindings = {
        row["round"]: row["task_ids"] for row in decisions
        if row.get("decision_kind") == "delegate"
    }
    report_bindings = {
        step["attempt"]: (
            graph["reference_waves"][step["attempt"] - 1]
            if step["attempt"] < branch_round
            else decision_bindings[step["attempt"]]
        )
        for step in trajectory if step.get("action") == "delegate_task"
    }
    visible_claim_state = audit_visible_claim_states(
        scene,
        [step for step in trajectory
         if not (step.get("result") or {}).get("pending")],
        task_ids_by_round=report_bindings,
    )
    accepted = stopped == "single_turn_planned"
    episode = {
        "task_id": scene["task_id"],
        "question": scene["question"],
        "format": "maestro_untrusted_single_turn.v1",
        "clean_reference_id": clean_identity["clean_reference_id"],
        "branch_round": branch_round,
        "trajectory": trajectory,
        "outcome": stopped,
        "recovered": False,
        "main_decision_rounds": len(trajectory),
    }
    audit = {
        "task_id": scene["task_id"],
        "branch_round": branch_round,
        "clean_reference_identity": clean_identity,
        "faults": injected,
        "counterfactual_unrepaired": {"status": "not_evaluated_single_turn"},
        "runtime": state.audit(),
        "visible_claim_state": visible_claim_state,
        "observable_issue_snapshots": observable_issue_snapshots,
        "event_scenario": scenario_plan,
        "event_timeline": scheduler.audit(),
        "fault_events_applied": [{
            "task_id": faults[0].task_id,
            "kind": faults[0].atom_kind or faults[0].mode,
            "actual_execution_round": branch_round - 1,
            "where": "incoming_feedback",
        }],
        "unrealized_future_fault_tasks": [],
        "decision_audit": decisions,
        "decision_reward_settings": {
            "max_parallel_tasks": max_parallel_tasks,
            "max_retries": max_retries,
            "model_alias": model_alias,
        },
        "evidence_plan": evidence_plan,
        "decision_reward_states": reward_states,
        "subagent_report_reviews": [{
            "task_id": item["task_id"],
            "attempt_version": 1,
            "round": branch_round - 1,
            "authoring": item["report_mode"],
        } for item in injected],
        "main_action_reviews": main_reviews,
        "completion_generation_mode": None,
        "execution_outcome": stopped,
        "execution_recovered": False,
        "language_review_accepted": accepted,
        "parallel_batches_verified": all(
            batch_has_independent_dependencies(
                state, item.get("task_ids", [])
            ) for item in decisions
            if item.get("decision_kind") == "delegate"
        ),
        "outcome": stopped,
        "decision_unit": "single_turn",
        "stage3_prior_history_preserved": True,
        "accepted_turns": len(turns) if accepted else 0,
        "private_only": True,
    }
    return episode, turns if accepted else [], audit
