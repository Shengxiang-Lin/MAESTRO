"""Round-local untrusted branch construction with dynamic O1 delegation.

A branch changes the latest completed Sub-Agent feedback batch before
Main's next decision. An optional previous report may expose an additional
record only when its own prior executor really produced that record.
The earlier dispatch, sources, and task contracts remain unchanged.
After branching Main sees received reports, not private fault labels, and
does not replay obsolete clean Teacher waves.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json

from environment.base_orchestration.action_author import (
    action_issues, author_delegate_action,
    author_complete_action, _AUTHOR_SYSTEM, _CRITIC_SYSTEM,
)
from environment.base_orchestration.o1_compat import (
    o1_history, o1_prompt, source_catalog,
)
from environment.base_orchestration.trajectory_builder import (
    _answer_text, _strict_reports, _main_local_terminal,
    clean_reference_identity,
)
from environment.untrusted_orchestration.runtime.faults import Fault, relation_rows
from environment.untrusted_orchestration.runtime.atomic_faults import AtomicFault
from environment.untrusted_orchestration.runtime.event_scheduler import EventScheduler
from .state_machine import (
    ExecutionState, deterministic_report, public_history_item,
)
from .reporting.author import naturalize_report
from environment.untrusted_orchestration.runtime.evidence_planning import schedule_evidence_requests
from .scenario.history_conflicts import expose_executed_prior_record
from .scenario.same_source_recheck import (
    patch_scene as patch_same_source_recheck_scene,
    patch_clean_episode as patch_same_source_recheck_episode,
)
from .scenario.handoff_conflicts import expose_prior_handoff
from .scenario.prior_observation import (
    propose_prior_observation, patch_scene as patch_prior_observation_scene,
    latest_scope_statement,
)


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


def _replay_observed_prefix(
    scene: dict, reviewed: dict, base_episode: dict, *,
    branch_round: int, faults: list[Fault],
    fault_origin_wave_index: int | None = None,
    prefix_report_reviews: list[dict] | None = None,
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
    origin = (branch_round - 2
              if fault_origin_wave_index is None
              else fault_origin_wave_index)
    if not 0 <= origin <= branch_round - 2:
        raise ValueError("fault_origin_must_precede_branch_decision")
    if not targets or not set(targets) <= set(waves[origin]):
        raise ValueError("fault_targets_must_be_in_declared_origin_batch")
    if len(targets) != len(faults):
        raise ValueError("duplicate_initial_fault_target")
    # Only an earlier executed error can affect an intermediate consumer.
    # Re-run those consumers against the poisoned product and replace their
    # reports with actual observed values; never copy Stage-3 clean answers.
    affected = set()
    if origin < branch_round - 2:
        probe = ExecutionState(scene, reviewed)
        for tid in targets:
            affected.update(probe._dependents(tid))

    state = ExecutionState(scene, reviewed)
    history, episode_prefix, fault_log = [], [], []
    for index in range(branch_round - 1):
        source_action = actions[index]
        previous_batch = source_action["result"]["subtask_results"]
        visible_results = []
        for slot, tid in enumerate(waves[index]):
            clean_result = previous_batch[slot]
            fault = targets.get(tid) if index == origin else None
            attempt = state.register(tid, fault=fault)
            # Descendants of an early bad result must be replayed using
            # their ACTUAL values. A newly delegated clean recheck must
            # likewise have its own executed report rather than inheriting
            # the earlier Sub-Agent's prose verbatim.
            reauthor = (
                fault is not None or tid in affected
                or "history_recheck_of" in state.tasks[tid]
            )
            if not reauthor:
                public = _as_public(clean_result)
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
                if prefix_report_reviews is not None:
                    prefix_report_reviews.append({
                        "task_id": tid,
                        "attempt_version": attempt.version,
                        "round": index + 1,
                        "authoring": deepcopy(status),
                    })
                if on_report_review is not None:
                    on_report_review(status)
                if fault is not None:
                    fault_log.append({
                        "task_id": tid, "mode": fault.mode,
                        "persistence": fault.persistence, "seed": fault.seed,
                        "injection_primitive": attempt.runtime.get("primitive"),
                        "execution": deepcopy(attempt.runtime),
                        "report_mode": status,
                        "origin_round": index + 1,
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
    if origin < branch_round - 2:
        # A frozen trusted Stage-3 action is not automatically causal after
        # an earlier report has changed. Reject intermediate instructions
        # that explicitly name clean-only source values which no public
        # report or original question has supplied at that point.
        from .reporting.surface import delivered_records

        def visible_values(report):
            finish = (report.get("finish_result") or {})
            records = delivered_records(finish.get("result"))
            if records is None:
                return set()
            return {
                str(value) for record in records
                for value in record["values"].values()
                if (type(value) in {str, int, float}
                    and len(str(value).strip()) >= 4)
            }

        for index in range(origin + 1, branch_round - 1):
            # The originally clean downstream reports can themselves
            # contain fields lost or changed in the poisoned replay.
            clean_only = set()
            for earlier in range(origin, index):
                for slot, tid in enumerate(waves[earlier]):
                    if tid not in targets and tid not in affected:
                        continue
                    clean_only.update(
                        visible_values(actions[earlier]["result"][
                            "subtask_results"][slot])
                        - visible_values(episode_prefix[earlier]["result"][
                            "subtask_results"][slot])
                    )
            action = episode_prefix[index]
            plan_text = json.dumps({
                "reasoning": action.get("reasoning"),
                "assignments": action.get("params"),
            }, ensure_ascii=False)
            known_text = str(scene.get("question", "")) + json.dumps([
                row["result"]["subtask_results"]
                for row in episode_prefix[:index]
            ], ensure_ascii=False)
            if any(value in plan_text and value not in known_text
                   for value in clean_only):
                raise ValueError(
                    "delayed_prefix_dispatch_references_unseen_clean_value"
                )
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
    max_post_fork_rounds: int | None = None,
    max_parallel_tasks: int = 4, max_revisions: int = 2,
    report_max_revisions: int | None = None,
    main_max_revisions: int | None = None,
    sub_model: str = "qwen3-8b", model_alias: str = "model_1",
    on_report_review=None, on_main_review=None,
    validated_reference: tuple[dict, dict] | None = None,
) -> tuple[dict, list[dict], dict]:
    """Returns (public episode, accepted *post-branch* turn labels, audit)."""
    if max_rounds < branch_round or max_parallel_tasks < 1:
        raise ValueError("invalid_branch_round_or_budget")
    if max_post_fork_rounds is not None and (
        type(max_post_fork_rounds) is not int or max_post_fork_rounds < 1
    ):
        raise ValueError("invalid_max_post_fork_rounds")
    if report_max_revisions is None:
        report_max_revisions = max_revisions
    if main_max_revisions is None:
        main_max_revisions = max_revisions
    if base_episode.get("task_id") != scene["task_id"]:
        raise ValueError("base_episode_wrong_scene")
    # The Stage-4A coordinator validates each frozen scene exactly once.
    # Public callers retain strict per-branch validation by default.
    if validated_reference is None:
        reviewed = _strict_reports(scene, stage2_rows)
        clean_identity = clean_reference_identity(scene, reviewed)
    else:
        reviewed, clean_identity = validated_reference
    declared_reference = base_episode.get("clean_reference_id")
    if (declared_reference is not None and
            declared_reference != clean_identity["clean_reference_id"]):
        raise ValueError("stage4_clean_reference_fingerprint_mismatch")
    # A new same-source recheck is an explicit, scenario-local Main
    # delegation, not a ghost report appended to the latest feedback.
    # Preserve original Stage-3 identity; mark the augmented prefix
    # separately in the private branch audit.
    same_source_recheck_plan = (
        (scenario_plan or {}).get("same_source_recheck_plan")
    )
    if same_source_recheck_plan is not None:
        base_episode = patch_same_source_recheck_episode(
            base_episode, scene, same_source_recheck_plan,
            max_parallel_tasks=max_parallel_tasks,
        )
        scene = patch_same_source_recheck_scene(
            scene, same_source_recheck_plan,
        )
    fault_origin_wave_index = (
        same_source_recheck_plan["older_wave_index"]
        if same_source_recheck_plan is not None
        and same_source_recheck_plan.get("construction")
            == "delayed_consumed_error_recovery"
        else branch_round - 2
    )
    prefix_report_reviews = []
    state, task_entries, trajectory, injected = _replay_observed_prefix(
        scene, reviewed, base_episode,
        branch_round=branch_round, faults=faults,
        fault_origin_wave_index=fault_origin_wave_index,
        prefix_report_reviews=prefix_report_reviews,
        report_author=report_author, report_critic=report_critic,
        report_revisions=report_max_revisions, model_alias=model_alias,
        on_report_review=on_report_review,
    )
    # All comparisons use actual previously delivered evidence. An early
    # fault may propagate before a later, Main-delegated clean witness;
    # preserve the independently replayed clean pair only in private audit.
    history_anchor = {"status": "not_applicable", "reason": "not_targeted"}
    # A source-legal, earlier *public* keyed input was chosen before the
    # atomic fault. Preserve that real prior WorkProduct and only make its
    # already stated field explicit; no new source access is introduced.
    if same_source_recheck_plan is not None:
        history_anchor = {
            "status": "generated",
            "reason": same_source_recheck_plan.get(
                "construction", "earlier_same_query_result"
            ) if fault_origin_wave_index != branch_round - 2
              else "earlier_same_query_result",
            "task_id": same_source_recheck_plan["older_task_id"],
            "target_task_id": same_source_recheck_plan["new_task"]["task_id"],
            "prior_round": same_source_recheck_plan["older_wave_index"] + 1,
            "construction": "later_main_delegated_same_source_recheck",
        }
    handoff_plan = (scenario_plan or {}).get("history_first_plan")
    if handoff_plan is not None:
        history_anchor = expose_prior_handoff(
            state, task_entries, trajectory, handoff_plan,
        )
    if (history_anchor["status"] != "generated"
            and scenario_plan is not None
            and scenario_plan.get("mode") == "event_driven"
            and len(faults) == 1
            and faults[0].atom_kind in {
                "wrong_retrieval", "wrong_entity_binding",
                "fabricated_claim", "computation_error",
            }):
        history_anchor = expose_executed_prior_record(
            state, task_entries, trajectory,
            latest_task_ids={faults[0].task_id},
        )
    prior_observation_patch = None
    if (history_anchor["status"] not in {"exposed", "generated"}
            and scenario_plan is not None
            and scenario_plan.get("mode") == "event_driven"
            and len(faults) == 1
            and faults[0].atom_kind in {
                "wrong_retrieval", "wrong_entity_binding",
                "fabricated_claim", "computation_error",
            }):
        prior_observation_patch, prior_diag = propose_prior_observation(
            state, faults[0].task_id, branch_round=branch_round,
            max_parallel_tasks=max_parallel_tasks,
        )
        if prior_observation_patch is not None:
            previous_scene = scene
            scene = patch_prior_observation_scene(scene, prior_observation_patch)
            state.scene = scene
            tid = prior_observation_patch["task_id"]
            state.tasks[tid] = prior_observation_patch["task"]
            state.attempts[tid] = []
            attempt = state.register(tid)
            index = prior_observation_patch["prior_wave_index"]
            prior_action = trajectory[index]
            prior_entry = task_entries[index]
            slot = len(prior_action["result"]["subtask_results"])
            current_wave = previous_scene["macro_graph"]["reference_waves"][
                branch_round - 2
            ]
            current_slot = current_wave.index(faults[0].task_id)
            original_assignment = trajectory[-1]["params"]["tasks"][current_slot]
            instruction = prior_observation_patch["instruction"]
            report = deterministic_report(
                state.tasks[tid], attempt,
                instruction=instruction, model_alias=model_alias,
            )
            report["subtask_index"] = slot
            state.set_public_report(tid, report, round_number=index + 1)
            prior_action["params"]["tasks"].append({
                "task_instruction": instruction,
                "context": (
                    "A prior independent source survey for the same "
                    "question. Return all observed source rows, without "
                    "assuming which row will be needed later."
                ),
                "model": model_alias,
                "tools": list(original_assignment["tools"]),
            })
            prior_action["result"]["subtask_results"].append(report)
            prior_action["result"]["total_subtasks"] += 1
            prior_action["result"]["total_steps"] += report["steps_taken"]
            prior_action["result"]["all_done"] = all(
                (item.get("finish_result") or {}).get("status") == "done"
                for item in prior_action["result"]["subtask_results"]
            )
            prior_entry["subtask_results"].append(
                public_history_item(report, slot)
            )
            prior_entry["total_steps"] += report["steps_taken"]
            # The newest Sub-Agent must explicitly state the source lookup
            # key it actually used. Otherwise an earlier table and a later
            # answer are not an *observable* same-record disagreement.
            latest_report = state.public_reports[faults[0].task_id]
            note = latest_scope_statement(prior_observation_patch)
            latest_report["trace_summary"] = (
                str(latest_report.get("trace_summary") or "").rstrip()
                + " " + note
            ).strip()
            state.set_public_report(
                faults[0].task_id, latest_report,
                round_number=branch_round - 1,
            )
            trajectory[-1]["result"]["subtask_results"][current_slot] = (
                latest_report
            )
            task_entries[-1]["subtask_results"][current_slot] = (
                public_history_item(latest_report, current_slot)
            )
            history_anchor = prior_diag
        else:
            history_anchor = prior_diag

    paired_clean_history = None
    if history_anchor["status"] in {"exposed", "generated"}:
        if fault_origin_wave_index < branch_round - 2:
            # Entire paired prefix, including the original downstream
            # values, comes from the unpoisoned source-local dispatches.
            paired_clean_history = deepcopy(
                base_episode["trajectory"][:branch_round - 1]
            )
        else:
            paired_clean_history = deepcopy(trajectory)
            last_wave = scene["macro_graph"]["reference_waves"][branch_round - 2]
            for fault in faults:
                slot = last_wave.index(fault.task_id)
                paired_clean_history[-1]["result"]["subtask_results"][slot] = (
                    deepcopy(base_episode["trajectory"][branch_round - 2][
                        "result"]["subtask_results"][slot])
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
        "actual_execution_round": fault_origin_wave_index + 1,
        "where": ("historical_consumed_feedback"
                  if fault_origin_wave_index < branch_round - 2
                  else "incoming_feedback"),
    } for fault in faults]
    catalog = source_catalog(scene)
    retry_counts = Counter()
    turns = []
    decisions = []
    # Actual earlier poisoned reports, dependent reports, and the late
    # clean witness must all be independently materialized in Stage 4B.
    report_reviews = prefix_report_reviews
    main_reviews = []
    evidence_plan = []
    decision_reward_states = []
    observable_issue_snapshots = []
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

    # A short branch is a *budget*, never permission to manufacture a
    # completed episode. If it expires, retain the partial execution in
    # private diagnostics and do not publish positive SFT decisions.
    replay_end = min(
        max_rounds,
        branch_round + max_post_fork_rounds - 1
        if max_post_fork_rounds is not None else max_rounds,
    )
    for round_number in range(branch_round, replay_end + 1):
        if state.total_cost > max_cost:
            stopped = "budget_exhausted"
            break
        scheduling = state.scheduling_state()
        observable_issue_snapshots.append({
            "decision_round": round_number,
            "issues": state.observable_issues(),
            "private_only": True,
        })
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
        from environment.untrusted_orchestration.rewards.decision_reward_state import capture_decision_state
        decision_reward_states.append(capture_decision_state(
            state, round_number=round_number, repairs=repairs,
            ready=ready, blocked=scheduling["blocked"],
            max_parallel_tasks=max_parallel_tasks, catalog=catalog,
            model_alias=model_alias, event_pending=event_pending,
            can_complete=can_complete,
        ))
        if can_complete:
            # A targeted recheck is not proof that two delivered independent
            # claims now agree. Otherwise the simulator could promote a
            # still-publicly-contested branch to positive SFT via Gold.
            if state.unresolved_cited_conflicts():
                stopped = "unresolved_visible_claim_conflict"
                break
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
        stopped = (
            "short_replay_horizon" if replay_end < max_rounds
            else "max_decision_rounds"
        )
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
    # For delayed propagation, positive supervision REQUIRES an observable
    # conflict, invalidation of a real pre-discovery consumer, and an actual
    # post-discovery re-execution of that consumer. A correct final answer
    # alone is insufficient.
    delayed_causal_audit = None
    if (same_source_recheck_plan is not None
            and same_source_recheck_plan.get("construction")
                == "delayed_consumed_error_recovery"):
        old_id = same_source_recheck_plan["older_task_id"]
        expected_consumers = set(
            same_source_recheck_plan["consumers_before_discovery"]
        )
        invalidated = {
            item["stale_task"] for item in state.invalidations
            if item["stale_task"] in expected_consumers
        }
        recomputed = {
            tid for tid in invalidated
            if len(state.attempts.get(tid, [])) >= 2
            and state.current[tid].status == "done"
        }
        # A dependency edge alone is not sufficient: at least one
        # completed intermediate WorkProduct must ACTUALLY differ from
        # its clean executable counterpart before the late witness.
        impacted = {
            tid for tid in expected_consumers
            if reviewed.get(tid) is not None
            and state.attempts.get(tid)
            and relation_rows(state.attempts[tid][0].product)
                != reviewed[tid].get("answer_with_provenance")
        }
        observable = any(
            event.get("kind") == "cross_round_report_disagreement"
            and set(event.get("compared_parent_tasks", [])) ==
                {old_id, same_source_recheck_plan["new_task"]["task_id"]}
            for event in state.detected_conflicts
        )
        delayed_causal_audit = {
            "fault_origin_round": fault_origin_wave_index + 1,
            "discovery_main_round": branch_round,
            "pre_discovery_consumers": sorted(expected_consumers),
            "impacted_consumers": sorted(impacted),
            "invalidated_consumers": sorted(invalidated),
            "recomputed_consumers": sorted(recomputed),
            "observable_conflict": observable,
        }
        if stopped == "recovered" and not (
            observable and impacted and invalidated
            and invalidated & impacted and invalidated <= recomputed
        ):
            stopped = "delayed_propagation_not_demonstrated"
    # LLM publication must retain both independently reviewed surfaces.
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
    from environment.untrusted_orchestration.inspection.visible_claim_state import audit_visible_claim_states
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
        "history_conflict_anchor": history_anchor,
        "prior_observation_patch": prior_observation_patch,
        "same_source_recheck_plan": same_source_recheck_plan,
        "delayed_causal_audit": delayed_causal_audit,
        "paired_clean_history": paired_clean_history,
        "counterfactual_unrepaired": before,
        "runtime": state.audit(),
        "visible_claim_state": visible_claim_state,
        "observable_issue_snapshots": observable_issue_snapshots,
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
