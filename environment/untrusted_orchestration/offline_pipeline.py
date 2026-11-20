"""Offline Stage-4B/4C: review execution reports first, then Main actions.

4A freezes evidence requests, immutable actual Attempt traces, batch ordering,
and a public-history-grounded deterministic reference schedule. 4B may change
only report language. 4C sees ONLY the finalized report history and prior Main
actions, never the private Attempt ledger or Gold. Evidence-bound assignments
remain constant in scope/tools but can reuse the newly authored public context.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Callable

from environment.base_orchestration.action_author import (
    _assignment, action_issues, proposed_template_action,
    author_delegate_action, author_complete_action,
)
from environment.base_orchestration.o1_compat import (
    o1_prompt, o1_history, source_catalog,
)
from environment.base_orchestration.trajectory_builder import _strict_reports
from environment.generate_question.execution.program_ir import Relation
from .reporting.report_author import naturalize_report
from .scenario.prior_observation import patch_scene as patch_prior_observation_scene
from .scenario.same_source_recheck import patch_scene as patch_same_source_recheck_scene
from .reporting.public_language import sanitize_public_report
from environment.untrusted_orchestration.orchestration.reasoning_fidelity import reasoning_evidence_issues, reasoning_action_issues
from environment.untrusted_orchestration.inspection.visible_history import visible_history_reasoning_issues
from environment.untrusted_orchestration.orchestration.main_decision_contract import (
    build_main_decision_contract, critic_denies_confirmed_conflict, repeated_query_conflicts,
)
from .state_machine import Attempt
from environment.untrusted_orchestration.inspection.visible_claim_state import audit_visible_claim_states
from .trajectory_builder import (
    _MAIN_DECISION_AUTHOR_SYSTEM, _MAIN_REASONING_CRITIC_SYSTEM, _history_batch,
    _as_public,
)


def _attempt_from_audit(task: dict, entry: dict) -> Attempt:
    runtime = entry["runtime"]
    # A previously executed attempt can become 'stale' or 'disputed' later.
    # Report the original delivery outcome rather than the *later* plan state.
    status = runtime.get("status", entry["status"])
    rows = entry["rows"]
    relation = None if status == "missing" else Relation(
        tuple(task["outputs"]["typed_relation"]), tuple({
            "values": deepcopy(row["values"]),
            "support": tuple(row["source_fact_refs"]),
        } for row in rows),
    )
    return Attempt(
        task_id=task["task_id"], version=entry["version"], status=status,
        product=relation,
        available_fields=tuple(runtime.get("available_fields", [])),
        used_parent_versions=entry["used_parent_versions"],
        runtime=deepcopy(runtime), cost=entry["cost"],
    )


def _refresh_visible_claim_supervision(
    scene: dict, episode: dict, audit: dict, *,
    observed_rounds: set[int] | None = None,
) -> None:
    """Recompute teacher-only evidence state from FINAL Main-visible reports.

    4A's preliminary text and 4B's finalized Sub-Agent prose are NOT
    interchangeable. When 4C stops at an invalid Main turn, only reports
    received before that failure may contribute to the diagnostic state.
    """
    fork = audit["branch_round"]
    plan = {d["round"]: d["task_ids"]
            for d in audit.get("decision_audit", [])
            if d.get("decision_kind") == "delegate"}
    waves = scene["macro_graph"]["reference_waves"]
    bindings, visible_steps = {}, []
    for step in episode["trajectory"]:
        if step.get("action") != "delegate_task":
            continue
        number = step["attempt"]
        if observed_rounds is not None and number not in observed_rounds:
            continue
        if number < fork:
            ids = waves[number - 1]
        else:
            ids = plan[number]
        bindings[number] = ids
        visible_steps.append(step)
    audit["visible_claim_state"] = audit_visible_claim_states(
        scene, visible_steps, task_ids_by_round=bindings,
    )


def author_branch_reports(scene: dict, stage2_rows: list[dict],
                          episode: dict, audit: dict, *,
                          author=None, critic=None, max_revisions: int = 2,
                          on_report=None, resume_reports=None,
                          on_report_checkpoint=None) -> tuple[dict, dict]:
    """4B authors *only* actual previously simulated Attempt returns.

    The original trusted prefix reports are reused verbatim. 4B does not
    schedule tasks, invoke Main, or redo the underlying poisoned program.
    """
    episode, audit = deepcopy(episode), deepcopy(audit)
    # Stage 4 changes the latest delivered report and may expose an
    # already-executed older record; it never adds a hidden verifier task.
    reviewed = _strict_reports(scene, stage2_rows)
    scene = patch_same_source_recheck_scene(
        scene, audit.get("same_source_recheck_plan"),
    )
    scene = patch_prior_observation_scene(
        scene, audit.get("prior_observation_patch"),
    )
    tasks = {t["task_id"]: t for t in scene["macro_graph"]["tasks"]}
    versions = audit["runtime"]["attempt_versions"]
    decision_ids = {
        d["round"]: d["task_ids"]
        for d in audit["decision_audit"] if d.get("decision_kind") == "delegate"
    }
    waves = scene["macro_graph"]["reference_waves"]
    fork = audit["branch_round"]
    review_rows = []
    saved_reports = list(resume_reports or [])
    if len(saved_reports) > len(audit["subagent_report_reviews"]):
        raise ValueError("stage4b_checkpoint_exceeds_report_count")
    for index, scheduled in enumerate(audit["subagent_report_reviews"]):
        tid, version, round_number = (
            scheduled["task_id"], scheduled["attempt_version"],
            scheduled["round"],
        )
        wave = (waves[round_number - 1] if round_number < fork
                else decision_ids[round_number])
        slot = wave.index(tid)
        step = next(
            x for x in episode["trajectory"]
            if x["action"] == "delegate_task" and x["attempt"] == round_number
        )
        old = step["result"]["subtask_results"][slot]
        record = next(
            v for v in versions[tid] if v["version"] == version
        )
        actual = _attempt_from_audit(tasks[tid], record)
        if index < len(saved_reports):
            cached = saved_reports[index]
            if (cached.get("ordinal") != index + 1
                    or cached.get("task_id") != tid
                    or cached.get("attempt_version") != version
                    or cached.get("round") != round_number
                    or cached.get("task_instruction") != old["task_instruction"]
                    or cached.get("model") != old["model"]):
                raise ValueError("stage4b_checkpoint_report_alignment_changed")
            report = deepcopy(cached["report"])
            status = deepcopy(cached["authoring"])
        else:
            report, status = naturalize_report(
                tasks[tid], actual,
                instruction=old["task_instruction"], model_alias=old["model"],
                reviewed=reviewed.get(tid), author=author, critic=critic,
                max_revisions=max_revisions,
            )
            if on_report_checkpoint is not None:
                on_report_checkpoint({
                    "ordinal": index + 1,
                    "task_id": tid,
                    "attempt_version": version,
                    "round": round_number,
                    "task_instruction": old["task_instruction"],
                    "model": old["model"],
                    "report": deepcopy(report),
                    "authoring": deepcopy(status),
                })
        report["subtask_index"] = slot
        step["result"]["subtask_results"][slot] = _as_public(
            sanitize_public_report(report)
        )
        review_rows.append({
            **scheduled, "authoring": deepcopy(status),
        })
        if round_number == fork - 1:
            for fault in audit["faults"]:
                if fault["task_id"] == tid:
                    fault["report_mode"] = deepcopy(status)
        if on_report is not None:
            on_report(status)
    audit["subagent_report_reviews"] = review_rows
    _refresh_visible_claim_supervision(scene, episode, audit)
    return episode, audit


def _public_done_null_fields(entries: list[dict]) -> set[str]:
    """Inspect the real O1 *structured* history, not its formatted prose.

    _history_batch/public_history_item flattens finish_result into top-level
    status/result/summary; the O1 prompt formatter does not produce JSON.
    This check uses exactly the report values already delivered to Main.
    """
    from environment.untrusted_orchestration.orchestration.main_decision_contract import _visible_finish
    missing = set()
    for batch in entries:
        for report in batch.get("subtask_results", []):
            if not isinstance(report, dict):
                continue
            finish = _visible_finish(report)
            if not isinstance(finish, dict) or finish.get("status") != "done":
                continue
            try:
                from .reporting.result_text import delivered_records
                rows = delivered_records(finish.get("result", ""))
            except (ValueError, TypeError):
                continue
            if not isinstance(rows, list) or not rows:
                continue
            records = [
                item.get("values", item) for item in rows
                if isinstance(item, dict)
            ]
            if not records or not all(isinstance(row, dict)
                                      for row in records):
                continue
            for field in set().union(*(set(row) for row in records)):
                if all(row.get(field) is None for row in records):
                    missing.add(field)
    return missing


def _visibility_issues(
    requests: list[dict], entries: list[dict], public_history: str,
) -> list[str]:
    """Validate evidence needs against Main's actually received history."""
    visible = public_history.casefold()
    delivered_null_fields = _public_done_null_fields(entries)
    missing = []
    for request in requests:
        for identifier in request.get("known_identifiers", []):
            if identifier.casefold() not in visible:
                missing.append(request["request_id"] + ":" + identifier)
        if request["operation"] in {
            "retry_missing_evidence", "complete_partial_evidence",
        }:
            if (not any(x in visible for x in (
                "missing", "partial", "no result", "unresolved",
                "not delivered", "incomplete", "did not return",
                "did not produce",
            )) and not set(request.get("target_fields") or [])
                            & delivered_null_fields):
                missing.append(request["request_id"] + ":missing_status")
        if request["operation"] == "verify_conflicting_evidence":
            from environment.untrusted_orchestration.runtime.evidence_planning import _IDENT
            for identifier in _IDENT.findall(
                request["main_visible_justification"]
            ):
                if identifier.casefold() not in visible:
                    missing.append(
                        request["request_id"] + ":unseen_conflict:" + identifier
                    )
    return missing


def _completion_from_visible(step: dict, latest: dict,
                             scene: dict) -> dict:
    terminal = scene["macro_graph"]["terminal_task_id"]
    tasks = {t["task_id"]: t for t in scene["macro_graph"]["tasks"]}
    refs = ([terminal] if terminal in latest else
            tasks[terminal]["depends_on"])
    quotes = []
    for tid in refs[-2:]:
        finish = (latest.get(tid) or {}).get("finish_result") or {}
        value = str(finish.get("result") or "").strip().replace("\n", " ")
        if value:
            quotes.append(value[:360].rstrip())
    if not quotes:
        raise ValueError("stage4c_completion_has_no_visible_result")
    reason = " ".join(
        'The returned finding states: "' + quote + '".'
        for quote in quotes
    ) + (
        " These returned findings provide the required evidence "
        "for the final answer."
    )
    return {
        "action": "complete", "reasoning": reason,
        "params": deepcopy(step["params"]),
    }


def _validate_evidence_dispatch(plan: dict, assignments: list[dict],
                                *, prompt: str) -> None:
    """Code-owned 4A/4B semantic and source contract, BEFORE Main authoring.

    The LLM must not be tasked with deciding whether a frozen Macro
    operator/route is valid. It can only explain a validated decision.
    """
    requests = {
        item["request_id"]: item for item in plan["evidence_requests"]
    }
    dispatches = plan["dispatch"]
    if (not dispatches or len(dispatches) != len(assignments)
            or len({x["execution_binding"]["macro_task_id"]
                    for x in dispatches}) != len(dispatches)):
        raise ValueError("stage4c_dispatch_cardinality_or_binding_invalid")
    covered = []
    for i, (dispatch, assignment) in enumerate(zip(dispatches, assignments)):
        contracted = dispatch["assignment_contract"]
        req_ids = dispatch["evidence_request_ids"]
        if not req_ids or any(x not in requests for x in req_ids):
            raise ValueError("stage4c_dispatch_unresolved_request")
        covered.extend(req_ids)
        if any(assignment[k] != contracted[k] for k in (
            "model", "tools", "source_ids", "output_fields",
            "focus_fields", "needs_upstream"
        )):
            raise ValueError("stage4c_dispatch_source_or_schema_changed")
        instruction = assignment["target_instruction"].casefold()
        for rid in req_ids:
            for field in requests[rid]["target_fields"]:
                if (field.casefold() not in instruction
                        and field.replace("_", " ").casefold()
                            not in instruction):
                    raise ValueError(
                        f"stage4c_dispatch_missing_requested_field:{i}:{field}"
                    )
        if not all(
            sid in assignment["target_instruction"] + assignment["context"]
            for sid in contracted["source_ids"]
        ):
            raise ValueError("stage4c_dispatch_source_not_named")
    if len(covered) != len(set(covered)) or set(covered) != set(requests):
        raise ValueError("stage4c_dispatch_not_exact_cover")
    issues = action_issues(
        proposed_template_action(assignments), assignments, prompt=prompt,
    )
    if issues:
        raise ValueError(
            "stage4c_dispatch_o1_contract_invalid:" + ",".join(issues)
        )


def author_branch_main(scene: dict, episode: dict, audit: dict, *,
                       author=None, critic=None,
                       max_revisions: int = 2,
                       max_rounds: int = 16,
                       sub_model: str = "qwen3-8b",
                       model_alias: str = "model_1",
                       on_main=None, resume_actions=None,
                       on_main_checkpoint=None) -> tuple[dict, list[dict], dict]:
    """4C writes all Main turns after 4B has completed, from public history."""
    episode, audit = deepcopy(episode), deepcopy(audit)
    # Trusted Stage-3 prefix reports may still contain synthetic Macro /
    # primitive IDs. Normalize their public PROSE before building either
    # Main's O1 prompt or the emitted SFT subtask_history. Never alter the
    # underlying WorkProduct / source provenance in the private ledger.
    for visible_step in episode["trajectory"]:
        if visible_step.get("action") != "delegate_task":
            continue
        report_list = (visible_step.get("result") or {}).get(
            "subtask_results", []
        )
        visible_step["result"]["subtask_results"] = [
            sanitize_public_report(report) for report in report_list
        ]
    scene = patch_same_source_recheck_scene(
        scene, audit.get("same_source_recheck_plan"),
    )
    scene = patch_prior_observation_scene(
        scene, audit.get("prior_observation_patch"),
    )
    fork = audit["branch_round"]
    plans = {p["round"]: p for p in audit.get("evidence_plan", [])}
    tasks = {t["task_id"]: t for t in scene["macro_graph"]["tasks"]}
    catalog = source_catalog(scene)
    entries, latest, turns, reviews = [], {}, [], []
    latest_rounds, latest_versions, latest_parent_versions = {}, {}, {}
    saved_actions = list(resume_actions or [])
    expected_actions = sum(
        step["attempt"] >= fork and step["action"] in {
            "delegate_task", "complete",
        } for step in episode["trajectory"]
    )
    if len(saved_actions) > expected_actions:
        raise ValueError("stage4c_checkpoint_exceeds_decision_count")
    failure = None
    for step in episode["trajectory"]:
        round_number = step["attempt"]
        if round_number >= fork and step["action"] in {
            "delegate_task", "complete",
        }:
            prompt = o1_prompt(
                scene, entries, attempt=round_number,
                max_attempts=max_rounds, sub_model=sub_model,
                model_alias=model_alias,
            )
            if step["action"] == "delegate_task":
                plan = plans[round_number]
                history = json.dumps(entries, ensure_ascii=False)
                issues = _visibility_issues(plan["evidence_requests"], entries, history)
                if issues:
                    failure = "stage4c_evidence_not_main_visible:" + ",".join(issues)
                    break
                assignments = []
                for dispatch in plan["dispatch"]:
                    tid = dispatch["execution_binding"]["macro_task_id"]
                    planned = dispatch["assignment_contract"]
                    # The fixed semantics and tool scope come from 4A.
                    # Stage-4B's actual report language supplies up-to-date
                    # parent context; never copy future or private records.
                    current = _assignment(
                        tasks[tid], catalog, latest, model_alias=model_alias,
                    )
                    current["target_instruction"] = planned["target_instruction"]
                    current["focus_fields"] = list(planned["focus_fields"])
                    current["context"] = (
                        current["context"] + " Current evidence need: "
                        + next(req["main_visible_justification"]
                               for req in plan["evidence_requests"]
                               if req["request_id"] in
                               dispatch["evidence_request_ids"])
                    )[:3500]
                    assignments.append(current)
                try:
                    _validate_evidence_dispatch(
                        plan, assignments, prompt=prompt,
                    )
                except ValueError as exc:
                    # Keep the *entire* preceding 4B language audit and
                    # prior 4C reviewed decisions. This is a publication
                    # failure, not a new execution failure or a reason to
                    # erase the successfully replayed 4A evidence.
                    failure = (
                        "stage4c_dispatch_validation_rejected:" + str(exc)
                    )
                    break
                # Both Author and Critic see the *observable* reason
                # for a pending check, not merely a list of output fields.
                # Without it the Critic can falsely deny a genuine
                # Result/Summary discrepancy in the delivered reports.
                visible_goals = [{
                    "operation": req["operation"],
                    "target_fields": list(req["target_fields"]),
                    "main_visible_justification":
                        req["main_visible_justification"],
                } for req in plan["evidence_requests"]]
                # Compile only the facts Main could read in its current
                # O1 history. A private planner's VERIFY label alone cannot
                # establish a contradiction: Result/Summary must differ in
                # one delivered public Sub-Agent report.
                decision_contract = build_main_decision_contract(
                    entries, visible_goals, number_of_tasks=len(assignments),
                )
                history_conflicts = repeated_query_conflicts(
                    tasks, latest, latest_rounds, latest_parent_versions,
                )
                decision_contract["visible_evidence"]["confirmed_repeated_query_conflicts"] = history_conflicts
                author_system = _MAIN_DECISION_AUTHOR_SYSTEM
                if history_conflicts:
                    author_system += (
                        "\nThe following already received reports describe the same "
                        "delegated source query with the same prerequisite input versions. "
                        "Their returned values still disagree. Explain the targeted check "
                        "without treating a calculated downstream answer as verification:\n"
                        + json.dumps(history_conflicts, ensure_ascii=False)
                    )
                if any(goal["operation"] == "derive_from_returned_evidence"
                       for goal in visible_goals):
                    author_system += (
                        "\nThis round performs a pending derivation. Describe how "
                        "the identified entity will be matched against the returned "
                        "records and how the requested comparison or calculation "
                        "will be performed. Keep the explanation focused on that "
                        "pending work. Do not assert that all prerequisites are "
                        "verified or retell older event-validation steps."
                    )
                guard_drafts = []
                def check_reasoning(text):
                    problems = (
                        reasoning_evidence_issues(text, visible_goals)
                        + reasoning_action_issues(
                            text, assignments, question=scene["question"],
                            visible_text=prompt + " " + json.dumps(
                                assignments, ensure_ascii=False
                            ) + " " + json.dumps(history_conflicts, ensure_ascii=False),
                        )
                        + visible_history_reasoning_issues(
                            text, decision_contract["prior_field_activity"])
                    )
                    if problems:
                        guard_drafts.append({"reasoning": text, "issues": list(problems)})
                    hints = {
                        "unobserved_identifier_in_reasoning":
                            "Copy identifiers exactly from the already received reports "
                            "or omit an unnecessary identifier; do not introduce a new or misspelled key.",
                        "observation_year_lookup_claims_date_coordinate_extraction":
                            "This lookup returns observation identifiers and years for counting. "
                            "Do not say it extracts dates, coordinates or location measurements.",
                        "intermediate_identifier_claimed_as_final_place_name":
                            "The entity_N value is an intermediate relationship identifier. "
                            "Explain its role as the next lookup key, not a final place name.",
                        "intermediate_field_claimed_as_question_output":
                            "Explain that this identifier enables an intermediate lookup. "
                            "Do not say the user explicitly requests this intermediate field.",
                        "date_pair_replaced_with_observation_date_extrema":
                            "Use the named date_a and date_b values. Observation-log years "
                            "are for the observation count, not the date-pair calculation.",
                        "source_free_action_claims_external_acquisition":
                            "Explain recomputation from already returned observations; "
                            "the current assignment performs no new external acquisition.",
                    }
                    return [
                        issue + (": " + hints[issue.split(":", 1)[0]]
                                 if issue.split(":", 1)[0] in hints else "")
                        for issue in problems
                    ]
                cached = (saved_actions[len(reviews)]
                          if len(reviews) < len(saved_actions) else None)
                if cached is not None:
                    if (cached.get("ordinal") != len(reviews) + 1
                            or cached.get("round") != round_number
                            or cached.get("kind") != "delegate"
                            or cached.get("prompt_sha256") != sha256(
                                prompt.encode("utf-8")
                            ).hexdigest()):
                        raise ValueError("stage4c_checkpoint_dispatch_drift")
                    action = deepcopy(cached["action"])
                    status = deepcopy(cached["authoring"])
                else:
                    action, status = author_delegate_action(
                        assignments=assignments, prompt=prompt,
                        author=author, critic=critic,
                        max_revisions=max_revisions,
                        author_system=author_system,
                        critic_system=_MAIN_REASONING_CRITIC_SYSTEM,
                        freeze_assignment_prose=True,
                        reasoning_goals=visible_goals,
                        decision_contract=decision_contract,
                        critic_conflict_objection=(
                            lambda review: critic_denies_confirmed_conflict(
                                review, decision_contract,
                            )
                        ),
                        reasoning_guard=check_reasoning if author is not None else None,
                    )
                    if guard_drafts:
                        status["reasoning_guard_drafts"] = guard_drafts
                # The final Main instruction/context must be exactly the
                # field-and-source-scoped 4A contract rebuilt against FINAL
                # 4B evidence, even if a fluent LLM drafted an unrelated
                # task objective. Never mutate the contract to fit the LLM.
                if action is not None and action.get("params") != (
                    proposed_template_action(assignments)["params"]
                ):
                    raise ValueError(
                        "stage4c_evidence_assignment_semantic_drift"
                    )
            else:
                expected = _completion_from_visible(step, latest, scene)
                cached = (saved_actions[len(reviews)]
                          if len(reviews) < len(saved_actions) else None)
                if cached is not None:
                    if (cached.get("ordinal") != len(reviews) + 1
                            or cached.get("round") != round_number
                            or cached.get("kind") != "complete"
                            or cached.get("prompt_sha256") != sha256(
                                prompt.encode("utf-8")
                            ).hexdigest()):
                        raise ValueError("stage4c_checkpoint_completion_drift")
                    action = deepcopy(cached["action"])
                    status = deepcopy(cached["authoring"])
                else:
                    action, status = author_complete_action(
                        template=expected, prompt=prompt,
                        delivered={
                            k: v for k, v in latest.items()
                            if isinstance(v.get("finish_result"), dict)
                        },
                        author=author, critic=critic,
                        max_revisions=max_revisions,
                    )
            if cached is None and on_main_checkpoint is not None:
                on_main_checkpoint({
                    "ordinal": len(reviews) + 1,
                    "round": round_number,
                    "kind": ("delegate" if step["action"] == "delegate_task"
                             else "complete"),
                    "prompt_sha256": sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "action": deepcopy(action),
                    "authoring": deepcopy(status),
                })
            reviews.append({
                "round": round_number,
                "kind": ("delegate" if step["action"] == "delegate_task"
                         else "complete"),
                "authoring": deepcopy(status),
            })
            if on_main is not None:
                on_main(status)
            if action is None:
                failure = "main_action_not_accepted"
                break
            step["reasoning"] = action["reasoning"]
            step["params"] = deepcopy(action["params"])
            if step["action"] == "delegate_task":
                # In this offline pipeline 4B was authored before 4C.
                # Rebind only the public *instruction metadata* to the
                # final Main-assigned language. Actual observed evidence,
                # report prose and frozen WorkProduct are unchanged.
                for report, task_action in zip(
                    step["result"]["subtask_results"],
                    action["params"]["tasks"],
                ):
                    report["task_instruction"] = task_action[
                        "task_instruction"
                    ]
                    report["model"] = task_action["model"]
            prefix = (
                (audit.get("event_scenario") or {}).get("scenario_id") or ""
            )
            sample_id = (
                prefix + f"-round-{round_number:02d}"
                if "-candidate-" in prefix else
                scene["task_id"] +
                f"-branch-{fork:02d}-round-{round_number:02d}"
            )
            turns.append({
                "sample_id": sample_id,
                "task_id": scene["task_id"], "round": round_number,
                "input": {
                    "subtask_history": o1_history(entries),
                },
                "output": deepcopy(action),
                "generation_mode": status["mode"],
            })
        if step["action"] == "delegate_task":
            ids = (scene["macro_graph"]["reference_waves"][round_number - 1]
                   if round_number < fork else
                   [d["execution_binding"]["macro_task_id"]
                    for d in plans[round_number]["dispatch"]])
            returned = step["result"]["subtask_results"]
            before_versions = dict(latest_versions)
            for tid, report in zip(ids, returned):
                latest[tid] = report
                latest_rounds[tid] = round_number
                latest_versions[tid] = latest_versions.get(tid, 0) + 1
                if all(parent in before_versions for parent in tasks[tid]["depends_on"]):
                    latest_parent_versions[tid] = {
                        parent: before_versions[parent] for parent in tasks[tid]["depends_on"]
                    }
                else:
                    # Unreported internal execution cannot prove that two
                    # publicly delegated queries used identical input versions.
                    latest_parent_versions.pop(tid, None)
            entries.append(_history_batch(round_number, returned))
    execution_outcome = audit["execution_outcome"]
    had_report_fallback = any(
        row["authoring"].get("mode") == "grounded_template_fallback"
        for row in audit["subagent_report_reviews"]
    )
    had_main_fallback = any(
        row["authoring"].get("mode") ==
        "teacher_grounded_completion_fallback"
        for row in reviews
    )
    if failure is not None:
        outcome = ("language_review_unaccepted"
                   if failure.startswith((
                       "stage4c_evidence_not_main_visible",
                       "stage4c_dispatch_validation_rejected",
                   ))
                   else failure)
    elif execution_outcome == "recovered" and (
        had_report_fallback or had_main_fallback
    ):
        outcome = "language_review_unaccepted"
    else:
        outcome = execution_outcome
    # Preserve each individually reviewed Main decision in the PRIVATE
    # per-round audit even when a later report or terminal result invalidates
    # the complete branch. The publisher may review *local* decisions
    # separately, but cannot silently promote them to globally correct reviewed-turn.
    # Use only the reports actually added to Main's O1 history.
    # A failed 4C turn does not grant access to this turn's future
    # simulated Sub-Agent returns, even if 4A/4B precomputed them.
    _refresh_visible_claim_supervision(
        scene, episode, audit,
        observed_rounds={item["attempt"] for item in entries},
    )
    # Stage4B/4C may replace the original 4A report prose. Refresh reward
    # states using the FINAL reports actually delivered to Main; never train
    # on 4A-only snapshots which might disagree with final visible history.
    if outcome == "recovered":
        from .decision_reward_replay import replay_reward_states
        settings = audit.get("decision_reward_settings") or {}
        try:
            audit["decision_reward_states"] = replay_reward_states(
                scene, episode, audit,
                max_parallel_tasks=settings.get("max_parallel_tasks", 4),
                max_retries=settings.get("max_retries", 1),
                model_alias=settings.get("model_alias", model_alias),
            )
            audit["decision_reward_replay_error"] = None
        except (ValueError, KeyError, TypeError) as exc:
            # Preserve accepted SFT turns; exclude ONLY unsafe reward rows.
            # The private publisher records the explicit missing coverage.
            audit["decision_reward_states"] = []
            audit["decision_reward_replay_error"] = str(exc)[:300]
    audit["candidate_local_turns"] = deepcopy(turns)
    audit["main_action_reviews"] = reviews
    audit["outcome"] = outcome
    audit["language_review_accepted"] = (
        failure is None and not had_report_fallback and not had_main_fallback
    )
    audit["language_evidence_issues"] = [failure] if failure else []
    audit["accepted_turns"] = len(turns) if outcome == "recovered" else 0
    episode["outcome"] = outcome
    episode["recovered"] = outcome == "recovered"
    return episode, turns if outcome == "recovered" else [], audit
