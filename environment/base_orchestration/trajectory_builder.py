"""Construct trusted Orchestra-o1 *per-decision* supervised examples.

One wave of independent Macro Tasks -> one Main 'delegate_task' action;
its reviewed Stage-2 responses become visible only in the NEXT round.
An executable Main-local terminal calculation may become 'complete'
without unnecessarily delegating another synthetic Sub-Agent.

The full teacher alignment is returned separately and never embedded in
a prompt or training label.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any

from environment.generate_question.auditing.quality import validate_replay
from environment.implement_subtask.realize import realize_scene
from environment.implement_subtask.public_surface import reviewed_public_view

from .o1_compat import (
    o1_prompt, o1_history, o1_prompt_from_turn, source_catalog,
    stage3_author_context,
)
from .action_author import (
    _assignment, action_issues, completion_issues,
    author_delegate_action, author_complete_action,
    STAGE3_REASONING_AUTHOR_SYSTEM, STAGE3_REASONING_CRITIC_SYSTEM,
    STAGE3_TASK_PROSE_AUTHOR_SYSTEM, STAGE3_TASK_PROSE_CRITIC_SYSTEM,
    stage3_reasoning_issues,
)
from .reasoning_fidelity import (
    build_round_evidence_contract, check_reasoning_fidelity,
)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def clean_reference_identity(scene: dict, reports: dict[str, dict]) -> dict:
    """Stable pairing identity for Stage-3 and Stage-4 clean histories.

    Depends on the frozen environment and ACTUALLY reviewed Stage-2 returns,
    not on optional LLM wording of Main actions. No private facts are
    published through the ID itself.
    """
    world_sha = _hash(scene["world"])
    graph_sha = _hash(scene["macro_graph"])
    report_sha = _hash({
        tid: reports[tid]["finish_result"]
        for tid in sorted(reports)
    })
    identity = _hash([
        scene["task_id"], world_sha, graph_sha, report_sha,
    ])
    return {
        "version": "maestro.clean_reference.v1",
        "clean_reference_id": "clean-" + identity[:24],
        "world_sha256": world_sha,
        "macro_graph_sha256": graph_sha,
        "stage2_public_reports_sha256": report_sha,
    }


def _strict_reports(scene: dict, rows: list[dict]) -> dict[str, dict]:
    if validate_replay(scene)["status"] != "valid":
        raise ValueError("stage3_invalid_frozen_scene")
    expected = {row["macro_task_id"]: row for row in realize_scene(scene)[
        "macro_tasks"
    ]}
    accepted = {}
    for row in rows:
        tid = row.get("macro_task_id")
        if tid in accepted or tid not in expected:
            raise ValueError("stage3_duplicate_or_unrecognized_macro")
        if row.get("scene_task_id") != scene["task_id"]:
            raise ValueError("stage3_report_wrong_scene")
        if row.get("report_authoring", {}).get("status") != "accepted":
            raise ValueError("stage3_report_not_reviewed")
        oracle = expected[tid]
        frozen_keys = [
            "answer", "answer_with_provenance", "depends_on",
            "authorized_source_ids", "output_schema",
        ]
        if "required_tools" in oracle:
            frozen_keys.append("required_tools")
        if any(row.get(key) != oracle[key] for key in frozen_keys):
            raise ValueError("stage3_report_not_grounded:" + tid)
        if "required_tools" in oracle:
            expected_steps = oracle["trace_summary"]["steps"]
            reported_steps = row.get("trace_summary", {}).get("steps")
            if not isinstance(reported_steps, list) or len(
                reported_steps
            ) != len(expected_steps):
                raise ValueError("stage3_report_tool_trace_missing:" + tid)
            for actual_step, expected_step in zip(reported_steps, expected_steps):
                if (
                    actual_step.get("evidence_acquisition") !=
                    expected_step.get("evidence_acquisition")
                    or actual_step.get("tool_executed") !=
                    expected_step.get("tool_executed")
                    or actual_step.get("simulated_tool_calls") !=
                    expected_step.get("simulated_tool_calls")
                ):
                    raise ValueError("stage3_report_acquisition_tool_drift:" + tid)
        finish = row.get("finish_result")
        if not isinstance(finish, dict) or finish.get("status") != "done":
            raise ValueError("stage3_report_not_done:" + tid)
        if (not isinstance(finish.get("result"), str)
                or not isinstance(finish.get("summary"), str)
                or not finish["result"].strip()
                or not finish["summary"].strip()):
            raise ValueError("stage3_report_lacks_public_result:" + tid)
        # Historical reports may omit narrative; the audited finish summary
        # is still a legitimate public trace summarization.
        narrative = row.get("trace_summary", {}).get("narrative")
        if narrative is not None and not isinstance(narrative, str):
            raise ValueError("stage3_invalid_public_trace_narrative")
        accepted[tid] = deepcopy(row)
    if set(accepted) != set(expected):
        raise ValueError("stage3_requires_complete_accepted_scene")
    return accepted


def _main_local_terminal(graph: dict) -> bool:
    terminal = next(row for row in graph["tasks"] if row["task_id"] ==
                    graph["terminal_task_id"])
    operations = [step["op"] for step in terminal["operator_chain"]]
    return (
        not terminal["source_ids"]
        and bool(set(operations) & {"derive_comparison", "derive_percentage"})
        and set(operations).issubset({
            "join", "project", "unique",
            "derive_comparison", "derive_percentage",
        })
    )


def _can_derive_local_answer(scene: dict, delivered: dict,
                             tasks: dict) -> bool:
    """Use *delivered* trusted parent facts, not a private future result."""
    if not _main_local_terminal(scene["macro_graph"]):
        return False
    terminal_id = scene["macro_graph"]["terminal_task_id"]
    terminal = tasks[terminal_id]
    parents = list(terminal["depends_on"])
    if len(parents) != 2 or not all(p in delivered for p in parents):
        return False
    plan = scene["problem_spec"].get("macro_plan", {})
    kind = plan.get("goal_resolution", {}).get("kind")
    if kind not in ("comparison", "calculation"):
        return False
    values = {}
    for parent in parents:
        row = delivered[parent]
        if len(row["answer"]) != 1:
            return False
        # A value in a private typed Oracle row is NOT sufficient to
        # justify a Main-local calculation: the Sub-Agent must actually
        # have communicated that value in the preceding visible message.
        visible_text = (
            row["finish_result"]["result"] + " " +
            row["finish_result"]["summary"]
        )
        for value in row["answer"][0].values():
            if type(value) is int and not re.search(
                r"(?<!\d)" + re.escape(str(value)) + r"(?!\d)",
                visible_text,
            ):
                return False
        values.update(row["answer"][0])
    first, second = (
        ("metric_a", "metric_b") if kind == "comparison"
        else ("completed_count", "total_count")
    )
    if first not in values or second not in values:
        return False
    from environment.generate_question.planning.goal_resolution import goal_values
    actual = goal_values(kind, values[first], values[second])
    field = plan["goal"]["answer_field"]
    return scene["gold_answer"] == [{field: actual}]


def _visible_terminal_answer(scene: dict, delivered: dict,
                             tasks: dict) -> bool:
    """Allow only answer-preserving terminal transforms over visible results.

    In archived Easy plans, an initial Sub-Agent may already identify all
    active Entry labels or the unique top-ranked Entry; a separate
    take/project/unique delegation adds no evidence. Do not generalize this
    to new-source, join, arithmetic, or otherwise nontrivial terminals.
    The comparison uses the *public finish result*, not a private Oracle
    value in the Sub-Agent's structured answer. Gold is only a verifier.
    """
    graph = scene["macro_graph"]
    terminal_id = graph["terminal_task_id"]
    if terminal_id in delivered:
        return False
    terminal = tasks[terminal_id]
    if terminal["source_ids"] or len(terminal["depends_on"]) != 1:
        return False
    operators = [step["op"] for step in terminal["operator_chain"]]
    if not operators or not set(operators).issubset(
        {"take", "project", "unique"}
    ):
        return False
    upstream = terminal["depends_on"][0]
    if upstream not in delivered:
        return False
    answer = scene["gold_answer"]
    if not answer or any(
        set(row) != {"entry_label"} or
        not isinstance(row["entry_label"], str) or
        not re.fullmatch(r"Entry-\d+", row["entry_label"])
        for row in answer
    ):
        return False
    public_result = delivered[upstream]["finish_result"]["result"]
    observed = set(re.findall(r"\bEntry-\d+\b", public_result))
    expected = {row["entry_label"] for row in answer}
    return bool(observed) and observed == expected


def _grounded_completion_reasoning(delivered: dict, *,
                                   shortcut: bool,
                                   main_compute: bool) -> str:
    """Phrase completion only from delivered Sub-Agent public feedback."""
    if not delivered:
        raise ValueError("completion_requires_visible_feedback")
    recent = list(delivered.values())[-2:] if main_compute else [
        list(delivered.values())[-1]
    ]
    excerpts = []
    for report in recent:
        finish = report["finish_result"]
        statement = finish["result"].strip().replace("\n", " ")
        # This is a quoted, already-visible Sub-Agent response, not the
        # private target answer or an imagined tool result.
        if statement:
            excerpts.append(statement[:360].rstrip())
    if not excerpts:
        raise ValueError("completion_lacks_public_evidence")
    cited = " ".join('The returned finding states: "' + part + '".'
                     for part in excerpts)
    if shortcut:
        conclusion = (
            "These observations already identify every requested entry. "
            "A further extraction or deduplication delegation would not "
            "add evidence, so I can finish using the reported result."
        )
    elif main_compute:
        conclusion = (
            "The required numerical inputs are now available in the "
            "earlier feedback. Their verified comparison or calculation "
            "determines the requested answer without another lookup."
        )
    else:
        conclusion = (
            "The completed investigation has supplied the requested "
            "finding with its supporting evidence. No additional "
            "Sub-Agent delegation is needed to answer."
        )
    return cited + " " + conclusion


def _answer_text(scene: dict) -> str:
    rows = scene["gold_answer"]
    if len(rows) == 1 and len(rows[0]) == 1:
        field, value = next(iter(rows[0].items()))
        if field == "completion_rate_percent":
            return str(value) + "%"
        return str(value)
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def _main_report(record: dict, assigned: dict, index: int,
                 *, model_alias: str) -> dict:
    """Return the *public* delegate_task result, never Oracle structure."""
    finish = deepcopy(record["finish_result"])
    narrative = record.get("trace_summary", {}).get("narrative")
    if not isinstance(narrative, str) or not narrative.strip():
        narrative = finish["summary"]
    return {
        "subtask_index": index,
        "task_instruction": assigned["task_instruction"],
        "model": model_alias,
        "steps_taken": len(record["trace_summary"]["steps"]),
        "statistics": {"max_steps": 30},
        "done": True,
        "finish_result": finish,
        "trace_summary": narrative,
        "cost": 0.0,  # Offline replay, not a measured LLM/Tool bill.
    }


def _history_item(sub_result: dict) -> dict:
    finish = sub_result["finish_result"]
    return {
        "status": finish["status"],
        "instruction": sub_result["task_instruction"],
        "model": sub_result["model"],
        "steps_taken": sub_result["steps_taken"],
        "max_steps": sub_result["statistics"]["max_steps"],
        "cost": sub_result["cost"],
        "result": finish["result"],
        "summary": finish["summary"],
        "trace_summary": sub_result["trace_summary"],
    }


def build_episode(
    scene: dict, stage2_rows: list[dict], *, author=None, critic=None,
    max_revisions: int = 2, max_attempts: int = 16,
    sub_model: str = "qwen3-8b", model_alias: str = "model_1",
    resume_rounds: list[dict] | None = None,
    on_round=None, on_episode_start=None,
    freeze_dispatch: bool = False,
) -> tuple[dict | None, list[dict], list[dict], dict]:
    """Return (public Episode, reviewed-turn turns, PRIVATE alignments, status)."""
    if max_attempts < 2:
        raise ValueError("stage3_max_attempts_too_small")
    if not isinstance(model_alias, str) or not model_alias:
        raise ValueError("stage3_invalid_model_alias")
    # The reviewed Stage-2 bank remains immutable. Main consumes a
    # non-destructive display copy. Stage-4 clean prefixes reuse it.
    reports = {
        tid: reviewed_public_view(row)
        for tid, row in _strict_reports(scene, stage2_rows).items()
    }
    graph = scene["macro_graph"]
    # Versioned V1.4 scenes train Main reasoning over a frozen
    # dependency-safe dispatch, not redundant action-structure synthesis.
    # Stage-1 was consolidated into the current Task Plan. The old
    # evidence_trust_version marker is no longer issued, so it must not
    # disable the explicitly requested clean frozen-dispatch baseline.
    freeze_trusted_dispatch = bool(freeze_dispatch)
    tasks = {row["task_id"]: row for row in graph["tasks"]}
    if len(tasks) != len(graph["tasks"]):
        raise ValueError("stage3_duplicate_graph_task")
    catalog = source_catalog(scene)
    waves = graph["reference_waves"]
    if not waves:
        raise ValueError("stage3_empty_reference_schedule")
    main_compute = (
        _main_local_terminal(graph)
        and _can_derive_local_answer(scene, reports, tasks)
    )
    if main_compute and waves[-1] != [graph["terminal_task_id"]]:
        raise ValueError("stage3_terminal_not_separate_last_wave")
    # A teacher shortcut is legal only if its exact public upstream report
    # will have returned before the terminal wave. Check again against
    # actually delivered history before emitting the final completion.
    preview_prior = {
        tid: reports[tid] for wave in waves[:-1] for tid in wave
    }
    visible_answer_shortcut = (
        waves[-1] == [graph["terminal_task_id"]]
        and _visible_terminal_answer(scene, preview_prior, tasks)
    )
    scheduled = (waves[:-1] if (main_compute or visible_answer_shortcut)
                 else waves)
    if len(scheduled) + 1 > max_attempts:
        return None, [], [], {
            "status": "rejected", "cause": "stage3_round_budget_insufficient",
            "rounds_required": len(scheduled) + 1,
        }

    saved_rounds = list(resume_rounds or [])
    total_rounds = len(scheduled) + 1
    if len(saved_rounds) > total_rounds:
        raise RuntimeError("stage3_checkpoint_has_extra_rounds")
    for index, saved in enumerate(saved_rounds, 1):
        if not isinstance(saved, dict) or saved.get("round") != index:
            raise RuntimeError("stage3_checkpoint_round_order_invalid")
    if on_episode_start is not None:
        on_episode_start(total_rounds, len(saved_rounds))

    task_entries, delivered = [], {}
    trajectory, examples, audit = [], [], []
    for wave_number, wave in enumerate(scheduled, 1):
        # Teacher uses the DAG to choose targets, but only existing
        # user-visible feedback enters the actual Main input prompt.
        if any(any(parent not in delivered for parent in tasks[tid]["depends_on"])
               for tid in wave):
            raise ValueError("stage3_wave_dependency_not_satisfied")
        available_before = list(delivered)
        history = o1_history(task_entries)
        main_prompt = o1_prompt(
            scene, task_entries, attempt=wave_number,
            max_attempts=max_attempts, sub_model=sub_model,
            model_alias=model_alias,
        )
        # Keep the exact O1 prompt for training/replay and its unchanged
        # identifier Guard. Do not send baseline tutorials/tool JSON schemas
        # to the prose Author or Critic in frozen-dispatch Stage 3.
        author_prompt = (
            stage3_author_context(
                scene, history, attempt=wave_number,
                max_attempts=max_attempts,
            ) if freeze_trusted_dispatch else main_prompt
        )
        assignment = [
            _assignment(
                tasks[tid], catalog, delivered, model_alias=model_alias,
                compile_contract=freeze_trusted_dispatch
            ) for tid in wave
        ]
        # An overlap witness is legal only for a true two-parent
        # identifier join and must be reconstructed from the ACTUALLY
        # delivered public Result strings (never the Oracle/Gold rows).
        visible_parent_groups = []
        for tid in wave:
            target = tasks[tid]
            parents = target["depends_on"]
            eligible = (
                len(parents) == 2
                and target["outputs"]["typed_relation"] == ["case_id"]
                and any(step["op"] == "join"
                        for step in target["operator_chain"])
            )
            visible_parent_groups.append(
                [
                    delivered[parent]["finish_result"]["result"]
                    for parent in parents
                ] if eligible else None
            )
        evidence_contract = (
            build_round_evidence_contract(
                assignment, delivered,
                has_later_delegation=wave_number < len(scheduled),
                upstream_result_groups=visible_parent_groups,
            ) if freeze_trusted_dispatch else None
        )
        saved = (saved_rounds[wave_number - 1]
                 if wave_number <= len(saved_rounds) else None)
        if saved is not None:
            prior_turn = saved.get("turn", {})
            prior_audit = saved.get("private_audit", {})
            if (not isinstance(prior_turn, dict)
                    or not isinstance(prior_audit, dict)
                    or prior_turn.get("input") != {"subtask_history": history}
                    or prior_audit.get("prompt_sha256") != _hash(main_prompt)
                    or o1_prompt_from_turn(
                        scene, prior_turn, max_attempts=max_attempts,
                        sub_model=sub_model, model_alias=model_alias,
                    ) != main_prompt):
                raise RuntimeError("stage3_checkpoint_prompt_mismatch")
            action = deepcopy(prior_turn.get("output"))
            expected_mode = (
                "llm_reviewed_contract_frozen" if freeze_trusted_dispatch
                and author is not None else
                "llm_reviewed" if author is not None else "template"
            )
            status = prior_audit.get("authoring")
            if (prior_turn.get("generation_mode") != expected_mode
                    or not isinstance(status, dict)
                    or status.get("mode") != prior_turn.get("generation_mode")
                    or status.get("status") != "accepted"
                    or action_issues(action, assignment, prompt=main_prompt)):
                raise RuntimeError("stage3_checkpoint_action_invalid")
        else:
            action, status = author_delegate_action(
                assignments=assignment, prompt=author_prompt,
                validation_prompt=main_prompt,
                author=author, critic=critic, max_revisions=max_revisions,
                freeze_assignment_prose=freeze_trusted_dispatch,
                reasoning_only_stage3=freeze_trusted_dispatch,
                # Freeze DAG dispatch and O1 tool/model slots, not the
                # natural-language instruction/context attached to each slot.
                naturalize_assignment_prose=freeze_trusted_dispatch,
                author_system=(
                    STAGE3_TASK_PROSE_AUTHOR_SYSTEM
                    if freeze_trusted_dispatch else None
                ),
                critic_system=(
                    STAGE3_TASK_PROSE_CRITIC_SYSTEM
                    if freeze_trusted_dispatch else None
                ),
                reasoning_guard=(
                    (lambda reasoning: (
                        stage3_reasoning_issues(reasoning, delivered)
                        + check_reasoning_fidelity(
                            reasoning, evidence_contract,
                        )
                    )) if freeze_trusted_dispatch else None
                ),
                round_evidence_contract=evidence_contract,
            )
            if action is None:
                return None, [], [], {
                    "status": "rejected", "cause": "main_action_not_accepted",
                    "round": wave_number, "authoring": status,
                }
        sub_results = []
        for index, (tid, sub_instruction) in enumerate(
                zip(wave, action["params"]["tasks"])):
            result = _main_report(reports[tid], sub_instruction, index,
                                  model_alias=model_alias)
            sub_results.append(result)
        batch_result = {
            "subtask_results": sub_results,
            "total_subtasks": len(sub_results),
            "total_cost": 0.0,
            "total_steps": sum(row["steps_taken"] for row in sub_results),
            "all_done": True,
        }
        action_record = {
            "attempt": wave_number,
            "action": "delegate_task",
            "reasoning": action["reasoning"],
            "params": deepcopy(action["params"]),
            "result": batch_result,
            "raw_response": json.dumps(action, ensure_ascii=False),
        }
        example = {
            "sample_id": scene["task_id"] + f"-round-{wave_number:02d}",
            "task_id": scene["task_id"], "round": wave_number,
            "input": {"subtask_history": history},
            "output": action,
                "generation_mode": status["mode"],
        }
        audit_row = {
            "task_id": scene["task_id"],
            "round": wave_number,
            "action": "delegate_task",
            "private_macro_alignment": list(wave),
            "visible_macros_before_round": available_before,
            "newly_delivered_macro_ids": list(wave),
            "prompt_sha256": _hash(main_prompt),
            "full_o1_prompt_chars": len(main_prompt),
            "author_context_chars": len(author_prompt),
            "action_sha256": _hash(action),
            "o1_history_sha256": _hash(history),
            "dependencies_satisfied": True,
            "authoring": status,
            **({
                "reasoning_fidelity_contract": evidence_contract,
                "reasoning_fidelity_checked": True,
            } if evidence_contract is not None else {}),
            "private_only": True,
        }
        examples.append(example)
        audit.append(audit_row)
        trajectory.append(action_record)
        for tid in wave:
            delivered[tid] = reports[tid]
        task_entries.append({
            "attempt": wave_number,
            "is_parallel_batch": True,
            "subtask_results": [_history_item(item) for item in sub_results],
            "total_cost": 0.0,
            "total_steps": batch_result["total_steps"],
        })
        checkpoint = {
            "round": wave_number, "turn": example, "private_audit": audit_row,
        }
        if saved is not None:
            if saved != checkpoint:
                raise RuntimeError("stage3_checkpoint_round_content_mismatch")
        elif on_round is not None:
            on_round(checkpoint)

    shortened = main_compute or visible_answer_shortcut
    required = (set(tasks) - {graph["terminal_task_id"]}
                if shortened else set(tasks))
    if set(delivered) != required:
        raise ValueError("stage3_incomplete_macro_schedule")
    if main_compute and not _can_derive_local_answer(scene, delivered, tasks):
        raise ValueError("stage3_local_resolution_not_derived_from_reports")
    if visible_answer_shortcut and not _visible_terminal_answer(
            scene, delivered, tasks):
        raise ValueError("stage3_visible_answer_shortcut_not_justified")
    completion_round = len(scheduled) + 1
    completion_prompt = o1_prompt(
        scene, task_entries, attempt=completion_round,
        max_attempts=max_attempts, sub_model=sub_model,
        model_alias=model_alias,
    )
    completion_history = o1_history(task_entries)
    completion_author_prompt = (
        stage3_author_context(
            scene, completion_history, attempt=completion_round,
            max_attempts=max_attempts,
        ) if freeze_trusted_dispatch else completion_prompt
    )
    completion_template = {
        "action": "complete",
        "reasoning": _grounded_completion_reasoning(
            delivered, shortcut=visible_answer_shortcut,
            main_compute=main_compute,
        ),
        "params": {"answer": _answer_text(scene)},
    }
    prior_final = (saved_rounds[completion_round - 1]
                   if completion_round <= len(saved_rounds) else None)
    if prior_final is not None:
        prior_turn = prior_final.get("turn", {})
        prior_audit = prior_final.get("private_audit", {})
        completion_status = prior_audit.get("authoring")
        final_action = deepcopy(prior_turn.get("output"))
        if (not isinstance(completion_status, dict)
                or completion_status.get("status") != "accepted"
                or prior_turn.get("input") != {
                    "subtask_history": o1_history(task_entries)}
                or prior_audit.get("prompt_sha256") != _hash(completion_prompt)
                or prior_turn.get("generation_mode")
                   != completion_status.get("mode")
                or o1_prompt_from_turn(
                    scene, prior_turn, max_attempts=max_attempts,
                    sub_model=sub_model, model_alias=model_alias,
                ) != completion_prompt):
            raise RuntimeError("stage3_checkpoint_completion_prompt_mismatch")
    else:
        final_action, completion_status = author_complete_action(
            template=completion_template, prompt=completion_author_prompt,
            delivered=delivered, author=author, critic=critic,
            max_revisions=max_revisions,
        )
    # The language Author never gets a deterministic fallback: exhausted
    # Completion review rejects the whole Scene without publishing partial
    # training data. Template-mode clean replay remains deterministic.
    if author is not None and (
        final_action is None
        or completion_status.get("status") != "accepted"
    ):
        return None, [], [], {
            "status": "rejected",
            "cause": "completion_action_not_accepted",
            "round": completion_round,
            "authoring": completion_status,
        }
    completion_errors = completion_issues(
        final_action, expected_answer=_answer_text(scene),
        delivered=delivered,
    )
    if completion_errors:
        raise ValueError(
            "stage3_invalid_visible_completion:" + ",".join(completion_errors)
        )
    final_example = {
        "sample_id": scene["task_id"] + f"-round-{completion_round:02d}",
        "task_id": scene["task_id"], "round": completion_round,
        "input": {"subtask_history": completion_history},
        "output": final_action,
        "generation_mode": completion_status["mode"],
    }
    final_audit = {
        "task_id": scene["task_id"], "round": completion_round,
        "action": "complete",
        "private_macro_alignment": (
            [graph["terminal_task_id"]] if shortened else []
        ),
        "visible_macros_before_round": list(delivered),
        "newly_delivered_macro_ids": [],
        "prompt_sha256": _hash(completion_prompt),
        "full_o1_prompt_chars": len(completion_prompt),
        "author_context_chars": len(completion_author_prompt),
        "action_sha256": _hash(final_action),
        "main_locally_computed": main_compute,
        "visible_answer_shortcut": visible_answer_shortcut,
        "authoring": completion_status,
        "dependencies_satisfied": True,
        "private_only": True,
    }
    examples.append(final_example)
    audit.append(final_audit)
    final_checkpoint = {
        "round": completion_round, "turn": final_example,
        "private_audit": final_audit,
    }
    if len(saved_rounds) >= completion_round:
        if saved_rounds[completion_round - 1] != final_checkpoint:
            raise RuntimeError("stage3_checkpoint_completion_mismatch")
    elif on_round is not None:
        on_round(final_checkpoint)
    trajectory.append({
        "attempt": completion_round,
        "action": "complete",
        "reasoning": final_action["reasoning"],
        "params": final_action["params"],
        "result": {"success": True, "done": True},
        "raw_response": json.dumps(final_action, ensure_ascii=False),
    })
    reference = clean_reference_identity(scene, reports)
    return {
        "task_id": scene["task_id"],
        "question": scene["question"],
        "format": "orchestra_o1_main_trajectory.v3",
        "clean_reference_id": reference["clean_reference_id"],
        "trajectory": trajectory,
        "main_decision_rounds": len(trajectory),
        "trusted_subagent_feedback": True,
    }, examples, audit, {
        "status": "accepted", "main_decision_rounds": len(trajectory),
        "parallel_delegation_rounds": len(scheduled),
        "main_local_goal_resolution": main_compute,
        "visible_answer_shortcut": visible_answer_shortcut,
        "completion_generation_mode": completion_status["mode"],
        "clean_reference_identity": reference,
    }
