"""Runnable difficulty, dependency, provenance and shortcut checks for V2.

These diagnostics distinguish useful evidence dependency from DAG node count;
they are still NOT proofs of globally shortest possible solution programs.
"""
from __future__ import annotations

import copy
import json

from .program_ir import (
    MissingSourceError, answer_values, evaluate_program, program_complexity,
)
from .compiler import (
    compile_macro_tasks, oracle_schedule, run_macro_tasks,
)
from .world import build_scene, generate_world, compile_program
from .question import guard_question
from .spec import spec_issues, VERSION
from .macro_contract import macro_question_contract


def _mutate(sources: dict, source_id: str, answer: list[dict]) -> dict:
    other = copy.deepcopy(sources)
    rows = other[source_id]["observations"]
    if source_id == "register":
        winning = {value.get("entry_label") for value in answer}
        if not winning:
            winning = {"Entry-01"}
        for row in rows:
            if row["values"]["entry_label"] in winning:
                row["values"]["active"] = False
        # Count answers have no entry label; toggle a known qualifying row.
        if not any(row["values"]["entry_label"] in winning for row in rows):
            rows[0]["values"]["active"] = False
    elif source_id == "glossary":
        rows[0]["values"]["category_name"] = "category not in annotations"
    elif source_id == "annotations":
        rows[0]["values"]["approved"] = False
    elif source_id == "clearance":
        rows[0]["values"]["cleared"] = False
    else:
        raise ValueError("unsupported_counterfactual_source:" + source_id)
    return other


def _mutate_answer_witness(sources: dict, source_id: str,
                           gold_workproducts: list[dict],
                           spec: dict | None = None) -> dict:
    """Flip ONE source observation that actually supports the Gold answer.

    New data should not depend on the position of F001, Entry-01, or a
    fixed category. Count/group-count output does not identify an entry,
    so use the cited source fact instead of an output label.
    """
    refs = {ref for item in gold_workproducts
            for ref in item["source_fact_refs"]
            if ref.startswith(source_id + ":")}
    if not refs:
        raise ValueError("missing_gold_source_witness:" + source_id)
    revised = copy.deepcopy(sources)
    rows = revised[source_id]["observations"]
    selected = next((
        row for row in rows if refs.intersection(row["fact_refs"])
    ), None)
    if selected is None:
        raise ValueError("referenced_source_fact_not_found:" + source_id)
    fields = selected["values"]
    if source_id == "register":
        fields["active"] = False
    elif source_id == "glossary":
        field = "group_id" if "group_id" in fields else "category_name"
        fields[field] = "__unmapped_intervention__"
    elif source_id == "annotations":
        pattern = (spec or {}).get("reasoning_pattern", "approval")
        if pattern == "approval":
            fields["approved"] = False
        else:
            from .pattern_world import modality_fields
            modality = revised[source_id]["modality"]
            target = modality_fields(modality, pattern)
            if isinstance(target, tuple):
                # A simulated time window moved well beyond all event steps.
                fields[target[0]] = 100000
                fields[target[1]] = 100100
            elif pattern == "status_disagreement":
                fields[target] = not fields[target]
            else:
                fields[target] = 100000
    elif source_id == "clearance":
        fields["cleared"] = False
    else:
        raise ValueError("unknown_mutation_source:" + source_id)
    return revised


def weak_single_source_baseline(scene: dict) -> list[dict]:
    """A measured shortcut attempt, NOT exhaustive program minimization."""
    rows = [
        observation["values"]
        for observation in scene["world"]["sources"]["register"]["observations"]
        if observation["values"]["active"] is True
    ]
    family = scene["problem_spec"]["family"]
    if family == "coverage_count":
        return [{"count": len(rows)}]
    if family == "category_distribution":
        return []  # Register lacks the required category crosswalk.
    if family == "priority":
        rows = sorted(rows, key=lambda x: x["recorded_priority"], reverse=True)[:1]
    if family == "chronology":
        rows = sorted(rows, key=lambda x: x["relative_step"])[:1]
    return sorted([{"entry_label": row["entry_label"]} for row in rows],
                  key=lambda item: item["entry_label"])


def _needed_operator_depth(program: dict) -> int:
    """Logical program depth excluding scan/project/unique presentation nodes."""
    depths = {}
    for node in program["nodes"]:
        deps = [depths[parent] for parent in node["inputs"]]
        depths[node["id"]] = max(deps, default=0) + (
            1 if node["op"] not in {"scan", "project", "unique"} else 0
        )
    return depths[program["output"]]


def audit_scene(scene: dict, *, strict: bool = True) -> dict:
    issues = []
    spec = scene["problem_spec"]
    program, world = scene["gold_program"], scene["world"]
    if spec_issues(spec):
        issues.extend(spec_issues(spec))
    if scene["task_id"] != spec["task_id"]:
        issues.append("task_id_mismatch")
    if not world.get("simulated_only"):
        issues.append("non_synthetic_world")
    if set(program.get("authorized_source_ids", [])) != set(world["sources"]):
        issues.append("source_authorization_mismatch")
    if scene.get("gold_answer") is None:
        issues.append("missing_gold_answer")
    if not isinstance(scene.get("public"), dict) or set(scene["public"]) & {
        "world", "gold_program", "gold_answer", "entities", "gold_trace",
    }:
        issues.append("privileged_world_leaked_to_public")
    if scene["public"].get("question") != scene["question"]:
        issues.append("public_question_mismatch")
    if not scene["question_origin"].startswith(("deterministic_", "llm_")):
        issues.append("unknown_question_origin")
    if scene["question_origin"] == "deterministic_frozen_problem_spec":
        issues.extend(guard_question(scene["question"], spec))
    try:
        execution = evaluate_program(program, sources=world["sources"])
        answer = answer_values(execution)
        if answer != scene["gold_answer"]:
            issues.append("gold_program_answer_mismatch")
        if execution["trace"] != scene["gold_trace"]:
            issues.append("gold_execution_trace_mismatch")
        graph = compile_macro_tasks(
        program, execution["trace"], compact=spec.get("version") == VERSION
    )
        if graph != scene["macro_graph"]:
            issues.append("macro_graph_not_program_compiled")
        if spec.get("version") == VERSION:
            try:
                macro_question_contract(spec, graph)
            except ValueError as exc:
                issues.append("question_macro_contract_mismatch:" + str(exc))
        macro = run_macro_tasks(graph, sources=world["sources"])
        if macro["answer"] != answer:
            issues.append("macro_executor_answer_mismatch")
        if macro["task_feedback"] != scene["task_feedback"]:
            issues.append("task_feedback_replay_mismatch")
        reference = oracle_schedule(scene)
        if reference["answer"] != answer:
            issues.append("orchestration_schedule_answer_mismatch")
    except Exception as exc:
        if strict:
            raise
        issues.append("execution_error:" + type(exc).__name__ + ":" + str(exc)[:140])
        return {"status": "invalid", "issues": issues}
    difficulty = spec["difficulty_target"]
    profile = program_complexity(program)
    profile["necessary_structural_operator_depth"] = _needed_operator_depth(program)
    levels = {
        "easy": (1, 0, 1),
        "medium": (3, 2, 3),
        "hard": (4, 3, 4),
    }
    source_count, joins, depth = levels[difficulty]
    if spec.get("version") == VERSION and graph["macro_task_count"] > 9:
        issues.append("macro_task_budget_exceeded")
    if profile["source_count"] < source_count:
        issues.append("insufficient_source_count")
    if profile["join_count"] < joins:
        issues.append("insufficient_join_count")
    if profile["necessary_structural_operator_depth"] < depth:
        issues.append("insufficient_operator_depth")
    if (difficulty != "easy" and not
            any(ref.startswith("annotations:")
                for row in macro["answer_with_provenance"]
                for ref in row["source_fact_refs"])):
        issues.append("missing_annotation_witness")
    if not answer:
        issues.append("unexpected_empty_answer")
    if (spec.get("version") == "maestro.problem_spec.v4"
            and spec["family"] == "evidence_reconcile"):
        if any(
            set(entry) != {"entry_label", "evidence_refs"}
            or not isinstance(entry.get("evidence_refs"), list)
            or not entry["evidence_refs"]
            for entry in answer
        ):
            issues.append("gold_evidence_output_contract_mismatch")
    if len({json.dumps(row, sort_keys=True) for row in answer}) != len(answer):
        issues.append("duplicate_answer_rows")
    one_source = weak_single_source_baseline(scene)
    shortcut = one_source == answer
    if difficulty in ("medium", "hard") and shortcut:
        issues.append("weak_single_source_shortcut_succeeds")
    counterfactual = {}
    for source_id in sorted(world["sources"]):
        removed = {
            sid: data for sid, data in world["sources"].items()
            if sid != source_id
        }
        try:
            evaluate_program(program, sources=removed)
            access_blocked = False
        except MissingSourceError:
            access_blocked = True
        if spec.get("version") == "maestro.problem_spec.v2":
            # Keep historical audit JSON byte-for-byte replay compatible.
            changed = _mutate(world["sources"], source_id, answer)
        else:
            changed = _mutate_answer_witness(
                world["sources"], source_id,
                macro["answer_with_provenance"], spec=spec,
            )
        altered = answer_values(evaluate_program(program, sources=changed))
        semantic_effect = altered != answer
        counterfactual[source_id] = {
            "access_is_required_by_program": access_blocked,
            "changing_source_changes_answer": semantic_effect,
            "mutated_answer": altered,
        }
        if not access_blocked:
            issues.append("missing_source_not_required:" + source_id)
        if difficulty != "easy" and not semantic_effect:
            issues.append("counterfactual_not_answer_sensitive:" + source_id)
    # Full-World "oracle can compute" is expected for ANY deterministic
    # program; unlike one-source/one-step baseline, it is NOT a difficulty
    # rejection criterion.
    return {
        "status": "valid" if not issues else "invalid",
        "issues": sorted(set(issues)),
        "difficulty_target": difficulty,
        "reasoning": profile,
        "orchestration": {
            "macro_task_count": graph["macro_task_count"],
            "critical_path_depth": graph["critical_path_depth"],
            "wave_count": len(graph["reference_waves"]),
            "max_parallel_tasks": max(map(len, graph["reference_waves"]), default=0),
            "oracle_action_count": len(reference["action_log"]),
            "oracle_action_cost": reference["total_cost"],
            "simulated_source_acquisition_count": len(world["sources"]),
        },
        "one_source_baseline": {
            "answer": one_source,
            "exact_match": shortcut,
            "policy": "register_only_naive_filter_optional_sort_no_join",
        },
        "counterfactual_source_tests": counterfactual,
        "replay_exact": True,
        "answer_provenance": macro["answer_with_provenance"],
        "limitations": [
            "Operator depth is validated program depth, not a global complexity lower bound.",
            "Counterfactual tests sample controlled changes, not every possible mutation.",
            "The weak single-source baseline is not an exhaustive shortcut search.",
            "Verifier checks evidence closure, not independent attack resistance.",
            "Image, video and audio are simulated annotations, not real perception.",
        ],
    }


def compile_and_validate(spec: dict, question: str) -> dict:
    scene = build_scene(spec, question)
    program = scene["gold_program"]
    graph = compile_macro_tasks(
        program, scene["gold_trace"], compact=spec.get("version") == VERSION
    )
    scene["macro_graph"] = graph
    scene["task_feedback"] = run_macro_tasks(
        graph, sources=scene["world"]["sources"]
    )["task_feedback"]
    report = audit_scene(scene)
    if report["status"] != "valid":
        raise ValueError(
            "V2 failed quality gates: " + ", ".join(report["issues"])
        )
    scene["difficulty_audit"] = report
    return scene


def validate_replay(scene: dict) -> dict:
    """Strict source-regeneration and full execution replay, not JSON tautology."""
    expected_world = generate_world(scene["problem_spec"])
    errors = []
    if expected_world != scene.get("world"):
        errors.append("world_not_reproducible_from_frozen_spec")
    expected_program = compile_program(scene["problem_spec"])
    if expected_program != scene.get("gold_program"):
        errors.append("program_not_reproducible_from_frozen_spec")
    report = audit_scene(scene, strict=False)
    errors.extend(report.get("issues", []))
    if scene.get("difficulty_audit"):
        if report.get("status") == "valid" and report != scene["difficulty_audit"]:
            errors.append("stored_difficulty_audit_mismatch")
    return {"status": "valid" if not errors else "invalid",
            "issues": sorted(set(errors)), "task_id": scene.get("task_id")}
