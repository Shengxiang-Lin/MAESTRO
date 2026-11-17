"""Runnable difficulty, dependency, provenance and shortcut checks for V2.

These diagnostics distinguish useful evidence dependency from DAG node count;
they are still NOT proofs of globally shortest possible solution programs.
"""
from __future__ import annotations

import copy
import json

from environment.generate_question.execution.program_ir import (
    MissingSourceError, answer_values, evaluate_program, program_complexity,
)
from environment.generate_question.execution.compiler import (
    compile_frozen_macro_plan,
    oracle_schedule, run_macro_tasks,
)
from environment.generate_question.execution.world import build_scene, generate_world, compile_program
from environment.generate_question.authoring.question import guard_question
from environment.generate_question.planning.spec import spec_issues, VERSION
from environment.generate_question.auditing.macro_contract import macro_question_contract
from environment.generate_question.execution.evidence_tools import validate_scene_evidence_tools


def _mutate_answer_witness(sources: dict, source_id: str,
                           gold_workproducts: list[dict],
                           spec: dict | None = None) -> dict:
    from environment.generate_question.execution.task_plan_runtime import mutate_witness
    return mutate_witness(sources, source_id, gold_workproducts, spec)


def weak_single_source_baseline(scene: dict) -> list[dict]:
    """Current Task Plan has no accessible register-only answer shortcut."""
    return []


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
        "evidence_ledger", "evidence_relation_audit",
    }:
        issues.append("privileged_world_leaked_to_public")
    if scene["public"].get("question") != scene["question"]:
        issues.append("public_question_mismatch")
    if "task_input" in spec and (
        scene["public"].get("task_input") != spec["task_input"]
    ):
        issues.append("public_task_input_not_from_frozen_spec")
    elif "task_input" not in spec and "task_input" in scene["public"]:
        issues.append("unexpected_public_task_input")
    if not scene["question_origin"].startswith(("deterministic_", "llm_")):
        issues.append("unknown_question_origin")
    if scene["question_origin"] == "deterministic_frozen_problem_spec":
        issues.extend(guard_question(scene["question"], spec, scene.get("macro_graph")))
    evidence_relation_audit = None
    try:
        execution = evaluate_program(program, sources=world["sources"])
        answer = answer_values(execution)
        if answer != scene["gold_answer"]:
            issues.append("gold_program_answer_mismatch")
        if execution["trace"] != scene["gold_trace"]:
            issues.append("gold_execution_trace_mismatch")
        graph = (
            compile_frozen_macro_plan(spec, program, execution["trace"])
        )
        if graph != scene["macro_graph"]:
            issues.append("macro_graph_not_program_compiled")
        issues.extend(validate_scene_evidence_tools(scene))
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
    if spec.get("reasoning_pattern") in {"task_plan"}:
        source_count, joins, depth = 2, 0, 2
    else:
        source_count, joins, depth = levels[difficulty]
    budget = 11 if spec.get("reasoning_pattern") == "task_plan" else 9
    if spec.get("version") == VERSION and graph["macro_task_count"] > budget:
        issues.append("macro_task_budget_exceeded")
    if profile["source_count"] < source_count:
        issues.append("insufficient_source_count")
    if profile["join_count"] < joins:
        issues.append("insufficient_join_count")
    if profile["necessary_structural_operator_depth"] < depth:
        issues.append("insufficient_operator_depth")
    if spec.get("reasoning_pattern") in {"task_plan"}:
        if (len(answer) != 1 or set(answer[0]) != {
            spec["macro_plan"]["goal"]["answer_field"]
        }):
            issues.append("macro_first_answer_contract")
        provenance = {
            ref.split(":")[0] for row in macro["answer_with_provenance"]
            for ref in row["source_fact_refs"]
        }
        if provenance != set(world["sources"]):
            issues.append("macro_first_missing_source_witness")
        from environment.generate_question.auditing.macro_contract import validate_handoffs
        handoff_report = validate_handoffs(scene)
        if not handoff_report["all_handoffs_necessary"]:
            issues.append("macro_first_redundant_macro_handoffs")
    operator_grammar_report = None
    information_report = None
    if (spec.get("macro_information_necessity_version")
            and spec.get("macro_plan", {}).get("information_necessity_policy")):
        from environment.generate_question.auditing.information_necessity import audit_candidate_information
        information_report = audit_candidate_information(scene)
        if not information_report["passed"]:
            issues.append("macro_branches_not_informationally_necessary")
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
    result = {
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
        **({"evidence_relation_audit": evidence_relation_audit}
           if evidence_relation_audit is not None else {}),
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
    if spec.get("reasoning_pattern") in {"task_plan"}:
        result["macro_first_audit"] = {
            "pattern": spec["macro_plan"].get("pattern", spec["reasoning_pattern"]),
            "frozen_macro_count": len(spec["macro_plan"]["macro_tasks"]),
            "actual_macro_count": graph["macro_task_count"],
            "macro_boundary_preserved": graph.get("plan_origin") == "macro_first",
            "primitive_ops_inside_macros": [
                len(t["operator_chain"]) for t in graph["tasks"]
            ],
            "handoff_counterfactuals": handoff_report,
            "source_counterfactuals_pass": all(
                x["changing_source_changes_answer"]
                for x in counterfactual.values()
            ),
            **({"operator_grammar_equivalence": operator_grammar_report}
               if operator_grammar_report is not None else {}),
            **({"candidate_information_necessity": information_report}
               if information_report is not None else {}),
        }
        if (spec.get("reasoning_pattern") == "macro_first_v2"
                and spec.get("scene_realism_version")):
            from environment.generate_question.auditing.difficulty_proxy import scene_difficulty_profile
            result["macro_first_audit"]["structural_difficulty_profile"] = (
                scene_difficulty_profile(
                    scene, information_report=information_report
                )
            )
    return result


def compile_and_validate(spec: dict, question: str) -> dict:
    scene = build_scene(spec, question)
    program = scene["gold_program"]
    graph = (
        compile_frozen_macro_plan(spec, program, scene["gold_trace"])
    )
    scene["macro_graph"] = graph
    if spec.get("semantic_evidence_contract_version"):
        from environment.generate_question.authoring.semantic_evidence import compile_semantic_contract
        scene["semantic_evidence_contract"] = compile_semantic_contract(
            spec, program=program
        )
    scene["task_feedback"] = run_macro_tasks(
        graph, sources=scene["world"]["sources"]
    )["task_feedback"]
    if spec.get("evidence_relation_version"):
        from environment.generate_question.auditing.evidence_relations import compile_evidence_ledger
        scene["evidence_ledger"] = compile_evidence_ledger(scene)
    report = audit_scene(scene)
    if report["status"] != "valid":
        raise ValueError(
            "Question generation failed quality gates: " + ", ".join(report["issues"])
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
    try:
        report = audit_scene(scene, strict=False)
        errors.extend(report.get("issues", []))
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        # A tampered world/spec may have no surviving witness; replay
        # must report invalid instead of raising inside counterfactuals.
        report = {"status": "invalid", "issues": []}
        errors.append("replay_audit_error:" + type(exc).__name__)
    if scene.get("difficulty_audit"):
        if report.get("status") == "valid" and report != scene["difficulty_audit"]:
            errors.append("stored_difficulty_audit_mismatch")
    return {"status": "valid" if not errors else "invalid",
            "issues": sorted(set(errors)), "task_id": scene.get("task_id")}
