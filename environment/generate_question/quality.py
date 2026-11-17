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
    compile_macro_tasks, compile_frozen_macro_plan,
    oracle_schedule, run_macro_tasks,
)
from .world import build_scene, generate_world, compile_program
from .question import guard_question
from .spec import spec_issues, VERSION
from .macro_contract import macro_question_contract
from .evidence_tools import validate_scene_evidence_tools


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
    if (spec or {}).get("reasoning_pattern") == "task_plan":
        from .task_plan_runtime import mutate_witness
        return mutate_witness(sources, source_id, gold_workproducts, spec)
    if (spec or {}).get("reasoning_pattern") == "task_graph_first_v2":
        from .task_graph_first_v2 import mutate_witness
        return mutate_witness(sources, source_id, gold_workproducts, spec)
    if (spec or {}).get("reasoning_pattern") == "task_graph_first_v1":
        from .task_graph_first import mutate_witness
        return mutate_witness(sources, source_id, gold_workproducts, spec)
    if (spec or {}).get("reasoning_pattern") == "typed_knowledge_v4":
        from .hybrid_world_v4 import mutate_witness
        return mutate_witness(sources, source_id, gold_workproducts, spec)
    if (spec or {}).get("reasoning_pattern") == "typed_knowledge_v3":
        from .hybrid_world import mutate_witness
        return mutate_witness(sources, source_id, gold_workproducts, spec)
    if (spec or {}).get("reasoning_pattern") == "typed_knowledge_v2":
        from .knowledge_macro_v2 import mutate_witness
        return mutate_witness(sources, source_id, gold_workproducts, spec)
    if (spec or {}).get("reasoning_pattern") == "typed_knowledge_v1":
        from .knowledge_macro import mutate_witness
        return mutate_witness(sources, source_id, gold_workproducts, spec)
    if (spec or {}).get("reasoning_pattern") == "macro_first_v2":
        from .macro_evidence_world import mutate_witness
        return mutate_witness(sources, source_id, gold_workproducts, spec)
    revised = copy.deepcopy(sources)
    rows = revised[source_id]["observations"]
    selected = next((
        row for row in rows if refs.intersection(row["fact_refs"])
    ), None)
    if selected is None:
        raise ValueError("referenced_source_fact_not_found:" + source_id)
    fields = selected["values"]
    if source_id == "illustration":
        fields["emblem_color"] = "__visual_clue_removed__"
    elif source_id == "index":
        fields["accession_id"] = "__unresolvable_accession__"
    elif source_id == "archive":
        fields["accession_year"] += 1
    elif source_id == "register":
        fields["active"] = False
    elif source_id == "glossary":
        field = "group_id" if "group_id" in fields else "category_name"
        fields[field] = "__unmapped_intervention__"
    elif source_id == "annotations":
        fields["approved"] = False
    elif source_id == "clearance":
        fields["cleared"] = False
    else:
        raise ValueError("unknown_mutation_source:" + source_id)
    return revised


def weak_single_source_baseline(scene: dict) -> list[dict]:
    """A measured shortcut attempt, NOT exhaustive program minimization."""
    if scene["problem_spec"].get("reasoning_pattern") in {"macro_first_v2", "typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3", "typed_knowledge_v4", "task_graph_first_v1", "task_graph_first_v2", "task_plan"}:
        # A visual clue is mandatory, neither keyed store is browsable.
        return []
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


def _audit_operator_grammar_equivalence(scene: dict, execution: dict) -> dict:
    """Compare EVERY Macro output with the pre-grammar canonical solver.

    A different final answer is insufficient: an intermediate Macro must
    hand off the same typed values with the same supporting fact references.
    This verifies concrete witness equivalence, not universal theorem-level
    equivalence across all possible World values.
    """
    from .operator_grammar import canonical_spec
    from .macro_planning import compile_macro_solvers

    canonical = compile_macro_solvers(canonical_spec(scene["problem_spec"]))
    baseline = evaluate_program(canonical, sources=scene["world"]["sources"])
    variant_groups = scene["gold_program"]["macro_solver_groups"]
    canonical_groups = canonical["macro_solver_groups"]
    if [g["macro_id"] for g in variant_groups] != [
        g["macro_id"] for g in canonical_groups
    ]:
        raise ValueError("operator_grammar_macro_group_identity_changed")

    def signature(product) -> list[tuple[str, tuple]]:
        return sorted(
            (json.dumps(row["values"], sort_keys=True, ensure_ascii=False),
             tuple(sorted(set(row["support"]))))
            for row in product.rows
        )

    per_macro = []
    for actual_group, baseline_group in zip(variant_groups, canonical_groups):
        observed = execution["products"][actual_group["output_node_id"]]
        expected = baseline["products"][baseline_group["output_node_id"]]
        equivalent = (
            set(observed.columns) == set(expected.columns)
            and signature(observed) == signature(expected)
        )
        per_macro.append({
            "macro_id": actual_group["macro_id"],
            "same_values_schema_and_factrefs": equivalent,
        })
    final_same = (
        answer_values(execution) == answer_values(baseline)
        and [tuple(x["source_fact_refs"]) for x in execution["answer"]]
        == [tuple(x["source_fact_refs"]) for x in baseline["answer"]]
    )
    return {
        "version": scene["problem_spec"]["operator_grammar_version"],
        "all_macro_outputs_equivalent": all(
            row["same_values_schema_and_factrefs"] for row in per_macro
        ),
        "final_answer_and_provenance_equivalent": final_same,
        "verified_macro_outputs": len(per_macro),
        "per_macro": per_macro,
        "scope": "Exact current synthetic Witness equivalence, not all Worlds",
    }


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
    if spec.get("semantic_evidence_contract_version"):
        from .semantic_evidence import (
            compile_semantic_contract, semantic_scenario_issues
        )
        expected_contract = compile_semantic_contract(spec, program=program)
        if scene.get("semantic_evidence_contract") != expected_contract:
            issues.append("semantic_evidence_contract_not_frozen")
        # Recheck *every* LLM-authored question, not only deterministic
        # references; the Critic alone is not a formal equivalence proof.
        issues.extend(guard_question(scene["question"], spec, scene.get("macro_graph")))
        issues.extend(semantic_scenario_issues(spec, expected_contract))
    evidence_relation_audit = None
    try:
        if spec.get("evidence_relation_version"):
            from .evidence_relations import (
                VERSION as RELATION_VERSION, compile_evidence_ledger,
            )
            if spec["evidence_relation_version"] != RELATION_VERSION:
                issues.append("unsupported_evidence_relation_version")
            ledger = compile_evidence_ledger(scene)
            if scene.get("evidence_ledger") != ledger:
                issues.append("evidence_ledger_not_reproducible")
            evidence_relation_audit = {
                "version": ledger["version"],
                "independent_comparable_pairs":
                    ledger["independent_comparable_pairs"],
                "verification_sensitive_pairs":
                    ledger["verification_sensitive_pairs"],
                "relation_counts": ledger["relation_counts"],
                "source_first_wave": ledger["source_first_wave"],
                **({"reference_wave_trust": {
                    "mode": ledger["reference_wave_trust"]["mode"],
                    "cross_wave_claim_pairs":
                        ledger["reference_wave_trust"]["cross_wave_claim_pairs"],
                    "answer_relevant_claim_pairs":
                        ledger["reference_wave_trust"]["answer_relevant_claim_pairs"],
                    "answer_relevant_cross_wave_pairs":
                        ledger["reference_wave_trust"][
                            "answer_relevant_cross_wave_pairs"],
                    "trust_state_transitions": len(
                        ledger["reference_wave_trust"]["transitions"]
                    ),
                }} if "reference_wave_trust" in ledger else {}),
            }
        elif "evidence_ledger" in scene:
            issues.append("unversioned_evidence_ledger")
        execution = evaluate_program(program, sources=world["sources"])
        answer = answer_values(execution)
        if answer != scene["gold_answer"]:
            issues.append("gold_program_answer_mismatch")
        if execution["trace"] != scene["gold_trace"]:
            issues.append("gold_execution_trace_mismatch")
        graph = (
            compile_frozen_macro_plan(spec, program, execution["trace"])
            if spec.get("reasoning_pattern") in {"macro_first_v2", "typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3", "typed_knowledge_v4", "task_graph_first_v1", "task_graph_first_v2", "task_plan"}
            else compile_macro_tasks(program, execution["trace"])
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
    if spec.get("reasoning_pattern") in {"macro_first_v2", "typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3", "typed_knowledge_v4", "task_graph_first_v1", "task_graph_first_v2", "task_plan"}:
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
    if spec.get("reasoning_pattern") in {"macro_first_v2", "typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3", "typed_knowledge_v4", "task_graph_first_v1", "task_graph_first_v2", "task_plan"}:
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
        from .macro_contract import validate_handoffs
        handoff_report = validate_handoffs(scene)
        if not handoff_report["all_handoffs_necessary"]:
            issues.append("macro_first_redundant_macro_handoffs")
    operator_grammar_report = None
    if spec.get("operator_grammar_version"):
        operator_grammar_report = _audit_operator_grammar_equivalence(
            scene, execution
        )
        if not operator_grammar_report["all_macro_outputs_equivalent"]:
            issues.append("typed_operator_grammar_changed_macro_workproduct")
        if not operator_grammar_report["final_answer_and_provenance_equivalent"]:
            issues.append("typed_operator_grammar_changed_final_answer")
    information_report = None
    if (spec.get("macro_information_necessity_version")
            and spec.get("macro_plan", {}).get("information_necessity_policy")):
        from .information_necessity import audit_candidate_information
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
    if spec.get("reasoning_pattern") in {"macro_first_v2", "typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3", "typed_knowledge_v4", "task_graph_first_v1", "task_graph_first_v2", "task_plan"}:
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
            from .difficulty_proxy import scene_difficulty_profile
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
        if spec.get("reasoning_pattern") in {"macro_first_v2", "typed_knowledge_v1", "typed_knowledge_v2", "typed_knowledge_v3", "typed_knowledge_v4", "task_graph_first_v1", "task_graph_first_v2", "task_plan"}
        else compile_macro_tasks(program, scene["gold_trace"])
    )
    scene["macro_graph"] = graph
    if spec.get("semantic_evidence_contract_version"):
        from .semantic_evidence import compile_semantic_contract
        scene["semantic_evidence_contract"] = compile_semantic_contract(
            spec, program=program
        )
    scene["task_feedback"] = run_macro_tasks(
        graph, sources=scene["world"]["sources"]
    )["task_feedback"]
    if spec.get("evidence_relation_version"):
        from .evidence_relations import compile_evidence_ledger
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
