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
    if (spec or {}).get("reasoning_pattern") == "macro_first_v2":
        from .macro_first_v2_world import mutate_witness
        return mutate_witness(sources, source_id, gold_workproducts, spec)
    if (spec or {}).get("reasoning_pattern") == "macro_first":
        from .macro_first_world import mutate_macro_first_witness
        return mutate_macro_first_witness(sources, source_id, gold_workproducts, spec)
    if (spec or {}).get("reasoning_pattern") == "evidence_graph":
        from .world_synthesizer import mutate_evidence_witness
        return mutate_evidence_witness(sources, source_id, gold_workproducts, spec)
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
    if scene["problem_spec"].get("reasoning_pattern") in {"clue_chain", "evidence_graph", "macro_first", "macro_first_v2"}:
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
    try:
        execution = evaluate_program(program, sources=world["sources"])
        answer = answer_values(execution)
        if answer != scene["gold_answer"]:
            issues.append("gold_program_answer_mismatch")
        if execution["trace"] != scene["gold_trace"]:
            issues.append("gold_execution_trace_mismatch")
        graph = (
            compile_frozen_macro_plan(spec, program, execution["trace"])
            if spec.get("reasoning_pattern") in {"macro_first", "macro_first_v2"}
            else compile_macro_tasks(program, execution["trace"])
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
    if spec.get("reasoning_pattern") == "clue_chain":
        # Information-acquisition depth replaces the static-join quota.
        source_count, joins, depth = 3, 0, 4
    elif spec.get("reasoning_pattern") == "evidence_graph":
        source_count, joins, depth = (3 if difficulty == "medium" else 4), 0, 3
    elif spec.get("reasoning_pattern") == "macro_first":
        source_count, joins, depth = (3 if difficulty == "medium" else 4), 0, 2
    elif spec.get("reasoning_pattern") == "macro_first_v2":
        # Do not equate a hard Macro with having more source tables:
        # complexity is driven by local evidence budgets and temporal logic.
        source_count, joins, depth = 2, 0, 2
    else:
        source_count, joins, depth = levels[difficulty]
    if spec.get("version") == VERSION and graph["macro_task_count"] > 9:
        issues.append("macro_task_budget_exceeded")
    if profile["source_count"] < source_count:
        issues.append("insufficient_source_count")
    if profile["join_count"] < joins:
        issues.append("insufficient_join_count")
    if profile["necessary_structural_operator_depth"] < depth:
        issues.append("insufficient_operator_depth")
    if spec.get("reasoning_pattern") in {"macro_first", "macro_first_v2"}:
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
        from .macro_first import validate_handoffs
        handoff_report = validate_handoffs(scene)
        if not handoff_report["all_handoffs_necessary"]:
            issues.append("macro_first_redundant_macro_handoffs")
    information_report = None
    if (spec.get("macro_information_necessity_version")
            and spec.get("macro_plan", {}).get("information_necessity_policy")):
        from .information_necessity import audit_candidate_information
        information_report = audit_candidate_information(scene)
        if not information_report["passed"]:
            issues.append("macro_branches_not_informationally_necessary")
    if spec.get("reasoning_pattern") == "evidence_graph":
        if profile.get("keyed_lookup_count", 0) < 1:
            issues.append("evidence_graph_no_keyed_acquisition")
        if (len(answer) != 1 or set(answer[0]) != {"accession_year"}):
            issues.append("evidence_graph_answer_contract")
        refs = {ref for row in macro["answer_with_provenance"]
                for ref in row["source_fact_refs"]}
        if not all(any(ref.startswith(sid + ":") for ref in refs)
                   for sid in world["sources"]):
            issues.append("evidence_graph_missing_source_witness")
    if spec.get("reasoning_pattern") == "clue_chain":
        if profile.get("keyed_lookup_count", 0) != 2:
            issues.append("clue_chain_requires_two_keyed_lookups")
        refs = {ref for row in macro["answer_with_provenance"]
                for ref in row["source_fact_refs"]}
        if not all(any(ref.startswith(sid + ":") for ref in refs)
                   for sid in ("illustration", "index", "archive")):
            issues.append("clue_chain_missing_three_source_provenance")
        if len(answer) != 1 or set(answer[0]) != {"accession_year"}:
            issues.append("clue_chain_answer_year_contract")
    elif (difficulty != "easy"
          and spec.get("reasoning_pattern") not in {
              "evidence_graph", "macro_first", "macro_first_v2"
          } and not
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
    acquisition_audit = {}
    if spec.get("reasoning_pattern") == "clue_chain":
        from .compiler import QuestionSession
        from .program_ir import _scan
        view = scene["public"]["available_source_descriptions"]
        if {row["source_id"]: row["access_mode"] for row in view} != {
            "archive": "keyed_lookup", "illustration": "browse",
            "index": "keyed_lookup",
        }:
            issues.append("clue_chain_public_access_policy_mismatch")
        denied = {}
        session = QuestionSession(scene)
        for sid in ("index", "archive"):
            try:
                session.read_source(sid)
                denied[sid] = False
            except MissingSourceError:
                denied[sid] = True
            if not denied[sid]:
                issues.append("clue_chain_full_read_bypass:" + sid)
            try:
                _scan(sid, world["sources"][sid])
                issues.append("clue_chain_scan_bypass:" + sid)
            except MissingSourceError:
                pass
        illustration = [
            row["values"] for row in
            world["sources"]["illustration"]["observations"]
        ]
        features = spec["investigation"]["visual_features"]
        target = [r for r in illustration if
                  all(r[field] == value for field, value in features.items())]
        if len(target) != 1:
            issues.append("clue_chain_target_not_unique")
        if any(
            sum(row[field] == value for row in illustration) < 2
            for field, value in features.items()
        ):
            issues.append("clue_chain_missing_visual_decoys")
        # Replacing JUST the visual marker with another genuine marker
        # must route retrieval to a different accession year.
        rerouted = copy.deepcopy(world["sources"])
        target_marker = target[0]["marker_code"] if target else None
        alternative = next((
            row["marker_code"] for row in illustration
            if row["marker_code"] != target_marker
        ), None)
        if target_marker is not None and alternative is not None:
            for row in rerouted["illustration"]["observations"]:
                if row["values"]["marker_code"] == target_marker:
                    row["values"]["marker_code"] = alternative
                    break
            rerouted_answer = answer_values(evaluate_program(
                program, sources=rerouted
            ))
            clue_sensitive = rerouted_answer != answer
            if not clue_sensitive:
                issues.append("clue_chain_marker_reroute_does_not_change_answer")
        else:
            clue_sensitive = False
            issues.append("clue_chain_missing_counterfactual_marker")
        acquisition_audit = {
            "keyed_lookup_count": profile.get("keyed_lookup_count", 0),
            "evidence_acquisition_depth": profile["keyed_lookup_count"],
            "full_read_blocked": denied,
            "unauthorized_full_read_success_rate": (
                sum(not value for value in denied.values())
                / max(1, len(denied))
            ),
            "clue_counterfactual_pass": clue_sensitive,
            "unique_visual_target_count": len(target),
            "nontrivial_decoys": True if len(target) == 1 and
            all(sum(r[field] == val for r in illustration) >= 2
                for field, val in features.items()) else False,
        }
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
    if spec.get("reasoning_pattern") == "clue_chain":
        result["evidence_acquisition"] = acquisition_audit
    if spec.get("reasoning_pattern") in {"macro_first", "macro_first_v2"}:
        result["macro_first_audit"] = {
            "pattern": spec["macro_plan"]["pattern"],
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
            **({"candidate_information_necessity": information_report}
               if information_report is not None else {}),
        }
        if spec.get("scene_realism_version"):
            from .difficulty_proxy import scene_difficulty_profile
            result["macro_first_audit"]["structural_difficulty_profile"] = (
                scene_difficulty_profile(
                    scene, information_report=information_report
                )
            )
    if spec.get("reasoning_pattern") == "evidence_graph":
        result["evidence_graph_audit"] = {
            "archetype": spec["evidence_graph"]["archetype"],
            "lookup_hops": spec["evidence_graph"]["lookup_hops"],
            "keyed_lookup_count": profile.get("keyed_lookup_count", 0),
            "source_counterfactuals_pass": all(
                x["changing_source_changes_answer"]
                for x in counterfactual.values()
            ),
            "effective_parallel_wave_width": max(
                map(len, graph["reference_waves"]), default=0
            ),
        }
    return result


def compile_and_validate(spec: dict, question: str) -> dict:
    scene = build_scene(spec, question)
    program = scene["gold_program"]
    graph = (
        compile_frozen_macro_plan(spec, program, scene["gold_trace"])
        if spec.get("reasoning_pattern") in {"macro_first", "macro_first_v2"}
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
    report = audit_scene(scene, strict=False)
    errors.extend(report.get("issues", []))
    if scene.get("difficulty_audit"):
        if report.get("status") == "valid" and report != scene["difficulty_audit"]:
            errors.append("stored_difficulty_audit_mismatch")
    return {"status": "valid" if not errors else "invalid",
            "issues": sorted(set(errors)), "task_id": scene.get("task_id")}
