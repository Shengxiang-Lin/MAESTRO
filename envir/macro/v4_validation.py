"""Teacher-side structural and evidence-level clean pilot checks.

Checks all EvidenceGraph leaves, named handoff fields, deterministic feedback
replay and the terminal evidence proof closure. Does not claim human question
quality, true multimodal perception, malicious evidence robustness or real
fact correctness.
"""
from __future__ import annotations

import copy
import json
import re

from .graph import compile_contracts, compile_waves, MacroPlanError
from .v4_taskfirst import author_question, task_level_brief
from .v4_question_review import validate_naturalized_question
from .v4_evidence import run_clean_tasks, EvidenceExecutionError


def validate_clean_scene(scene: dict) -> list[str]:
    errors = []
    if scene.get("schema") != "maestro.clean_taskfirst.v4":
        errors.append("invalid_schema")
    if scene.get("construction_status") != "clean_task_evidence_replayed":
        errors.append("incorrect_construction_status")
    if any(k in scene for k in ("gold_answer", "final_gold_answer", "candidate_recovery_actions")):
        errors.append("answer_or_untrusted_state_in_clean_scene")
    plan = scene.get("macro_graph") or {}
    tasks = plan.get("tasks") or []
    edges = plan.get("edges") or []
    if not 3 <= len(tasks) <= 9:
        errors.append("macro_task_count_out_of_bounds")
    if any("final_assessment" == t.get("work_type") for t in tasks):
        errors.append("unnecessary_agent_final_answer_task")
    try:
        copy_of_tasks = copy.deepcopy(tasks)
        compile_contracts(copy_of_tasks, edges)
        for left, right in zip(tasks, copy_of_tasks):
            if left["inputs"] != right["inputs"]:
                errors.append("typed_handoff_contract_changed")
        if compile_waves(tasks) != plan.get("reference_waves"):
            errors.append("reference_waves_changed")
    except (ValueError, TypeError, KeyError, MacroPlanError):
        errors.append("invalid_task_dependency_graph")
    if scene.get("scheduler") != {
        "max_concurrent_subagents": 10, "runtime_variants_generated": False
    }:
        errors.append("runtime_profile_should_be_fixed")
    source_truth = scene.get("world") or {}
    if source_truth.get("source_truth") != "deterministic_simulated_only":
        errors.append("source_truth_claim_wrong")
    if source_truth.get("real_multimodal_files") is not False:
        errors.append("false_real_media_claim")
    sources = source_truth.get("sources", {})
    for sid, source in sources.items():
        if source.get("source_id") != sid or len(source.get("facts", [])) < 3:
            errors.append("missing_source_facts")
        keys = set()
        for fact in source.get("facts", []):
            if fact.get("verification") != "consistent_synthetic_record":
                errors.append("unclean_source_record")
            if fact.get("field") in keys:
                errors.append("duplicated_fact_field")
            keys.add(fact.get("field"))
    grants = source_truth.get("source_grants", {})
    if set(grants) != {t["task_id"] for t in tasks}:
        errors.append("grants_missing_or_extra_tasks")
    for t in tasks:
        if len(set(grants.get(t["task_id"], []))) != len(grants.get(t["task_id"], [])):
            errors.append("duplicated_task_source_grants")
        if not set(grants.get(t["task_id"], [])) <= set(sources):
            errors.append("unauthorized_source_grant")
    brief = task_level_brief(scene)
    if brief != scene.get("question_brief"):
        errors.append("question_brief_not_task_derived")
    try:
        meta = scene["task_plan_provenance"]
        recomputed = author_question(
            brief, ordinal=meta["ordinal"], seed=meta["seed"])
        if "question_authoring" in scene:
            # The deterministic Question is the immutable semantic reference.
            # A model-accepted naturalized Question is allowed only with
            # approved review/audit tied to this SAME TaskGraph contract.
            errors.extend(validate_naturalized_question(scene))
        elif scene["question"] != recomputed:
            errors.append("question_not_derived_from_tasks")
    except (KeyError, TypeError, ValueError):
        errors.append("question_provenance_unavailable")
    if re.search(r"\b(?:S\d+|subtask|KoPL|first agent|second agent)\b",
                 scene.get("question", ""), re.I):
        errors.append("public_question_leaks_private_structure")
    public = scene.get("public") or {}
    expected_public = {
        "task_id", "question", "available_source_descriptions",
        "max_concurrent_subagents", "status",
    }
    if set(public) != expected_public:
        errors.append("public_projection_keys")
    if public.get("question") != scene.get("question"):
        errors.append("public_question_drift")
    if public.get("max_concurrent_subagents") != 10:
        errors.append("bad_public_concurrency")
    if public.get("status") != "synthetic_clean_world":
        errors.append("public_false_grounding_status")
    if any("facts" in doc or "observations" in doc
           for doc in public.get("available_source_descriptions", [])):
        errors.append("private_source_facts_exposed")
    if any(key in json.dumps(public) for key in (
            '"macro_graph"', '"source_priors"', '"task_evidence_graphs"',
            '"feedback_bank"', '"gold_answer"')):
        errors.append("teacher_only_data_leaked")
    expected_descriptions = [
        {"source_id": sid, "title": source["title"],
         "modality_hint": source["modality_hint"],
         "representation": source["representation"]}
        for sid, source in sorted(sources.items())
    ]
    if public.get("available_source_descriptions") != expected_descriptions:
        errors.append("source_description_projection_drift")
    try:
        expected = run_clean_tasks(scene, source_truth)
        for key in (
            "task_evidence_graphs", "feedback_bank", "goal_evidence_coverage"
        ):
            if expected[key] != scene.get(key):
                errors.append("clean_evidence_or_feedback_replay_mismatch:" + key)
        ledger = scene.get("goal_evidence_coverage", {})
        if not ledger.get("evidence_level") or not ledger.get("can_finish_cleanly"):
            errors.append("evidence_sufficiency_not_proven")
        if ledger.get("required_source_fact_refs") != ledger.get("supported_source_fact_refs"):
            errors.append("terminal_source_evidence_coverage_drift")
        for task in tasks:
            tid = task["task_id"]
            graph = scene["task_evidence_graphs"][tid]
            feedback = scene["feedback_bank"][tid]
            if graph["status"] != "clean_complete" or feedback["status"] != "completed":
                errors.append("incomplete_clean_task")
            nodes = graph["nodes"]
            if len(nodes) < 3 or any(n["status"] != "complete" for n in nodes):
                errors.append("missing_per_task_evidence_nodes")
            if {n["evidence_id"] for n in nodes} != set(graph["required_evidence_ids"]):
                errors.append("incomplete_evidence_graph")
            if [t["evidence_id"] for t in feedback["summary_trace"]] != [
                n["evidence_id"] for n in nodes
            ]:
                errors.append("summary_trace_not_linked_to_evidence")
            for n in nodes:
                if n["kind"] == "source_observation":
                    if n["source_id"] not in grants[tid]:
                        errors.append("source_fact_not_authorized")
                if not set(n["depends_on"]) <= set(graph["required_evidence_ids"]):
                    errors.append("evidence_graph_unknown_dependency")
    except (EvidenceExecutionError, KeyError, ValueError, TypeError, IndexError) as exc:
        errors.append("clean_replay_failed:" + type(exc).__name__ + ":" + str(exc)[:130])
    return sorted(set(errors))
