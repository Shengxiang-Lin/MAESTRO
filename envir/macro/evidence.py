"""Clean per-Task EvidenceGraph and source-derived Sub-Agent Feedback.

Every task owns multiple evidence nodes. Source evidence is copied only from
sources specifically granted to that task; handoff evidence is extracted only
from completed upstream WorkProducts. Synthesis is deterministic and cites its
input nodes. The graph is INTERNAL, not an extra layer of Main-Agent tasks.

Clean-only: no missing, corrupted, conflicting, retry or backend switching
states are constructed. Feedback is a work product, not a final Gold answer.
"""
from __future__ import annotations

import copy
import re

from .graph import compile_waves


class EvidenceExecutionError(ValueError):
    pass


def _candidate_rows(inputs: dict, permitted: set, world: dict) -> list[dict]:
    """Propagate deduplicated, explicitly synthetic observations downstream."""
    rows, seen = [], set()
    for product in inputs.values():
        for item in product.get("candidate_observations", []):
            if item["source_id"] not in seen:
                rows.append(copy.deepcopy(item))
                seen.add(item["source_id"])
    for sid in sorted(permitted):
        facts = {row["field"]: row["value"] for row in world["sources"][sid]["facts"]}
        if sid in seen:
            continue
        excerpt = facts.get("original_context_excerpt")
        criterion_terms = {
            word for word in re.findall(r"[a-z]{4,}", facts["question_criterion"].lower())
            if word not in {"which", "where", "when", "what", "from",
                            "with", "that", "have", "their", "were", "would"}
        }
        matched_terms = (
            sorted(criterion_terms & set(re.findall(r"[a-z]{4,}", excerpt.lower())))
            if isinstance(excerpt, str) else []
        )
        rows.append({
            "source_id": sid,
            "original_doc_id": world["sources"][sid].get("original_doc_id"),
            "original_context_matched_terms": matched_terms,
            "original_context_available": isinstance(excerpt, str),
            "modality": facts["source_modality"],
            "candidate_key": facts["candidate_key"],
            "criterion": facts["question_criterion"],
            "score": int(facts["synthetic_attribute_score"]),
            "attribute_present": bool(facts["synthetic_attribute_flag"]),
            "time_index": int(facts["synthetic_time_index"]),
            "group": facts["synthetic_group_key"],
            "entity_key": facts.get("synthetic_entity_key", facts["candidate_key"]),
            "reference_targets": list(facts.get(
                "synthetic_reference_target_source_ids", [])),
            "precedes_targets": list(facts.get(
                "synthetic_precedes_source_ids", [])),
            "provenance": "deterministic_synthetic_observation",
        })
        seen.add(sid)
    return sorted(rows, key=lambda row: row["source_id"])


def _execute_operation(task: dict, rows: list[dict],
                       *, world: dict | None = None) -> dict:
    """A small executable typed work semantics over *synthetic* observations.

    These computations are useful for trace inspection and corruption tests,
    but are NOT original KQA KoPL execution or an official MMQA Gold answer.
    """
    if not rows:
        raise EvidenceExecutionError("operation_has_no_candidate_observations")
    spec = task.get("operation_spec", {})
    op = spec.get("op", task["work_type"]).lower()
    family = spec.get("question_operation", "lookup")
    groups, entities = {}, {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
        entities.setdefault(row.get("entity_key", row["candidate_key"]), []).append(row)
    # Group/category overlap does not establish same entity identity.
    # Directed explicit references are separate from both notions.
    row_by_source = {row["source_id"]: row for row in rows}
    matched_groups = {
        group: sorted(r["candidate_key"] for r in members)
        for group, members in sorted(groups.items())
        if len({r["source_id"] for r in members}) >= 2
    }
    matched_group_sources = {
        group: sorted(r["source_id"] for r in members)
        for group, members in sorted(groups.items())
        if group in matched_groups
    }
    same_entity_links = [
        {"entity_key": key,
         "source_ids": sorted(r["source_id"] for r in members),
         "candidate_observations": sorted(r["candidate_key"] for r in members)}
        for key, members in sorted(entities.items())
        if len({r["source_id"] for r in members}) >= 2
    ]
    reference_links = sorted(
        [
            {"from_source_id": row["source_id"],
             "to_source_id": target,
             "from_candidate": row["candidate_key"],
             "to_candidate": row_by_source[target]["candidate_key"]}
            for row in rows
            for target in row.get("reference_targets", [])
            if target in row_by_source and target != row["source_id"]
        ],
        key=lambda link: (link["from_source_id"], link["to_source_id"]),
    )
    precedence_links = sorted(
        [
            {"before_source_id": row["source_id"],
             "after_source_id": target}
            for row in rows
            for target in row.get("precedes_targets", [])
            if target in row_by_source and target != row["source_id"]
        ],
        key=lambda link: (link["before_source_id"], link["after_source_id"]),
    )
    scoped = set(row_by_source)
    relation_witnesses = [
        copy.deepcopy(relation)
        for relation in (world or {}).get("relations", [])
        if set(relation["source_ids"]) <= scoped
    ]
    selected = [row["candidate_key"] for row in rows if row["attribute_present"]]
    ranked = sorted(rows, key=lambda row: (-row["score"], row["candidate_key"]))
    by_time = sorted(rows, key=lambda row: (row["time_index"], row["candidate_key"]))
    result = {
        "operation": op,
        "candidate_count": len(rows),
        "source_count": len({row["source_id"] for row in rows}),
        "source_ids": [row["source_id"] for row in rows],
        "simulation_only": True,
        "question_operation": family,
        "relation_evidence": relation_witnesses,
        "relation_counts": {
            "shared_category_pairs": sum(
                r["kind"] == "shared_category" for r in relation_witnesses),
            "shared_entity_pairs": sum(
                r["kind"] == "same_entity" for r in relation_witnesses),
            "explicit_references": len(reference_links),
            "explicit_precedence": len(precedence_links),
        },
        "original_context_source_count": sum(
            bool(row.get("original_context_available")) for row in rows
        ),
        "original_context_matched_terms": {
            row["source_id"]: list(row.get("original_context_matched_terms", []))
            for row in rows if row.get("original_context_available")
        },
    }
    if op.startswith("source_extract_"):
        result["extracted"] = [
            {"candidate": row["candidate_key"], "score": row["score"],
             "attribute_present": row["attribute_present"]}
            for row in rows
        ]
    elif op in {"align_candidates", "compose", "reference_chain_resolution",
                "cross_source_evidence_alignment"}:
        result["category_memberships"] = {
            key: sorted(row["candidate_key"] for row in members)
            for key, members in sorted(groups.items())
        }
        result["same_entity_links"] = same_entity_links
        result["reference_links"] = reference_links
        result["linkage_policy"] = (
            "explicit_source_references_distinct_from_identity_and_category")
    elif op in {"intersection", "intersect", "record_matching"}:
        result["matched_groups"] = matched_groups
        result["matched_group_source_ids"] = matched_group_sources
        result["match_policy"] = "category_overlap_across_distinct_sources_not_entity_identity"
    elif op in {"compare", "comparison", "comparative_assessment"}:
        result["top_candidate"] = ranked[0]["candidate_key"]
        result["ranked_candidates"] = [row["candidate_key"] for row in ranked]
    elif op in {"temporal", "timeline_reconstruction"}:
        result["ordered_candidates"] = [
            row["candidate_key"] for row in by_time
        ]
        result["earliest_time_index"] = by_time[0]["time_index"]
        result["explicit_precedence_links"] = precedence_links
        result["temporal_policy"] = "relative_order_not_real_calendar_dates"
    elif op in {"aggregation", "candidate_aggregation", "count"}:
        result["selected_count"] = len(selected)
        result["total_count"] = len(rows)
    elif op in {"prioritization", "candidate_prioritization"}:
        result["priority_order"] = [row["candidate_key"] for row in ranked]
        result["priority_scores"] = {
            row["candidate_key"]: row["score"] for row in ranked
        }
    elif op in {"feasibility", "candidate_feasibility"}:
        result["eligibility_rule"] = "attribute_present and score >= 50"
        result["eligible_candidates"] = [
            row["candidate_key"] for row in rows
            if row["attribute_present"] and row["score"] >= 50
        ]
    elif op in {"verify_evidence"}:
        result["source_handle_unique"] = (
            len({row["source_id"] for row in rows}) == len(rows))
        result["all_observations_simulated"] = all(
            row["provenance"] == "deterministic_synthetic_observation"
            for row in rows)
    elif op in {"answer_synthesis"}:
        # A structured *demonstration* result, not a Gold answer claim.
        result["candidate_selection"] = (
            [by_time[0]["candidate_key"]]
            if family == "temporal" else
            [ranked[0]["candidate_key"]]
            if family == "comparison" else
            [row["candidate_key"] for row in ranked]
            if family == "prioritization" else
            [row["candidate_key"] for row in rows
             if row["attribute_present"] and row["score"] >= 50]
            if family == "feasibility" else
            list(selected)
        )
        # The final GoalReport must expose the FULL user-requested
        # work product. Intermediate scores/groups are not a substitute.
        if family == "comparison":
            result["ranked_candidates"] = [
                row["candidate_key"] for row in ranked]
            result["top_candidate"] = ranked[0]["candidate_key"]
        if family == "prioritization":
            result["priority_order"] = [
                row["candidate_key"] for row in ranked]
            result["priority_scores"] = {
                row["candidate_key"]: row["score"] for row in ranked}
        if family == "temporal":
            result["ordered_candidates"] = [
                row["candidate_key"] for row in by_time]
            result["earliest_time_index"] = by_time[0]["time_index"]
        if family == "lookup":
            result["selected_candidates"] = sorted(selected)
        if family == "feasibility":
            result["eligible_candidates"] = sorted(
                row["candidate_key"] for row in rows
                if row["attribute_present"] and row["score"] >= 50)
            result["eligibility_rule"] = "positive_flag=true AND score>=50"
        if family == "aggregation":
            result["selected_count"] = len(selected)
            result["total_count"] = len(rows)
        if family == "intersection":
            result["matched_groups"] = matched_groups
            result["matched_group_source_ids"] = matched_group_sources
            result["match_policy"] = (
                "category_overlap_across_distinct_sources_not_entity_identity")
        if family == "composition":
            result["reference_links"] = reference_links
            result["same_entity_links"] = same_entity_links
            result["linkage_policy"] = (
                "directed_reference_links_do_not_imply_shared_entity")
        if family == "temporal":
            result["explicit_precedence_links"] = precedence_links
        result["answer_status"] = "synthetic_evidence_only_not_gold_answer"
    else:
        result["selected_candidates"] = list(selected)
    return result


def _task_feedback(task: dict, *, world: dict, completed: dict,
                   tasks: dict) -> tuple[dict, dict]:
    tid = task["task_id"]
    evidence = []
    source_support = set()
    inputs = {}
    case_reference = world["case_reference"]
    permitted = set(world["source_grants"].get(tid, []))
    for slot, binding in task["inputs"].items():
        parent_id = binding["from_task"]
        if parent_id not in completed:
            raise EvidenceExecutionError("unavailable_upstream_work_product:" + parent_id)
        output_slot = binding["output_slot"]
        expected_type = binding["type"]
        if tasks[parent_id]["outputs"].get(output_slot) != expected_type:
            raise EvidenceExecutionError("typed_handoff_mismatch:" + slot)
        parent_result = completed[parent_id]["answer"].get(output_slot)
        if not parent_result or parent_result.get("artifact_type") != expected_type:
            raise EvidenceExecutionError("upstream_work_product_missing:" + slot)
        if parent_result["case_reference"] != case_reference:
            raise EvidenceExecutionError("cross_case_handoff")
        inputs[slot] = parent_result
        source_support.update(parent_result["source_fact_refs"])
        # Distinct evidence units inside the consumer: upstream identity/scope
        # and the actual findings delivered by the upstream work package.
        evidence.append({
            "evidence_id": f"{tid}.E{len(evidence)+1}",
            "kind": "upstream_scope",
            "input_slot": slot,
            "from_task": parent_id,
            "artifact_type": expected_type,
            "statement": (
                f"Received {expected_type} from {parent_id}, covering "
                f"synthetic case {parent_result['case_reference']}."
            ),
            "depends_on": [],
            "upstream_source_fact_refs": list(parent_result["source_fact_refs"]),
            "status": "complete",
        })
        evidence.append({
            "evidence_id": f"{tid}.E{len(evidence)+1}",
            "kind": "upstream_finding",
            "input_slot": slot,
            "from_task": parent_id,
            "artifact_type": expected_type,
            "statement": parent_result["finding_statement"],
            "depends_on": [evidence[-1]["evidence_id"]],
            "upstream_source_fact_refs": list(parent_result["source_fact_refs"]),
            "status": "complete",
        })
    # The source grants come from the composed Task, not from the public
    # question and not from an omniscient private-answer oracle.
    for source_id in sorted(permitted):
        if source_id not in world["sources"]:
            raise EvidenceExecutionError("missing_granted_clean_source:" + source_id)
        source = world["sources"][source_id]
        for fact in source["facts"]:
            if fact["verification"] not in {
                "consistent_synthetic_record",
                "original_mmqa_text_excerpt_not_pixels",
            }:
                raise EvidenceExecutionError("nonclean_fact")
            ref = f"{source_id}:{fact['fact_id']}"
            source_support.add(ref)
            evidence.append({
                "evidence_id": f"{tid}.E{len(evidence)+1}",
                "kind": "source_observation",
                "source_id": source_id,
                "fact_id": fact["fact_id"],
                "field": fact["field"],
                "value": copy.deepcopy(fact["value"]),
                "statement": fact["statement"],
                "depends_on": [],
                "status": "complete",
            })
    if not evidence:
        raise EvidenceExecutionError("unsubstantiated_macro_task:" + tid)
    if inputs:
        # Derived evidence is a checked statement about upstream work
        # coverage/scope. This is a Sub-Agent internal operation, not an
        # extra macro-level delegation.
        upstream_scope_ids = [
            e["evidence_id"] for e in evidence
            if e["kind"] == "upstream_scope"
        ]
        evidence.append({
            "evidence_id": f"{tid}.E{len(evidence)+1}",
            "kind": "verified_scope_alignment",
            "statement": (
                f"All {len(inputs)} upstream work products refer to "
                f"the same review case {case_reference}."
            ),
            "depends_on": upstream_scope_ids,
            "verified_case_reference": case_reference,
            "status": "complete",
        })
    base_ids = [e["evidence_id"] for e in evidence]
    candidates = _candidate_rows(inputs, permitted, world)
    computed = _execute_operation(task, candidates, world=world)
    operator_id = f"{tid}.E{len(evidence)+1}"
    evidence.append({
        "evidence_id": operator_id,
        "kind": "typed_operation_evaluation",
        "operation": computed["operation"],
        "result": copy.deepcopy(computed),
        "statement": (
            f"Executed {computed['operation']} over "
            f"{computed['candidate_count']} synthetic candidate observation(s) "
            f"from {computed['source_count']} original source handle(s)."
        ),
        "depends_on": base_ids,
        "source_fact_refs": sorted(source_support),
        "status": "complete",
    })
    if any(item["status"] != "complete" for item in evidence):
        raise EvidenceExecutionError("incomplete_clean_evidence")
    if not source_support:
        raise EvidenceExecutionError("no_transitive_source_support")
    # This is an evidence claim about the assembled work product, not a
    # fabricated adjudication of regulatory, medical or other real facts.
    input_descriptions = [
        f"{slot} ({value['artifact_type']})" for slot, value in inputs.items()
    ]
    local_descriptions = [
        world["sources"][sid]["role"] for sid in sorted(permitted)
    ]
    inputs_text = ", ".join(input_descriptions + local_descriptions)
    resulting_statement = (
        f"{task['work_type']} for {case_reference} produced a typed synthetic "
        f"work product from {inputs_text}; operation={computed['operation']}; "
        f"candidate_count={computed['candidate_count']}."
    )
    synthesis_id = f"{tid}.E{len(evidence)+1}"
    evidence.append({
        "evidence_id": synthesis_id,
        "kind": "work_synthesis",
        "statement": resulting_statement,
        "depends_on": [operator_id],
        "source_fact_refs": sorted(source_support),
        "status": "complete",
    })
    # Report substantive primary findings: the root observations where
    # available, and inherited upstream conclusions where there are no
    # independent local source observations. Do not simply echo task goals.
    local_statements = [
        e["statement"] for e in evidence
        if e["kind"] == "source_observation"
        and e.get("field") not in {"case_reference", "review_date"}
    ]
    inherited_statements = [
        e["statement"] for e in evidence
        if e["kind"] == "upstream_finding"
    ]
    primary_statements = (local_statements + inherited_statements)[:3]
    result_brief = {
        k: v for k, v in computed.items()
        if k not in {"source_ids", "operation", "question_operation"}
    }
    finding_statement = (
        f"{computed['operation']} evaluated {computed['candidate_count']} "
        f"distinct simulated source candidates; "
        f"result={str(result_brief)[:280]}. "
        f"No original MMQA answer is asserted."
    )
    artifact_slot, artifact_type = next(iter(task["outputs"].items()))
    work_product = {
        "artifact_type": artifact_type,
        "work_type": task["work_type"],
        "case_reference": case_reference,
        "finding_statement": finding_statement,
        "operation_result": copy.deepcopy(computed),
        "candidate_observations": candidates,
        "evidence_refs": [e["evidence_id"] for e in evidence],
        "source_fact_refs": sorted(source_support),
        "synthesis_evidence_id": synthesis_id,
        "coverage": "all_required_internal_evidence_complete",
    }
    trace = []
    for item in evidence:
        if item["kind"] == "source_observation":
            origin = f"{item['source_id']}:{item['fact_id']}"
            action = "ExtractSourceEvidence"
        elif item["kind"] == "upstream_scope":
            origin = f"completed_work:{item['from_task']}"
            action = "BindWorkProductScope"
        elif item["kind"] == "upstream_finding":
            origin = f"completed_work:{item['from_task']}"
            action = "ReviewUpstreamFinding"
        elif item["kind"] == "verified_scope_alignment":
            origin = "verified_upstream_work_product_keys"
            action = "CheckWorkProductAlignment"
        elif item["kind"] == "typed_operation_evaluation":
            origin = "synthetic_candidate_observations_and_completed_work"
            action = "ExecuteTypedEvidenceOperation"
        else:
            origin = "internal_evidence_combination"
            action = "SynthesizeWorkProduct"
        trace.append({
            "evidence_id": item["evidence_id"],
            "action": action,
            "origin": origin,
            "finding": item["statement"],
            **({
                "operation": item["operation"],
                "result": copy.deepcopy(item["result"]),
            } if item["kind"] == "typed_operation_evaluation" else {}),
        })
    feedback = {
        "task_id": tid,
        "status": "completed",
        "answer": {artifact_slot: work_product},
        "summary": (
            f"{finding_statement} The resulting {artifact_type} is supported "
            f"by {len(source_support)} source fact reference(s)."
        ),
        "summary_trace": trace,
        "evidence_refs": work_product["evidence_refs"],
        "source_fact_refs": work_product["source_fact_refs"],
        "verification": "clean_simulated_evidence_replay",
    }
    graph = {
        "task_id": tid,
        "status": "clean_complete",
        "nodes": evidence,
        "terminal_evidence_id": synthesis_id,
        "required_evidence_ids": [item["evidence_id"] for item in evidence],
        "source_grants": sorted(permitted),
        "input_handoffs": {
            k: {"from_task": task["inputs"][k]["from_task"],
                "artifact_type": task["inputs"][k]["type"]}
            for k in inputs
        },
    }
    return graph, feedback


def run_clean_tasks(plan: dict, world: dict) -> dict:
    tasks_list = plan["macro_graph"]["tasks"]
    tasks = {t["task_id"]: t for t in tasks_list}
    if len(tasks_list) != len(tasks):
        raise EvidenceExecutionError("duplicate_macro_task_ids")
    waves = compile_waves(tasks_list)
    if waves != plan["macro_graph"]["reference_waves"]:
        raise EvidenceExecutionError("invalid_reference_waves")
    completed, graphs = {}, {}
    for wave in waves:
        if len(wave) > 10:
            raise EvidenceExecutionError("concurrency_limit_exceeded")
        earlier = dict(completed)
        wave_results = {}
        for tid in wave:
            graph, feedback = _task_feedback(
                tasks[tid], world=world, completed=earlier, tasks=tasks)
            graphs[tid] = graph
            wave_results[tid] = feedback
        completed.update(wave_results)
    terminal = plan["macro_graph"]["terminal_handoffs"][0]
    product = completed[terminal["from_task"]]["answer"][terminal["output_slot"]]
    if product["artifact_type"] != terminal["type"]:
        raise EvidenceExecutionError("wrong_terminal_work_product")
    # Evidence-level success: terminal output has a valid proof closure of
    # ALL required leaf source observations across its ancestor work.
    ancestors = set()
    def visit(task_id):
        if task_id in ancestors:
            return
        ancestors.add(task_id)
        for previous in tasks[task_id]["depends_on"]:
            visit(previous)
    visit(terminal["from_task"])
    expected_refs = set()
    for tid in ancestors:
        for source_id in world["source_grants"][tid]:
            for fact in world["sources"][source_id]["facts"]:
                expected_refs.add(f"{source_id}:{fact['fact_id']}")
    observed_refs = set(product["source_fact_refs"])
    valid_evidence = all(
        graph["status"] == "clean_complete" and
        all(n["status"] == "complete" for n in graph["nodes"])
        for graph in graphs.values()
    )
    success = valid_evidence and expected_refs == observed_refs
    if not success:
        raise EvidenceExecutionError("insufficient_evidence_proof")
    return {
        "task_evidence_graphs": graphs,
        "feedback_bank": completed,
        "goal_evidence_coverage": {
            "evidence_level": True,
            "can_finish_cleanly": True,
            "required_source_fact_refs": sorted(expected_refs),
            "supported_source_fact_refs": sorted(observed_refs),
            "terminal_task_id": terminal["from_task"],
            "terminal_evidence_id": graphs[terminal["from_task"]]["terminal_evidence_id"],
            "policy": "all_required_clean_evidence_proven_not_gold_answer",
        },
        "replay_mode": "deterministic_clean_source_and_handoff_execution",
    }
