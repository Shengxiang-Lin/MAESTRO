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

from .graph import compile_waves


class EvidenceExecutionError(ValueError):
    pass


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
            if fact["verification"] != "consistent_synthetic_record":
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
    base_ids = [e["evidence_id"] for e in evidence]
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
        f"{task['work_type']} for {case_reference} synthesized the documented "
        f"findings from {inputs_text}."
    )
    synthesis_id = f"{tid}.E{len(evidence)+1}"
    evidence.append({
        "evidence_id": synthesis_id,
        "kind": "work_synthesis",
        "statement": resulting_statement,
        "depends_on": base_ids,
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
    finding_statement = " ".join(primary_statements) or resulting_statement
    artifact_slot, artifact_type = next(iter(task["outputs"].items()))
    work_product = {
        "artifact_type": artifact_type,
        "work_type": task["work_type"],
        "case_reference": case_reference,
        "finding_statement": finding_statement,
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
        else:
            origin = "internal_evidence_combination"
            action = "SynthesizeWorkProduct"
        trace.append({
            "evidence_id": item["evidence_id"],
            "action": action,
            "origin": origin,
            "finding": item["statement"],
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
