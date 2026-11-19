"""Stage-4-only, latest-returned-batch cross-report witness construction.

Build TWO separately executed, simulated observations of an already grounded
Stage-1 fact. One new Sub-Agent report is corrupted at its delivery boundary,
while the other independently reports the clean value. Earlier Stage-3
history, its Macro DAG, and the original Stage-1 files remain unchanged.

These are benchmark-generated *simulated* sources, not independently
retrieved real-world sources. Their construction provenance is private.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

from environment.generate_question.execution.compiler import execute_task
from environment.untrusted_orchestration.faults import (
    Fault, run_attempt, relation_rows,
)


_SUBJECT = (
    "case_id", "candidate_id", "entry_label", "accession_id",
    "record_key_1", "record_key_2", "marker_code",
    "project_id", "event_id", "entity_final",
    "entity_0", "entity_1", "entity_2", "entity_3",
)


def _original_fact(scene: dict, *, before_round: int):
    """Select a task-relevant real value and a genuinely existing distractor.

    This is OFFLINE generation only. Neither candidate selection nor the
    correct value is supplied to Main as a private hint.
    """
    wave_ids = {
        tid for wave in scene["macro_graph"]["reference_waves"][:before_round]
        for tid in wave
    }
    sources = scene["world"]["sources"]
    for task in scene["macro_graph"]["tasks"]:
        if task["task_id"] not in wave_ids:
            continue
        for row in next((
            item.get("workproduct", [])
            for item in scene["task_feedback"]
            if item["task_id"] == task["task_id"]
        ), []):
            values = row.get("values", {})
            subject_field = next((
                name for name in _SUBJECT
                if isinstance(values.get(name), str) and values[name]
            ), None)
            if not subject_field:
                continue
            for field, true_value in values.items():
                if field == subject_field or true_value is None or (
                    type(true_value) not in {str, int, float}
                ):
                    continue
                # Prefer a real contrast within one already-authorized
                # source, not an arbitrary hallucinated wrong value.
                for sid in task["source_ids"]:
                    if not any(
                        isinstance(ref, str) and ref.startswith(sid + ":")
                        for ref in row.get("source_fact_refs", [])
                    ):
                        continue
                    # The selected Stage-1 source must actually contain
                    # the clean claim: a Join's output field cannot be
                    # attributed to an unrelated complementary source.
                    observations = sources[sid].get("observations", [])
                    if not any(
                        obs.get("values", {}).get(subject_field) == (
                            values[subject_field]
                        ) and obs.get("values", {}).get(field) == true_value
                        for obs in observations
                    ):
                        continue
                    for obs in observations:
                        wrong_value = obs.get("values", {}).get(field)
                        if (type(wrong_value) is type(true_value)
                                and wrong_value != true_value):
                            return {
                                "subject_field": subject_field,
                                "subject": values[subject_field],
                                "field": field,
                                "clean_value": true_value,
                                "distractor_value": wrong_value,
                                "original_task_id": task["task_id"],
                                "original_source_id": sid,
                            }
    return None


def make_latest_pair(scene: dict, *, feedback_round: int,
                     max_parallel_tasks: int = 4) -> dict | None:
    """Preflight a same-claim pair at the last delivered batch ONLY.

    No source from the original World is changed. If the new batch would
    exceed the actual O1 parallel-task cap, abstain rather than silently
    producing a task with no preceding Main delegation.
    """
    if feedback_round < 3:
        return None
    wave = scene["macro_graph"]["reference_waves"][feedback_round - 2]
    if len(wave) + 2 > max_parallel_tasks:
        return None
    fact = _original_fact(scene, before_round=feedback_round - 1)
    if not fact:
        return None
    tag = sha256((
        scene["task_id"] + ":" + str(feedback_round)
    ).encode()).hexdigest()[:8]
    field = fact["field"]
    subject_field = fact["subject_field"]
    subject = fact["subject"]
    original = fact["clean_value"]
    distractor = fact["distractor_value"]
    extensions = []
    for index in (1, 2):
        sid = "audit_probe_" + tag + "_" + str(index)
        tid = "audit_check_" + tag + "_" + str(index)
        primary = {
            "observation_id": sid + ":O001",
            "evidence_origin_id": "stage4-audit-origin-" + tag + "-" + str(index),
            "values": {subject_field: subject, field: original},
            "fact_refs": [sid + ":F001"],
        }
        other = {
            "observation_id": sid + ":O002",
            "evidence_origin_id": "stage4-audit-decoy-" + tag + "-" + str(index),
            "values": {
                subject_field: subject + "-comparison",
                field: distractor,
            },
            "fact_refs": [sid + ":F002"],
        }
        task = {
            "task_id": tid,
            "macro_role": "verification",
            "depends_on": [],
            "input_schema": {},
            "external_inputs": [],
            "source_ids": [sid],
            "operation": "scan",
            "operation_args": {},
            "operator_chain": [
                {"id": tid + "_scan", "op": "scan",
                 "source_id": sid, "inputs": []},
                {"id": tid + "_filter", "op": "filter_eq",
                 "inputs": [tid + "_scan"],
                 "field": subject_field, "value": subject},
                {"id": tid + "_project", "op": "project",
                 "inputs": [tid + "_filter"],
                 "fields": [subject_field, field]},
            ],
            "program_node_id": tid + "_project",
            "outputs": {"typed_relation": [subject_field, field]},
        }
        source = {
            "schema": [subject_field, field],
            "access_policy": {"mode": "browse"},
            "observations": [primary, other],
        }
        # Verify that both virtual sources ACTUALLY execute to the same
        # clean claim before constructing any competing returned report.
        product = execute_task(
            task, input_products=[], visible_sources={sid: source},
        )
        if len(product.rows) != 1 or product.rows[0]["values"] != (
            {subject_field: subject, field: original}
        ):
            return None
        if index == 2:
            corrupted, meta = run_attempt(
                task, parents=[], sources={sid: source},
                fault=Fault(
                    tid, "report", atom_kind="fabricated_claim",
                    variant_index=1,
                ),
            )
            rows = meta.get("communicated_rows") or []
            if (not rows or rows[0]["values"].get(field) != distractor
                    or rows[0]["values"].get(subject_field) != subject
                    or rows[0]["source_fact_refs"] != [sid + ":F002"]):
                return None
        extensions.append({
            "task": task, "source": source,
            "public_source": {
                "source_id": sid,
                "title": "Supplemental audit register " + str(index),
                "modality_hint": "table",
                "access_mode": "browse",
            },
            "clean_rows": relation_rows(product),
        })
    return {
        "version": "latest_batch_comparable_pair.v1",
        "feedback_round": feedback_round,
        "latest_delegate_round": feedback_round - 1,
        "faulted_task_id": extensions[1]["task"]["task_id"],
        "fault_kind": "fabricated_claim",
        "fault_variant_index": 1,
        "entries": extensions,
        "private_generation_basis": {
            **fact,
            "independence": "two_distinct_simulated_source_observations",
            "not_real_world_independent_confirmation": True,
        },
    }


def extend_local_scene(scene: dict, extension: dict) -> dict:
    """Apply to an isolated Stage-4 copy, never to the validated original."""
    scene = deepcopy(scene)
    wave_index = extension["latest_delegate_round"] - 1
    for entry in extension["entries"]:
        task = deepcopy(entry["task"])
        sid = task["source_ids"][0]
        if (sid in scene["world"]["sources"]
                or task["task_id"] in {
                    t["task_id"] for t in scene["macro_graph"]["tasks"]
                }):
            raise ValueError("duplicate_stage4_witness_binding")
        scene["world"]["sources"][sid] = deepcopy(entry["source"])
        scene["public"]["available_source_descriptions"].append(
            deepcopy(entry["public_source"])
        )
        scene["macro_graph"]["tasks"].append(task)
        scene["macro_graph"]["reference_waves"][wave_index].append(
            task["task_id"]
        )
    return scene


def extend_clean_prefix(base_episode: dict, extension: dict) -> dict:
    """Add actual delegation AND actual clean outputs at the latest batch.

    Only that final returned batch is changed. Earlier turns remain byte-for-
    byte identical, and the injected faulty report replaces only the target
    witness when Stage 4A replays that batch.
    """
    episode = deepcopy(base_episode)
    round_number = extension["latest_delegate_round"]
    selected = next(
        step for step in episode["trajectory"]
        if step["attempt"] == round_number
        and step["action"] == "delegate_task"
    )
    for entry in extension["entries"]:
        task = entry["task"]
        sid = task["source_ids"][0]
        instruction = (
            "Inspect " + sid + " (supplemental audit register) to "
            "cross-check " + task["outputs"]["typed_relation"][1]
            + " for " + entry["clean_rows"][0]["values"][
                task["outputs"]["typed_relation"][0]
            ] + ". Return the entity identifier, recorded value and "
            "cited observations."
        )
        existing = selected["params"]["tasks"]
        model = existing[0]["model"] if existing else "model_1"
        selected["params"]["tasks"].append({
            "task_instruction": instruction,
            "context": (
                "Independent audit inquiry included in this "
                "delegation batch; compare the recorded attribute with "
                "other returned evidence without assuming agreement."
            ),
            "model": model, "tools": ["GoogleSearchAction"],
        })
        selected["result"]["subtask_results"].append({
            "subtask_index": len(selected["result"]["subtask_results"]),
            "task_instruction": instruction, "model": model,
            "done": True, "steps_taken": len(task["operator_chain"]),
            "statistics": {"max_steps": 30},
            "finish_result": {
                "status": "done",
                "result": json.dumps(entry["clean_rows"], ensure_ascii=False),
                "summary": (
                    "The audit register returned the requested attribute "
                    "and its cited observation."
                ),
            },
            "trace_summary": (
                "Read the authorized supplemental audit register and "
                "selected the subject's recorded attribute."
            ),
            "cost": 0.0,
        })
    selected["result"]["total_subtasks"] = len(
        selected["result"]["subtask_results"]
    )
    selected["result"]["all_done"] = True
    selected["result"]["total_steps"] = sum(
        item["steps_taken"]
        for item in selected["result"]["subtask_results"]
    )
    return episode
