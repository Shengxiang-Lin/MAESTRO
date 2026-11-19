"""Branch-local latest-feedback witness for targeted cross-report conflicts.

A single already-executed, latest-batch faulty report is contrasted against
a second, executable Stage-4 SIMULATED observation. The older clean history
is NEVER rewritten. Witness source values are constructed from the frozen
clean operator result, not from guessed text or a fabricated FactRef. This
is an independently *instantiated synthetic source*, not evidence of
empirical independence or an existing Stage-1 witness.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

from .faults import relation_rows, run_attempt
from .visible_claim_extraction import SUBJECT_FIELDS


def _scalar(value) -> bool:
    return type(value) in (str, int, float, bool) and (
        not isinstance(value, str) or bool(value.strip())
    )


def propose_witness_patch(state, target: str, *, branch_round: int,
                          diagnostic: dict | None = None) -> dict | None:
    """Find ONE literal, same-subject same-field faulty/clean difference.

    Only the most recently faulted report is examined. A composite source,
    absent FactRef, missing cited origin, or different entity is not a
    comparable fact and must not generate a verification witness.
    """
    def skip(reason: str):
        if diagnostic is not None:
            diagnostic.update(status="ineligible", reason=reason)
        return None

    task = state.tasks[target]
    # A composite Macro may return a *single-source* cited assertion.
    # Decide eligibility per delivered claim, not by Macro source count.
    if not task.get("source_ids"):
        return skip("no_direct_source")
    received = state.public_reports.get(target, {})
    finish = received.get("finish_result") or {}
    if finish.get("status") != "done":
        return skip("no_completed_public_result")
    try:
        delivered = json.loads(finish.get("result", ""))
    except (ValueError, TypeError):
        return skip("result_not_typed_json")
    if not isinstance(delivered, list):
        return skip("result_not_typed_json")
    parents = [state.communicated_product(p) for p in task["depends_on"]]
    if any(parent is None for parent in parents):
        return skip("missing_executable_parent")
    try:
        clean, _ = run_attempt(
            task, parents=parents,
            sources={sid: state.scene["world"]["sources"][sid]
                     for sid in task["source_ids"]},
        )
    except (ValueError, TypeError, KeyError, ZeroDivisionError):
        return skip("clean_execution_not_replayable")
    clean_rows = [item["values"] for item in relation_rows(clean)]
    reason = "no_comparable_cited_claim"
    for item in delivered:
        if not isinstance(item, dict) or not isinstance(item.get("values"), dict):
            continue
        wrong = item["values"]
        refs = item.get("source_fact_refs")
        cited_sources = {
            ref.split(":", 1)[0] for ref in refs
            if isinstance(ref, str) and ":" in ref
        } if isinstance(refs, list) else set()
        origins = (
            {state.origin_by_ref.get(ref) for ref in refs}
            if isinstance(refs, list) else set()
        )
        if (not refs or len(cited_sources) != 1 or len(origins) != 1
                or None in origins or len(cited_sources & set(task["source_ids"])) != 1):
            reason = "no_single_origin_direct_source_claim"
            continue
        reason = "no_explicit_subject"
        subject_field = next((field for field in SUBJECT_FIELDS
                              if isinstance(wrong.get(field), str)
                              and wrong[field].strip()), None)
        if not subject_field:
            continue
        same_subject = [
            row for row in clean_rows
            if row.get(subject_field) == wrong[subject_field]
        ]
        if len(same_subject) != 1:
            reason = "subject_not_unique_in_clean_result"
            continue
        reason = "no_differing_scalar_attribute"
        for field, original in same_subject[0].items():
            reported = wrong.get(field)
            if (field == subject_field or field not in wrong
                    or not _scalar(original) or not _scalar(reported)
                    or type(original) is not type(reported)
                    or original == reported):
                continue
            token = sha256(
                (target + ":" + subject_field + ":" + str(wrong[subject_field])
                 + ":" + field + ":" + str(branch_round)).encode("utf-8")
            ).hexdigest()[:12]
            source_id = "crosscheck_" + token
            task_id = "zz_aux_" + token
            source = {
                "schema": [subject_field, field],
                "access_mode": "browse",
                "access_policy": {"mode": "browse"},
                "modality": "text",
                "title": "Independent comparison register",
                "observations": [{
                    "observation_id": source_id + ".OBS001",
                    "values": {
                        subject_field: wrong[subject_field],
                        field: original,
                    },
                    "fact_refs": [source_id + ":F001"],
                    "evidence_origin_id": source_id + ".independent_observation",
                    "observed_content": (
                        "The comparison register lists " +
                        str(wrong[subject_field]) + " with " + field.replace("_", " ")
                        + " " + str(original) + "."
                    ),
                }],
            }
            op = "read_" + token
            macro = {
                "task_id": task_id,
                "macro_role": "verification",
                "depends_on": [],
                "source_ids": [source_id],
                "external_inputs": [],
                "input_schema": {},
                "operator_chain": [{
                    "id": op, "op": "scan", "inputs": [],
                    "source_id": source_id,
                }],
                "program_node_id": op,
                "outputs": {"typed_relation": [subject_field, field]},
            }
            if diagnostic is not None:
                diagnostic.update(
                    status="generated", reason="row_level_comparable_claim",
                    macro_source_count=len(task["source_ids"]),
                )
            return {
                "version": 1,
                "source": source,
                "source_id": source_id,
                "task": macro,
                "task_id": task_id,
                "feedback_round": branch_round - 1,
                "compare_with": target,
                "claim": {
                    "subject": wrong[subject_field],
                    "subject_field": subject_field,
                    "attribute": field,
                    "faulty_delivered_value": reported,
                    "clean_observation_value": original,
                },
                "construction": "stage4_synthetic_comparison_source",
                "empirically_independent": False,
            }
    return skip(reason)


def patch_scene(scene: dict, patch: dict | None) -> dict:
    """Materialize only this branch's extra source and Macro; do not mutate Stage 1."""
    if not patch:
        return scene
    result = deepcopy(scene)
    tid, sid = patch["task_id"], patch["source_id"]
    if (tid in {t["task_id"] for t in result["macro_graph"]["tasks"]}
            or sid in result["world"]["sources"]):
        raise ValueError("duplicate_latest_batch_witness")
    result["world"]["sources"][sid] = deepcopy(patch["source"])
    result["macro_graph"]["tasks"].append(deepcopy(patch["task"]))
    result["macro_graph"]["reference_waves"][
        patch["feedback_round"] - 1
    ].append(tid)
    result["public"]["available_source_descriptions"].append({
        "source_id": sid,
        "title": patch["source"]["title"],
        "modality_hint": "text",
        "access_mode": "browse",
    })
    return result
