"""Execute a second observation of an existing frozen evidence task.

Both reports use the same source and query contract. Agreement does not
establish independent corroboration. No comparison value or new source is
constructed from the clean answer.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256


def propose_witness_patch(state, target: str, *, branch_round: int,
                          diagnostic: dict | None = None) -> dict | None:
    task = state.tasks[target]
    if not task.get("source_ids"):
        if diagnostic is not None:
            diagnostic.update(status="ineligible", reason="no_direct_source")
        return None
    if any(parent not in state.current for parent in task["depends_on"]):
        if diagnostic is not None:
            diagnostic.update(status="ineligible", reason="missing_executable_parent")
        return None
    token = sha256((target + ":" + str(branch_round)).encode()).hexdigest()[:12]
    repeat = deepcopy(task)
    repeat["task_id"] = "zz_aux_" + token
    repeat["verification_target_task_id"] = target
    repeat["macro_role"] = "verification"
    if diagnostic is not None:
        diagnostic.update(
            status="generated", reason="existing_source_repeat_observation",
            macro_source_count=len(task["source_ids"]),
        )
    return {
        "version": 2,
        "task": repeat,
        "task_id": repeat["task_id"],
        "feedback_round": branch_round - 1,
        "compare_with": target,
        "construction": "repeat_existing_frozen_query",
        "empirically_independent": False,
        "source_ids": list(task["source_ids"]),
        "claim": {"output_fields": list(task["outputs"]["typed_relation"])},
    }


def patch_scene(scene: dict, patch: dict | None) -> dict:
    """Add a branch-local executable observation, supporting older saved patches."""
    if not patch:
        return scene
    result = deepcopy(scene)
    tid = patch["task_id"]
    if tid in {t["task_id"] for t in result["macro_graph"]["tasks"]}:
        raise ValueError("duplicate_latest_batch_witness")
    if patch.get("source") is not None:
        sid = patch["source_id"]
        if sid in result["world"]["sources"]:
            raise ValueError("duplicate_latest_batch_witness")
        result["world"]["sources"][sid] = deepcopy(patch["source"])
        result["public"]["available_source_descriptions"].append({
            "source_id": sid, "title": patch["source"]["title"],
            "modality_hint": "text", "access_mode": "browse",
        })
    result["macro_graph"]["tasks"].append(deepcopy(patch["task"]))
    result["macro_graph"]["reference_waves"][
        patch["feedback_round"] - 1
    ].append(tid)
    return result
