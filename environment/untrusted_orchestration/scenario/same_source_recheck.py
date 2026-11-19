"""Causal same-source repeat observations in a later feedback wave.

A previous, publicly returned Sub-Agent can be re-invoked in the next
selected wave using the *same* authorized source and the same inputs.
An injected executor/report fault may yield a conflicting observation.
The new Sub-Agent must be present in the earlier Main delegate action;
adding only a returned report would be a fabricated, unrequested task.
This is a scenario-local branch augmentation, not the frozen Stage-3 run.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

from environment.untrusted_orchestration.runtime.atomic_faults import AtomicFault
from environment.untrusted_orchestration.runtime.faults import (
    FaultUnsupported, relation_rows, run_attempt,
)
from environment.generate_question.execution.program_ir import ProgramError
from environment.untrusted_orchestration.inspection.visible_claim_extraction import SUBJECT_FIELDS


RECHECK_FAULT_KINDS = {"wrong_retrieval", "computation_error", "fabricated_claim"}


def _recheck_task(original: dict, branch_round: int) -> dict:
    suffix = sha256(
        (original["task_id"] + ":" + str(branch_round)).encode()
    ).hexdigest()[:12]
    new_id = "history_recheck_" + suffix
    task = deepcopy(original)
    task["task_id"] = new_id
    local_ids = {
        step["id"]: new_id + "_" + str(index)
        for index, step in enumerate(task["operator_chain"])
    }
    for step in task["operator_chain"]:
        step["inputs"] = [local_ids.get(value, value)
                          for value in step.get("inputs", [])]
        step["id"] = local_ids[step["id"]]
    task["program_node_id"] = local_ids[task["program_node_id"]]
    task["macro_role"] = "verification"
    return task


def _reported_scalar_really_changes(
    clean_rows: list[dict], faulty_rows: list[dict],
    old_report: dict,
) -> bool:
    """Require the same explicit subject and a changed non-key field.

    A swapped entity is not incorrectly called a same-entity conflict.
    A field not actually present in the older public answer cannot be
    supplied retroactively from the private World.
    """
    result = (old_report.get("finish_result") or {}).get("result")
    summary = (old_report.get("finish_result") or {}).get("summary")
    if (old_report.get("finish_result") or {}).get("status") != "done":
        return False
    public = str(result or "") + " " + str(summary or "")
    for clean in clean_rows:
        values = clean["values"]
        # Match the EXACT subject-field precedence used by the public
        # claim parser. Treating a secondary entity label as the subject
        # could misclassify a wrong case_id as a same-entity disagreement.
        subject_field = next((
            key for key in SUBJECT_FIELDS
            if isinstance(values.get(key), str) and values[key].strip()
        ), None)
        if subject_field is None or str(values[subject_field]) not in public:
            continue
        for faulty in faulty_rows:
            other = faulty["values"]
            if other.get(subject_field) != values[subject_field]:
                continue
            for changed in sorted(set(values) & set(other) - {subject_field}):
                expected, got = values[changed], other[changed]
                if (type(expected) in {str, int, float, bool}
                        and type(got) is type(expected)
                        and expected != got and str(expected) in public
                        and clean.get("source_fact_refs")
                        and faulty.get("source_fact_refs")):
                    return True
    return False


def discover_same_source_rechecks(
    scene: dict, round_plan: dict, products: dict, *,
    base_episode: dict | None, atomic_fault_kinds: list[str],
    max_parallel_tasks: int,
) -> list[dict]:
    """Derive latest-wave rechecks only from earlier *visible* observations.

    The fault is validated against an actually executed clone. Same-source
    claims are disagreements, not independent votes. No new source, tool
    grant or hidden key is added.
    """
    if base_episode is None:
        return []
    waves = scene["macro_graph"]["reference_waves"]
    latest_index = round_plan["branch_round"] - 2
    if latest_index <= 0 or len(waves[latest_index]) >= max_parallel_tasks:
        return []
    original_tasks = {
        item["task_id"]: item for item in scene["macro_graph"]["tasks"]
    }
    delegates = {
        action["attempt"] - 1: action
        for action in base_episode.get("trajectory", [])
        if action.get("action") == "delegate_task"
    }
    results = []
    for earlier_index in range(latest_index):
        action = delegates.get(earlier_index)
        if action is None:
            continue
        for slot, older_id in enumerate(waves[earlier_index]):
            older = original_tasks[older_id]
            if (not older.get("source_ids")
                    or not older.get("operator_chain")
                    or not isinstance(older.get("outputs", {}).get(
                        "typed_relation"), list)):
                continue
            old_report = action["result"]["subtask_results"][slot]
            clean_rows = relation_rows(products[older_id])
            if not clean_rows:
                continue
            clone = _recheck_task(older, round_plan["branch_round"])
            if clone["task_id"] in original_tasks:
                continue
            sources = {
                sid: scene["world"]["sources"][sid]
                for sid in clone["source_ids"]
            }
            parents = [products[pid] for pid in clone["depends_on"]]
            # The recheck itself must produce exactly the prior real
            # observation before any fault can be considered.
            try:
                repeated, _ = run_attempt(
                    clone, parents=parents, sources=sources,
                )
            except (FaultUnsupported, ProgramError, KeyError, TypeError, ValueError):
                continue
            if relation_rows(repeated) != clean_rows:
                continue

            for kind in atomic_fault_kinds:
                if kind not in RECHECK_FAULT_KINDS:
                    continue
                # All candidate replacements come from existing authorized
                # observations. Try the finite provenance-bound options,
                # not an invented arbitrary string or numeric offset.
                variants = (sum(len(source.get("observations", []))
                                for source in sources.values())
                            * max(1, len(clone["outputs"]["typed_relation"]))
                            if kind == "fabricated_claim" else 2)
                for variant in range(max(1, variants)):
                    fault = AtomicFault(
                        kind, clone["task_id"],
                        origin_round=round_plan["branch_round"] - 1,
                        variant_index=variant,
                    )
                    try:
                        result, runtime = run_attempt(
                            clone, parents=parents, sources=sources,
                            fault=fault.as_legacy_fault(),
                        )
                    except (FaultUnsupported, ProgramError, KeyError, ValueError, TypeError):
                        continue
                    changed = runtime.get("communicated_rows")
                    if changed is None:
                        changed = relation_rows(result)
                    if not _reported_scalar_really_changes(
                        clean_rows, changed, old_report,
                    ):
                        continue
                    results.append({
                        "older_task_id": older_id,
                        "older_wave_index": earlier_index,
                        "new_task": clone,
                        "fault_kind": kind,
                        "fault_variant": variant,
                        "new_wave_index": latest_index,
                        "construction": "later_main_delegated_same_source_recheck",
                    })
                    # One naturally different error per fault family and
                    # older observation; extra variants are equivalent.
                    break
    return results


def patch_scene(scene: dict, patch: dict | None) -> dict:
    """Apply an already-vetted extra Main assignment to the fork wave."""
    if patch is None:
        return scene
    result = deepcopy(scene)
    task = deepcopy(patch["new_task"])
    tid = task["task_id"]
    if tid in {item["task_id"] for item in result["macro_graph"]["tasks"]}:
        raise ValueError("duplicate_history_recheck_task")
    earlier = {
        t for wave in result["macro_graph"]["reference_waves"][
            :patch["new_wave_index"]] for t in wave
    }
    if not set(task["depends_on"]) <= earlier:
        raise ValueError("history_recheck_future_dependency")
    result["macro_graph"]["tasks"].append(task)
    result["macro_graph"]["reference_waves"][
        patch["new_wave_index"]].append(tid)
    return result


def patch_clean_episode(
    base_episode: dict, scene: dict, patch: dict | None, *,
    max_parallel_tasks: int,
) -> dict:
    """Return a paired branch prefix with a real, visible recheck dispatch.

    The clean output is copied from the identical earlier executed Macro;
    the replay engine independently checks identical execution in discovery.
    We explicitly mark this as an augmented prefix, not the original
    frozen Stage-3 teacher trajectory.
    """
    if patch is None:
        return base_episode
    result = deepcopy(base_episode)
    wave = scene["macro_graph"]["reference_waves"][patch["new_wave_index"]]
    if len(wave) >= max_parallel_tasks:
        raise ValueError("history_recheck_parallel_capacity_full")
    actions = {
        step["attempt"] - 1: step for step in result["trajectory"]
        if step.get("action") == "delegate_task"
    }
    previous = actions[patch["older_wave_index"]]
    newer = actions[patch["new_wave_index"]]
    old_slot = scene["macro_graph"]["reference_waves"][
        patch["older_wave_index"]].index(patch["older_task_id"])
    older_report = previous["result"]["subtask_results"][old_slot]
    older_assignment = previous["params"]["tasks"][old_slot]
    fields = patch["new_task"]["outputs"]["typed_relation"]
    instruction = (
        "Independently recheck the earlier Sub-Agent's reported "
        + ", ".join(fields) + " using the same authorized source and "
        "previously available inputs. Return the actual finding and citations."
    )
    assignment = deepcopy(older_assignment)
    assignment["task_instruction"] = instruction
    assignment["context"] = (
        "Previously returned evidence should be checked independently; "
        "do not assume that it is correct. " + str(
            older_assignment.get("context", "")
        )
    )[:3500]
    new_report = deepcopy(older_report)
    new_report["task_instruction"] = instruction
    new_report["subtask_index"] = len(wave)
    newer["params"]["tasks"].append(assignment)
    newer["result"]["subtask_results"].append(new_report)
    newer["result"]["total_subtasks"] = len(
        newer["result"]["subtask_results"]
    )
    newer["result"]["total_steps"] = sum(
        row["steps_taken"] for row in newer["result"]["subtask_results"]
    )
    newer["result"]["all_done"] = all(
        (row.get("finish_result") or {}).get("status") == "done"
        for row in newer["result"]["subtask_results"]
    )
    return result
