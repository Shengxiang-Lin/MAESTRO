"""Optional Stage-4 historical evidence, executed before the faulty batch.

One source-backed lookup can be appended to an existing *earlier* dispatch,
when that wave has spare capacity. Unlike the removed same-batch witness,
the older observation is delivered first; the latest report can then disagree
with it. No World fact, FactRef, or unexecuted observation is fabricated.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

from ..faults import relation_rows, run_attempt


def _scalar(value) -> bool:
    return type(value) in (str, int, float, bool) and (
        not isinstance(value, str) or bool(value.strip())
    )


def propose_prior_observation(state, target: str, *,
                              branch_round: int,
                              max_parallel_tasks: int) -> tuple[dict | None, dict]:
    """Find a real source record comparable to the *newest* faulty delivery.

    First find a clean source-backed field and a unique selector in that
    source. Then compare its exact value against the latest delivered report.
    This is an offline scene construction step, never a Main-side oracle.
    The earlier lookup is itself an executable scan/filter/project.
    """
    def skip(reason: str):
        return None, {"status": "ineligible", "reason": reason}

    if branch_round <= 2:
        return skip("no_earlier_feedback_wave")
    task = state.tasks[target]
    if not task.get("source_ids"):
        return skip("no_authorized_direct_source")
    if state.public_status(target) != "done":
        return skip("latest_report_not_completed")
    latest_rows = state._public_result_rows(state.public_reports[target])
    if not latest_rows or len(latest_rows) != 1:
        return skip("latest_result_not_single_row")

    available_waves = state.scene["macro_graph"]["reference_waves"]
    # The faulty batch is at index branch_round - 2. A genuine historical
    # observation must precede that batch, never be a same-batch duplicate.
    eligible_wave = next((
        index for index in range(branch_round - 3, -1, -1)
        if len(available_waves[index]) < max_parallel_tasks
    ), None)
    if eligible_wave is None:
        return skip("earlier_dispatch_parallel_capacity_full")

    parents = [state.communicated_product(tid) for tid in task["depends_on"]]
    if any(parent is None for parent in parents):
        return skip("missing_clean_parent")
    world = state.scene["world"]["sources"]
    try:
        clean, _ = run_attempt(
            task, parents=parents,
            sources={sid: world[sid] for sid in task["source_ids"]},
        )
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return skip("frozen_query_not_replayable")
    clean_rows = relation_rows(clean)
    if len(clean_rows) != 1:
        return skip("clean_result_not_single_row")

    faulty = latest_rows[0]  # _public_result_rows strips the citation wrapper
    clean_values = clean_rows[0]["values"]
    fields = task["outputs"]["typed_relation"]
    for field in fields:
        true_value = clean_values.get(field)
        if (field not in faulty or not _scalar(true_value)
                or type(faulty[field]) is not type(true_value)
                or faulty[field] == true_value):
            continue
        for source_id in task["source_ids"]:
            source = world[source_id]
            observations = source.get("observations", [])
            for observation in observations:
                values = observation.get("values") or {}
                if values.get(field) != true_value:
                    continue
                # Require an identifying predicate other than the answer
                # field. This must select ONE actual source observation.
                keys = sorted(
                    (key for key, val in values.items()
                     if key != field and _scalar(val)
                     and sum(
                         1 for other in observations
                         if (other.get("values") or {}).get(key) == val
                     ) == 1),
                    key=lambda key: (key not in (
                        "case_id", "entity_0", "entity_1", "entity_2",
                        "project_id", "record_id", "observation_id",
                    ), key),
                )
                if not keys:
                    continue
                key = keys[0]
                token = sha256(
                    (target + ":" + str(branch_round) + ":" + source_id
                     + ":" + field + ":" + str(values[key])).encode()
                ).hexdigest()[:12]
                tid = "hist_" + token
                history_task = {
                    "task_id": tid,
                    "depends_on": [],
                    "source_ids": [source_id],
                    "external_inputs": [],
                    "input_schema": {},
                    "operator_chain": [
                        {"id": "hist_scan", "op": "scan", "inputs": [],
                         "source_id": source_id},
                        {"id": "hist_filter", "op": "filter_eq",
                         "inputs": ["hist_scan"],
                         "field": key, "value": values[key]},
                        {"id": "hist_project", "op": "project",
                         "inputs": ["hist_filter"], "fields": [field]},
                    ],
                    "program_node_id": "hist_project",
                    "outputs": {"typed_relation": [field]},
                    "macro_role": "historical_source_observation",
                    "history_compare_target_task_id": target,
                    "history_compare_field": field,
                    "history_scope_key": [key, values[key]],
                }
                try:
                    observed, _ = run_attempt(
                        history_task, parents=[],
                        sources={source_id: source},
                    )
                except (ValueError, KeyError, TypeError, ZeroDivisionError):
                    continue
                observed_rows = relation_rows(observed)
                if (len(observed_rows) != 1
                        or observed_rows[0]["values"].get(field) != true_value
                        or not observed_rows[0]["source_fact_refs"]):
                    continue
                instruction = (
                    "Check the " + source_id + " source for "
                    + key.replace("_", " ") + " = " + str(values[key])
                    + ". Return the observed " + field.replace("_", " ")
                    + " value with its source citation."
                )
                return {
                    "task": history_task,
                    "task_id": tid,
                    "prior_wave_index": eligible_wave,
                    "target_task_id": target,
                    "instruction": instruction,
                    "source_id": source_id,
                    "field": field,
                    "scope_key": [key, values[key]],
                    "construction": "earlier_executable_source_lookup",
                }, {
                    "status": "generated",
                    "reason": "prior_source_observation_matches_clean_query",
                    "task_id": tid,
                    "target_task_id": target,
                    "prior_round": eligible_wave + 1,
                    "field": field,
                }
    return skip("no_comparable_unique_source_observation")


def patch_scene(scene: dict, patch: dict | None) -> dict:
    """Rehydrate the previously executed historical investigation for 4B/4C."""
    if not patch:
        return scene
    new = deepcopy(scene)
    task = deepcopy(patch["task"])
    if task["task_id"] in {x["task_id"] for x in new["macro_graph"]["tasks"]}:
        raise ValueError("duplicate_prior_observation_task")
    index = patch["prior_wave_index"]
    if not 0 <= index < len(new["macro_graph"]["reference_waves"]):
        raise ValueError("invalid_prior_observation_wave")
    new["macro_graph"]["tasks"].append(task)
    new["macro_graph"]["reference_waves"][index].append(task["task_id"])
    return new


def comparison_rows(state) -> list[dict]:
    """Read only publicly delivered paired reports, not their private truth."""
    comparisons = []
    for tid, task in state.tasks.items():
        target = task.get("history_compare_target_task_id")
        field = task.get("history_compare_field")
        if not target or target not in state.tasks or not field:
            continue
        if state.public_status(tid) != "done" or state.public_status(target) != "done":
            continue
        first = state._public_result_rows(state.public_reports[tid])
        second = state._public_result_rows(state.public_reports[target])
        if not first or not second or len(first) != 1 or len(second) != 1:
            continue
        earlier = first[0].get(field)
        latest = second[0].get(field)
        if not _scalar(earlier) or type(latest) is not type(earlier):
            continue
        if (state.public_report_rounds.get(tid, 0)
                >= state.public_report_rounds.get(target, 0)):
            continue
        if earlier == latest:
            continue
        from ..reporting.result_text import delivered_records
        prior_full = delivered_records(
            state.public_reports[tid]["finish_result"].get("result", "")
        ) or []
        latest_full = delivered_records(
            state.public_reports[target]["finish_result"].get("result", "")
        ) or []
        comparisons.append({
            "older_task": tid, "latest_task": target,
            "subject": "source_lookup:" + str(task["history_scope_key"]),
            "attribute": field, "earlier_value": earlier,
            "latest_value": latest,
            "older_round": state.public_report_rounds[tid],
            "latest_round": state.public_report_rounds[target],
            "older_fact_refs": (prior_full[0].get("source_fact_refs", [])
                                if prior_full else []),
            "latest_fact_refs": (latest_full[0].get("source_fact_refs", [])
                                 if latest_full else []),
            "comparison_kind": "earlier_source_lookup",
        })
    return comparisons
