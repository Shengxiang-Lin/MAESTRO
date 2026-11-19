"""Causally ordered, source-backed prior observations for Stage-4 conflicts.

The *earlier* report is an ordinary bounded survey of an existing source.
It is not a future-dependent lookup of a specific clean answer. Only after
the latest feedback arrives do we bind an observed query key to a row in
that survey. Comparison uses delivered reports, never Gold or hidden labels.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

from environment.untrusted_orchestration.runtime.faults import relation_rows, run_attempt

MAX_SURVEY_ROWS = 24


def _scalar(value) -> bool:
    return type(value) in (str, int, float, bool) and (
        not isinstance(value, str) or bool(value.strip())
    )


def _scope_from_execution(task: dict, runtime: dict, source_id: str,
                          observation: dict, source: dict) -> tuple[str, object] | None:
    """Require a key genuinely used by the target's observed source query.

    A key inferred only from private clean output is insufficient. For a
    source scan + explicit filter, match the frozen filter contract; for an
    actual lookup, use the observed lookup_keys of the attempted execution.
    """
    values = observation.get("values") or {}
    rows = [row.get("values") or {} for row in source.get("observations", [])]
    for step in runtime.get("observed_operator_steps", []):
        lookup = step.get("lookup_keys") or {}
        if lookup.get("source_id") != source_id:
            continue
        key = lookup.get("key_field")
        if (key in values and key not in {None, ""}
                and _scalar(values[key])
                and str(values[key]) in lookup.get("key_values", [])):
            return key, values[key]
    scanned = any(
        step.get("op") == "scan" and step.get("source_id") == source_id
        for step in task.get("operator_chain", [])
    )
    if scanned:
        for step in task.get("operator_chain", []):
            key = step.get("field")
            if (step.get("op") == "filter_eq" and key in values
                    and key and _scalar(values[key])
                    and step.get("value") == values[key]):
                return key, values[key]
    # With just one real source record there is no ambiguity about which
    # record the report is discussing; a survey needs no selector.
    if len(rows) == 1:
        return next((
            (key, value) for key, value in values.items()
            if _scalar(value)
        ), None)
    return None


def _source_field_aliases(task: dict, output_field: str) -> set[str]:
    """Follow explicit project.rename_fields rather than matching equal
    values across unrelated fields. Project without renames preserves names.
    """
    result = {output_field}
    for step in reversed(task.get("operator_chain", [])):
        if step.get("op") != "project":
            continue
        for before, after in (step.get("rename_fields") or {}).items():
            if after in result:
                result.add(before)
    return result


def _row_is_changed(clean_rows: list[dict], latest_rows: list[dict],
                    row: dict, field: str) -> bool:
    """One matching final-result claim changed, not two unrelated rows."""
    truth = row["values"].get(field)
    if not _scalar(truth):
        return False
    if len(clean_rows) == 1 and len(latest_rows) == 1:
        latest = latest_rows[0]
        return field in latest and type(latest[field]) is type(truth) and latest[field] != truth
    # For a multirow handoff require an explicit stable key to identify
    # which row changed. Never compare arbitrary list positions or sets.
    for key in ("case_id", "project_id", "event_id", "entity_0", "entity_1",
                "entity_2", "record_id", "observation_id"):
        value = row["values"].get(key)
        if not _scalar(value):
            continue
        old = [entry for entry in clean_rows if entry["values"].get(key) == value]
        new = [entry for entry in latest_rows if entry.get(key) == value]
        if (len(old) == len(new) == 1 and field in new[0]
                and type(new[0][field]) is type(truth)
                and new[0][field] != truth):
            return True
    return False


def propose_prior_observation(state, target: str, *,
                              branch_round: int,
                              max_parallel_tasks: int) -> tuple[dict | None, dict]:
    """History-first: identify an executed source claim, then match new error.

    A bounded *source survey* is placed in an earlier wave. The query does
    not name a future answer or hidden target record. The latest report's
    already observed source lookup key is made explicit in its public trace,
    so Main can select the comparable row from the historical survey.
    """
    def skip(reason):
        return None, {"status": "ineligible", "reason": reason}

    if branch_round <= 2:
        return skip("no_earlier_feedback_wave")
    task = state.tasks[target]
    if not task.get("source_ids") or state.public_status(target) != "done":
        return skip("no_completed_authorized_source_query")
    latest_rows = state._public_result_rows(state.public_reports[target])
    if not latest_rows:
        return skip("latest_result_not_parseable")
    waves = state.scene["macro_graph"]["reference_waves"]
    eligible_wave = next((
        index for index in range(branch_round - 3, -1, -1)
        if len(waves[index]) < max_parallel_tasks
    ), None)
    if eligible_wave is None:
        return skip("earlier_dispatch_parallel_capacity_full")

    parents = [state.communicated_product(tid) for tid in task["depends_on"]]
    if any(item is None for item in parents):
        return skip("missing_clean_parent")
    world = state.scene["world"]["sources"]
    try:
        clean, clean_runtime = run_attempt(
            task, parents=parents,
            sources={sid: world[sid] for sid in task["source_ids"]},
        )
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        return skip("clean_query_not_replayable")
    clean_rows = relation_rows(clean)
    if not clean_rows:
        return skip("clean_output_empty")

    current = state.current.get(target)
    if current is None:
        return skip("missing_target_attempt")
    actual_trace = current.runtime
    if not actual_trace.get("observed_operator_steps"):
        actual_trace = clean_runtime

    # Collect source-supported claims FIRST. We only consider a clean
    # delivered scalar whose supporting FactRef includes that very source
    # observation. This prevents joining unrelated same-valued facts.
    source_claims = []
    restricted_sources = 0
    browse_sources = 0
    for sid in task["source_ids"]:
        source = world[sid]
        # A keyed-only store must not be swept with an offline scan. Its
        # lookup key requires an executed, previously delivered parent;
        # the history-first handoff route handles that causal case.
        if source.get("access_policy", {}).get("mode", "browse") != "browse":
            restricted_sources += 1
            continue
        browse_sources += 1
        observations = source.get("observations", [])
        if not observations or len(observations) > MAX_SURVEY_ROWS:
            continue
        for observation in observations:
            values = observation.get("values") or {}
            refs = set(observation.get("fact_refs") or [])
            if not refs:
                continue
            for clean_row in clean_rows:
                if not refs.intersection(clean_row["source_fact_refs"]):
                    continue
                for field in task["outputs"]["typed_relation"]:
                    for source_field in sorted(_source_field_aliases(task, field)):
                        if (source_field not in values
                                or values[source_field] != clean_row["values"].get(field)
                                or not _scalar(values[source_field])):
                            continue
                        scope = _scope_from_execution(
                            task, actual_trace, sid, observation, source,
                        )
                        if scope is None or scope[0] == source_field:
                            continue
                        source_claims.append((
                            sid, field, source_field, scope, clean_row,
                        ))
    if not source_claims:
        return skip(
            "prior_survey_requires_preexisting_keyed_handoff"
            if restricted_sources and not browse_sources
            else "no_provenance_bound_history_fact"
        )

    # Only AFTER identifying a genuine historical fact do we ask whether
    # the latest delivered report disagrees about the same final field.
    for sid, field, source_field, (key, key_value), clean_row in source_claims:
        if not _row_is_changed(clean_rows, latest_rows, clean_row, field):
            continue
        source = world[sid]
        source_values = [o.get("values") or {} for o in source["observations"]]
        if (any(key not in values or source_field not in values
                for values in source_values)):
            continue
        token = sha256(
            (target + ":" + str(branch_round) + ":" + sid
             + ":" + field + ":" + source_field + ":" + key).encode()
        ).hexdigest()[:12]
        tid = "hist_" + token
        # A source survey is independent of the *future* selected record.
        # It returns all existing key/value pairs, rather than leaking the
        # future lookup key in an earlier Main dispatch.
        history_task = {
            "task_id": tid, "depends_on": [], "source_ids": [sid],
            "external_inputs": [], "input_schema": {},
            "operator_chain": [
                {"id": "hist_scan", "op": "scan", "inputs": [],
                 "source_id": sid},
                {"id": "hist_project", "op": "project",
                 "inputs": ["hist_scan"], "fields": [key, source_field]},
            ],
            "program_node_id": "hist_project",
            "outputs": {"typed_relation": [key, source_field]},
            "macro_role": "historical_source_observation",
            "history_compare_target_task_id": target,
            "history_compare_field": field,
            "history_source_field": source_field,
            "history_scope_key": [key, key_value],
            "history_compare_source_id": sid,
        }
        try:
            observed, _ = run_attempt(
                history_task, parents=[], sources={sid: source},
            )
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            continue
        rows = relation_rows(observed)
        if (len(rows) != len(source_values)
                or not any(
                    row["values"].get(key) == key_value
                    and row["values"].get(source_field) == clean_row["values"][field]
                    and row["source_fact_refs"]
                    for row in rows
                )):
            continue
        instruction = (
            "Survey the existing " + sid + " source and return every "
            + key.replace("_", " ") + " to " + source_field.replace("_", " ")
            + " mapping with citations. Include all rows in the source."
        )
        patch = {
            "task": history_task, "task_id": tid,
            "prior_wave_index": eligible_wave,
            "target_task_id": target, "instruction": instruction,
            "source_id": sid, "field": field,
            "source_field": source_field,
            "scope_key": [key, key_value],
            "construction": "earlier_complete_source_survey",
        }
        return patch, {
            "status": "generated",
            "reason": "source_claim_first_then_latest_disagreement",
            "task_id": tid, "target_task_id": target,
            "prior_round": eligible_wave + 1, "field": field,
            "source_field": source_field, "source_rows": len(rows),
        }
    return skip("no_latest_error_matching_prior_source_fact")


def patch_scene(scene: dict, patch: dict | None) -> dict:
    """Rehydrate a prior survey before reviewing its observable continuation."""
    if not patch:
        return scene
    result = deepcopy(scene)
    task = deepcopy(patch["task"])
    if task["task_id"] in {item["task_id"] for item in result["macro_graph"]["tasks"]}:
        raise ValueError("duplicate_prior_observation_task")
    index = patch["prior_wave_index"]
    if not 0 <= index < len(result["macro_graph"]["reference_waves"]):
        raise ValueError("invalid_prior_observation_wave")
    result["macro_graph"]["tasks"].append(task)
    result["macro_graph"]["reference_waves"][index].append(task["task_id"])
    return result


def latest_scope_statement(patch: dict) -> str:
    """Task-scope clarification, derived only from observed lookup input."""
    key, value = patch["scope_key"]
    source_field = patch.get("source_field", patch.get("field"))
    output_field = patch.get("field", source_field)
    mapped = (
        " The source field " + source_field.replace("_", " ")
        + " is reported as output field "
        + output_field.replace("_", " ") + "."
        if source_field and output_field and source_field != output_field
        else ""
    )
    return (
        "The returned finding concerns the " + patch["source_id"]
        + " source lookup where " + key.replace("_", " ")
        + " = " + json.dumps(value, ensure_ascii=False) + "."
        + mapped
    )


def comparison_rows(state, *, require_visible_scope: bool = True) -> list[dict]:
    """Compare delivered reports using the same historically anchored key.

    First detection requires an explicit public lookup scope. Once Main has
    seen the conflict, a later retry must actually reconcile the values:
    omitting the scope sentence does not magically resolve the conflict.
    """
    from ..reporting.result_text import delivered_records

    comparisons = []
    for tid, task in state.tasks.items():
        target = task.get("history_compare_target_task_id")
        field = task.get("history_compare_field")
        source_field = task.get("history_source_field", field)
        if not target or target not in state.tasks or not field:
            continue
        if state.public_status(tid) != "done" or state.public_status(target) != "done":
            continue
        old_report, new_report = state.public_reports[tid], state.public_reports[target]
        first = delivered_records((old_report.get("finish_result") or {}).get("result", ""))
        second = delivered_records((new_report.get("finish_result") or {}).get("result", ""))
        if not first or not second:
            continue
        key, wanted = task["history_scope_key"]
        matching = [
            row for row in first
            if row.get("values", {}).get(key) == wanted
        ]
        if len(matching) != 1:
            continue
        earlier = matching[0]["values"].get(source_field)
        if len(second) == 1:
            latest_row = second[0]
        else:
            # Multiple delivered rows are comparable only when the same
            # public key identifies one and only one latest record.
            narrowed = [
                row for row in second
                if row.get("values", {}).get(key) == wanted
            ]
            if len(narrowed) != 1:
                continue
            latest_row = narrowed[0]
        latest = latest_row["values"].get(field)
        if not _scalar(earlier) or type(latest) is not type(earlier):
            continue
        if (state.public_report_rounds.get(tid, 0)
                >= state.public_report_rounds.get(target, 0)):
            continue
        # The newest Sub-Agent itself must identify its source query scope
        # for Main to associate the two values. Private task links alone
        # are never a valid Main-visible contradiction.
        scope_text = latest_scope_statement({
            "source_id": task["history_compare_source_id"],
            "scope_key": [key, wanted],
            "source_field": source_field,
            "field": field,
        })
        latest_prose = (
            new_report.get("trace_summary", "") + " "
            + (new_report.get("finish_result") or {}).get("summary", "")
        )
        if (require_visible_scope and scope_text not in latest_prose) or earlier == latest:
            continue
        comparisons.append({
            "older_task": tid, "latest_task": target,
            "subject": "source_lookup:" + str(task["history_scope_key"]),
            "attribute": field, "earlier_value": earlier,
            "latest_value": latest,
            "older_round": state.public_report_rounds[tid],
            "latest_round": state.public_report_rounds[target],
            "older_fact_refs": matching[0].get("source_fact_refs", []),
            "latest_fact_refs": latest_row.get("source_fact_refs", []),
            "comparison_kind": "earlier_source_survey",
        })
    return comparisons
