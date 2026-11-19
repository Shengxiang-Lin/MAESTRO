"""Stage-4 Sub-Agent factual contract compiled from one *executed* attempt.

This module never reads the clean Oracle, injected-fault label, or future
evidence. All delivered fields are executor-owned; complete snapshots can
certify inter-step identity discrepancies, while positive observations from
previewed rows never justify a claim that absent rows are missing.
The contract is private to normal report authoring/validation. Deliberately
untrusted report-only faults bypass it without being silently repaired.
"""
from __future__ import annotations

from copy import deepcopy
import json

from environment.implement_subtask.reporting import (
    _readable_value, _readable_verified_answer,
)
from ..state_machine import Attempt, relation_rows
from .report_evidence import observed_handoff_discrepancies


def _trace_facts(observed: list[dict]) -> list[dict]:
    facts = []
    for index, step in enumerate(observed):
        operation = step.get("operation")
        if not operation:
            continue
        output_count = step.get("output_row_count")
        preview = step.get("sampled_result_rows") or []
        complete = (
            step.get("execution_status") == "done"
            and type(output_count) is int
            and len(preview) == output_count
            and all(isinstance(row, dict) and
                    isinstance(row.get("values"), dict) for row in preview)
        )
        facts.append({
            "step_index": index,
            "operation": operation,
            "execution_status": step.get("execution_status"),
            "output_row_count": (
                output_count if type(output_count) is int else None
            ),
            "complete_result_snapshot": complete,
        })
    return facts


def _partial_observation_facts(observed: list[dict], *,
                               unresolved_fields: list[str],
                               delivered_rows: list[dict]) -> list[dict]:
    """Positive observations only; never infer absence from a preview."""
    fields = list(dict.fromkeys(unresolved_fields))
    delivered_values: dict[str, list] = {}
    for row in delivered_rows:
        for key, value in row.get("values", {}).items():
            delivered_values.setdefault(key, []).append(value)
    for key, values in delivered_values.items():
        if values and all(v is None for v in values) and key not in fields:
            fields.append(key)
    facts = []
    for field in fields:
        if field == "remaining_rows":
            continue
        # A partially populated delivered column is not globally
        # unavailable. Do not claim its existing non-null rows vanished.
        if any(v is not None for v in delivered_values.get(field, [])):
            continue
        values = []
        operations = []
        for step in observed:
            if step.get("execution_status") != "done":
                continue
            for entry in step.get("sampled_result_rows") or []:
                if not isinstance(entry, dict):
                    continue
                value = (entry.get("values") or {}).get(field)
                if value is None:
                    continue
                if value not in values:
                    values.append(value)
                if step.get("operation") not in operations:
                    operations.append(step.get("operation"))
        if values:
            facts.append({
                "field": field, "observed_values_sample": values[:4],
                "observed_in_operations": operations,
                "final_delivery_is_unavailable": True,
                "interpretation": (
                    "Intermediate evidence was observed, but this field "
                    "is NOT available in the delivered WorkProduct."
                ),
            })
    return facts


def _fact_spine(contract: dict) -> dict:
    """A compact, code-owned scaffold; no model decides what occurred."""
    status = contract["delivery_status"]
    missing = contract["unresolved_fields"]
    values_by_field: dict[str, list] = {}
    for row in contract["actual_delivered_rows"]:
        for key, value in row["values"].items():
            values_by_field.setdefault(key, []).append(value)
    null_fields = sorted(
        key for key, values in values_by_field.items()
        if values and all(v is None for v in values)
    )
    delivery = {
        "status": status,
        "result": contract["actual_delivered_rows"],
        "result_text": (
            render_delivered_result(contract)
            if contract["finish_result_exists"] else None
        ),
        "unresolved_fields": missing,
        "null_fields": null_fields,
        "rows_complete": contract["rows_complete"],
    }
    mandatory = []
    if status == "partial":
        fields = sorted(set(missing + null_fields))
        if fields:
            mandatory.append(
                "The final handoff is partial; these fields are "
                "unavailable or incomplete in the delivered WorkProduct: "
                + ", ".join(x.replace("_", " ") for x in fields) + "."
            )
        elif not contract["rows_complete"]:
            mandatory.append(
                "The final handoff contains only a partial set of rows."
            )
        else:
            mandatory.append(
                "The final handoff is partial and requires supplementary evidence."
            )
    for witness in contract["observed_handoff_discrepancies"][:2]:
        mandatory.append(
            "The recorded " + str(witness["earlier_operation"])
            + " output included " + ", ".join(
                witness["earlier_observed_identities"][:5]
            ) + ", but the subsequent " + str(witness["later_operation"])
            + " output included " + ", ".join(
                witness["later_unmatched_identities"][:5]
            ) + ". This observed identity discrepancy remains unverified."
        )
    for fact in contract["partial_intermediate_observations"][:2]:
        mandatory.append(
            "An intermediate " + "/".join(
                str(x) for x in fact["observed_in_operations"][:2]
            ) + " step observed " +
            fact["field"].replace("_", " ") + " values, but this field "
            "was not provided in the final handoff."
        )
    if not contract["finish_result_exists"]:
        mandatory.append(
            "No usable final result was delivered by this attempt."
        )
    return {
        "execution": _trace_facts(contract["observed_operator_steps"]),
        "delivery": delivery,
        "observable_discrepancies":
            contract["observed_handoff_discrepancies"],
        "partial_intermediate_observations":
            contract["partial_intermediate_observations"],
        "mandatory_summary_clauses": mandatory,
        "verification_status": (
            "No independent correctness verification is established "
            "by this attempt's return alone."
        ),
    }


def render_delivered_result(contract: dict) -> str:
    """Render only actually delivered rows, preserving citations when typed."""
    if contract.get("result_encoding") == "cited_record_prose_v1":
        from .result_text import render_records
        return render_records(contract["actual_delivered_rows"])
    if contract.get("result_encoding") == "typed_evidence_json_v1":
        # Preserve Stage-2's exact, fully cited Main-visible Result encoding.
        # These rows belong to THIS executed delivery, not the clean Oracle.
        # Missing/partial fields remain null or absent; explain a partial
        # handoff in Summary rather than appending non-JSON text to Result.
        rows = contract["actual_delivered_rows"]
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("values"), dict)
            or not isinstance(item.get("source_fact_refs"), list)
            for item in rows
        ):
            raise ValueError("typed_delivery_missing_provenance")
        return json.dumps([{
            "values": item["values"],
            "source_fact_refs": list(item["source_fact_refs"]),
        } for item in rows], ensure_ascii=False, separators=(",", ":"))
    rows = [row["values"] for row in contract["actual_delivered_rows"]]
    if len(rows) == 1 and len(rows[0]) <= 5:
        pairs = ", ".join(
            f"{str(k).replace('_', ' ')}: {_readable_value(v)}"
            for k, v in rows[0].items()
        )
        result = "The delivered record contains " + pairs + "."
    elif 2 <= len(rows) <= 10 and all(len(row) <= 5 for row in rows):
        # The Author no longer recreates the immutable result. Preserve
        # actual delivered identities and fields rather than reducing a
        # multi-record result to an uninformative row count.
        records = [
            ", ".join(
                f"{str(key).replace('_', ' ')}: {_readable_value(value)}"
                for key, value in row.items()
            )
            for row in rows
        ]
        detailed = (
            f"The handoff contains {len(rows)} delivered records: "
            + "; ".join(records) + "."
        )
        result = (
            detailed if len(detailed) <= 1200
            else _readable_verified_answer(rows)
        )
    else:
        result = _readable_verified_answer(rows)
    if contract["delivery_status"] == "partial":
        missing = contract["unresolved_fields"]
        if "remaining_rows" in missing:
            result += " This is an incomplete delivered row set."
        elif missing:
            result += (
                " The following values were not delivered: "
                + ", ".join(x.replace("_", " ") for x in missing) + "."
            )
    return result


def build_report_contract(
    task: dict, attempt: Attempt, instruction: str, *,
    result_encoding: str | None = None,
) -> dict:
    """Build the Stage-2-shaped contract from THIS execution, never Oracle.

    In the real runner observed_operator_steps comes from run_attempt's
    per-primitive post-injection trace. Legacy hand-made test Attempts may
    lack steps; planned_operator_steps are then clearly marked as plans.
    """
    delivered = relation_rows(attempt.product)
    # Present the actual trace with Stage-2-style evidence examples, while
    # removing preview-truncation counters that can be mistaken for real
    # missing rows. Full runtime evidence remains unchanged in private audit.
    observed = []
    for raw in attempt.runtime.get("observed_operator_steps", []):
        step = deepcopy(raw)
        for key in ("remaining_rows_omitted", "evidence_atoms_truncated",
                    "evidence_atoms_total", "remaining_atoms_omitted"):
            step.pop(key, None)
        if "evidence_atoms" in step:
            step["sampled_evidence_atoms"] = step.pop("evidence_atoms")
        step["evidence_preview_only"] = True
        observed.append(step)
    planned = [
        {"operation": node["op"], "direct_source": node.get("source_id")}
        for node in task.get("operator_chain", [])
    ]
    executed = [
        row["operation"] for row in observed
        if row.get("execution_status") == "done"
    ]
    refs = sorted({
        ref for row in delivered for ref in row["source_fact_refs"]
    })
    refs = sorted(set(refs) | {
        ref for step in observed
        for ref in step.get("supporting_fact_refs", [])
    })
    contract = {
        "assigned_task_instruction": instruction,
        "result_encoding": result_encoding,
        "observed_operator_steps": observed,
        "observed_handoff_discrepancies":
            observed_handoff_discrepancies(observed),
        "planned_operator_steps": planned,
        "trace_from_actual_execution": bool(observed),
        "executed_operations_in_order": executed,
        # As in Stage 2, the actual complete delivered relation is the
        # authoritative answer. Truncation applies to evidence PREVIEWS,
        # not to execution. Do not hide the rest of the delivered relation.
        "actual_delivered_rows": delivered,
        "total_delivered_rows": len(delivered),
        "delivered_relation_is_complete_in_prompt": True,
        "evidence_samples_are_preview_only": True,
        "output_fields": (
            list(attempt.product.columns) if attempt.product else []
        ),
        "authorized_direct_sources": list(task["source_ids"]),
        "available_provenance_sources": sorted({
            ref.split(":", 1)[0] for ref in refs
        }),
        "returned_source_fact_refs": refs,
        "available_fields": list(attempt.available_fields),
        "unresolved_fields": list(attempt.runtime.get("missing_fields", [])),
        "rows_complete": attempt.runtime.get("rows_complete", True),
        "delivery_issue": (
            "partial_field_delivery"
            if attempt.runtime.get("delivery_fault_kind") == "field_not_delivered"
            else "incomplete_delivered_record_set"
            if attempt.runtime.get("delivery_fault_kind") == "row_set_truncated"
            else None
        ),
        "execution_ended": True,
        "delivery_status": attempt.status,
        "finish_result_exists": attempt.product is not None,
        "delivery_semantics": (
            "The task returned NO finish_result. Only actually completed "
            "operations can be described."
            if attempt.product is None else
            "An internal operator may have observed more than the final "
            "handoff delivered. The final delivered WorkProduct is PARTIAL: "
            "report only available fields/rows, not missing values."
            if attempt.status == "partial" else
            "The complete returned WorkProduct is authoritative. Sample "
            "previews do NOT limit which rows the executor processed."
        ),
        "attempt_version": attempt.version,
        "used_parent_versions": attempt.used_parent_versions,
        "upstream_handoffs": list(task.get("depends_on", [])),
        "stage_scope_note": (
            "Only actually observed operations and this delivered result "
            "may be reported. Later Macro operations are not yet performed."
        ),
    }
    contract["partial_intermediate_observations"] = (
        _partial_observation_facts(
            observed, unresolved_fields=contract["unresolved_fields"],
            delivered_rows=delivered,
        ) if attempt.status == "partial" else []
    )
    contract["fact_spine"] = _fact_spine(contract)
    return contract




