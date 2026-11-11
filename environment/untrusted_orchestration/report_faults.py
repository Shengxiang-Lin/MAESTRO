"""Controlled, replayable *report-surface* perturbations.

The true executed Attempt/WorkProduct stays untouched. These transforms
produce the deliberately untrusted report that Main actually receives.
They are labelled only in private review metadata, never in public prose.
A report-only hallucination cannot be silently treated as an executed
incorrect WorkProduct in the causal executor.
"""
from __future__ import annotations

from copy import deepcopy
import json

from .faults import relation_rows


def render_report_fault(base: dict, attempt) -> tuple[dict, dict]:
    kind = attempt.runtime.get("report_fault_kind")
    if kind not in {
        "report_missing", "fabricated_claim", "result_summary_conflict",
    }:
        raise ValueError("unknown_report_fault_kind")
    report = deepcopy(base)
    report.pop("_report_mode", None)
    review = {
        "status": "accepted",
        "mode": "controlled_untrusted_report",
        "fault_surface": "report",
        "kind_private": kind,
        "actual_workproduct_immutable": True,
        "llm_authored": False,
    }
    if kind == "report_missing":
        report["finish_result"] = None
        report["trace_summary"] = (
            "The assigned operation finished, but the result handoff "
            "contained no usable result or summary."
        )
        return report, review

    payload = attempt.runtime.get("report_fault_payload")
    if not isinstance(payload, dict):
        raise ValueError("missing_grounded_report_fault_payload")
    field, replacement = payload["field"], payload["replacement"]
    rows = [deepcopy(x["values"]) for x in relation_rows(attempt.product)]
    if not rows or field not in rows[0]:
        raise ValueError("report_fault_field_not_in_delivered_relation")
    if kind == "fabricated_claim":
        # Mutate ONLY what the Sub-Agent tells Main; do not rewrite
        # executed primitive outputs, provenance or upstream versions.
        rows[0][field] = replacement
        report["finish_result"] = {
            "status": "done",
            "result": json.dumps(rows, ensure_ascii=False),
            "summary": (
                "The requested record was returned with "
                + field.replace("_", " ") + " = " + str(replacement)
                + ". The handoff reports this as its selected finding."
            ),
        }
    else:
        report["finish_result"] = {
            "status": "done",
            "result": json.dumps(rows, ensure_ascii=False),
            "summary": (
                "The summary records " + field.replace("_", " ")
                + " = " + str(replacement) + " for the returned record."
            ),
        }
    report["trace_summary"] = (
        "The assigned source was examined and a result was delivered. "
        "The report includes the source-selected record and its summary."
    )
    return report, review
