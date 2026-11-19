"""Stage-4 cross-round claims: compare received reports, never private Gold.

Optional history augmentation is limited to an actual previous executable
WorkProduct. It adds a missing, already-executed record to its own previous
report; it does not fabricate another source, task, or independent verifier.
"""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from ..faults import relation_rows
from ..result_text import render_records
from ..visible_claim_extraction import extract_visible_claims, SUBJECT_FIELDS

if TYPE_CHECKING:
    from ..state_machine import ExecutionState


def public_claims(state: "ExecutionState") -> list[dict]:
    """Only explicit, delivered Result assertions with a real task binding."""
    collected = []
    for tid, report in state.public_reports.items():
        task = state.tasks.get(tid)
        if task is None or state.public_status(tid) != "done":
            continue
        sources = task.get("source_ids", [])
        claims, _ = extract_visible_claims(
            report, output_fields=task["outputs"]["typed_relation"],
            source_id=sources[0] if len(sources) == 1 else None,
        )
        for claim in claims:
            if (claim.get("surface") != "result"
                    or not claim.get("declared_output_field")
                    or claim["subject"].startswith("source:")):
                continue
            # A composite JOIN may cite several observation roots. Its
            # numeric result is not an independent claim about an individual
            # source record; do not turn it into a rival assertion.
            refs = claim.get("fact_refs") or []
            if refs:
                origins = {state.origin_by_ref.get(ref) for ref in refs}
                if len(origins) != 1 or None in origins:
                    continue
            collected.append({
                **claim, "task_id": tid,
                "observed_round": state.public_report_rounds.get(tid, 0),
            })
    return collected


def latest_history_disagreements(state: "ExecutionState") -> list[dict]:
    """Same subject/field, older batch vs latest batch, conflicting values.

    A repeated source is a report discrepancy, not independent corroboration.
    Do not compare a task with its own ancestor/descendant: a transformed
    result may legitimately change a field without being a rival assertion.
    """
    claims = public_claims(state)
    if not claims:
        return []
    latest = max(state.public_report_rounds.values(), default=0)
    previous = [row for row in claims if row["observed_round"] < latest]
    current = [row for row in claims if row["observed_round"] == latest]
    result = []
    for new in current:
        for old in previous:
            if (new["task_id"] == old["task_id"]
                    or old["task_id"] in state._dependents(new["task_id"])
                    or new["task_id"] in state._dependents(old["task_id"])
                    or (new["subject"], new["attribute"], new["valid_scope"])
                       != (old["subject"], old["attribute"], old["valid_scope"])
                    or (type(new["value"]), new["value"])
                       == (type(old["value"]), old["value"])):
                continue
            result.append({
                "older_task": old["task_id"], "latest_task": new["task_id"],
                "subject": new["subject"], "attribute": new["attribute"],
                "earlier_value": old["value"], "latest_value": new["value"],
                "older_round": old["observed_round"],
                "latest_round": new["observed_round"],
                "older_fact_refs": old.get("fact_refs", []),
                "latest_fact_refs": new.get("fact_refs", []),
            })
    return result


def expose_executed_prior_record(
    state: "ExecutionState", task_entries: list[dict],
    episode_prefix: list[dict], *, latest_task_ids: set[str],
) -> dict:
    """Make one *already-executed* prior row explicit if it reveals a conflict.

    Only the previous task's own WorkProduct is used, and only a field that
    the task actually produced can be included. No source, dispatch or field
    is invented. This is scenario-local history presentation, not an edit
    to the saved Stage-3 clean reference.
    """
    diagnostic = {"status": "not_applicable", "reason": "no_supported_prior_row"}
    if len(task_entries) < 2 or latest_history_disagreements(state):
        return diagnostic
    from ..state_machine import public_history_item
    claims = [row for row in public_claims(state)
              if row["task_id"] in latest_task_ids]
    if not claims:
        return diagnostic

    for prior_round in range(len(task_entries) - 1, 0, -1):
        for slot, delivered in enumerate(
                episode_prefix[prior_round - 1]["result"]["subtask_results"]):
            tid = next((
                name for name, item in state.public_reports.items()
                if item is delivered
            ), None)
            if tid is None:
                # Source action reports are deep-copied; resolve by their
                # actual round/slot using previously registered task order.
                prior_ids = [
                    name for name, rnd in state.public_report_rounds.items()
                    if rnd == prior_round
                ]
                if slot >= len(prior_ids):
                    continue
                tid = prior_ids[slot]
            attempt = state.current.get(tid)
            if attempt is None or attempt.product is None:
                continue
            rows = relation_rows(attempt.product)
            if not rows:
                continue
            fields = set(state.tasks[tid]["outputs"]["typed_relation"])
            matches = False
            for actual in rows:
                values = actual["values"]
                subject = next((key for key in SUBJECT_FIELDS
                                if isinstance(values.get(key), str)
                                and values[key].strip()), None)
                if subject is None:
                    continue
                if any(
                    claim["subject"] == values[subject]
                    and claim["attribute"] in fields
                    and claim["attribute"] in values
                    and type(claim["value"]) is type(values[claim["attribute"]])
                    and claim["value"] != values[claim["attribute"]]
                    for claim in claims
                ):
                    matches = True
                    break
            if not matches:
                continue
            report = deepcopy(delivered)
            finish = report.get("finish_result")
            if not isinstance(finish, dict) or finish.get("status") != "done":
                continue
            report["finish_result"]["result"] = render_records([
                {"values": dict(row["values"]),
                 "source_fact_refs": list(row["source_fact_refs"])}
                for row in rows
            ])
            episode_prefix[prior_round - 1]["result"]["subtask_results"][slot] = report
            task_entries[prior_round - 1]["subtask_results"][slot] = (
                public_history_item(report, slot)
            )
            state.set_public_report(tid, report, round_number=prior_round)
            return {
                "status": "exposed", "reason": "previous_actual_workproduct",
                "task_id": tid, "round": prior_round,
                "source": "executed_prior_task_only",
            }
    return diagnostic
