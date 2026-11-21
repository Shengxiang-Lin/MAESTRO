"""Read-only comparison of already delivered, source-grounded reports.

Unlike the retired history-first generator, no prior Main action, source
observation or Sub-Agent report is ever synthesized or edited.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from .visible_claim_extraction import extract_visible_claims

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


