"""Visible, operation-local evidence for Sub-Agent report consistency.

Only the executed attempt's *returned step snapshots* are examined. A
sampled_result_rows preview is complete for a step only when its length equals
the executor's output_row_count. Never infer absence from a truncated preview.
These witnesses explain why a report may truthfully describe intermediate
output that differs from a later delivered WorkProduct; they do NOT prove the
cause, clean answer, or attack origin.
"""
from __future__ import annotations

from typing import Any

# These operators do not introduce a new entity identity. Joins, lookups,
# aggregations and transforms are deliberately excluded.
_ID_PRESERVING = frozenset({
    "project", "take", "sort", "filter_eq", "filter_ne",
    "filter_gt", "filter_ge", "filter_lt", "filter_le",
    "deduplicate",
})
_IDENTITY_FIELDS = (
    "candidate_id", "case_id", "entry_id", "record_id",
    "entity_id", "item_id", "marker_code", "record_key_1",
    "record_key_2",
)


def _complete_step_rows(step: dict) -> list[dict] | None:
    """Read complete *executed* step outputs, not incomplete preview sets."""
    if step.get("execution_status") != "done":
        return None
    rows = step.get("sampled_result_rows")
    count = step.get("output_row_count")
    if (not isinstance(rows, list) or type(count) is not int
            or count != len(rows)):
        return None
    if any(not isinstance(r, dict) or
           not isinstance(r.get("values"), dict) for r in rows):
        return None
    return [r["values"] for r in rows]


def observed_handoff_discrepancies(steps: list[dict]) -> list[dict[str, Any]]:
    """Find *provable*, same-identity discrepancies across adjacent steps.

    A projection/filter of rows cannot generate a previously absent
    candidate identity. For non-identity-preserving operators, incomplete
    snapshots, or ambiguous columns, no inference is made.
    """
    witnesses = []
    for earlier, later in zip(steps, steps[1:]):
        if later.get("operation") not in _ID_PRESERVING:
            continue
        previous_rows = _complete_step_rows(earlier)
        current_rows = _complete_step_rows(later)
        if previous_rows is None or current_rows is None:
            continue
        if not previous_rows or not current_rows:
            continue
        shared = [
            field for field in _IDENTITY_FIELDS
            if all(field in row and row[field] is not None
                   for row in previous_rows + current_rows)
        ]
        if not shared:
            continue
        # Prefer the canonical identity; secondary key fields can differ
        # even when the primary identity is the same.
        field = shared[0]
        prev_ids = {str(row[field]) for row in previous_rows}
        late_ids = {str(row[field]) for row in current_rows}
        unmatched = sorted(late_ids - prev_ids)
        if not unmatched:
            continue
        witnesses.append({
            "observation_type": "identity_preserving_step_discrepancy",
            "earlier_operation": earlier["operation"],
            "later_operation": later["operation"],
            "identity_field": field,
            "earlier_observed_identities": sorted(prev_ids),
            "later_unmatched_identities": unmatched,
            "basis": (
                "Both adjacent executed step snapshots include every "
                "output row; the later operation preserves identity."
            ),
            "interpretation": (
                "The later observed output differs from its earlier "
                "input identity set. It is valid to report this mismatch "
                "without asserting a cause or a correct replacement."
            ),
        })
    return witnesses
