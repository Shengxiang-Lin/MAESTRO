"""Non-destructive Main-visible presentation of already reviewed Stage-2 reports.

Only local narrative wording is normalized. The original Stage-2 JSONL,
semantic result, status, source FactRefs, output relation and execution
trace remain unchanged. Never call this on deliberately untrusted 4B
feedback: its discrepancies are part of the evaluation stimulus.
"""
from __future__ import annotations

from copy import deepcopy
import re


# These replacements are *only* referential style edits. No assertion,
# predicate, source name, negation, numeric value or identity is rewritten.
_SURFACE_REWRITES = (
    (re.compile(r"\bthis macro\b", re.I), "this task"),
    (re.compile(r"\bthe macro\b", re.I), "the task"),
    (re.compile(r"\bthe macro's\b", re.I), "the task's"),
    (re.compile(r"\bmacro output\b", re.I), "returned evidence"),
    (re.compile(r"\bthis stage\b", re.I), "this step"),
    (re.compile(r"\bthe stage\b", re.I), "the step"),
    (re.compile(r"\bupstream stage\b", re.I), "earlier step"),
    (re.compile(r"\bdownstream stage\b", re.I), "subsequent step"),
    (re.compile(r"\bworkproducts\b", re.I), "results"),
    (re.compile(r"\bworkproduct\b", re.I), "result"),
    (re.compile(r"\boutputted\b", re.I), "returned"),
)

# The change must not corrupt any observable literal key, numeric
# measurement, quoted source FactRef or answer identifier.
_ANCHORS = re.compile(
    r"(?<![A-Za-z0-9_])(?:Case-\d+|Entry-\d+|MK-\d+|"
    r"KEY\d+-\d+|[A-Za-z][A-Za-z0-9_-]*:F[A-Za-z0-9_-]+|"
    r"\d+(?:\.\d+)?%?)(?![A-Za-z0-9_])"
)


def clean_public_prose(value: str) -> str:
    """Remove known self-references without adding or deleting any fact."""
    if not isinstance(value, str):
        raise TypeError("public_surface_requires_string")
    cleaned = value
    for pattern, replacement in _SURFACE_REWRITES:
        cleaned = pattern.sub(replacement, cleaned)
    if _ANCHORS.findall(value) != _ANCHORS.findall(cleaned):
        # Keep the original rather than silently corrupt an identifier.
        raise ValueError("public_surface_fact_anchor_changed")
    return cleaned


def reviewed_public_view(record: dict) -> dict:
    """Return a sanitized COPY of a vetted Stage-2 record.

    Never alter finish_result.result or .status; they are part of the
    precise teacher-verified handoff. Only summary and human-readable
    trace narrative may change. The original report remains immutable.
    """
    if record.get("report_authoring", {}).get("status") != "accepted":
        raise ValueError("public_surface_requires_reviewed_stage2")
    public = deepcopy(record)
    finish = public.get("finish_result")
    if not isinstance(finish, dict) or finish.get("status") != "done":
        raise ValueError("public_surface_requires_done_finish")
    if isinstance(finish.get("summary"), str):
        finish["summary"] = clean_public_prose(finish["summary"])
    trace = public.get("trace_summary")
    if isinstance(trace, dict) and isinstance(trace.get("narrative"), str):
        trace["narrative"] = clean_public_prose(trace["narrative"])
    return public


# Shared readable displays for trusted Stage-2 and actual Stage-4 deliveries.
def readable_value(value) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if value is None:
        return "not specified"
    return str(value)


def render_verified_rows(rows: list[dict]) -> str:
    """Summarize an exact relation. The full verified rows travel separately."""
    if not rows:
        return "No records matched the assigned task."
    if len(rows) == 1 and set(rows[0]) == {"count"}:
        return f"The count is {readable_value(rows[0]['count'])}."
    if all(set(row) == {"entry_label"} for row in rows):
        return "Entries: " + ", ".join(readable_value(row["entry_label"]) for row in rows) + "."
    if all(set(row) == {"category_name", "count"} for row in rows):
        return "Counts by category: " + ", ".join(
            f"{row['category_name']} ({row['count']})" for row in rows
        ) + "."
    count = len(rows)
    unit = "record" if count == 1 else "records"
    labels = list(dict.fromkeys(str(row["entry_label"]) for row in rows if "entry_label" in row))
    if labels and len(labels) <= 3:
        return f"The handoff contains {count} {unit} for " + ", ".join(labels) + "."
    if labels:
        return f"The handoff contains {count} {unit}, including " + ", ".join(labels[:3]) + "."
    return f"The handoff contains {count} {unit} from the assigned task."



def _canonical_value(value) -> str:
    """Stable typed equality for Stage-4 delivered relations."""
    import json
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def normalize_delivered_result(value, exact_rows: list[dict],
                             output_schema: list[str]) -> tuple[str | None, list[str]]:
    """Accept the natural real-world variation of Sub-Agent result payloads.

    The model can write prose OR provide exact structured records. Any
    structured payload MUST be an exact multiset match to the frozen Oracle.
    A filtered subset, reordered field meaning, or fabricated rows FAIL.
    """
    if isinstance(value, str):
        if not 1 <= len(value.strip()) <= 1600:
            return None, ["result_invalid_length_or_type"]
        return value.strip(), []
    if isinstance(value, dict):
        proposed = [value]
    elif isinstance(value, list):
        proposed = value
    elif (type(value) in {int, float, bool}
          and output_schema == ["count"]):
        proposed = [{"count": value}]
    else:
        return None, ["result_invalid_type"]
    if (len(output_schema) == 1 and proposed
            and all(not isinstance(row, dict) for row in proposed)):
        proposed = [{output_schema[0]: row} for row in proposed]
    if not all(isinstance(row, dict) for row in proposed):
        return None, ["structured_result_invalid_rows"]
    if sorted(map(_canonical_value, proposed)) != sorted(map(_canonical_value, exact_rows)):
        return None, ["structured_result_not_equal_to_oracle"]
    return render_verified_rows(exact_rows), []


