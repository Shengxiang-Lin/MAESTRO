"""Publicly checkable work evidence from the *actual* attempted operators.

There is no reference to Gold, the injected atom, or the clean result here.
Each clause is rendered for both clean and corrupted attempts. The Main
verifier parses ONLY text that was actually delivered in a Sub-Agent report.
"""
from __future__ import annotations

import datetime
import json
import re


_NUMBER = r"-?(?:0|[1-9]\d*)(?:\.\d+)?"
_VALUE = r'(?:"(?:\\.|[^"\\])*"|' + _NUMBER + r")"
_ASSOCIATION = re.compile(
    r"For the record association, I paired key (" + _VALUE
    + r") from the first input with key (" + _VALUE
    + r") from the second input\."
)
_ARITHMETIC = re.compile(
    r"For the (absolute difference|ratio|percentage|date gap) calculation, "
    r"I used (" + _VALUE + r") and (" + _VALUE
    + r"), rounded to (\d+) decimal places, and reported ("
    + _VALUE + r")\."
)
_COUNT = re.compile(
    r"For the (row|group) count, I examined (\d+) records "
    r"and reported (" + _VALUE + r")\."
)


def _scalar(value):
    if type(value) in {str, int, float}:
        return value
    return None


def _unique_support_match(rows, refs):
    """Identify one witnessed input row by source provenance, not by Gold."""
    matching = [
        row for row in rows
        if row["support"] and set(row["support"]) <= refs
    ]
    return matching[0] if len(matching) == 1 else None


def observed_work_checks(step: dict, inputs: list, output) -> list[dict]:
    """Describe observable inputs and outputs at the attempted operator."""
    kind = step["op"]
    if kind == "join" and len(inputs) == 2 and len(output.rows) == 1:
        refs = set(output.rows[0]["support"])
        a = _unique_support_match(inputs[0].rows, refs)
        b = _unique_support_match(inputs[1].rows, refs)
        if a is None or b is None:
            return []
        left = _scalar(a["values"].get(step["left_on"]))
        right = _scalar(b["values"].get(step["right_on"]))
        if left is not None and right is not None:
            return [{
                "kind": "association", "left": left, "right": right,
                "source_fact_refs": sorted(refs),
            }]
        return []
    if len(inputs) != 1 or len(output.rows) != 1:
        return []
    src = inputs[0]
    result = output.rows[0]["values"]
    op = kind
    if op in {"count", "group_count"}:
        if "count" in result and type(result["count"]) is int:
            return [{
                "kind": "row_count" if op == "count" else "group_count",
                "input_count": len(src.rows), "reported": result["count"],
            }]
        return []
    if len(src.rows) != 1:
        return []
    row = src.rows[0]["values"]
    if op == "goal_compute":
        spec = step.get("goal_operation", {})
        if spec.get("op") == "arithmetic":
            op = "arithmetic"
            function = spec.get("function")
            places = spec.get("round_digits", 0)
            fields = step["input_fields"][-2:]
        elif spec.get("op") == "temporal_difference":
            op, function, places = "date_gap", "date_gap", 0
            fields = step["input_fields"][-2:]
        else:
            return []
    elif op == "derive_arithmetic":
        function = step["operation"]
        places = step.get("round_digits", 0)
        fields = [step["left_field"], step["right_field"]]
        op = "arithmetic"
    elif op in {"derive_date_gap", "derive_percentage"}:
        function = "date_gap" if op == "derive_date_gap" else "percentage"
        places = 0
        fields = [step["left_field"], step["right_field"]]
    else:
        return []
    if function not in {"absolute_difference", "ratio", "percentage", "date_gap"}:
        return []
    first, second = (_scalar(row.get(field)) for field in fields)
    reported = _scalar(result.get(step["output_field"]))
    if first is None or second is None or type(reported) not in {int, float}:
        return []
    if function != "date_gap" and (
        type(first) not in {int, float}
        or type(second) not in {int, float}
    ):
        return []
    if function == "date_gap" and not all(
        isinstance(x, str) for x in (first, second)
    ):
        return []
    return [{
        "kind": function, "first": first, "second": second,
        "round_digits": places if type(places) is int else 0,
        "reported": reported,
    }]


def public_work_clauses(observed: list[dict]) -> list[str]:
    """Grounded factual statements, identical in clean and faulted mode."""
    clauses = []
    for step in observed:
        for item in step.get("public_work_checks", []):
            kind = item["kind"]
            if kind == "association":
                clauses.append(
                    "For the record association, I paired key "
                    + json.dumps(item["left"], ensure_ascii=False)
                    + " from the first input with key "
                    + json.dumps(item["right"], ensure_ascii=False)
                    + " from the second input."
                )
            elif kind in {"row_count", "group_count"}:
                clauses.append(
                    "For the " + ("row" if kind == "row_count" else "group")
                    + " count, I examined " + str(item["input_count"])
                    + " records and reported " + str(item["reported"]) + "."
                )
            else:
                clauses.append(
                    "For the " + kind.replace("_", " ")
                    + " calculation, I used "
                    + json.dumps(item["first"], ensure_ascii=False)
                    + " and " + json.dumps(item["second"], ensure_ascii=False)
                    + ", rounded to " + str(item["round_digits"])
                    + " decimal places, and reported "
                    + str(item["reported"]) + "."
                )
    return clauses[:4]


def visible_work_discrepancy(public_text: str) -> str | None:
    """Re-evaluate only the exact numerical/identity claims in public prose.

    The source sentence is the evidence, not the private task graph. No
    fault mode, source World, executor trace or clean Oracle is consulted.
    """
    for match in _ASSOCIATION.finditer(public_text):
        left, right = (json.loads(match.group(i)) for i in (1, 2))
        if left != right:
            return (
                "The report associated the first record key "
                + str(left) + " with the second record key " + str(right)
                + ". These identifiers disagree; verify the association."
            )
    for match in _ARITHMETIC.finditer(public_text):
        op = match.group(1)
        left, right = (json.loads(match.group(i)) for i in (2, 3))
        places = int(match.group(4))
        reported = json.loads(match.group(5))
        try:
            if op == "date gap":
                expected = abs((
                    datetime.date.fromisoformat(right)
                    - datetime.date.fromisoformat(left)
                ).days)
            elif type(left) in {int, float} and type(right) in {int, float}:
                if op == "absolute difference":
                    expected = abs(left - right)
                elif op == "ratio" and right:
                    expected = round(left / right, places)
                elif op == "percentage" and type(left) is int and type(right) is int and right > 0:
                    expected = (100 * left) // right
                else:
                    continue
            else:
                continue
            if abs(float(expected) - float(reported)) > 1e-9:
                return (
                    "The report states a " + op + " calculation using "
                    + str(left) + " and " + str(right) + ", but gives "
                    + str(reported) + " rather than " + str(expected)
                    + ". Recompute the numerical finding."
                )
        except (ValueError, TypeError, OverflowError):
            continue
    for match in _COUNT.finditer(public_text):
        count, reported = int(match.group(2)), json.loads(match.group(3))
        if type(reported) in {int, float} and reported != count:
            return (
                "The reported count " + str(reported)
                + " disagrees with the " + str(count)
                + " examined records stated in the same report."
            )
    return None
