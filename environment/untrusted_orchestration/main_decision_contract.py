"""Main-visible evidence contract for 4C, grounded in public O1 history.

Read the flattened public_history_item() report emitted by _history_batch();
raw Sub-Agent reports with nested finish_result remain supported for
existing test fixtures and diagnostic callers. Never consume hidden Gold.
"""
from __future__ import annotations
import json
import re

from .visible_history import public_field_activity

_FIELD = re.compile(r"disagree\s+about\s+([A-Za-z][A-Za-z0-9 _-]{0,75})\s*:", re.I)
_VALUE = (
    r"(?:Case-\d+|Entry-\d+|Dossier-[A-Za-z0-9]+|"
    r"(?:ENT|Entity)-[A-Za-z0-9]+|MK-\d+|RK-\d+|"
    r"KEY[12]-\w+|true|false|yes|no|-?\d+(?:\.\d+)?)"
)


def _normalize_field(field: str) -> str:
    return re.sub(r"[\s_-]+", "_", field.strip().lower())


def _field_values(text: str, field: str) -> set[str]:
    """Read explicit field values in one or several delivered rows.

    A summary may legitimately mention only one of several returned
    rows. We only infer a contradiction if a *claimed* explicit value
    is absent from the entire Result set. No hidden source is consulted.
    """
    if not isinstance(text, str):
        return set()
    try:
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = [rows]
        if isinstance(rows, list) and rows and all(
            isinstance(row, dict) for row in rows
        ):
            values = set()
            for row in rows:
                payload = row.get("values") if isinstance(
                    row.get("values"), dict
                ) else row
                for key, value in payload.items():
                    if (_normalize_field(key) == _normalize_field(field)
                            and value is not None
                            and not isinstance(value, (list, dict))):
                        values.add(str(value).strip().lower())
            if values:
                return values
    except (ValueError, TypeError):
        pass
    matches = set()
    for form in {field, field.replace("_", " "), field.replace(" ", "_")}:
        pattern = re.compile(
            r"(?<!\w)" + re.escape(form)
            + r"\s*(?::|=|is|was)\s*(" + _VALUE + r")\b", re.I,
        )
        matches.update(m.group(1).lower() for m in pattern.finditer(text))
    return matches


def _visible_finish(report: dict) -> dict | None:
    """Read the *actual O1 history schema*, with legacy raw-report support.

    _history_batch() calls public_history_item(), which flattens the
    Sub-Agent's finish_result into top-level status/result/summary fields.
    The legacy raw report instead keeps those fields under finish_result.
    Neither representation may imply a missing handoff merely because
    the other representation's wrapper is absent.
    """
    nested = report.get("finish_result")
    if isinstance(nested, dict):
        return nested
    if any(key in report for key in ("status", "result", "summary")):
        return {
            "status": report.get("status", "done"),
            "result": report.get("result"),
            "summary": report.get("summary"),
        }
    return None


def build_main_decision_contract(history: list[dict], goals: list[dict],
                                 *, number_of_tasks: int) -> dict:
    """A verification request is not proof; confirm values in one public report."""
    batches = [item for item in history
               if isinstance(item, dict) and isinstance(item.get("attempt"), int)
               and isinstance(item.get("subtask_results"), list)]
    latest = max((item["attempt"] for item in batches), default=None)
    recent = next((b["subtask_results"] for b in reversed(batches)
                   if b["attempt"] == latest), [])
    fields = set()
    for goal in goals:
        if goal.get("operation") != "verify_conflicting_evidence":
            continue
        fields.update(_normalize_field(f) for f in goal.get("target_fields", [])
                      if isinstance(f, str))
        m = _FIELD.search(goal.get("main_visible_justification", ""))
        if m:
            fields.add(_normalize_field(m.group(1)))
    conflicts, gaps = [], []
    for position, report in enumerate(recent):
        if not isinstance(report, dict):
            continue
        finish = _visible_finish(report)
        if finish is None:
            gaps.append({"report_index": position, "status": "missing_finish_result"})
            continue
        status = finish.get("status", "done")
        if status in {"partial", "missing", "stale"}:
            gaps.append({"report_index": position, "status": status})
        for field in sorted(fields):
            result_values = _field_values(finish.get("result"), field)
            summary_values = _field_values(finish.get("summary"), field)
            # Summary may accurately cite only a subset of the Result.
            # An absent value is a genuine *observable* discrepancy.
            unsupported = summary_values - result_values
            if not result_values or not unsupported:
                continue
            conflicts.append({
                "field": field,
                "result_value": ", ".join(sorted(result_values)[:8]),
                "summary_value": ", ".join(sorted(unsupported)[:4]),
                "report_index": position,
                "observed_round": latest,
                "basis": "same visible Sub-Agent Result and Summary",
                "state": "unverified",
            })
    return {
        "contract_version": "v10_4_3",
        "visible_evidence": {"latest_round": latest, "delivery_gaps": gaps,
                             "confirmed_result_summary_conflicts": conflicts},
        "prior_field_activity": public_field_activity(history, goals),
        "pending_requests": [{
            "operation": g.get("operation"),
            "target_fields": list(g.get("target_fields", [])),
            "justification": g.get("main_visible_justification", ""),
            "status": "requested_not_executed",
        } for g in goals],
        "next_decision": {"action": "delegate_task",
                          "number_of_tasks": number_of_tasks},
    }


def critic_denies_confirmed_conflict(review: dict, contract: dict) -> bool:
    if (not isinstance(review, dict) or review.get("verdict") != "fail"
            or not contract["visible_evidence"]["confirmed_result_summary_conflicts"]):
        return False
    reason = (str(review.get("reason", "")) + " " +
              str(review.get("repair_hint", ""))).lower()
    return bool(re.search(
        r"(?:invented|invents?|fabricated|fabricates?|unsupported|no|hallucinated)\s+"
        r"(?:a\s+|factual\s+|specific\s+|direct\s+)*"
        r"(?:conflict|contradiction|mismatch|discrepancy)"
        r"|(?:conflict|contradiction|mismatch|discrepancy)\s+"
        r"(?:is not present|does not exist)"
        r"|summary\s+(?:is|was)\s+(?:a\s+)?hallucination"
        r"|result\s+(?:is|was)\s+(?:already\s+)?verified"
        r"|(?:no|not\s+a|without\s+(?:a|any))\s+(?:actual\s+)?"
        r"(?:conflict|contradiction|mismatch|discrepancy)"
        r"|(?:claim|assertion|statement)\s+(?:that\s+)?"
        r"(?:the\s+)?(?:result|summary).{0,45}"
        r"(?:conflict|contradict|disagree|mismatch).{0,50}"
        r"(?:unsupported|unfounded|invented|false)",
        reason,
    ))
