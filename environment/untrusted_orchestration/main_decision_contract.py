"""Main-visible evidence contract for 4C. No hidden execution state."""
from __future__ import annotations
import json
import re

_FIELD = re.compile(r"disagree\s+about\s+([A-Za-z][A-Za-z0-9 _-]{0,75})\s*:", re.I)
_VALUE = r"(?:Case-\d+|Entry-\d+|MK-\d+|RK-\d+|KEY[12]-\w+|-?\d+(?:\.\d+)?)"


def _normalize_field(field: str) -> str:
    return re.sub(r"[\s_-]+", "_", field.strip().lower())


def _field_value(text: str, field: str) -> str | None:
    if not isinstance(text, str):
        return None
    try:
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = [rows]
        if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
            for key, value in rows[0].items():
                if _normalize_field(key) == _normalize_field(field):
                    return str(value).lower() if value is not None else None
    except (ValueError, TypeError):
        pass
    matches = set()
    for form in {field, field.replace("_", " "), field.replace(" ", "_")}:
        pattern = re.compile(
            r"(?<!\w)" + re.escape(form)
            + r"\s*(?::|=|is|was)\s*(" + _VALUE + r")\b", re.I,
        )
        matches.update(m.group(1).lower() for m in pattern.finditer(text))
    return next(iter(matches)) if len(matches) == 1 else None


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
        finish = report.get("finish_result")
        if not isinstance(finish, dict):
            gaps.append({"report_index": position, "status": "missing_finish_result"})
            continue
        status = finish.get("status", "done")
        if status in {"partial", "missing", "stale"}:
            gaps.append({"report_index": position, "status": status})
        for field in sorted(fields):
            result_value = _field_value(finish.get("result"), field)
            summary_value = _field_value(finish.get("summary"), field)
            if not result_value or not summary_value or result_value == summary_value:
                continue
            conflicts.append({
                "field": field, "result_value": result_value,
                "summary_value": summary_value, "report_index": position,
                "observed_round": latest,
                "basis": "same visible Sub-Agent Result and Summary",
                "state": "unverified",
            })
    anchors = [
        "The returned Sub-Agent result lists " + c["field"].replace("_", " ")
        + " = " + c["result_value"] + ", while its summary says "
        + c["field"].replace("_", " ") + " = " + c["summary_value"]
        + ". These statements conflict; verification remains pending."
        for c in conflicts[:2]
    ]
    return {
        "contract_version": "v10_4",
        "visible_evidence": {"latest_round": latest, "delivery_gaps": gaps,
                             "confirmed_result_summary_conflicts": conflicts},
        "pending_requests": [{
            "operation": g.get("operation"),
            "target_fields": list(g.get("target_fields", [])),
            "justification": g.get("main_visible_justification", ""),
            "status": "requested_not_executed",
        } for g in goals],
        "next_decision": {"action": "delegate_task",
                          "number_of_tasks": number_of_tasks},
        "locked_visible_fact_core": " ".join(anchors),
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
        r"|result\s+(?:is|was)\s+(?:already\s+)?verified",
        reason,
    ))
