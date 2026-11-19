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


def _cross_report_visible_disagreements(
    batches: list[dict], requested_fields: set[str], latest: int | None,
) -> list[dict]:
    """Detect same-subject/value disagreement from public cited Result rows.

    This tests observable disagreement, NOT factual truth or independent
    evidence origins. True provenance independence is validated in Stage
    4A; a 4C prose reviewer is not permitted to consult that private map.
    """
    from .visible_claim_extraction import SUBJECT_FIELDS
    grouped: dict[tuple[str, str], list[dict]] = {}
    for batch in batches:
        for report in batch["subtask_results"]:
            if not isinstance(report, dict):
                continue
            finish = _visible_finish(report)
            if finish is None or finish.get("status") != "done":
                continue
            try:
                records = json.loads(finish.get("result") or "")
            except (TypeError, ValueError):
                continue
            if not isinstance(records, list):
                continue
            for item in records:
                if not isinstance(item, dict):
                    continue
                values = item.get("values", item)
                refs = item.get("source_fact_refs", [])
                if (not isinstance(values, dict)
                        or not isinstance(refs, list)):
                    continue
                sid_set = {
                    ref.split(":", 1)[0] for ref in refs
                    if isinstance(ref, str) and ":" in ref
                }
                subject_field = next((
                    key for key in SUBJECT_FIELDS
                    if isinstance(values.get(key), str) and values[key]
                ), None)
                if subject_field is None or len(sid_set) != 1:
                    continue
                for field, value in values.items():
                    if (field == subject_field
                            or _normalize_field(field) not in requested_fields
                            or type(value) not in {str, int, float, bool}):
                        continue
                    grouped.setdefault(
                        (values[subject_field], _normalize_field(field)), []
                    ).append({
                        "source_id": next(iter(sid_set)),
                        "value": value,
                        "round": batch["attempt"],
                    })
    disagreements = []
    for (subject, field), rows in sorted(grouped.items()):
        for i, earlier in enumerate(rows):
            for later in rows[i + 1:]:
                if (earlier["source_id"] == later["source_id"]
                        or (type(earlier["value"]), earlier["value"])
                           == (type(later["value"]), later["value"])
                        or latest not in (earlier["round"], later["round"])):
                    continue
                disagreements.append({
                    "subject": subject, "field": field,
                    "reported_values": [
                        earlier["value"], later["value"],
                    ],
                    "source_ids": [
                        earlier["source_id"], later["source_id"],
                    ],
                    "observed_rounds": [
                        earlier["round"], later["round"],
                    ],
                    "basis": "two distinct cited sources in visible Result",
                    "status": "disputed_not_verified",
                })
                break
            else:
                continue
            break
    return disagreements


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
    cross_reports = _cross_report_visible_disagreements(
        batches, fields, latest,
    )
    return {
        "contract_version": "v10_4_3",
        "visible_evidence": {
            "latest_round": latest, "delivery_gaps": gaps,
            "confirmed_result_summary_conflicts": conflicts,
            "confirmed_cross_report_disagreements": cross_reports,
        },
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
    evidence = contract.get("visible_evidence") or {}
    if (not isinstance(review, dict) or review.get("verdict") != "fail"
            or not (
                evidence.get("confirmed_result_summary_conflicts")
                or evidence.get("confirmed_cross_report_disagreements")
            )):
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
