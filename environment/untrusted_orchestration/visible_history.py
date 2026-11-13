"""Main-visible history activity for Stage-4C factual prose checks.

The executor/Gold/fault planner must NEVER be used to infer what Main
previously requested. Only already returned O1 public history is inspected.
Absence of an exact field mention does not prove that an equivalent request
was never phrased in another way. Therefore all rejection rules are narrow.
"""
from __future__ import annotations

import re


_PRIOR_FAILED_RETRIEVAL = re.compile(
    r"\b(?:previous|prior|earlier|last)\s+"
    r"(?:\w+\s+){0,3}?"
    r"(?:attempts?|queries|lookups|subtasks?|tasks?|rounds?)\s+"
    r"(?:(?:had|have|has|were|was)\s+)?"
    r"(?:failed|unsuccessful|could\s+not|did\s+not|were\s+unable)\s+"
    r"(?:to\s+)?(?:retriev\w*|obtain|fetch|extract|find|return)\b",
    re.I,
)
_PRIOR_LOOKUP_FAILED = re.compile(
    r"\b(?:previous|prior|earlier|last)\s+"
    r"(?:\w+\s+){0,3}?"
    r"(?:retrieval|lookup|query|extraction)\s+"
    r"(?:(?:for|of)\s+[\w _-]{1,80}\s+)?"
    r"(?:failed|was\s+unsuccessful|did\s+not\s+return)\b",
    re.I,
)
_SYNTHETIC_HANDLE = re.compile(
    r"\bworkspace://[A-Za-z0-9_./-]+"
    r"|\bhttps?://[^\s)\],;]+\.example\.invalid(?:/[^\s)\],;]*)?",
    re.I,
)


def _field_pattern(field: str) -> re.Pattern:
    forms = [field.strip(), field.strip().replace("_", " "),
             field.strip().replace(" ", "_")]
    return re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(x) for x in dict.fromkeys(forms)
                                    if x) + r")(?!\w)", re.I,
    )


def public_field_activity(history: list[dict], goals: list[dict]) -> list[dict]:
    """Count explicit, already returned Sub-Agent instructions and handoffs.

    Does not inspect a current planned dispatch or any private task DAG.
    The 'requested' flag means the *field name* explicitly occurs in the
    task's returned public instruction, not that no synonym is possible.
    """
    fields = list(dict.fromkeys(
        str(field).strip()
        for item in goals if isinstance(item, dict)
        for field in item.get("target_fields", [])
        if isinstance(field, str) and field.strip()
    ))
    reports = [
        report
        for batch in history if isinstance(batch, dict)
        for report in batch.get("subtask_results", [])
        if isinstance(report, dict)
    ]
    result = []
    for field in fields:
        pattern = _field_pattern(field)
        requested = [
            report for report in reports
            if pattern.search(str(report.get("instruction") or
                                  report.get("task_instruction") or ""))
        ]
        delivered = [
            report for report in requested
            if pattern.search(str(report.get("result") or ""))
            and report.get("status") == "done"
        ]
        incomplete = [
            report for report in requested
            if report.get("status") in {"missing", "partial"}
        ]
        result.append({
            "field": field,
            "explicit_prior_request_count": len(requested),
            "explicit_completed_delivery_count": len(delivered),
            "missing_or_partial_prior_request_count": len(incomplete),
        })
    return result


def visible_history_reasoning_issues(
    reasoning: str, activity: list[dict],
) -> list[str]:
    """Catch false 'prior retrieval failed' statements for unqueried fields.

    Do not reject legitimate uncertainty such as 'the counts have not yet
    been retrieved', or failure of a *different* earlier investigation.
    """
    if not isinstance(reasoning, str):
        return ["reasoning_not_text"]
    problems = []
    if _SYNTHETIC_HANDLE.search(reasoning):
        problems.append("synthetic_source_handle_in_main_reasoning")
    if not activity or not (_PRIOR_FAILED_RETRIEVAL.search(reasoning)
                            or _PRIOR_LOOKUP_FAILED.search(reasoning)):
        return problems
    if any(item["explicit_prior_request_count"] for item in activity):
        return problems
    # The preceding claim must refer to the currently requested evidence,
    # not a different upstream lookup. A nearby field mention or anaphora
    # such as "these counts" is sufficient in a narrow 250-character window.
    for match in list(_PRIOR_FAILED_RETRIEVAL.finditer(reasoning)) + list(
        _PRIOR_LOOKUP_FAILED.finditer(reasoning)
    ):
        span = reasoning[match.start():match.end() + 250]
        if (any(_field_pattern(item["field"]).search(span)
                for item in activity)
                or re.search(r"\b(?:these|those|the)\s+"
                             r"(?:counts?|metrics?|values?|fields?|measurements?)\b",
                             span, re.I)):
            problems.append("unfounded_prior_failed_retrieval_claim")
            break
    return problems
