"""Deterministic semantic evidence-request -> executable-route projection."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List

_SAMPLING = {"reuse_ok", "prefer_diverse", "require_diverse"}
_LEGACY_POLICY_TO_SAMPLING = {
    "reuse_allowed": "reuse_ok",
    "prefer_independent": "prefer_diverse",
    "require_independent": "require_diverse",
}


def _text(value: Any, limit: int = 1200) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _dedup(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in output:
            output.append(text)
    return output


def _normalize_request(task: Dict[str, Any]) -> Dict[str, Any]:
    request = task.get("evidence_request") or {}
    if not isinstance(request, dict) or not request:
        legacy = task.get("evidence_constraint") or {}
        request = legacy if isinstance(legacy, dict) else {}
    if not request:
        return {}
    based_on = _dedup(request.get("based_on") or request.get("independent_of") or [])
    sampling = str(request.get("sampling") or "").strip().lower()
    if sampling not in _SAMPLING:
        legacy_policy = str(request.get("route_policy") or request.get("policy") or "").strip().lower()
        sampling = _LEGACY_POLICY_TO_SAMPLING.get(legacy_policy, "")
    if sampling not in _SAMPLING:
        sampling = "prefer_diverse" if based_on else "reuse_ok"
    return {
        "target_key": _text(request.get("target_key") or request.get("belief_key"), 120),
        "target_requirement": _text(request.get("target_requirement"), 900),
        "sampling": sampling,
        "based_on": based_on,
        "blind": bool(request.get("blind", False)),
    }


def _failure_result(task: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    reason = str(report.get("reason") or "required diverse evidence route unavailable")
    return {
        "task_instruction": str(task.get("task_instruction") or ""),
        "model": str(task.get("model") or ""),
        "tools_assigned": [],
        "preferred_tools": [],
        "steps_taken": 0,
        "done": False,
        "cost": 0.0,
        "usage": {"total_input_tokens": 0, "total_output_tokens": 0, "total_tokens": 0, "total_cost": 0.0, "call_count": 0},
        "error": f"evidence_projection_unavailable: {reason}",
        "finish_result": {"status": "partial", "result": "", "summary": f"The requested strict evidence-diversity projection could not be executed: {reason}."},
        "execution_projection": deepcopy(report),
    }


def project_evidence_requests(params: Dict[str, Any], *, ledger: Any, available_tools: Iterable[str], provider_tools: Iterable[str], strict_diversity: bool = True, blind_requery: bool = True) -> Dict[str, Any]:
    raw_tasks = (params or {}).get("tasks") or []
    available = _dedup(available_tools)
    provider_set = set(_dedup(provider_tools))
    executable: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    records = {str(item.get("result_id") or ""): item for item in getattr(ledger, "results", []) if isinstance(item, dict) and item.get("result_id")}

    for task_index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, dict):
            continue
        task = deepcopy(raw_task)
        request = _normalize_request(task)
        if request:
            task["evidence_request"] = deepcopy(request)
        task.pop("evidence_constraint", None)
        based_on = list(request.get("based_on") or [])
        sampling = str(request.get("sampling") or "reuse_ok")
        blind = bool(request.get("blind", False))
        target_key = str(request.get("target_key") or "")
        target_requirement = str(request.get("target_requirement") or "")
        report: Dict[str, Any] = {
            "task_index": task_index,
            "target_key": target_key,
            "target_requirement": target_requirement,
            "sampling": sampling,
            "based_on": based_on,
            "blind": blind,
            "status": "ordinary" if not request else "projected",
            "resolved_refs": [],
            "unknown_refs": [],
            "referenced_provider_ids": [],
            "preferred_provider_ids": [],
            "fallback_provider_ids": [],
            "blocked_provider_ids": [],
            "requested_tool_ids": _dedup(task.get("tools") or []),
            "execution_tool_ids": [],
            "execution_provider_ids": [],
            "warnings": [],
        }
        resolved_refs = [ref for ref in based_on if ref in records]
        unknown_refs = [ref for ref in based_on if ref not in records]
        report["resolved_refs"] = resolved_refs
        report["unknown_refs"] = unknown_refs
        if sampling == "require_diverse" and based_on and not resolved_refs:
            report.update(status="unavailable", reason="none of the based_on result IDs are present in the visible ledger")
            reports.append(report)
            failures.append(_failure_result(task, report))
            continue
        if sampling == "prefer_diverse" and based_on and not resolved_refs:
            report["status"] = "preference_unresolved"
            report["warnings"].append("based_on_refs_not_resolved")

        referenced_providers: List[str] = []
        for ref in resolved_refs:
            for tool in records[ref].get("executed_tool_ids") or []:
                tool = str(tool)
                if tool in provider_set and tool not in referenced_providers:
                    referenced_providers.append(tool)
        report["referenced_provider_ids"] = referenced_providers
        requested = _dedup(task.get("tools") or [])
        candidates = requested if requested else list(available)
        candidate_providers = [tool for tool in candidates if tool in provider_set]
        alternate_providers = [tool for tool in candidate_providers if tool not in set(referenced_providers)]
        reused_providers = [tool for tool in candidate_providers if tool in set(referenced_providers)]
        if request and not requested:
            task["tools"] = list(candidates)

        if sampling == "require_diverse" and strict_diversity:
            feasible = [tool for tool in candidates if tool not in set(referenced_providers)]
            feasible_providers = [tool for tool in feasible if tool in provider_set]
            if referenced_providers and not feasible_providers:
                report.update(
                    status="unavailable",
                    reason="no alternate external provider remains after excluding the provider path of " + ", ".join(resolved_refs),
                    blocked_provider_ids=referenced_providers,
                    execution_tool_ids=[],
                    execution_provider_ids=[],
                )
                reports.append(report)
                failures.append(_failure_result(task, report))
                continue
            task["tools"] = feasible
            task["preferred_tools"] = feasible_providers
            report["blocked_provider_ids"] = referenced_providers
            report["preferred_provider_ids"] = feasible_providers
            report["status"] = "strict_diversity"
        elif sampling == "prefer_diverse":
            if alternate_providers:
                task["preferred_tools"] = alternate_providers
                report["status"] = "diversity_preferred"
            elif referenced_providers:
                report["status"] = "reuse_fallback"
            report["preferred_provider_ids"] = alternate_providers
            report["fallback_provider_ids"] = reused_providers
        else:
            report["status"] = "reuse_ok" if request else "ordinary"

        report["execution_tool_ids"] = deepcopy(task.get("tools") or [])
        report["execution_provider_ids"] = [tool for tool in report["execution_tool_ids"] if tool in provider_set]
        if blind and blind_requery:
            if target_requirement:
                task["task_instruction"] = "Resolve the following requirement afresh from the original question and permitted tools, without assuming or confirming any candidate value from earlier results: " + target_requirement
                task["context"] = ""
                report["blind_requery_applied"] = True
            else:
                report["blind_requery_applied"] = False
                report["warnings"].append("blind_target_missing")
        else:
            report["blind_requery_applied"] = False

        projection = {
            "target_key": target_key,
            "sampling": sampling,
            "based_on": resolved_refs,
            "referenced_provider_ids": deepcopy(report["referenced_provider_ids"]),
            "preferred_provider_ids": deepcopy(report["preferred_provider_ids"]),
            "fallback_provider_ids": deepcopy(report["fallback_provider_ids"]),
            "blocked_provider_ids": deepcopy(report["blocked_provider_ids"]),
            "blind": bool(report.get("blind_requery_applied", False)),
        }
        task["execution_projection"] = projection
        task["route_compilation"] = deepcopy(projection)
        reports.append(report)
        executable.append(task)

    events = getattr(ledger, "events", None)
    if isinstance(events, list):
        events.append({"event": "execution_projection", "projection_reports": deepcopy(reports), "executable_tasks": len(executable), "unavailable_tasks": len(failures)})
    return {
        "execution_params": {"tasks": executable},
        "projection_reports": reports,
        "projection_failures": failures,
        "route_reports": reports,
        "route_failures": failures,
    }


def compile_evidence_routes(params: Dict[str, Any], *, ledger: Any, available_tools: Iterable[str], provider_tools: Iterable[str], provider_disjoint: bool = True, enforce_blind: bool = True) -> Dict[str, Any]:
    return project_evidence_requests(
        params,
        ledger=ledger,
        available_tools=available_tools,
        provider_tools=provider_tools,
        strict_diversity=provider_disjoint,
        blind_requery=enforce_blind,
    )
