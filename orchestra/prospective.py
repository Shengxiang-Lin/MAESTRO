"""Normalization for MAESTRO epistemic/prospective decisions."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List

from .config import ProspectiveConfig
from .modes import ORCHESTRATION_MODES

_READINESS = {"ready", "not_ready", "uncertain"}
_ACTIONS = {"delegate_task", "complete"}
_EPISTEMIC_STATUS = {"unresolved", "provisional", "supported"}
_SAMPLING = {"reuse_ok", "prefer_diverse", "require_diverse"}
_CONSEQUENCE_LEVELS = {"low", "medium", "high"}
_LEGACY_POLICY_TO_SAMPLING = {
    "reuse_allowed": "reuse_ok",
    "prefer_independent": "prefer_diverse",
    "require_independent": "require_diverse",
}


def _text(value: Any, limit: int = 1000) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


def _strings(value: Any, *, limit: int, item_limit: int = 500) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = _text(item, item_limit)
        if text and text not in output:
            output.append(text)
        if len(output) >= limit:
            break
    return output


def _allowed(value: Any, choices: Iterable[str]) -> str:
    choices = [str(x) for x in choices]
    candidate = str(value or "").strip()
    return candidate if candidate in choices else (choices[0] if choices else candidate)


def _normalize_epistemic_state(raw: Any, config: ProspectiveConfig) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    bindings_raw = raw.get("bindings") or raw.get("variables") or []
    bindings: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in bindings_raw if isinstance(bindings_raw, list) else []:
        if not isinstance(item, dict):
            continue
        key = _text(item.get("key") or item.get("name"), 120)
        predicate = _text(item.get("predicate") or item.get("meaning"), 500)
        if not key or key in seen_keys:
            continue
        status = str(item.get("status") or "unresolved").strip().lower()
        if status not in _EPISTEMIC_STATUS:
            status = "unresolved"
        bindings.append({
            "key": key,
            "predicate": predicate,
            "value": _text(item.get("value") or item.get("candidate"), 500),
            "status": status,
            "evidence_refs": _strings(item.get("evidence_refs") or item.get("supporting_results"), limit=int(config.max_evidence_refs), item_limit=100),
            "open_issue": _text(item.get("open_issue") or item.get("uncertainty"), 600),
        })
        seen_keys.add(key)
        if len(bindings) >= int(config.max_epistemic_bindings):
            break
    return {"bindings": bindings}


def _normalize_prospective_state(raw: Any, config: ProspectiveConfig) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    dependencies: List[Dict[str, Any]] = []
    for item in raw.get("critical_dependencies") or raw.get("fragile_dependencies") or []:
        if not isinstance(item, dict):
            continue
        belief_key = _text(item.get("belief_key") or item.get("key"), 120)
        reason = _text(item.get("reason"), 500)
        consequence = _text(item.get("consequence_if_wrong"), 700)
        evidence_gap = _text(item.get("evidence_gap"), 600)
        legacy_claim = _text(item.get("claim"), 400)
        legacy_refs = _strings(item.get("evidence_refs"), limit=int(config.max_evidence_refs), item_limit=100)
        if not belief_key and legacy_claim:
            belief_key = "legacy_claim"
        if not belief_key and not reason and not consequence and not evidence_gap:
            continue
        dep = {"belief_key": belief_key, "reason": reason, "consequence_if_wrong": consequence, "evidence_gap": evidence_gap}
        if legacy_claim:
            dep["legacy_claim"] = legacy_claim
        if legacy_refs:
            dep["legacy_evidence_refs"] = legacy_refs
        dependencies.append(dep)
        if len(dependencies) >= int(config.max_critical_dependencies):
            break
    readiness = str(raw.get("completion_readiness") or "uncertain").strip().lower()
    if readiness not in _READINESS:
        readiness = "uncertain"
    return {
        "future_requirements": _strings(raw.get("future_requirements") or raw.get("remaining_requirements"), limit=int(config.max_future_requirements), item_limit=500),
        "critical_dependencies": dependencies,
        "desired_next_evidence_state": _text(raw.get("desired_next_evidence_state") or raw.get("predicted_progress") or raw.get("future_summary"), 900),
        "completion_readiness": readiness,
    }


def _normalize_orchestration_mode(raw: Any, action: str) -> tuple[str, List[str]]:
    candidate = str(raw or "").strip().lower()
    warnings: List[str] = []
    if candidate not in ORCHESTRATION_MODES:
        warnings.append("missing_or_invalid_orchestration_mode")
        candidate = "complete" if action == "complete" else "advance"
    if action == "complete" and candidate != "complete":
        warnings.append("complete_action_with_noncomplete_mode")
    if action == "delegate_task" and candidate == "complete":
        warnings.append("delegate_action_with_complete_mode")
    return candidate, warnings


def _normalize_consequence_assessment(
    raw: Any,
    *,
    orchestration_mode: str,
) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    effort = str(raw.get("cost_level") or raw.get("effort") or "").strip().lower()
    if effort not in _CONSEQUENCE_LEVELS:
        effort = ""
    return {
        "chosen_mode": orchestration_mode,
        "expected_gain": _text(
            raw.get("expected_gain") or raw.get("expected_progress"),
            700,
        ),
        "risk_if_skipped": _text(
            raw.get("risk_if_skipped") or raw.get("risk"),
            700,
        ),
        "cost_note": _text(
            raw.get("cost_note") or raw.get("cost"),
            500,
        ),
        "cost_level": effort,
    }


def _normalize_evidence_request(raw: Any, config: ProspectiveConfig) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    if not raw:
        return {}
    target_key = _text(raw.get("target_key") or raw.get("belief_key"), 120)
    target_requirement = _text(raw.get("target_requirement"), 900)
    based_on = _strings(raw.get("based_on") or raw.get("independent_of"), limit=int(config.max_evidence_refs), item_limit=100)
    blind = bool(raw.get("blind", False))
    sampling = str(raw.get("sampling") or "").strip().lower()
    if sampling not in _SAMPLING:
        legacy = str(raw.get("route_policy") or raw.get("policy") or "").strip().lower()
        sampling = _LEGACY_POLICY_TO_SAMPLING.get(legacy, "")
    if sampling not in _SAMPLING:
        sampling = "prefer_diverse" if based_on else "reuse_ok"
    if sampling == "reuse_ok" and not target_key and not target_requirement and not based_on and not blind:
        return {}
    return {"target_key": target_key, "target_requirement": target_requirement, "sampling": sampling, "based_on": based_on, "blind": blind}


def _normalize_tasks(raw: Any, *, config: ProspectiveConfig, selection_field: str, choices: Iterable[str], tool_names: Iterable[str]) -> List[Dict[str, Any]]:
    tool_set = {str(name) for name in tool_names}
    tasks: List[Dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        instruction = _text(item.get("task_instruction") or item.get("instruction"), 1200)
        if not instruction:
            continue
        tools = [tool for tool in _strings(item.get("tools"), limit=8, item_limit=120) if tool in tool_set]
        task = {"task_instruction": instruction, selection_field: _allowed(item.get(selection_field), choices), "tools": tools, "context": _text(item.get("context"), 2000)}
        request = _normalize_evidence_request(item.get("evidence_request"), config) or _normalize_evidence_request(item.get("evidence_constraint"), config)
        if request:
            task["evidence_request"] = request
        tasks.append(task)
        if len(tasks) >= int(config.max_tasks_per_action):
            break
    return tasks


def normalize_prospective_decision(raw: Any, *, config: ProspectiveConfig, selection_field: str, choices: Iterable[str], tool_names: Iterable[str]) -> Dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    epistemic_state = _normalize_epistemic_state(raw.get("epistemic_state") or raw.get("belief_state") or {}, config)
    prospective_state = _normalize_prospective_state(raw.get("prospective_state") or raw.get("future_state") or {}, config)
    action = str(raw.get("action") or "").strip()
    warnings: List[str] = []
    if action not in _ACTIONS:
        warnings.append("invalid_action")
        action = "orchestration_error"

    orchestration_mode, mode_warnings = _normalize_orchestration_mode(
        raw.get("orchestration_mode") or raw.get("macro_mode") or raw.get("mode"),
        action,
    )
    warnings.extend(mode_warnings)
    consequence_assessment = _normalize_consequence_assessment(
        raw.get("consequence_assessment") or raw.get("consequence") or {},
        orchestration_mode=orchestration_mode,
    )

    binding_keys = {item["key"] for item in epistemic_state["bindings"]}
    for dep in prospective_state["critical_dependencies"]:
        key = dep.get("belief_key")
        if key and key != "legacy_claim" and key not in binding_keys:
            warnings.append(f"unknown_critical_belief:{key}")
    params = deepcopy(raw.get("params") or {})
    if not isinstance(params, dict):
        params = {}
    if action == "delegate_task":
        tasks_raw = params.get("tasks") if params.get("tasks") is not None else raw.get("tasks")
        tasks = _normalize_tasks(tasks_raw, config=config, selection_field=selection_field, choices=choices, tool_names=tool_names)
        params = {"tasks": tasks}
        if not tasks:
            warnings.append("empty_delegate_batch")
            action = "orchestration_error"
            params = {}
    elif action == "complete":
        answer = str(params.get("answer") or raw.get("answer") or "").strip()
        params = {"answer": answer}
        if not answer:
            warnings.append("empty_complete_answer")
            action = "orchestration_error"
            params = {}
        if config.completion_consistency_check:
            if prospective_state["completion_readiness"] != "ready":
                warnings.append("complete_while_not_ready")
            if prospective_state["future_requirements"]:
                warnings.append("complete_with_future_requirements")
            unresolved = {dep.get("belief_key") for dep in prospective_state["critical_dependencies"] if dep.get("belief_key")}
            for binding in epistemic_state["bindings"]:
                if binding["key"] in unresolved and binding["status"] == "unresolved":
                    warnings.append(f"complete_with_unresolved_binding:{binding['key']}")
    return {
        "epistemic_state": epistemic_state,
        "prospective_state": prospective_state,
        "orchestration_mode": orchestration_mode,
        "consequence_assessment": consequence_assessment,
        "action": action,
        "params": params,
        "reasoning": _text(raw.get("reasoning"), 1200),
        "validation_warnings": list(dict.fromkeys(warnings)),
    }
