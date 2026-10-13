"""Persistent evidence-grounded state for MAESTRO prospective orchestration."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import re
from typing import Any, Dict, List
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_URL_RE = re.compile(r"https?://[^\s<>'\"`]+", re.IGNORECASE)
_MEDIA_RE = re.compile(r"(?:^|[\\/])([^\\/\s]+\.(?:mp3|wav|m4a|aac|flac|ogg|mp4|mov|avi|mkv|webm|jpg|jpeg|png|webp))\b", re.IGNORECASE)
_TRACKING_KEYS = {"fbclid", "gclid", "ref", "source"}
_SOURCE_PARAM_KEYS = {"url", "uri", "path", "file", "file_path", "image", "image_path", "audio", "audio_path", "video", "video_path"}


def canonical_source_id(url: str) -> str:
    value = str(url or "").strip().rstrip(".,);]}>")
    if not value:
        return ""
    try:
        parts = urlsplit(value if "://" in value else f"https://{value}")
    except ValueError:
        return ""
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""
    path = re.sub(r"/{2,}", "/", parts.path or "/").rstrip("/")
    kept_query = [(key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True) if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_KEYS]
    query = urlencode(kept_query)
    canonical = urlunsplit(("", host, path, query, "")).lstrip("//")
    return f"url:{canonical}"


def extract_source_ids(text: Any) -> List[str]:
    value = str(text or "")
    sources = {canonical_source_id(match.group(0)) for match in _URL_RE.finditer(value)}
    sources.discard("")
    for match in _MEDIA_RE.finditer(value):
        sources.add(f"media:{match.group(1).lower()}")
    return sorted(sources)


def _source_domains(source_ids: List[str]) -> List[str]:
    domains = set()
    for source_id in source_ids:
        if source_id.startswith("url:"):
            domain = source_id[4:].split("/", 1)[0].strip().lower()
            if domain:
                domains.add(domain)
    return sorted(domains)


def _compact_text(*values: Any, limit: int = 700) -> str:
    text = "\n".join(str(value or "").strip() for value in values if value)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _source_ids_from_param_value(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        for nested in value.values():
            output.update(_source_ids_from_param_value(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            output.update(_source_ids_from_param_value(nested))
    else:
        output.update(extract_source_ids(value))
    return output


def trace_control_provenance(trace: Any) -> Dict[str, List[str]]:
    executed_tools: set[str] = set()
    attested_sources: set[str] = set()
    for step in trace if isinstance(trace, list) else []:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if not isinstance(action, dict):
            continue
        tool = str(action.get("action") or action.get("tool") or action.get("name") or "").strip()
        if tool and tool.lower() not in {"finish", "complete"}:
            executed_tools.add(tool)
        params = action.get("params")
        if not isinstance(params, dict):
            continue
        for key, value in params.items():
            normalized_key = str(key or "").strip().lower()
            if normalized_key in _SOURCE_PARAM_KEYS or normalized_key.endswith("_url") or normalized_key.endswith("_uri") or normalized_key.endswith("_path"):
                attested_sources.update(_source_ids_from_param_value(value))
    return {"executed_tool_ids": sorted(executed_tools), "attested_source_ids": sorted(attested_sources)}


class ObservedStateLedger:
    """Append-only evidence history plus revisable Main-authored epistemic state."""

    def __init__(self, *, max_attempts: int, max_source_ids_per_result: int = 12, max_belief_history: int = 4, max_evidence_attempts_per_target: int = 6) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.max_source_ids_per_result = max(1, int(max_source_ids_per_result))
        self.max_belief_history = max(1, int(max_belief_history))
        self.max_evidence_attempts_per_target = max(1, int(max_evidence_attempts_per_target))
        self.attempt = 0
        self.results: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []
        self.belief_snapshots: List[Dict[str, Any]] = []
        self.evidence_attempts: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []

    def clone(self) -> "ObservedStateLedger":
        return deepcopy(self)

    def set_attempt(self, attempt: int) -> None:
        self.attempt = max(0, int(attempt))

    @staticmethod
    def _subtasks(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        batch = result.get("subtask_results")
        if isinstance(batch, list):
            return [item for item in batch if isinstance(item, dict)]
        return [result] if isinstance(result, dict) else []

    def observe_feedback(self, result: Dict[str, Any], attempt: int) -> List[str]:
        self.set_attempt(attempt)
        observed: List[str] = []
        for subtask_index, subtask in enumerate(self._subtasks(result), start=1):
            finish = subtask.get("finish_result") or {}
            if not isinstance(finish, dict):
                finish = {}
            error = subtask.get("error") or finish.get("error")
            status = "error" if error else str(finish.get("status") or ("done" if subtask.get("done") else "partial")).lower()
            result_id = f"result_{len(self.results) + 1}"
            visible_text = _compact_text(finish.get("result"), finish.get("summary"), subtask.get("trace_summary"))
            assigned_tools = subtask.get("tools_assigned") or []
            if isinstance(assigned_tools, str):
                assigned_tools = [assigned_tools]
            provenance = trace_control_provenance(subtask.get("trace") or [])
            model = str(subtask.get("model") or "unknown")
            context_fp = str(subtask.get("context_fingerprint") or "")
            if not context_fp:
                normalized = " ".join(str(subtask.get("context") or "").lower().split())
                context_fp = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12] if normalized else "empty"
            attested_source_ids = provenance["attested_source_ids"][: self.max_source_ids_per_result]
            reported_source_ids = extract_source_ids(visible_text)[: self.max_source_ids_per_result]
            source_ids = sorted(set(attested_source_ids) | set(reported_source_ids))
            source_domains = _source_domains(source_ids)
            record = {
                "result_id": result_id,
                "attempt": int(attempt),
                "subtask_index": int(subtask_index),
                "status": status,
                "task": str(subtask.get("task_instruction") or "")[:500],
                "model": model,
                "assigned_tool_ids": sorted({str(value) for value in assigned_tools if value}),
                "executed_tool_ids": provenance["executed_tool_ids"],
                "attested_source_ids": attested_source_ids,
                "reported_source_ids": reported_source_ids,
                "source_domains": source_domains,
                "context_fingerprint": context_fp,
                "visible_excerpt": visible_text,
                "usage": deepcopy(subtask.get("usage") or {}),
            }
            if error:
                record["error"] = str(error)[:500]
            self.results.append(record)
            self.events.append({"event": "result_observed", **deepcopy(record)})
            observed.append(result_id)
        return observed

    def record_decision(self, *, attempt: int, epistemic_state: Dict[str, Any], prospective_state: Dict[str, Any], action: str, params: Dict[str, Any], reasoning: str, validation_warnings: List[str]) -> Dict[str, Any]:
        belief_snapshot = {"attempt": int(attempt), "bindings": deepcopy((epistemic_state or {}).get("bindings") or [])}
        self.belief_snapshots.append(belief_snapshot)
        self.events.append({"event": "epistemic_revision", **deepcopy(belief_snapshot)})
        record = {
            "attempt": int(attempt),
            "epistemic_state": deepcopy(epistemic_state or {"bindings": []}),
            "prospective_state": deepcopy(prospective_state),
            "action": str(action),
            "params": deepcopy(params),
            "reasoning": str(reasoning or ""),
            "validation_warnings": list(validation_warnings or []),
        }
        self.decisions.append(record)
        self.events.append({"event": "decision", **deepcopy(record)})
        return deepcopy(record)

    def record_evidence_attempts(self, *, attempt: int, execution_params: Dict[str, Any], projection_reports: List[Dict[str, Any]], result_ids: List[str]) -> None:
        tasks = (execution_params or {}).get("tasks") or []
        result_by_id = {str(item.get("result_id") or ""): item for item in self.results}
        for index, task in enumerate(tasks if isinstance(tasks, list) else []):
            if not isinstance(task, dict):
                continue
            request = task.get("evidence_request") or {}
            if not isinstance(request, dict) or not request:
                continue
            result_ref = result_ids[index] if index < len(result_ids) else ""
            observed = result_by_id.get(result_ref, {})
            projection = task.get("execution_projection") or task.get("route_compilation") or {}
            record = {
                "attempt": int(attempt),
                "target_key": str(request.get("target_key") or "")[:120],
                "target_requirement": str(request.get("target_requirement") or "")[:500],
                "sampling": str(request.get("sampling") or "reuse_ok"),
                "based_on": list(request.get("based_on") or []),
                "blind": bool(request.get("blind", False)),
                "result_ref": result_ref,
                "status": str(observed.get("status") or "unknown"),
                "executed_tool_ids": list(observed.get("executed_tool_ids") or []),
                "source_domains": list(observed.get("source_domains") or []),
                "projection": deepcopy(projection),
            }
            self.evidence_attempts.append(record)
            self.events.append({"event": "evidence_attempt", **deepcopy(record)})
        for report in projection_reports or []:
            if not isinstance(report, dict) or report.get("status") != "unavailable":
                continue
            record = {
                "attempt": int(attempt),
                "target_key": str(report.get("target_key") or "")[:120],
                "target_requirement": str(report.get("target_requirement") or "")[:500],
                "sampling": str(report.get("sampling") or "require_diverse"),
                "based_on": list(report.get("based_on") or []),
                "blind": bool(report.get("blind", False)),
                "result_ref": "",
                "status": "unavailable",
                "executed_tool_ids": [],
                "source_domains": [],
                "projection": deepcopy(report),
            }
            self.evidence_attempts.append(record)
            self.events.append({"event": "evidence_attempt", **deepcopy(record)})

    def _evidence_dynamics(self, *, max_results: int) -> Dict[str, Any]:
        visible = self.results[-max(1, int(max_results)) :]
        source_to_results: Dict[str, List[str]] = defaultdict(list)
        domain_to_results: Dict[str, List[str]] = defaultdict(list)
        for item in self.results:
            result_id = str(item.get("result_id") or "")
            for source_id in item.get("attested_source_ids") or []:
                source_to_results[str(source_id)].append(result_id)
            for domain in item.get("source_domains") or []:
                domain_to_results[str(domain)].append(result_id)
        repeated_sources = [{"source_id": source_id, "result_ids": result_ids, "count": len(result_ids)} for source_id, result_ids in sorted(source_to_results.items()) if len(result_ids) > 1]
        repeated_domains = [{"domain": domain, "result_ids": result_ids, "count": len(result_ids)} for domain, result_ids in sorted(domain_to_results.items()) if len(result_ids) > 1]
        profiles = []
        for item in visible:
            source_ids = sorted(set(item.get("attested_source_ids") or []) | set(item.get("reported_source_ids") or []))
            domains = sorted(set(item.get("source_domains") or []))
            profiles.append({
                "result_id": item.get("result_id"),
                "source_count": len(source_ids),
                "source_domain_count": len(domains),
                "tool_count": len(set(item.get("executed_tool_ids") or [])),
                "model_count": 1 if item.get("model") else 0,
                "context_count": 1 if item.get("context_fingerprint") else 0,
                "source_ids": source_ids[: self.max_source_ids_per_result],
                "source_domains": domains,
                "model": item.get("model"),
                "executed_tool_ids": item.get("executed_tool_ids") or [],
            })
        all_sources = {str(source) for item in self.results for source in item.get("attested_source_ids") or []}
        all_domains = {str(domain) for item in self.results for domain in item.get("source_domains") or []}
        return {
            "episode_counts": {
                "distinct_sources": len(all_sources),
                "distinct_source_domains": len(all_domains),
                "distinct_models": len({str(item.get("model") or "unknown") for item in self.results}),
                "distinct_tools": len({str(tool) for item in self.results for tool in item.get("executed_tool_ids") or []}),
                "distinct_contexts": len({str(item.get("context_fingerprint") or "unknown") for item in self.results}),
            },
            "result_profiles": profiles,
            "repeated_sources": repeated_sources[-12:],
            "repeated_source_domains": repeated_domains[-12:],
        }

    def _evidence_attempt_history(self) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        labels: Dict[str, Dict[str, str]] = {}
        for item in self.evidence_attempts:
            target_key = str(item.get("target_key") or "").strip()
            requirement = str(item.get("target_requirement") or "").strip()
            group_key = target_key or requirement
            if not group_key:
                continue
            grouped[group_key].append(item)
            labels[group_key] = {"target_key": target_key, "target_requirement": requirement}
        output = []
        for group_key, attempts in grouped.items():
            output.append({**labels[group_key], "attempt_count": len(attempts), "attempts": deepcopy(attempts[-self.max_evidence_attempts_per_target :])})
        return output[-12:]

    def _coverage_summary(self) -> Dict[str, Any]:
        models = {str(item.get("model") or "unknown") for item in self.results}
        tools = {str(tool) for item in self.results for tool in item.get("executed_tool_ids") or []}
        sources = {str(source) for item in self.results for source in item.get("attested_source_ids") or []}
        contexts = {str(item.get("context_fingerprint") or "unknown") for item in self.results}
        return {"distinct_models": len(models), "distinct_executed_tools": len(tools), "distinct_attested_sources": len(sources), "distinct_contexts": len(contexts)}

    def prompt_snapshot(self, max_chars: int = 9000, *, max_results: int = 12) -> Dict[str, Any]:
        current_epistemic = {"bindings": deepcopy(self.belief_snapshots[-1].get("bindings") or [])} if self.belief_snapshots else {"bindings": []}
        state: Dict[str, Any] = {
            "budget": {"attempt": self.attempt, "max_attempts": self.max_attempts, "remaining_attempts": max(0, self.max_attempts - self.attempt)},
            "current_epistemic_state": current_epistemic,
            "belief_revision_history": deepcopy(self.belief_snapshots[-self.max_belief_history :]),
            "evidence_attempt_history": self._evidence_attempt_history(),
            "coverage_summary": self._coverage_summary(),
            "evidence_dynamics": self._evidence_dynamics(max_results=max_results),
            "result_index": deepcopy(self.results[-max(1, int(max_results)) :]),
            "status_counts": dict(Counter(item["status"] for item in self.results)),
            "last_decision": deepcopy(self.decisions[-1]) if self.decisions else {},
        }
        limit = max(1000, int(max_chars))
        while len(json.dumps(state, ensure_ascii=False)) > limit and state["result_index"]:
            state["result_index"].pop(0)
        while len(json.dumps(state, ensure_ascii=False)) > limit and len(state["belief_revision_history"]) > 1:
            state["belief_revision_history"].pop(0)
        while len(json.dumps(state, ensure_ascii=False)) > limit and state["evidence_attempt_history"]:
            state["evidence_attempt_history"].pop(0)
        if len(json.dumps(state, ensure_ascii=False)) > limit:
            state["evidence_dynamics"]["result_profiles"] = []
        if len(json.dumps(state, ensure_ascii=False)) > limit:
            state["evidence_dynamics"]["repeated_sources"] = []
            state["evidence_dynamics"]["repeated_source_domains"] = []
        if len(json.dumps(state, ensure_ascii=False)) > limit:
            state["last_decision"] = {}
        return state

    def audit_trace(self) -> Dict[str, Any]:
        return {
            "version": "epistemic_projection_v4",
            "results": deepcopy(self.results),
            "belief_snapshots": deepcopy(self.belief_snapshots),
            "evidence_attempts": deepcopy(self.evidence_attempts),
            "decisions": deepcopy(self.decisions),
            "events": deepcopy(self.events),
        }
