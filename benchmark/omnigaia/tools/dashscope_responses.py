"""Receive provider usage before returning tools; recover stored responses by ID."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime
from urllib.parse import quote

import requests

from base.engine.cost_monitor import (
    get_current_monitor, record_api_response, record_unmeasured_call,
)

_audit_lock = threading.Lock()
_TERMINAL = {"completed", "incomplete", "failed", "cancelled"}


def _audit(event: dict) -> None:
    monitor = get_current_monitor()
    if monitor is None or not monitor.journal_path:
        return
    path = monitor.journal_path.with_name(
        monitor.journal_path.name.replace("api_usage_", "api_requests_", 1)
    )
    if path == monitor.journal_path:
        path = path.with_name(path.stem + "_requests.jsonl")
    try:
        with _audit_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "timestamp": datetime.now().isoformat(), "task_id": monitor.task_id,
                    **event,
                }, ensure_ascii=False) + "\n")
    except OSError as exc:
        logging.getLogger(__name__).warning("Could not persist request audit: %s", exc)


def _sse_events(lines):
    data, event_type = [], ""
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line:
            if data:
                payload = "\n".join(data)
                if payload != "[DONE]":
                    obj = json.loads(payload)
                    if event_type and "type" not in obj:
                        obj["type"] = event_type
                    yield obj
            data, event_type = [], ""
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif line.startswith("event:"):
            event_type = line[6:].strip()
    if data and "\n".join(data) != "[DONE]":
        obj = json.loads("\n".join(data))
        if event_type and "type" not in obj:
            obj["type"] = event_type
        yield obj


def _close(response) -> None:
    if response is not None:
        try:
            response.close()
        except Exception:
            pass


def request_response(*, api_key: str, base_url: str, model: str, prompt: str,
                     category: str, idle_timeout: float = 90,
                     total_timeout: float = 300, recovery_attempts: int = 3) -> dict:
    """One generation POST; retrieval retries never submit another generation.

    A dropped stream without a final usage report remains unmeasured. Observed
    tool events are diagnostics, not invented token usage or billing counts.
    """
    endpoint = base_url.rstrip("/") + "/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": prompt, "tools": [{"type": "web_search"}],
               "stream": True, "store": True}
    started = time.monotonic()
    metadata = {
        "client_request_id": uuid.uuid4().hex, "model": model, "category": category,
        "input_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "response_id": None, "provider_request_id": None,
        "observed_completed_searches": 0, "transport": "responses_stream",
    }
    _audit({**metadata, "phase": "started"})

    def bounded_lines(lines):
        for line in lines:
            if time.monotonic() - started > total_timeout:
                raise TimeoutError("Response exceeded total time limit")
            yield line
    obj = None
    http = None
    items, completed_searches = {}, set()
    terminal_received = False
    status_code = None
    transport_error = None
    try:
        http = requests.post(endpoint, headers=headers, json=payload, stream=True,
                             timeout=(10, idle_timeout))
        status_code = http.status_code
        metadata["http_status"] = status_code
        metadata["provider_request_id"] = http.headers.get("x-request-id") or http.headers.get("x-dashscope-request-id")
        content_type = http.headers.get("content-type", "").lower()
        if "text/event-stream" not in content_type:
            obj = http.json()
            if not isinstance(obj, dict):
                raise RuntimeError("Provider response is not a JSON object")
            metadata["response_id"] = obj.get("id")
            terminal_received = obj.get("status") not in ("in_progress", "queued")
        else:
            if status_code >= 400:
                raise RuntimeError(f"Provider HTTP {status_code}")
            http.encoding = "utf-8"
            for event in _sse_events(bounded_lines(http.iter_lines(chunk_size=1, decode_unicode=True))):
                response = event.get("response")
                if isinstance(response, dict) and response.get("id") and not metadata["response_id"]:
                    metadata["response_id"] = response["id"]
                    _audit({**metadata, "phase": "response_created"})
                item = event.get("item")
                kind = event.get("type", "")
                if isinstance(item, dict) and kind == "response.output_item.done":
                    index = event.get("output_index", len(items))
                    items[index] = item
                    if item.get("type") == "web_search_call" and item.get("status") == "completed":
                        completed_searches.add(item.get("id") or str(index))
                elif kind == "response.web_search_call.completed":
                    completed_searches.add(event.get("item_id") or str(event.get("output_index")))
                observed = len(completed_searches)
                if observed != metadata["observed_completed_searches"]:
                    metadata["observed_completed_searches"] = observed
                    _audit({**metadata, "phase": "search_progress"})
                if kind in ("response.completed", "response.incomplete", "response.failed"):
                    if not isinstance(response, dict):
                        raise RuntimeError("Terminal stream event has no response object")
                    obj = dict(response)
                    if not obj.get("output") and items:
                        obj["output"] = [items[k] for k in sorted(items)]
                    terminal_received = True
                    break
            if not terminal_received:
                raise RuntimeError("Response stream ended before terminal usage")
    except Exception as exc:
        transport_error = exc
        metadata["error_type"] = type(exc).__name__
        metadata["error_message"] = str(exc).replace(api_key, "[redacted]")[:300]
        _audit({**metadata, "phase": "transport_failed"})
    finally:
        _close(http)

    # GET reads the stored response. It must not repeat a paid POST on timeout.
    if not terminal_received and metadata["response_id"]:
        for attempt in range(recovery_attempts):
            retrieved = None
            try:
                retrieved = requests.get(endpoint + "/" + quote(metadata["response_id"], safe=""),
                                         headers=headers, timeout=(5, 10))
                if retrieved.status_code == 200:
                    candidate = retrieved.json()
                    if isinstance(candidate, dict) and candidate.get("status") in _TERMINAL:
                        obj, terminal_received = candidate, True
                        metadata["recovered_by_response_id"] = True
                        _audit({**metadata, "phase": "response_recovered"})
                        break
                _audit({**metadata, "phase": "retrieval_pending", "retrieval_attempt": attempt + 1,
                        "retrieval_http_status": retrieved.status_code})
            except Exception as exc:
                _audit({**metadata, "phase": "retrieval_failed", "error_type": type(exc).__name__})
            finally:
                _close(retrieved)
            if attempt + 1 < recovery_attempts:
                time.sleep(2)

    metadata["elapsed_seconds"] = round(time.monotonic() - started, 3)
    if terminal_received and isinstance(obj, dict):
        metadata["response_status"] = obj.get("status")
        metadata["response_id"] = obj.get("id") or metadata["response_id"]
        if obj.get("usage") is not None:
            estimate = record_api_response(model, obj, category=category, base_url=base_url,
                                           search_requested=True, request_details=metadata)
            _audit({**metadata, "phase": "usage_recorded", "coverage_complete": estimate["coverage_complete"]})
        else:
            record_unmeasured_call(model, category, category + "_request_failed_without_usage",
                                   request_details=metadata)
            _audit({**metadata, "phase": "usage_unavailable"})
        if ((status_code and status_code >= 400) or obj.get("code") or obj.get("error")
                or obj.get("status") in {"failed", "cancelled"}):
            raise RuntimeError(f"Provider response failed; HTTP={status_code}, response_id={metadata['response_id']}")
        return obj

    record_unmeasured_call(model, category, category + "_request_failed_without_usage",
                           request_details=metadata)
    _audit({**metadata, "phase": "usage_unavailable"})
    raise RuntimeError(f"Response usage unavailable; response_id={metadata['response_id']}, "
                       f"error={type(transport_error).__name__}") from transport_error
