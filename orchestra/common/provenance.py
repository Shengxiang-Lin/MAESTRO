"""Trusted runtime provenance helpers shared by O1-facing telemetry."""
from __future__ import annotations
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
    kept_query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_KEYS]
    canonical = urlunsplit(("", host, path, urlencode(kept_query), "")).lstrip("//")
    return f"url:{canonical}"


def extract_source_ids(text: Any) -> List[str]:
    value = str(text or "")
    sources = {canonical_source_id(m.group(0)) for m in _URL_RE.finditer(value)}
    sources.discard("")
    for match in _MEDIA_RE.finditer(value):
        sources.add(f"media:{match.group(1).lower()}")
    return sorted(sources)


def _source_ids_from_param_value(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        for nested in value.values(): output.update(_source_ids_from_param_value(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value: output.update(_source_ids_from_param_value(nested))
    else:
        output.update(extract_source_ids(value))
    return output


def trace_control_provenance(trace: Any) -> Dict[str, List[str]]:
    executed_tools, attested_sources = set(), set()
    for step in trace if isinstance(trace, list) else []:
        if not isinstance(step, dict): continue
        action = step.get("action")
        if not isinstance(action, dict): continue
        tool = str(action.get("action") or action.get("tool") or action.get("name") or "").strip()
        if tool and tool.lower() not in {"finish", "complete"}: executed_tools.add(tool)
        params = action.get("params")
        if not isinstance(params, dict): continue
        for key, value in params.items():
            k = str(key or "").strip().lower()
            if k in _SOURCE_PARAM_KEYS or k.endswith("_url") or k.endswith("_uri") or k.endswith("_path"):
                attested_sources.update(_source_ids_from_param_value(value))
    return {"executed_tool_ids": sorted(executed_tools), "attested_source_ids": sorted(attested_sources)}
