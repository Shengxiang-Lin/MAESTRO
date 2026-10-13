from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os
import re

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig
from base.engine.cost_monitor import record_api_response, record_unmeasured_call

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_BASE_URL = "https://google.serper.dev/search"
DASHSCOPE_FALLBACK_MODEL = "qwen3.7-plus"
DASHSCOPE_DEFAULT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_EMPTY_HIT = [{"content": "No good Google Search Result was found", "source": "None"}]


def _get_dashscope_cfg() -> tuple[str, str, str]:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name = DASHSCOPE_FALLBACK_MODEL
    try:
        cfg = LLMsConfig.default().get(DASHSCOPE_FALLBACK_MODEL)
        if cfg and cfg.key:
            api_key, base_url = cfg.key, cfg.base_url
            model_name = cfg.model or model_name
    except Exception:
        pass
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
    base_url = base_url or os.getenv("DASHSCOPE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_API_BASE") or DASHSCOPE_DEFAULT_BASE
    return api_key, base_url, model_name


async def _dashscope_web_search(query: str, k: int, gl: str, hl: str, timeout_s: int = 40) -> List[Dict[str, str]]:
    import aiohttp
    api_key, base_url, model = _get_dashscope_cfg()
    if not api_key:
        raise RuntimeError("GoogleSearchAction: no DashScope/OpenAI-compatible API key available")
    endpoint = base_url.rstrip("/") + "/responses"
    payload = {
        "model": model,
        "input": f"Use web_search for: {query}\nReturn terse factual snippets. Region={gl}, language={hl}.",
        "tools": [{"type": "web_search"}],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json=payload, headers=headers) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {text[:600]}")
        obj = json.loads(text)
    except Exception:
        record_unmeasured_call(model, "search", "search_request_failed_without_usage")
        raise
    record_api_response(model, obj, category="search", base_url=base_url,
                        search_requested=True)
    if obj.get("code"):
        raise RuntimeError(f"DashScope API error: {obj.get('code')} {obj.get('message', '')}")

    sources: List[str] = []
    paragraphs: List[str] = []
    for block in obj.get("output") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "web_search_call":
            for src in (block.get("action") or {}).get("sources") or []:
                url = str(src.get("url") or "").strip() if isinstance(src, dict) else ""
                if url and url not in sources:
                    sources.append(url)
        elif block.get("type") == "message":
            for item in block.get("content") or []:
                if isinstance(item, dict) and item.get("type") in ("text", "output_text"):
                    paragraphs.extend(p.strip() for p in str(item.get("text") or "").splitlines() if len(p.strip()) >= 16)
    hits: List[Dict[str, str]] = []
    for idx, paragraph in enumerate(paragraphs[:k]):
        urls = re.findall(r"https?://[^\s)\]>\"']+", paragraph)
        source = urls[0] if urls else (sources[idx] if idx < len(sources) else "None")
        hits.append({"content": paragraph, "source": source})
    if len(hits) < k:
        used = {h["source"] for h in hits}
        for url in sources:
            if url not in used:
                hits.append({"content": f"Additional web_search source: {url}", "source": url})
                if len(hits) >= k:
                    break
    return hits or list(_EMPTY_HIT)


def _serper_parse_results(results: Dict[str, Any], k: int) -> List[Dict[str, str]]:
    snippets: List[Dict[str, str]] = []
    ans = results.get("answerBox") or {}
    if isinstance(ans, dict):
        if ans.get("answer"):
            return [{"content": str(ans["answer"]), "source": "None"}]
        if ans.get("snippet"):
            return [{"content": str(ans["snippet"]).replace("\n", " "), "source": "None"}]
    for item in (results.get("organic") or [])[:max(1, k)]:
        if item.get("snippet"):
            snippets.append({"content": str(item["snippet"]), "source": str(item.get("link") or "None")})
    return snippets or list(_EMPTY_HIT)


async def _serper_search(query: str, k: int, gl: str, hl: str) -> List[Dict[str, str]]:
    import aiohttp
    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY is empty")
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(SERPER_BASE_URL, json={"q": query, "num": k, "gl": gl, "hl": hl}, headers=headers) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}: {text[:600]}")
    record_unmeasured_call("serper", "search", "external_provider_price_not_configured")
    return _serper_parse_results(json.loads(text), k)


class GoogleSearchAction(BaseAction):
    name: str = "GoogleSearchAction"
    description: str = "Web search returning snippets and source URLs."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "k": {"type": "integer", "default": 5},
            "gl": {"type": "string", "default": "us"},
            "hl": {"type": "string", "default": "en"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def __call__(self, **kwargs) -> Any:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return {"success": False, "output": None, "error": "empty query", "metrics": {}}
        try:
            k = max(1, int(kwargs.get("k", 5)))
        except Exception:
            k = 5
        gl, hl = kwargs.get("gl", "us") or "us", kwargs.get("hl", "en") or "en"
        errors: List[str] = []
        parsed = None
        if os.getenv("GOOGLE_SEARCH_USE_SERPER", "").strip().lower() in ("1", "true", "yes", "on"):
            try:
                parsed = await _serper_search(query, k, gl, hl)
            except Exception as exc:
                errors.append(f"Serper: {exc}")
        if parsed is None:
            try:
                parsed = await _dashscope_web_search(query, k, gl, hl)
            except Exception as exc:
                errors.append(f"DashScope web_search: {exc}")
        if not parsed:
            return {"success": False, "output": None, "error": " ; ".join(errors) or "no results", "metrics": {}}
        return {"success": True, "output": parsed, "error": None, "metrics": {}}
