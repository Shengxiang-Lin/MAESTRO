from __future__ import annotations

from typing import Any, Dict, List, Optional
import asyncio
import json
import os

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig
from base.engine.cost_monitor import record_unmeasured_call
from benchmark.omnigaia.tools.dashscope_responses import request_response
from benchmark.omnigaia.tools.search_response import parse_search_response

SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
SERPER_BASE_URL = "https://google.serper.dev/search"
DEFAULT_SEARCH_MODEL = "qwen3.7-flash"
DASHSCOPE_DEFAULT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_EMPTY_HIT = [{"content": "No good Google Search Result was found", "source": "None"}]


def _get_dashscope_cfg(model_name: str) -> tuple[str, str, str]:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    runtime_model = model_name
    try:
        cfg = LLMsConfig.default().get(model_name)
        if cfg and cfg.key:
            api_key, base_url = cfg.key, cfg.base_url
            runtime_model = cfg.model or runtime_model
    except Exception:
        pass
    api_key = (
        api_key
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
        or ""
    )
    base_url = (
        base_url
        or os.getenv("DASHSCOPE_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_API_BASE")
        or DASHSCOPE_DEFAULT_BASE
    )
    return api_key, base_url, runtime_model


async def _dashscope_web_search(
    query: str,
    k: int,
    gl: str,
    hl: str,
    model_name: str = DEFAULT_SEARCH_MODEL,
    timeout_s: int = 40,
) -> List[Dict[str, str]]:
    api_key, base_url, model = _get_dashscope_cfg(model_name)
    if not api_key:
        raise RuntimeError(
            "GoogleSearchAction: no DashScope/OpenAI-compatible API key available"
        )
    prompt = (
        f"Use your built-in web_search tool to perform a WEB SEARCH for:\n"
        f"  query = {query}\n"
        f"  desired hits (k) = {k}, region gl={gl}, language hl={hl}\n\n"
        "Rules:\n"
        "  - ONLY cite facts returned by the web_search tool — no guessing.\n"
        "  - Keep snippets terse, factual, and self-contained.\n"
        "  - If web_search returns zero results, just say so honestly.\n"
    )
    obj = await asyncio.to_thread(
        request_response,
        api_key=api_key,
        base_url=base_url,
        model=model,
        prompt=prompt,
        category="search",
        idle_timeout=timeout_s,
    )
    return parse_search_response(obj)


def _serper_parse_results(results: Dict[str, Any], k: int) -> List[Dict[str, str]]:
    snippets: List[Dict[str, str]] = []
    ans = results.get("answerBox") or {}
    if isinstance(ans, dict):
        if ans.get("answer"):
            return [{"content": str(ans["answer"]), "source": "None"}]
        if ans.get("snippet"):
            return [
                {
                    "content": str(ans["snippet"]).replace("\n", " "),
                    "source": "None",
                }
            ]
    for item in (results.get("organic") or [])[: max(1, k)]:
        if item.get("snippet"):
            snippets.append(
                {
                    "content": str(item["snippet"]),
                    "source": str(item.get("link") or "None"),
                }
            )
    return snippets or list(_EMPTY_HIT)


async def _serper_search(
    query: str,
    k: int,
    gl: str,
    hl: str,
) -> List[Dict[str, str]]:
    import aiohttp

    if not SERPER_API_KEY:
        raise RuntimeError("SERPER_API_KEY is empty")
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            SERPER_BASE_URL,
            json={"q": query, "num": k, "gl": gl, "hl": hl},
            headers=headers,
        ) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}: {text[:600]}")
    record_unmeasured_call(
        "serper",
        "search",
        "external_provider_price_not_configured",
    )
    return _serper_parse_results(json.loads(text), k)


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


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
    provider: Optional[str] = None
    model_name: str = DEFAULT_SEARCH_MODEL

    async def __call__(self, **kwargs) -> Any:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return {
                "success": False,
                "output": None,
                "error": "empty query",
                "metrics": {},
            }
        try:
            k = max(1, int(kwargs.get("k", 5)))
        except Exception:
            k = 5
        gl = kwargs.get("gl", "us") or "us"
        hl = kwargs.get("hl", "en") or "en"

        provider = str(self.provider or "").strip().lower()
        if provider:
            try:
                if provider == "serper":
                    parsed = await _serper_search(query, k, gl, hl)
                elif provider == "dashscope":
                    parsed = await _dashscope_web_search(
                        query,
                        k,
                        gl,
                        hl,
                        self.model_name,
                    )
                else:
                    raise ValueError(f"Unsupported search provider: {provider}")
            except Exception as exc:
                return {
                    "success": False,
                    "output": None,
                    "error": f"{provider}: {exc}",
                    "metrics": {},
                }
            return {
                "success": True,
                "output": parsed,
                "error": None,
                "metrics": {},
            }

        # Backward compatibility for direct construction without YAML provider.
        errors: List[str] = []
        parsed = None
        if _env_enabled("GOOGLE_SEARCH_USE_SERPER"):
            try:
                parsed = await _serper_search(query, k, gl, hl)
            except Exception as exc:
                errors.append(f"Serper: {exc}")
        if parsed is None:
            try:
                parsed = await _dashscope_web_search(
                    query,
                    k,
                    gl,
                    hl,
                    self.model_name,
                )
            except Exception as exc:
                errors.append(f"DashScope web_search: {exc}")
        if not parsed:
            return {
                "success": False,
                "output": None,
                "error": " ; ".join(errors) or "no results",
                "metrics": {},
            }
        return {
            "success": True,
            "output": parsed,
            "error": None,
            "metrics": {},
        }
