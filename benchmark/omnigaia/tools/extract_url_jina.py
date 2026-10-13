from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional
import asyncio
import os
import requests
from openai import AsyncOpenAI

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig
from base.engine.logs import logger


class ExtractUrlContentAction(BaseAction):
    name: str = "ExtractUrlContentAction"
    description: str = "Fetch/extract facts from a web URL. Uses DashScope web_search by default; Jina Reader is optional."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {"url": {"type": "string"}, "browse_query": {"type": "string"}},
        "required": ["url", "browse_query"],
        "additionalProperties": False,
    }
    DEFAULT_MODEL: ClassVar[str] = "qwen3.7-plus"
    FALLBACK_MODEL: ClassVar[str] = "qwen3.7-plus"

    def _get_llm_config(self, model_name: Optional[str] = None) -> tuple:
        model_name = model_name or self.DEFAULT_MODEL
        try:
            cfg = LLMsConfig.default().get(model_name)
            if cfg:
                return cfg.key, cfg.base_url, cfg.model
        except Exception:
            pass
        return (
            os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
            os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_API_BASE") or "https://api.openai.com/v1",
            model_name,
        )

    def _get_dashscope_cfg(self) -> tuple[str, str]:
        try:
            cfg = LLMsConfig.default().get(self.FALLBACK_MODEL)
            if cfg and cfg.key:
                return cfg.key, cfg.base_url
        except Exception:
            pass
        return (
            os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or "",
            os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def _dashscope_fallback_direct(self, url: str, browse_query: str, timeout: int = 70) -> Optional[tuple[str, str]]:
        api_key, base_url = self._get_dashscope_cfg()
        if not api_key:
            return None
        endpoint = base_url.rstrip("/") + "/responses"
        prompt = (
            f"Use web_search to retrieve facts from or about this URL: {url}\n"
            f"Question: {browse_query}\n"
            "Answer concisely and only from retrieved evidence."
        )
        payload = {"model": self.FALLBACK_MODEL, "input": prompt, "tools": [{"type": "web_search"}]}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            r = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            obj = r.json()
        except Exception:
            return None
        if r.status_code != 200 or obj.get("code"):
            return None
        texts, sources = [], []
        for item in obj.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for c in item.get("content") or []:
                    if isinstance(c, dict) and c.get("type") in ("text", "output_text") and c.get("text"):
                        texts.append(str(c["text"]))
            elif item.get("type") == "web_search_call":
                for src in (item.get("action") or {}).get("sources") or []:
                    if isinstance(src, dict) and src.get("url"):
                        sources.append(str(src["url"]))
        answer = "\n".join(texts).strip()
        if not answer:
            return None
        return answer, "\n".join(sources[:8])

    async def __call__(self, **kwargs) -> Any:
        url = str(kwargs.get("url") or "").strip()
        browse_query = str(kwargs.get("browse_query") or "").strip()
        if not url or not browse_query:
            return {"success": False, "output": None, "error": "url and browse_query are required", "metrics": {}}
        if not url.startswith(("http://", "https://")):
            return {"success": False, "output": None, "error": "only http/https URLs are supported", "metrics": {}}

        if os.getenv("EXTRACT_URL_USE_JINA", "").strip().lower() in ("1", "true", "yes", "on"):
            jina_key = os.getenv("JINA_API_KEY")
            if jina_key:
                try:
                    text = await asyncio.to_thread(
                        lambda: requests.get(
                            f"https://r.jina.ai/{url}",
                            headers={"Authorization": f"Bearer {jina_key}"},
                            timeout=10,
                        ).text
                    )
                    if text.strip():
                        api_key, base_url, model = self._get_llm_config()
                        if api_key:
                            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                            completion = await client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": f"Source:\n{text[:120000]}\n\nQuestion: {browse_query}"}],
                            )
                            answer = completion.choices[0].message.content
                            if answer:
                                return {"success": True, "output": {"url": url, "answer": answer.strip(), "source_preview": text[:2000]}, "error": None, "metrics": {}}
                except Exception as exc:
                    logger.warning(f"[ExtractUrl] Jina path failed: {exc}")

        try:
            result = await asyncio.to_thread(self._dashscope_fallback_direct, url, browse_query)
        except Exception as exc:
            result = None
            logger.warning(f"[ExtractUrl] DashScope fallback failed: {exc}")
        if result is None:
            return {"success": False, "output": None, "error": "failed to extract targeted URL content", "metrics": {}}
        answer, preview = result
        return {"success": True, "output": {"url": url, "answer": answer, "source_preview": preview}, "error": None, "metrics": {}}
