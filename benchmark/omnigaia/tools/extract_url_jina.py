from __future__ import annotations

from typing import Any, Dict, Optional
import asyncio
import os
import requests
from openai import AsyncOpenAI

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig
from base.engine.logs import logger
from base.engine.cost_monitor import record_api_response, record_unmeasured_call
from benchmark.omnigaia.tools.dashscope_responses import request_response


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class ExtractUrlContentAction(BaseAction):
    name: str = "ExtractUrlContentAction"
    description: str = (
        "Fetch/extract facts from a web URL. Uses DashScope web_search by default; "
        "Jina Reader is optional."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "browse_query": {"type": "string"},
        },
        "required": ["url", "browse_query"],
        "additionalProperties": False,
    }
    provider: Optional[str] = None
    model_name: str = "qwen3.7-flash"

    def _get_llm_config(self, model_name: Optional[str] = None) -> tuple:
        model_name = model_name or self.model_name
        try:
            cfg = LLMsConfig.default().get(model_name)
            if cfg:
                return cfg.key, cfg.base_url, cfg.model
        except Exception:
            pass
        return (
            os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"),
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_API_BASE")
            or "https://api.openai.com/v1",
            model_name,
        )

    def _get_dashscope_cfg(self) -> tuple[str, str, str]:
        """Return provider-local DashScope credentials and endpoint.

        provider='dashscope' is intentionally isolated from the shared model
        registry so a Paratera model alias cannot replace the DashScope base URL
        or credentials used by the Responses web_search API.
        """
        return (
            os.getenv("DASHSCOPE_API_KEY") or "",
            os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            self.model_name,
        )

    def _dashscope_fallback_direct(
        self,
        url: str,
        browse_query: str,
        timeout: int = 70,
    ) -> Optional[tuple[str, str]]:
        api_key, base_url, model = self._get_dashscope_cfg()
        if not api_key:
            return None
        prompt = (
            "You MUST use your built-in `web_search` tool to retrieve up-to-date "
            f"information from this specific URL (or pages linked from it): {url}\n\n"
            "Goal: answer the following question strictly based on facts found "
            f"at / about that URL.\n\n"
            f"Question: {browse_query}\n\n"
            "Rules:\n"
            "  - Do NOT guess or hallucinate; only cite what web_search returns.\n"
            "  - Give a concise, self-contained answer first (numbers should be "
            "digits, no prose around them when the question asks for an integer).\n"
            "  - Then, on a NEW LINE starting with exactly '===SOURCES===', paste "
            "the 3-6 most informative snippets returned by web_search (include "
            "year, counts, page titles) so the caller can verify your work.\n"
        )
        try:
            obj = request_response(
                api_key=api_key,
                base_url=base_url,
                model=model,
                prompt=prompt,
                category="browse",
                idle_timeout=timeout,
            )
        except Exception:
            return None
        texts, web_search_actions = [], []
        for item in obj.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for content in item.get("content") or []:
                    if (
                        isinstance(content, dict)
                        and content.get("type") in ("text", "output_text")
                        and content.get("text")
                    ):
                        texts.append(str(content["text"]))
            elif item.get("type") == "web_search_call":
                action = item.get("action") or {}
                for key in ("query", "queries"):
                    if action.get(key):
                        web_search_actions.append(f"query: {action[key]}")
                for source in action.get("sources") or []:
                    if isinstance(source, dict) and source.get("url"):
                        web_search_actions.append(f"src: {source['url']}")
        full_text = "\n".join(texts).strip()
        if not full_text:
            return None
        if "===SOURCES===" in full_text:
            answer, preview = full_text.split("===SOURCES===", 1)
            answer, preview = answer.strip(), preview.strip()
        else:
            words = full_text.split()
            answer = " ".join(words[:200]).strip()
            preview_rest = " ".join(words[200:]).strip()
            preview = (
                "(no ===SOURCES=== marker; raw answer tail + queries/sources)\n"
                + (preview_rest + "\n" if preview_rest else "")
                + "\n".join(web_search_actions)
            ).strip()
        if not answer:
            return None
        return answer, preview[:2500]

    async def _jina_direct(
        self,
        url: str,
        browse_query: str,
    ) -> Optional[Dict[str, Any]]:
        jina_key = os.getenv("JINA_API_KEY")
        if not jina_key:
            return None
        try:
            response = await asyncio.to_thread(
                requests.get,
                f"https://r.jina.ai/{url}",
                headers={"Authorization": f"Bearer {jina_key}"},
                timeout=10,
            )
            text = response.text
            record_unmeasured_call(
                "jina-reader",
                "browse",
                "external_provider_price_not_configured",
            )
            if not text.strip():
                return None
            api_key, base_url, model = self._get_llm_config()
            if not api_key:
                return None
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Source:\n{text[:120000]}\n\n"
                            f"Question: {browse_query}"
                        ),
                    }
                ],
            )
            record_api_response(
                model,
                completion,
                category="browse",
                base_url=base_url,
            )
            answer = completion.choices[0].message.content
            if not answer:
                return None
            return {
                "success": True,
                "output": {
                    "url": url,
                    "answer": answer.strip(),
                    "source_preview": text[:2000],
                },
                "error": None,
                "metrics": {},
            }
        except Exception as exc:
            logger.warning(f"[ExtractUrl] Jina path failed: {exc}")
            return None

    async def _dashscope_direct(
        self,
        url: str,
        browse_query: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            result = await asyncio.to_thread(
                self._dashscope_fallback_direct,
                url,
                browse_query,
            )
        except Exception as exc:
            result = None
            logger.warning(f"[ExtractUrl] DashScope fallback failed: {exc}")
        if result is None:
            return None
        answer, preview = result
        preview_line = f"(DashScope web_search direct fallback) {preview[:2000]}"
        return {
            "success": True,
            "output": {
                "url": url,
                "answer": answer,
                "source_preview": preview_line,
            },
            "error": None,
            "metrics": {},
        }

    async def __call__(self, **kwargs) -> Any:
        url = str(kwargs.get("url") or "").strip()
        browse_query = str(kwargs.get("browse_query") or "").strip()
        if not url or not browse_query:
            return {
                "success": False,
                "output": None,
                "error": "url and browse_query are required",
                "metrics": {},
            }
        if not url.startswith(("http://", "https://")):
            return {
                "success": False,
                "output": None,
                "error": "only http/https URLs are supported",
                "metrics": {},
            }

        provider = str(self.provider or "").strip().lower()
        if provider:
            if provider == "jina":
                result = await self._jina_direct(url, browse_query)
            elif provider == "dashscope":
                result = await self._dashscope_direct(url, browse_query)
            else:
                return {
                    "success": False,
                    "output": None,
                    "error": f"unsupported browse provider: {provider}",
                    "metrics": {},
                }
            if result is not None:
                return result
            return {
                "success": False,
                "output": None,
                "error": f"{provider} browse failed",
                "metrics": {},
            }

        # Backward compatibility for direct construction without YAML provider:
        # preserve the old Jina-first-then-DashScope behavior.
        if _env_enabled("EXTRACT_URL_USE_JINA"):
            result = await self._jina_direct(url, browse_query)
            if result is not None:
                return result
        result = await self._dashscope_direct(url, browse_query)
        if result is not None:
            return result
        return {
            "success": False,
            "output": None,
            "error": "failed to extract targeted URL content",
            "metrics": {},
        }
