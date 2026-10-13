from __future__ import annotations

from typing import Any, Dict, Optional
import base64
import os
from pathlib import Path
from openai import AsyncOpenAI

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig


class ImageAnalysisAction(BaseAction):
    name: str = "ImageAnalysisAction"
    description: str = "Analyze a local image or image URL with a vision-capable model."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "image_path": {"type": "string"},
        },
        "required": ["query", "image_path"],
        "additionalProperties": False,
    }
    DEFAULT_MODEL: str = "claude-4-5-sonnet"

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
            os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL") or "https://newapi.deepwisdom.ai/v1",
            model_name,
        )

    async def __call__(self, **kwargs) -> Any:
        query = kwargs.get("query")
        image_path = kwargs.get("image_path")
        if not query or not image_path:
            return {"success": False, "output": None, "error": "Both query and image_path are required.", "metrics": {}}
        api_key, base_url, model = self._get_llm_config()
        if not api_key:
            return {"success": False, "output": None, "error": "No API key configured", "metrics": {}}
        try:
            if str(image_path).startswith(("http://", "https://")):
                image_url = str(image_path)
            else:
                path = Path(image_path).expanduser()
                data = base64.b64encode(path.read_bytes()).decode("utf-8")
                ext = path.suffix.lower()
                mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/webp" if ext == ".webp" else "image/gif" if ext == ".gif" else "image/png"
                image_url = f"data:{mime};base64,{data}"
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            completion = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": str(query)},
                ]}],
            )
            content = completion.choices[0].message.content if completion.choices else None
        except Exception as exc:
            return {"success": False, "output": None, "error": f"Image analysis failed: {exc}", "metrics": {}}
        if not content:
            return {"success": False, "output": None, "error": "Model returned empty response", "metrics": {}}
        return {"success": True, "output": content.strip(), "error": None, "metrics": {}}
