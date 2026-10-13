from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig
from base.engine.logs import logger
from base.engine.cost_monitor import record_api_response, record_unmeasured_call

MAX_SEGMENT_SECONDS = 30
QWEN_OMNI_NATIVE_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
QWEN_OMNI_AUDIO_EXTS = {"mp3", "wav", "m4a", "aac", "flac", "ogg", "opus", "amr", "webm"}


def get_wav_duration(wav_data: bytes) -> float:
    try:
        import wave
        import io
        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            rate = wf.getframerate()
            return wf.getnframes() / rate if rate else 0.0
    except Exception:
        return 0.0


def split_wav_segments(wav_data: bytes, segment_seconds: float = 30) -> list[bytes]:
    # Kept for compatibility with VideoAnalysisAction; current native endpoint
    # can generally process the extracted clip directly.
    return [wav_data]


class ParseAudioAction(BaseAction):
    name: str = "ParseAudioAction"
    description: str = "Audio understanding via DashScope Qwen Omni native multimodal endpoint."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "audio_path": {"type": "string"},
        },
        "required": ["query", "audio_path"],
        "additionalProperties": False,
    }
    model_name: str = "qwen3.5-omni-flash"

    def _get_credentials(self) -> tuple[str, str]:
        try:
            cfg = LLMsConfig.default().get(self.model_name)
            if cfg and cfg.key:
                return cfg.key, cfg.model
        except Exception:
            pass
        return (
            os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            self.model_name,
        )

    async def _native_multimodal_call(self, api_key: str, model: str, audio_path: str, query: str, timeout: int = 240) -> Optional[str]:
        ext = Path(audio_path).suffix.lower().lstrip(".")
        mime = f"audio/{ext}" if ext in QWEN_OMNI_AUDIO_EXTS else "audio/mp3"
        with open(audio_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "model": model,
            "input": {"messages": [{"role": "user", "content": [
                {"audio": f"data:{mime};base64,{b64}"},
                {"text": query},
            ]}]},
            "parameters": {"temperature": 0.1, "top_p": 0.9, "enable_thinking": False},
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as sess:
            async with sess.post(QWEN_OMNI_NATIVE_ENDPOINT, headers=headers, json=payload) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    record_unmeasured_call(model, "audio", "audio_http_error_without_usage")
                    logger.error(f"[ParseAudio] qwen-omni HTTP {resp.status}: {text[:500]}")
                    return None
        obj = json.loads(text)
        record_api_response(model, obj, category="audio", base_url=QWEN_OMNI_NATIVE_ENDPOINT,
                            audio_input=True)
        texts: List[str] = []
        for choice in (obj.get("output") or {}).get("choices") or []:
            for item in ((choice.get("message") or {}).get("content") or []):
                if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                    texts.append(item["text"].strip())
        if not texts:
            output = obj.get("output") or {}
            if isinstance(output, dict) and isinstance(output.get("text"), str) and output["text"].strip():
                texts.append(output["text"].strip())
        return "\n".join(texts) if texts else None

    async def __call__(self, **kwargs) -> Any:
        query = kwargs.get("query")
        audio_path = kwargs.get("audio_path")
        if not query or not audio_path:
            return {"success": False, "output": None, "error": "Both query and audio_path are required.", "metrics": {}}
        path = Path(audio_path).expanduser()
        if not path.is_file():
            return {"success": False, "output": None, "error": f"audio_path not found: {audio_path!r}", "metrics": {}}
        api_key, model = self._get_credentials()
        if not api_key:
            return {"success": False, "output": None, "error": "No DashScope API key available.", "metrics": {}}
        try:
            answer = await self._native_multimodal_call(api_key, model, str(path), str(query))
        except Exception as exc:
            record_unmeasured_call(model, "audio", "audio_call_failed_without_complete_usage")
            return {"success": False, "output": None, "error": f"qwen-omni call failed: {exc}", "metrics": {"stage": "call", "model": model}}
        if not answer:
            return {"success": False, "output": None, "error": "qwen-omni returned empty answer.", "metrics": {"stage": "empty", "model": model}}
        return {"success": True, "output": answer, "error": None, "metrics": {"stage": "ok", "model": model}}
