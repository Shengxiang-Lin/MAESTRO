from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from openai import AsyncOpenAI

from base.agent.base_action import BaseAction
from base.engine.async_llm import LLMsConfig, llm_retry
from base.engine.logs import logger
from benchmark.omnigaia.tools.audio_analysis import QWEN_OMNI_NATIVE_ENDPOINT, QWEN_OMNI_AUDIO_EXTS

_FFPROBE_PATH_CACHE: Optional[str] = None
_FFMPEG_PATH_CACHE: Optional[str] = None


class VideoAnalysisAction(BaseAction):
    name: str = "VideoAnalysisAction"
    description: str = "Analyze video frames and, when present, the audio track; combine both into an answer."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "video_path": {"type": "string"},
            "max_frames": {"type": "integer", "default": 8},
            "start_time": {"type": "number", "default": 0},
            "end_time": {"type": "number"},
            "analyze_audio": {"type": "boolean", "default": True},
        },
        "required": ["query", "video_path"],
        "additionalProperties": False,
    }
    VISION_MODEL: str = "gpt-5"
    AUDIO_MODEL: str = "gpt-4o-audio-preview"
    SUMMARY_MODEL: str = "gpt-5"

    @staticmethod
    def _find_binary(name: str) -> Optional[str]:
        found = shutil.which(name)
        if found:
            return found
        candidate = Path(sys.executable).resolve().parent / name
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return str(candidate)
        for directory in ("/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"):
            candidate = Path(directory) / name
            if candidate.exists() and os.access(str(candidate), os.X_OK):
                return str(candidate)
        return None

    @staticmethod
    def _get_ffprobe() -> Optional[str]:
        global _FFPROBE_PATH_CACHE
        if _FFPROBE_PATH_CACHE is None:
            _FFPROBE_PATH_CACHE = VideoAnalysisAction._find_binary("ffprobe") or ""
        return _FFPROBE_PATH_CACHE or None

    @staticmethod
    def _get_ffmpeg() -> Optional[str]:
        global _FFMPEG_PATH_CACHE
        if _FFMPEG_PATH_CACHE is None:
            _FFMPEG_PATH_CACHE = VideoAnalysisAction._find_binary("ffmpeg") or ""
        return _FFMPEG_PATH_CACHE or None

    def _get_llm_config(self, model_name: Optional[str] = None) -> tuple:
        model_name = model_name or self.VISION_MODEL
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

    def _get_video_duration(self, video_path: str) -> Optional[float]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return count / fps if fps > 0 and count > 0 else None

    def _extract_frames(self, video_path: str, max_frames: int = 8, start_time: float = 0, end_time: Optional[float] = None) -> List[str]:
        duration = self._get_video_duration(video_path) or 300
        end_time = min(end_time if end_time is not None else duration, duration)
        start_time = max(0, start_time)
        if start_time >= end_time:
            start_time = max(0, end_time - 10)
        interval = max(1.0, (end_time - start_time) / max_frames)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        current = int(start_time * fps)
        frame_interval = max(1, int(interval * fps))
        end_frame = int(end_time * fps)
        frames: List[str] = []
        while len(frames) < max_frames and current < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current)
            ok, frame = cap.read()
            if not ok:
                break
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames.append(base64.b64encode(buffer.tobytes()).decode("utf-8"))
            current += frame_interval
        cap.release()
        return frames

    @staticmethod
    def _check_has_audio_track(video_path: str) -> bool:
        ffprobe = VideoAnalysisAction._get_ffprobe()
        if not ffprobe:
            return False
        try:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path],
                capture_output=True, text=True, timeout=30,
            )
            return "audio" in result.stdout
        except Exception:
            return False

    def _extract_audio_from_video(self, video_path: str, start_time: float = 0, end_time: Optional[float] = None) -> Optional[str]:
        ffmpeg = self._get_ffmpeg()
        if not ffmpeg:
            return None
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        cmd = [ffmpeg, "-y", "-i", video_path]
        if start_time > 0:
            cmd += ["-ss", str(start_time)]
        if end_time is not None:
            cmd += ["-t", str(max(0.1, end_time - start_time))]
        cmd += ["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0 or os.path.getsize(output_path) < 100:
                os.unlink(output_path)
                return None
            return output_path
        except Exception:
            try:
                os.unlink(output_path)
            except OSError:
                pass
            return None

    async def _transcribe_audio(self, wav_path: str, query: str) -> Optional[str]:
        try:
            cfg = LLMsConfig.default().get(self.AUDIO_MODEL)
            api_key = cfg.key if cfg else None
            model = cfg.model if cfg else "qwen3.5-omni-plus"
        except Exception:
            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
            model = "qwen3.5-omni-plus"
        if not api_key:
            return None
        import aiohttp
        with open(wav_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "model": model,
            "input": {"messages": [{"role": "user", "content": [
                {"audio": f"data:audio/wav;base64,{b64}"},
                {"text": f"Transcribe and analyze this video audio. Context: {query}"},
            ]}]},
            "parameters": {"temperature": 0.1, "top_p": 0.9, "enable_thinking": False},
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240)) as sess:
                async with sess.post(QWEN_OMNI_NATIVE_ENDPOINT, headers=headers, json=payload) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        return None
            obj = json.loads(text)
            chunks: List[str] = []
            for choice in (obj.get("output") or {}).get("choices") or []:
                for item in ((choice.get("message") or {}).get("content") or []):
                    if isinstance(item, dict) and item.get("text"):
                        chunks.append(str(item["text"]).strip())
            return "\n".join(c for c in chunks if c) or None
        except Exception:
            return None

    async def _combine_visual_and_audio(self, visual: str, audio: str, query: str) -> str:
        api_key, base_url, model = self._get_llm_config(self.SUMMARY_MODEL)
        if not api_key:
            return f"[Visual Analysis]\n{visual}\n\n[Audio Transcription]\n{audio}"
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        prompt = f"Visual:\n{visual}\n\nAudio:\n{audio}\n\nQuestion: {query}\nIntegrate both into one answer."
        try:
            completion = await llm_retry(
                client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                label=f"VideoAnalysis.combine({model})",
            )
            return (completion.choices[0].message.content or "").strip() or f"{visual}\n\n{audio}"
        except Exception:
            return f"{visual}\n\n{audio}"

    async def __call__(self, **kwargs) -> Any:
        query = kwargs.get("query")
        video_path = kwargs.get("video_path")
        if not query or not video_path:
            return {"success": False, "output": None, "error": "Both query and video_path are required.", "metrics": {}}
        if not Path(video_path).exists():
            return {"success": False, "output": None, "error": f"Video file not found: {video_path}", "metrics": {}}
        max_frames = min(int(kwargs.get("max_frames", 8)), 16)
        start_time = float(kwargs.get("start_time", 0))
        end_time = float(kwargs["end_time"]) if kwargs.get("end_time") is not None else None
        analyze_audio = kwargs.get("analyze_audio", True)
        api_key, base_url, model = self._get_llm_config(self.VISION_MODEL)

        visual = None
        frames = self._extract_frames(video_path, max_frames, start_time, end_time)
        if frames and api_key:
            content: List[Dict[str, Any]] = [{"type": "text", "text": f"Analyze these video frames. {query}"}]
            content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}} for frame in frames]
            try:
                client = AsyncOpenAI(api_key=api_key, base_url=base_url)
                completion = await llm_retry(
                    client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    label=f"VideoAnalysis.visual({model})",
                )
                visual = (completion.choices[0].message.content or "").strip() or None
            except Exception as exc:
                logger.warning(f"[VideoAnalysis] visual analysis failed: {exc}")

        audio = None
        has_audio = bool(analyze_audio and self._check_has_audio_track(video_path))
        if has_audio:
            wav = self._extract_audio_from_video(video_path, start_time, end_time)
            if wav:
                try:
                    audio = await self._transcribe_audio(wav, str(query))
                finally:
                    try:
                        os.unlink(wav)
                    except OSError:
                        pass

        if visual and audio:
            output = await self._combine_visual_and_audio(visual, audio, str(query))
            kind = "visual+audio"
        elif visual:
            output, kind = visual, "visual_only"
        elif audio:
            output, kind = audio, "audio_only"
        else:
            return {"success": False, "output": None, "error": "Failed to analyze video frames and audio.", "metrics": {}}
        return {
            "success": True,
            "output": output,
            "error": None,
            "metrics": {"frames_analyzed": len(frames), "has_audio_track": has_audio, "audio_transcribed": bool(audio), "analysis_type": kind},
        }
