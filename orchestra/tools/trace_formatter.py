"""
Trace Formatter - Abstracted trace formatting utility

Provides extensible trace formatting interface supporting different benchmark action/observation formats.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Protocol


class StepLike(Protocol):
    action: Dict[str, Any]
    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any]


class ActionFormatter(ABC):
    @property
    @abstractmethod
    def action_type(self) -> str:
        ...

    @abstractmethod
    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        ...


class ObservationFormatter(ABC):
    @abstractmethod
    def can_format(self, obs: Dict[str, Any]) -> bool:
        ...

    @abstractmethod
    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        ...


class FinishActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "finish"

    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        status = params.get("status", "done")
        msg = params.get("message", "")[:60]
        return f'finish(status="{status}", msg="{msg}")'


class GoogleSearchActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "GoogleSearchAction"

    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        return f'GoogleSearch(query="{params.get("query", "")[:80]}")'


class ExtractUrlActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "ExtractUrlContentAction"

    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        return f'ExtractUrl(url="{params.get("url", "")[:60]}", query="{params.get("browse_query", "")[:40]}")'


class ExecuteCodeActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "ExecuteCodeAction"

    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        code = params.get("code", "")[:80].replace("\n", " ")
        return f'ExecuteCode(code="{code}...")'


class ImageAnalysisActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "ImageAnalysisAction"

    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        return f'ImageAnalysis(query="{params.get("query", "")[:60]}", image="{params.get("image_path", "")[:40]}")'


class VideoAnalysisActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "VideoAnalysisAction"

    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        return f'VideoAnalysis(query="{params.get("query", "")[:60]}", video="{params.get("video_path", "")[:40]}", frames={params.get("max_frames", 8)}, audio={params.get("analyze_audio", True)})'


class ParseAudioActionFormatter(ActionFormatter):
    @property
    def action_type(self) -> str:
        return "ParseAudioAction"

    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        return f'ParseAudio(query="{params.get("query", "")[:60]}", audio="{params.get("audio_path", "")[:40]}")'


class SuccessObservationFormatter(ObservationFormatter):
    def can_format(self, obs: Dict[str, Any]) -> bool:
        return "success" in obs

    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        return f"success={obs.get('success', False)}", str(obs.get("output", obs.get("error", "")))


class FallbackActionFormatter(ActionFormatter):
    def __init__(self, action_type: str = "unknown"):
        self._action_type = action_type

    @property
    def action_type(self) -> str:
        return self._action_type

    def format(self, params: Dict[str, Any], max_len: int = 100) -> str:
        return f'{self._action_type}({list(params.keys())[:3]})'


class FallbackObservationFormatter(ObservationFormatter):
    def can_format(self, obs: Dict[str, Any]) -> bool:
        return True

    def format(self, obs: Dict[str, Any], max_len: int = 300) -> tuple[str, str]:
        return "", str(obs)


class TraceFormatter:
    def __init__(self):
        self._action_formatters: Dict[str, ActionFormatter] = {}
        self._obs_formatters: List[ObservationFormatter] = []
        self._fallback_obs_formatter = FallbackObservationFormatter()

    def register_action_formatter(self, formatter: ActionFormatter) -> "TraceFormatter":
        self._action_formatters[formatter.action_type] = formatter
        return self

    def register_obs_formatter(self, formatter: ObservationFormatter) -> "TraceFormatter":
        self._obs_formatters.append(formatter)
        return self

    def format_action(self, action: Dict[str, Any], max_len: int = 100) -> str:
        action_type = action.get("action", "unknown")
        params = action.get("params", {})
        formatter = self._action_formatters.get(action_type)
        return formatter.format(params, max_len) if formatter else FallbackActionFormatter(action_type).format(params, max_len)

    def format_observation(self, obs: Any, max_len: int = 300) -> tuple[str, str]:
        if not isinstance(obs, dict):
            return "", str(obs)
        for formatter in self._obs_formatters:
            if formatter.can_format(obs):
                return formatter.format(obs, max_len)
        return self._fallback_obs_formatter.format(obs, max_len)

    def format_trace(self, trace: List[StepLike], max_output_len: int = 300) -> str:
        if not trace:
            return "No steps executed"
        lines = []
        for i, step in enumerate(trace, 1):
            lines.append(f"Step {i}: {self.format_action(step.action)}")
            status_line, output = self.format_observation(step.observation, max_output_len)
            if status_line:
                lines.append(f"  → {status_line}")
            if len(output) > max_output_len:
                output = output[:max_output_len] + f"...[+{len(output)-max_output_len} chars]"
            lines.append(f"  → output: {output.replace(chr(10), ' ').strip()}")
            lines.append("")
        return "\n".join(lines)


def create_gaia_formatter() -> TraceFormatter:
    return (
        TraceFormatter()
        .register_action_formatter(GoogleSearchActionFormatter())
        .register_action_formatter(ExtractUrlActionFormatter())
        .register_action_formatter(ExecuteCodeActionFormatter())
        .register_action_formatter(ImageAnalysisActionFormatter())
        .register_action_formatter(VideoAnalysisActionFormatter())
        .register_action_formatter(ParseAudioActionFormatter())
        .register_action_formatter(FinishActionFormatter())
        .register_obs_formatter(SuccessObservationFormatter())
    )
