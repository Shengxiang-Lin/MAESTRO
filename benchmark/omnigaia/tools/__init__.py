"""OmniGAIA benchmark tools package."""
from __future__ import annotations

from typing import Mapping

from benchmark.omnigaia.tools.google_search import GoogleSearchAction
from benchmark.omnigaia.tools.execute_code import ExecuteCodeAction
from benchmark.omnigaia.tools.extract_url_jina import ExtractUrlContentAction
from benchmark.omnigaia.tools.multimodal_toolkit import ImageAnalysisAction
from benchmark.omnigaia.tools.audio_analysis import ParseAudioAction
from benchmark.omnigaia.tools.video_analysis import VideoAnalysisAction

DEFAULT_TOOL_MODELS = {
    "search": "qwen3.7-flash",
    "browse": "qwen3.7-flash",
    "image_analysis": "qwen3.7-flash",
    "audio_analysis": "qwen3.5-omni-flash",
    "video_vision": "qwen3.7-flash",
    "video_audio": "qwen3.5-omni-flash",
    "video_summary": "qwen3.7-flash",
}


def resolve_tool_models(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    models = dict(DEFAULT_TOOL_MODELS)
    if overrides:
        unknown = sorted(set(overrides) - set(DEFAULT_TOOL_MODELS))
        if unknown:
            raise ValueError(
                "Unknown tool_models keys: " + ", ".join(unknown)
                + ". Expected: " + ", ".join(sorted(DEFAULT_TOOL_MODELS))
            )
        models.update({str(key): str(value) for key, value in overrides.items()})
    return models


def build_omnigaia_tools(tool_models: Mapping[str, str] | None = None):
    """Build the shared OmniGAIA tool set using explicit model roles."""
    models = resolve_tool_models(tool_models)
    return [
        GoogleSearchAction(model_name=models["search"]),
        ExecuteCodeAction(),
        ExtractUrlContentAction(model_name=models["browse"]),
        ImageAnalysisAction(model_name=models["image_analysis"]),
        ParseAudioAction(model_name=models["audio_analysis"]),
        VideoAnalysisAction(
            vision_model=models["video_vision"],
            audio_model=models["video_audio"],
            summary_model=models["video_summary"],
        ),
    ]


__all__ = [
    "DEFAULT_TOOL_MODELS",
    "resolve_tool_models",
    "build_omnigaia_tools",
    "GoogleSearchAction",
    "ExecuteCodeAction",
    "ExtractUrlContentAction",
    "ImageAnalysisAction",
    "ParseAudioAction",
    "VideoAnalysisAction",
]
