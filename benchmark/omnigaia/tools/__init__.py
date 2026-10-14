"""OmniGAIA benchmark tools package."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from benchmark.omnigaia.tools.google_search import GoogleSearchAction
from benchmark.omnigaia.tools.execute_code import ExecuteCodeAction
from benchmark.omnigaia.tools.extract_url_jina import ExtractUrlContentAction
from benchmark.omnigaia.tools.multimodal_toolkit import ImageAnalysisAction
from benchmark.omnigaia.tools.audio_analysis import ParseAudioAction
from benchmark.omnigaia.tools.video_analysis import VideoAnalysisAction

DEFAULT_TOOL_MODELS = {
    "image_analysis": "qwen3.7-flash",
    "audio_analysis": "qwen3.5-omni-flash",
    "video_vision": "qwen3.7-flash",
    "video_audio": "qwen3.5-omni-flash",
    "video_summary": "qwen3.7-flash",
}
DEFAULT_SEARCH_CONFIG = {
    "provider": "dashscope",
    "model": "qwen3.7-flash",
}
DEFAULT_BROWSE_CONFIG = {
    "provider": "dashscope",
    "model": "qwen3.7-flash",
}
_SEARCH_PROVIDERS = {"dashscope", "serper"}
_BROWSE_PROVIDERS = {"dashscope", "jina"}
_LEGACY_WEB_MODEL_KEYS = {"search", "browse"}


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_tool_models(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Resolve non-web model roles while accepting legacy search/browse keys."""
    models = dict(DEFAULT_TOOL_MODELS)
    if overrides:
        unknown = sorted(
            set(overrides) - set(DEFAULT_TOOL_MODELS) - _LEGACY_WEB_MODEL_KEYS
        )
        if unknown:
            raise ValueError(
                "Unknown tool_models keys: " + ", ".join(unknown)
                + ". Expected: " + ", ".join(sorted(DEFAULT_TOOL_MODELS))
            )
        models.update(
            {
                str(key): str(value)
                for key, value in overrides.items()
                if key in DEFAULT_TOOL_MODELS
            }
        )
    return models


def _resolve_web_config(
    role: str,
    raw: Mapping[str, Any] | None,
    *,
    legacy_model: str | None = None,
) -> dict[str, str]:
    if role == "search":
        defaults = DEFAULT_SEARCH_CONFIG
        allowed = _SEARCH_PROVIDERS
        legacy_provider = "serper" if _env_enabled("GOOGLE_SEARCH_USE_SERPER") else "dashscope"
    elif role == "browse":
        defaults = DEFAULT_BROWSE_CONFIG
        allowed = _BROWSE_PROVIDERS
        legacy_provider = "jina" if _env_enabled("EXTRACT_URL_USE_JINA") else "dashscope"
    else:
        raise ValueError(f"Unknown web tool role: {role}")

    if raw is None:
        config = {
            "provider": legacy_provider,
            "model": str(legacy_model or defaults["model"]),
        }
    else:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{role} must be a mapping with provider/model")
        config = {
            "provider": str(raw.get("provider") or defaults["provider"]).strip().lower(),
            "model": str(raw.get("model") or legacy_model or defaults["model"]).strip(),
        }

    if config["provider"] not in allowed:
        raise ValueError(
            f"Unsupported {role}.provider={config['provider']!r}; "
            f"expected one of: {', '.join(sorted(allowed))}"
        )
    if role == "browse" and not config["model"]:
        raise ValueError("browse.model is required")
    if role == "search" and config["provider"] == "dashscope" and not config["model"]:
        raise ValueError("search.model is required for provider='dashscope'")
    return config


def load_web_tool_configs(config_path: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load provider-aware search/browse config with legacy YAML compatibility."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    tool_models = raw.get("tool_models") or {}
    if not isinstance(tool_models, Mapping):
        tool_models = {}
    search = _resolve_web_config(
        "search",
        raw.get("search"),
        legacy_model=(str(tool_models.get("search")) if tool_models.get("search") else None),
    )
    browse = _resolve_web_config(
        "browse",
        raw.get("browse"),
        legacy_model=(str(tool_models.get("browse")) if tool_models.get("browse") else None),
    )
    return search, browse


def build_omnigaia_tools(
    tool_models: Mapping[str, str] | None = None,
    *,
    search: Mapping[str, Any] | None = None,
    browse: Mapping[str, Any] | None = None,
):
    """Build the shared OmniGAIA tool set using explicit model/provider roles."""
    models = resolve_tool_models(tool_models)
    search_cfg = _resolve_web_config("search", search)
    browse_cfg = _resolve_web_config("browse", browse)
    return [
        GoogleSearchAction(
            provider=search_cfg["provider"],
            model_name=search_cfg["model"],
        ),
        ExecuteCodeAction(),
        ExtractUrlContentAction(
            provider=browse_cfg["provider"],
            model_name=browse_cfg["model"],
        ),
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
    "DEFAULT_SEARCH_CONFIG",
    "DEFAULT_BROWSE_CONFIG",
    "resolve_tool_models",
    "load_web_tool_configs",
    "build_omnigaia_tools",
    "GoogleSearchAction",
    "ExecuteCodeAction",
    "ExtractUrlContentAction",
    "ImageAnalysisAction",
    "ParseAudioAction",
    "VideoAnalysisAction",
]
