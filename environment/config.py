"""YAML configuration and per-role model resolution for environment CLIs.

Both remote APIs and already-running local vLLM servers use the same
OpenAI-compatible chat-completions client. No API tokens are stored in YAML.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ENV_DIR = Path(__file__).resolve().parent
REPO_DIR = ENV_DIR.parent
DEFAULT_CONFIG = ENV_DIR / "config.yaml"
ROLE_SELECTORS = {
    "question_author": ("question_generation", "author_profile"),
    "question_critic": ("question_generation", "critic_profile"),
    "trajectory_teacher": ("trajectory_simulation", "teacher_profile"),
}
PATH_ARGUMENTS = {
    "manifest", "authoring_inputs", "worlds", "capabilities",
    "output_dir", "output", "realized", "rejected_output",
    "kqa_train", "kqa_val", "yago_facts", "yago_schema",
    "yago_taxonomy", "mmqa_train", "mmqa_dev",
}


class EnvironmentConfigError(ValueError):
    """Raised when the YAML has an invalid setting or model profile."""


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise EnvironmentConfigError(f"environment config not found: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise EnvironmentConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise EnvironmentConfigError("environment/config.yaml must contain a YAML mapping")
    if payload.get("version") != 1:
        raise EnvironmentConfigError("environment/config.yaml must declare version: 1")
    models = payload.get("models", {})
    if not isinstance(models, Mapping):
        raise EnvironmentConfigError("models must be a mapping of named profiles")
    return dict(payload)


def _extract_config_arg(argv: list[str]) -> tuple[str, list[str]]:
    """Allow --config before or after an argparse subcommand."""
    config_path = str(DEFAULT_CONFIG)
    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        part = argv[index]
        if part == "--config":
            if index + 1 == len(argv):
                raise EnvironmentConfigError("--config requires a YAML file path")
            config_path = argv[index + 1]
            index += 2
            continue
        if part.startswith("--config="):
            config_path = part.partition("=")[2]
            index += 1
            continue
        cleaned.append(part)
        index += 1
    return config_path, cleaned


def _walk_parsers(parser: argparse.ArgumentParser):
    yield parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                yield from _walk_parsers(child)


def _configured_value(name: str, value: Any, target: int) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.replace("{target}", str(target))
        if name in PATH_ARGUMENTS:
            path = Path(value).expanduser()
            return str((REPO_DIR / path).resolve()) if not path.is_absolute() else str(path)
    return value


def parse_with_config(
    parser: argparse.ArgumentParser,
    section: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Load defaults from YAML, then let explicit CLI flags take precedence."""
    raw = list(sys.argv[1:] if argv is None else argv)
    config_path, cleaned = _extract_config_arg(raw)
    config = load_config(config_path)
    values = config.get(section, {})
    if not isinstance(values, Mapping):
        raise EnvironmentConfigError(f"{section} must be a YAML mapping")
    construction = config.get("task_construction", {})
    values = dict(values)
    if section in {"question_generation", "trajectory_simulation"} and "seed" not in values:
        values["seed"] = construction.get("seed", 17)
    # Pool-mode artifacts carry the *pool capacity* in their filenames.
    # The frozen manifest still records the actual accepted world count.
    if construction.get("all_valid_candidates", False):
        target = int(construction.get("candidate_pool_size") or 0)
        if target < 1:
            raise EnvironmentConfigError(
                "task_construction.candidate_pool_size must be positive "
                "when all_valid_candidates is enabled"
            )
    else:
        target = int(construction.get("target", 100))
    recognized: set[str] = set()
    for current in _walk_parsers(parser):
        names = {action.dest for action in current._actions}
        recognized.update(names)
        defaults = {
            key: _configured_value(key, value, target)
            for key, value in values.items()
            if key in names and key not in {"help", "config", "command"}
        }
        if defaults:
            current.set_defaults(**defaults)
            # YAML-supplied inputs remove the need to repeat required path flags.
            for action in current._actions:
                if action.dest in defaults and defaults[action.dest] is not None:
                    action.required = False
    extras = {"author_profile", "critic_profile", "teacher_profile"}
    unknown = set(values).difference(recognized | extras)
    if unknown:
        raise EnvironmentConfigError(
            f"unknown {section} configuration key(s): {', '.join(sorted(unknown))}"
        )
    if not any(action.dest == "config" for action in parser._actions):
        parser.add_argument(
            "--config", default=str(DEFAULT_CONFIG),
            help="environment YAML (default: environment/config.yaml)",
        )
    args = parser.parse_args(cleaned)
    args.config = str(Path(config_path).expanduser().resolve())
    args._environment_config = config
    args._explicit_options = {
        item.split("=", 1)[0] for item in cleaned if item.startswith("--")
    }
    return args


def resolve_model(args: argparse.Namespace, role: str) -> dict[str, Any]:
    """Resolve the model used by one Author, Critic or Teacher role.

    A profile with backend=api or backend=vllm is accessed via an
    OpenAI-compatible HTTP endpoint. backend=passthrough needs no network.
    Explicit legacy CLI options override profile values.
    """
    if role not in ROLE_SELECTORS:
        raise EnvironmentConfigError(f"unknown environment model role: {role}")
    section, selector = ROLE_SELECTORS[role]
    config = getattr(args, "_environment_config", {}) or {}
    section_cfg = config.get(section, {}) or {}
    profile_name = section_cfg.get(selector)
    if profile_name:
        models = config.get("models", {})
        if profile_name not in models:
            raise EnvironmentConfigError(
                f"{section}.{selector} refers to missing model profile: {profile_name}"
            )
        profile = models[profile_name]
        if not isinstance(profile, Mapping):
            raise EnvironmentConfigError(f"model profile {profile_name} must be a mapping")
        profile = dict(profile)
    else:
        profile = {
            "backend": "passthrough" if getattr(args, "provider", "passthrough") == "passthrough" else "api",
            "model": getattr(args, "model", None),
            "base_url": getattr(args, "base_url", None),
            "api_key_env": getattr(args, "api_key_env", None),
            "json_mode": True,
        }
    explicit = getattr(args, "_explicit_options", set())
    if "--provider" in explicit and args.provider == "passthrough":
        profile["backend"] = "passthrough"
    if role == "question_critic" and "--critic-model" in explicit:
        profile["model"] = args.critic_model
    elif role != "question_critic" and "--model" in explicit:
        profile["model"] = args.model
    if "--base-url" in explicit:
        profile["base_url"] = args.base_url
    if "--api-key-env" in explicit:
        profile["api_key_env"] = args.api_key_env
    if "--no-json-mode" in explicit and args.no_json_mode:
        profile["json_mode"] = False

    backend = str(profile.get("backend", "")).lower()
    if backend == "passthrough":
        return {"backend": "passthrough", "provider": "passthrough",
                "model": None, "base_url": None, "api_key": None, "json_mode": False}
    if backend not in {"api", "vllm"}:
        raise EnvironmentConfigError(
            f"profile {profile_name or '(CLI)'}: backend must be api, vllm or passthrough"
        )
    model = profile.get("model")
    base_url = profile.get("base_url")
    if not model or not base_url:
        raise EnvironmentConfigError(
            f"profile {profile_name or '(CLI)'} requires model and base_url"
        )
    env_name = profile.get("api_key_env")
    api_key = os.environ.get(str(env_name)) if env_name else None
    if backend == "api" and env_name and not api_key:
        raise EnvironmentConfigError(
            f"set {env_name} before calling API model profile {profile_name}"
        )
    return {
        "backend": backend,
        "provider": "openai-compatible",
        "model": str(model),
        "base_url": str(base_url),
        "api_key": api_key,
        "json_mode": bool(profile.get("json_mode", True)),
    }
