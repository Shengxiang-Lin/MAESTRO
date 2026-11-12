#!/usr/bin/env python3
"""Configuration-driven launcher for the original Orchestra-o1 verl GRPO method.

The training algorithm and reward dimensions live in the unchanged
train_grpo_qwen3_8b.sh and reward_fn.py; this launcher only resolves YAML,
safe provider authentication, preflight, and environment variables.
"""
from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = DIR / "config.yaml"
ENGINE = DIR / "train_grpo_qwen3_8b.sh"
ENV_FIELDS = {
    "nnodes": "NNODES",
    "n_gpus_per_node": "N_GPUS_PER_NODE",
    "train_batch_size": "TRAIN_BSZ",
    "ppo_mini_batch_size": "MINI_BSZ",
    "max_prompt_length": "MAX_PROMPT_LEN",
    "max_response_length": "MAX_RESP_LEN",
    "max_token_len_per_gpu": "MAX_TOKEN_LEN_PER_GPU",
    "rollout_n": "ROLLOUT_N",
    "learning_rate": "LR",
    "epochs": "EPOCHS",
    "save_freq": "SAVE_FREQ",
    "tensor_parallel_size": "TP_SIZE",
    "gpu_memory_utilization": "GPU_MEM_UTIL",
    "ray_num_cpus": "RAY_NUM_CPUS",
}
FLOAT_FIELDS = {"learning_rate", "gpu_memory_utilization"}
ALLOWED_SECTIONS = {"version", "paths", "judge", "training", "logging"}


def root_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected a non-empty repository-relative or absolute path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing DA-GRPO config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("DA-GRPO config requires version: 1")
    if set(raw) != ALLOWED_SECTIONS:
        raise ValueError(f"DA-GRPO config must contain exactly {sorted(ALLOWED_SECTIONS)}")
    for section in ("paths", "judge", "training", "logging"):
        if not isinstance(raw[section], dict):
            raise ValueError(f"{section} must be a YAML mapping")
    if set(raw["paths"]) != {"verl_dir", "model_path", "data_dir"}:
        raise ValueError("paths must contain verl_dir, model_path and data_dir")
    if set(raw["judge"]) != {"registry", "model", "api_key_env", "enable_thinking"}:
        raise ValueError("judge must contain registry, model, api_key_env, enable_thinking")
    if set(raw["training"]) != set(ENV_FIELDS):
        raise ValueError("training fields do not match the Orchestra-o1 launcher contract")
    if set(raw["logging"]) != {"backends"}:
        raise ValueError("logging must contain backends only")
    judge = raw["judge"]
    if type(judge["enable_thinking"]) is not bool:
        raise ValueError("judge.enable_thinking must be boolean")
    key_env = judge["api_key_env"]
    if not isinstance(key_env, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key_env):
        raise ValueError("judge.api_key_env must be a safe uppercase environment variable name")
    if not isinstance(judge["model"], str) or not judge["model"]:
        raise ValueError("judge.model is required")

    registry = root_path(judge["registry"])
    if not registry.is_file():
        raise FileNotFoundError(f"Missing shared model registry: {registry}")
    model_registry = yaml.safe_load(registry.read_text(encoding="utf-8"))
    models = model_registry.get("models") if isinstance(model_registry, dict) else None
    model = models.get(judge["model"]) if isinstance(models, dict) else None
    if not isinstance(model, dict) or model.get("backend") != "api" or model.get("api_type") != "openai":
        raise ValueError(f"Judge model {judge['model']!r} must be a shared OpenAI-compatible API model")
    api_base = model.get("base_url")
    if not isinstance(api_base, str) or not api_base.startswith("https://"):
        raise ValueError("The shared Judge provider must have an HTTPS base_url")

    settings = {}
    for name, env_name in ENV_FIELDS.items():
        value = raw["training"][name]
        if name in FLOAT_FIELDS:
            if type(value) not in (int, float) or value <= 0:
                raise ValueError(f"training.{name} must be positive")
            if name == "gpu_memory_utilization" and value > 1:
                raise ValueError("training.gpu_memory_utilization must be <= 1")
        elif type(value) is not int or value <= 0:
            raise ValueError(f"training.{name} must be a positive integer")
        settings[env_name] = str(value)

    if raw["training"]["n_gpus_per_node"] % raw["training"]["tensor_parallel_size"] != 0:
        raise ValueError("training.tensor_parallel_size must divide n_gpus_per_node")
    backends = raw["logging"]["backends"]
    if (not isinstance(backends, list) or not backends
        or len(set(map(str, backends))) != len(backends)
        or any(name not in {"console", "file", "wandb"} for name in backends)):
        raise ValueError("logging.backends must be a unique list from console, file, wandb")

    paths = {name: root_path(value) for name, value in raw["paths"].items()}
    return {
        "config": path, "paths": paths, "settings": settings,
        "judge": {
            "model": judge["model"], "api_base": api_base,
            "api_key_env": key_env, "enable_thinking": judge["enable_thinking"],
        },
        "backends": backends,
    }


def gpu_count() -> int | None:
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, capture_output=True, timeout=8, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return len([line for line in done.stdout.splitlines() if line.strip()])


def check_environment(cfg: dict) -> dict[str, bool]:
    paths = cfg["paths"]
    missing = {}
    for label, path in (
        ("model", paths["model_path"]),
        ("verl_checkout", paths["verl_dir"]),
        ("train.parquet", paths["data_dir"] / "train.parquet"),
        ("expert_decisions.jsonl", paths["data_dir"] / "expert_decisions.jsonl"),
        ("training_shell", ENGINE),
    ):
        missing[label] = path.is_file() if label in {
            "train.parquet", "expert_decisions.jsonl", "training_shell"
        } else path.is_dir()
    missing["verl_import"] = importlib.util.find_spec("verl") is not None
    count = gpu_count()
    needed = int(cfg["settings"]["N_GPUS_PER_NODE"])
    missing["gpu_count"] = count is not None and count >= needed
    return missing


def get_key(key_env: str, *, allow_prompt: bool) -> str | None:
    key = os.environ.get(key_env) or os.environ.get("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()
    if not allow_prompt:
        return None
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"No {key_env} in environment and stdin is not a terminal. "
            f"Set {key_env} securely, or run the launcher interactively."
        )
    value = getpass.getpass(f"Enter {key_env} (hidden, not saved): ").strip()
    if not value:
        raise RuntimeError("Judge API key cannot be empty")
    return value


def build_env(cfg: dict, api_key: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(cfg["settings"])
    env["VERL_DIR"] = str(cfg["paths"]["verl_dir"])
    env["MODEL_PATH"] = str(cfg["paths"]["model_path"])
    env["DATA_DIR"] = str(cfg["paths"]["data_dir"])
    judge = cfg["judge"]
    env[judge["api_key_env"]] = api_key
    env["OPENAI_API_KEY"] = api_key   # Original reward_fn.py reads OpenAI-compatible env.
    env["OPENAI_BASE_URL"] = judge["api_base"]
    env["LLM_JUDGE_MODEL"] = judge["model"]
    env["LLM_JUDGE_ENABLE_THINKING"] = str(judge["enable_thinking"]).lower()
    env["TRAINER_LOGGER"] = json.dumps(cfg["backends"], separators=(",", ":"))
    return env


def test_judge(cfg: dict, env: dict[str, str]) -> None:
    import requests
    judge = cfg["judge"]
    payload = {
        "model": judge["model"],
        "messages": [{"role": "user", "content": "Return only this JSON: {\"ok\": true}"}],
        "temperature": 0.0,
        "max_tokens": 256,
        "enable_thinking": judge["enable_thinking"],
    }
    try:
        result = requests.post(
            judge["api_base"].rstrip("/") + "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {env['OPENAI_API_KEY']}"},
            timeout=45,
        )
        result.raise_for_status()
        content = result.json()["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Judge returned an empty text response")
        print(f"[da-grpo] Judge API OK: {judge['model']}; response: {content[:120]}")
    except requests.HTTPError as error:
        status = error.response.status_code if error.response is not None else "unknown"
        raise RuntimeError(f"Judge HTTP {status} (check model, API key and endpoint region)") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved nonsecret settings and checks; no model call/training")
    parser.add_argument("--check", action="store_true",
                        help="Require verl, model, data and GPU preflight; no model call/training")
    parser.add_argument("--test-judge", action="store_true",
                        help="Make one small authenticated Qwen Judge API request; no training")
    args, overrides = parser.parse_known_args(argv)
    cfg = load_config(args.config)
    judge = cfg["judge"]
    checks = check_environment(cfg)
    safe = {
        "config": str(cfg["config"]),
        "paths": {k: str(v) for k, v in cfg["paths"].items()},
        "judge": {
            "model": judge["model"], "api_base": judge["api_base"],
            "api_key_env": judge["api_key_env"],
            "api_key_available": bool(get_key(judge["api_key_env"], allow_prompt=False)),
            "enable_thinking": judge["enable_thinking"],
        },
        "training_env": cfg["settings"],
        "loggers": cfg["backends"],
        "checks": checks,
        "detected_gpus": gpu_count(),
    }
    print(json.dumps(safe, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if args.check or not args.test_judge:
        if not all(checks.values()):
            raise RuntimeError(
                "DA-GRPO preflight failed: " +
                ", ".join(name for name, ok in checks.items() if not ok) +
                ". Install verl from its GitHub checkout, configure GPUs, "
                "and verify train.parquet and expert_decisions.jsonl."
            )
        if args.check:
            print("[da-grpo] Local preflight OK (no training started).")
            return 0
    api_key = get_key(judge["api_key_env"], allow_prompt=True)
    assert api_key is not None
    env = build_env(cfg, api_key)
    if args.test_judge:
        test_judge(cfg, env)
        return 0
    print("[da-grpo] Starting original Orchestra-o1 verl trainer. Credentials are not logged.")
    return subprocess.call(["bash", str(ENGINE), *overrides], cwd=str(ROOT), env=env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"[da-grpo] ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
