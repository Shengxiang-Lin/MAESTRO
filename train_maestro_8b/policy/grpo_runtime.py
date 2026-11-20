"""Standalone MAESTRO Stage2 GRPO runtime helpers.

Preserves the reference Unsloth / TRL training mechanics with dedicated vLLM Server Mode.
No imports, configuration or file paths from baseline/da-grpo are needed.
Only GPU-free configuration/data/runtime functions live in this module;
train_maestro_8b.policy.da_backend constructs and executes GRPOTrainer.
"""
from __future__ import annotations

import getpass
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "grpo_config.yaml"

# The dataset/tokenization package lives at the repository root. A path-launched
# Ensure the shared MAESTRO dataset utilities are importable from the repo root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def repo_path(name: str | Path) -> Path:
    value = Path(name).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    source = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("version") != 2:
        raise ValueError("MAESTRO Stage2 requires Unsloth/TRL config version: 2")
    if set(cfg) != {"version", "paths", "judge", "model", "rollout", "distributed", "training"}:
        raise ValueError("Invalid top-level configuration sections")
    for name in ("paths", "judge", "model", "rollout", "distributed", "training"):
        if not isinstance(cfg[name], dict):
            raise ValueError(f"{name} must be a mapping")

    paths = cfg["paths"]
    if set(paths) != {"model", "train_parquet", "output_dir"}:
        raise ValueError("Invalid paths configuration")
    for key in paths:
        paths[key] = repo_path(paths[key])

    judge = cfg["judge"]
    if set(judge) != {"registry", "model", "api_key_env", "enable_thinking", "workers"}:
        raise ValueError("Invalid judge configuration")
    judge["registry"] = repo_path(judge["registry"])
    registry = yaml.safe_load(judge["registry"].read_text(encoding="utf-8"))
    model_info = registry.get("models", {}).get(judge["model"], {})
    if model_info.get("backend") != "api" or model_info.get("api_type") != "openai":
        raise ValueError("Judge requires an OpenAI-compatible remote model")
    judge["base_url"] = model_info["base_url"]
    if not judge["base_url"].startswith("https://"):
        raise ValueError("Judge base_url must be HTTPS")
    if type(judge["enable_thinking"]) is not bool or not 1 <= judge["workers"] <= 16:
        raise ValueError("Invalid judge thinking/workers setting")
    if not isinstance(judge["api_key_env"], str) or not judge["api_key_env"].isidentifier():
        raise ValueError("Invalid judge API key env name")

    model = cfg["model"]
    if set(model) != {"dtype", "max_seq_length", "load_in_4bit",
                      "lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules"}:
        raise ValueError("Invalid model configuration")
    if model["dtype"] != "bfloat16" or type(model["load_in_4bit"]) is not bool:
        raise ValueError("Expected BF16 model and a boolean load_in_4bit flag")
    if model["load_in_4bit"]:
        raise ValueError("Baseline uses BF16 LoRA; quantized QLoRA is a separate experiment")
    if not 0 <= model["lora_dropout"] < 1 or not model["lora_target_modules"]:
        raise ValueError("Invalid LoRA settings")
    for key in ("max_seq_length", "lora_rank", "lora_alpha"):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"Invalid model.{key}")

    rollout = cfg["rollout"]
    expected = {"num_generations", "max_prompt_tokens",
                "max_completion_tokens", "temperature", "top_p",
                "gpu_memory_utilization", "max_num_seqs", "kv_cache_dtype",
                "disable_flashinfer", "enforce_eager", "standby",
                "standby_util_override"}
    if set(rollout) != expected:
        raise ValueError("Invalid rollout configuration")
    for key in ("num_generations", "max_prompt_tokens",
                "max_completion_tokens", "max_num_seqs"):
        if type(rollout[key]) is not int or rollout[key] < 1:
            raise ValueError(f"Invalid rollout.{key}")
    if rollout["num_generations"] < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    if rollout["max_num_seqs"] < rollout["num_generations"]:
        raise ValueError("max_num_seqs must support at least one GRPO candidate group")
    if rollout["max_prompt_tokens"] + rollout["max_completion_tokens"] > model["max_seq_length"]:
        raise ValueError("Maximum prompt + completion exceeds Unsloth model context")
    if not 0 < rollout["gpu_memory_utilization"] < 1:
        raise ValueError("Invalid vLLM memory fraction")
    if rollout["temperature"] != 1.0 or rollout["top_p"] != 1.0:
        raise ValueError("O1 reference sampling requires temperature=top_p=1")
    if rollout["disable_flashinfer"] is not True:
        raise ValueError("RTX 6000D profile must disable FlashInfer in Unsloth")
    if rollout["kv_cache_dtype"] != "auto":
        raise ValueError(
            "RTX 6000D + Unsloth FLASH_ATTN requires kv_cache_dtype: auto "
            "(FP8 KV cache unsupported in the selected attention backend)"
        )
    if any(type(rollout[key]) is not bool for key in
           ("enforce_eager", "standby", "standby_util_override")):
        raise ValueError("Invalid rollout boolean")
    if rollout["standby"] and not rollout["standby_util_override"]:
        raise ValueError(
            "With Unsloth Standby, set standby_util_override: true so "
            "gpu_memory_utilization is not silently forced to ~0.879"
        )

    distributed = cfg["distributed"]
    required_topology = {"vllm_mode", "train_gpus", "rollout_gpu",
                         "train_gpu_start", "server_host", "server_port"}
    if set(distributed) != required_topology:
        raise ValueError("Invalid distributed Stage2 configuration")
    if distributed["vllm_mode"] not in ("server", "colocate"):
        raise ValueError("vllm_mode must be server or colocate")
    for key in ("train_gpus", "rollout_gpu", "train_gpu_start", "server_port"):
        val = distributed[key]
        # GPU indices are zero-based for both roles. Only train_gpus and
        # server_port are positive counts; disjointness is checked below.
        minimum = 0 if key in ("rollout_gpu", "train_gpu_start") else 1
        if type(val) is not int or val < minimum:
            raise ValueError(f"Invalid distributed.{key}")
    if distributed["server_port"] > 65535:
        raise ValueError("Invalid vLLM server port")
    if distributed["server_host"] not in ("localhost", "127.0.0.1"):
        raise ValueError("Dedicated vLLM server must bind to localhost")
    train_gpu_ids = range(distributed["train_gpu_start"],
                          distributed["train_gpu_start"] + distributed["train_gpus"])
    if (distributed["vllm_mode"] == "server" and
            distributed["rollout_gpu"] in train_gpu_ids):
        raise ValueError("Rollout and training GPUs must not overlap")
    if distributed["vllm_mode"] == "colocate" and distributed["train_gpus"] != 1:
        raise ValueError("Legacy colocate path supports one GPU only")
    train = cfg["training"]
    required = {"per_device_train_batch_size", "gradient_accumulation_steps", "epochs", "max_steps", "learning_rate", "weight_decay", "warmup_ratio",
                "clip_range", "kl_coef", "loss_type", "optim",
                "max_grad_norm", "save_steps", "logging_steps", "seed"}
    if set(train) != required:
        raise ValueError("Invalid training configuration")
    if any(type(train[key]) is not int or train[key] < 1 for key in ("per_device_train_batch_size", "gradient_accumulation_steps")):
        raise ValueError("Training microbatch and gradient accumulation must be positive integers")
    if train["loss_type"] != "dapo":
        raise ValueError("MAESTRO uses token-normalized DAPO-style GRPO loss")
    if train["optim"] != "adamw_8bit":
        raise ValueError("Single-GPU profile uses 8-bit AdamW optimizer")
    for key in ("epochs", "save_steps", "logging_steps"):
        if type(train[key]) is not int or train[key] < 1:
            raise ValueError(f"Invalid training.{key}")
    if type(train["max_steps"]) is not int or (train["max_steps"] != -1 and train["max_steps"] < 1):
        raise ValueError("max_steps must be -1 (epochs) or positive")
    if train["learning_rate"] <= 0 or train["kl_coef"] < 0 or not 0 <= train["clip_range"] < 1:
        raise ValueError("Invalid learning rate / KL / clipping")
    if not 0 <= train["warmup_ratio"] < 1:
        raise ValueError("Invalid warmup_ratio")
    cfg["_source"] = source
    return cfg


def batch_plan(cfg: dict) -> dict[str, int | None]:
    """Full/global completion accounting; B is the per-device microbatch."""
    rollout, training, dist = cfg["rollout"], cfg["training"], cfg["distributed"]
    b = training["per_device_train_batch_size"]
    a = training["gradient_accumulation_steps"]
    g = rollout["num_generations"]
    global_completions = dist["train_gpus"] * b * a
    if global_completions % g:
        raise ValueError(
            f"Global completion batch {global_completions} must be divisible by "
            f"num_generations={g}; adjust training microbatch/accumulation"
        )
    return {
        "per_device_train_batch_size": b,
        "gradient_accumulation_steps": a,
        "generation_batch_size": None,  # TRL derives the distributed batch.
        "num_generations": g,
        "train_gpus": dist["train_gpus"],
        "global_completions_per_step": global_completions,
        "prompts_per_step": global_completions // g,
    }


def ranked_path(path: str | Path) -> Path:
    """Isolate per-rank reward diagnostics/cache writes under DDP."""
    dest = Path(path)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if not 0 <= rank < world:
        raise ValueError("Invalid distributed RANK/WORLD_SIZE")
    if world == 1:
        return dest
    return dest.with_name(f"{dest.stem}.rank{rank:02d}{dest.suffix}")


def environment_checks(cfg: dict) -> dict[str, bool]:
    result = {name: importlib.util.find_spec(name) is not None for name in
              ("unsloth", "trl", "vllm", "datasets", "torch", "pyarrow", "bitsandbytes")}
    result["model"] = cfg["paths"]["model"].is_dir()
    result["train_parquet"] = cfg["paths"]["train_parquet"].is_file()
    return result


def show_config(cfg: dict) -> None:
    checks = environment_checks(cfg)
    print(json.dumps({
        "backend": "Unsloth + TRL GRPOTrainer + " + cfg["distributed"]["vllm_mode"] + " vLLM",
        "config": str(cfg["_source"]),
        "model": str(cfg["paths"]["model"]),
        "train_parquet": str(cfg["paths"]["train_parquet"]),
        "output_dir": str(cfg["paths"]["output_dir"]),
        "batch_plan": batch_plan(cfg),
        "distributed": cfg["distributed"],
        "epochs": cfg["training"]["epochs"],
        "loss_type": cfg["training"]["loss_type"],
        "beta": cfg["training"]["kl_coef"],
        "disable_flashinfer": cfg["rollout"]["disable_flashinfer"],
        "kv_cache_dtype": cfg["rollout"]["kv_cache_dtype"],
        "vllm_gpu_memory_utilization": cfg["rollout"]["gpu_memory_utilization"],
        "vllm_max_num_seqs": cfg["rollout"]["max_num_seqs"],
        "standby_util_override": cfg["rollout"]["standby_util_override"],
        "judge": {
            "model": cfg["judge"]["model"],
            "base_url": cfg["judge"]["base_url"],
            "api_key_available": bool(os.environ.get(cfg["judge"]["api_key_env"])),
            "enable_thinking": cfg["judge"]["enable_thinking"],
        },
        "checks": checks,
    }, ensure_ascii=False, indent=2))


def judge_key(name: str) -> str:
    key = (os.environ.get(name) or "").strip()
    if not key:
        if not sys.stdin.isatty():
            raise RuntimeError(f"Set {name} before noninteractive training")
        key = getpass.getpass(f"Enter {name} (hidden): ").strip()
    if not key:
        raise ValueError("Judge API key cannot be blank")
    return key


def setup_judge(cfg: dict, key: str) -> None:
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = cfg["judge"]["base_url"]
    os.environ["LLM_JUDGE_MODEL"] = cfg["judge"]["model"]
    os.environ["LLM_JUDGE_ENABLE_THINKING"] = str(cfg["judge"]["enable_thinking"]).lower()


def probe_judge(cfg: dict, key: str) -> None:
    import requests
    cfg_j = cfg["judge"]
    response = requests.post(
        cfg_j["base_url"].rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": cfg_j["model"],
            "messages": [{"role": "user", "content": "Reply only: OK"}],
            "temperature": 0,
            "max_tokens": 32,
            "enable_thinking": cfg_j["enable_thinking"],
        },
        timeout=45,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    if not content or not str(content).strip():
        raise RuntimeError("Judge returned empty response")
    print(f"[MAESTRO/GRPO] Judge {cfg_j['model']} HTTP OK", flush=True)


def load_dataset_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError("Empty MAESTRO Stage2 Parquet")
    result = []
    for i, row in enumerate(rows):
        messages = row.get("prompt")
        info = row.get("extra_info")
        reward_data = row.get("reward_model")
        if (not isinstance(messages, list) or len(messages) != 1 or
            messages[0].get("role") != "user" or
            not isinstance(messages[0].get("content"), str) or
            not isinstance(info, dict) or not info.get("has_expert") or
            not info.get("question") or not isinstance(reward_data, dict)):
            raise ValueError(f"Invalid dataset row {i}: expected O1 prompt/reward-only metadata")
        result.append({
            "prompt": messages,
            # The following columns are passed ONLY to the reward callable.
            # They must never be interpolated into the prompt.
            "data_source": row.get("data_source") or "maestro",
            "ground_truth": str(reward_data.get("ground_truth") or ""),
            "extra_info_json": json.dumps(info, ensure_ascii=False),
        })
    return result


def verify_prepared_global_context(train_parquet: Path) -> dict:
    """Fail closed if prepared evaluator sidecars were replaced after conversion."""
    base = Path(train_parquet).parent
    path = base / "summary.json"
    if not path.is_file():
        raise FileNotFoundError("Missing prepared GRPO summary: " + str(path))
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("reward_mode") != "global":
        raise ValueError("Prepared GRPO reward mode is not global")
    expected = summary.get("private_context_file_sha256")
    if not isinstance(expected, dict) or set(expected) != {
        "private/scenes.jsonl", "private/claim_state_supervision.jsonl"
    }:
        raise ValueError("Prepared GRPO missing frozen Global Judge evidence hashes")
    from environment.data_export.source_io import digest
    for name, recorded in expected.items():
        current = base / name
        if not current.is_file() or digest(current) != recorded:
            raise ValueError("Prepared Global Judge evidence checksum mismatch: " + name)
    return expected

def check_prompt_lengths(rows: list[dict], tokenizer: Any, limit: int) -> None:
    # Fail instead of silently truncating evidence or Sub-Agent history.
    from dataset.tokenization import extract_token_ids

    overlong = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True,
        )
        ids = extract_token_ids(encoded)
        if len(ids) > limit:
            overlong.append((json.loads(row["extra_info_json"]).get("sample_id"), len(ids)))
    if overlong:
        raise ValueError(
            f"{len(overlong)} prompts exceed {limit} tokens; examples {overlong[:5]}"
        )


def configure_runtime(cfg: dict) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Unsloth's vLLM loader can unconditionally re-enable the FlashInfer
    # top-k/top-p sampler on SM120, overriding VLLM_USE_FLASHINFER_SAMPLER=0.
    # UNSLOTH_VLLM_NO_FLASHINFER is Unsloth's supported hard opt-out: it
    # disables the sampler AND excludes FlashInfer from attention selection
    # (including worker subprocesses). Set this before importing Unsloth.
    os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # Not a recognized setting in this vLLM 0.32.0 build: it was ignored
    # and the installed default selected FLASHINFER despite TRITON_ATTN.
    os.environ.pop("VLLM_ATTENTION_BACKEND", None)
    os.environ["UNSLOTH_VLLM_STANDBY"] = (
        "1" if cfg["distributed"]["vllm_mode"] == "colocate" and cfg["rollout"]["standby"] else "0"
    )
    # Otherwise Unsloth silently replaces the requested 0.35 utilization with
    # ~0.879 on 83 GiB GPUs. Its official override preserves this budget.
    os.environ["UNSLOTH_VLLM_STANDBY_UTIL_OVERRIDE"] = (
        "1" if cfg["rollout"]["standby_util_override"] else "0"
    )


def disable_vllm_flashinfer_autotune() -> None:
    """Disable vLLM 0.32's FlashInfer autotuner under Unsloth's opt-out.

    Unsloth's UNSLOTH_VLLM_NO_FLASHINFER masks the 'flashinfer' package so
    vLLM cannot select its kernels. vLLM 0.32 nonetheless enables its
    FlashInfer *autotuner* through the default O2 optimization preset, even
    in enforce_eager mode. During worker initialization the autotune-cache
    synchronization then imports flashinfer.autotuner and crashes because
    the masked package is not importable.

    Unsloth's load_vllm entrypoint does not expose vLLM's
    enable_flashinfer_autotune EngineArgs field. Adjust the official vLLM
    in-memory optimization presets before the engine is constructed, rather
    than patching site-packages or creating an unsafe fake FlashInfer module.
    """
    from vllm.config.vllm import OPTIMIZATION_LEVEL_TO_CONFIG

    if not OPTIMIZATION_LEVEL_TO_CONFIG:
        raise RuntimeError("vLLM optimization presets unavailable")
    for preset in OPTIMIZATION_LEVEL_TO_CONFIG.values():
        kernel = preset.get("kernel_config")
        if not isinstance(kernel, dict) or "enable_flashinfer_autotune" not in kernel:
            raise RuntimeError(
                "vLLM optimization preset changed; cannot safely disable "
                "FlashInfer autotuning for this Unsloth/vLLM combination"
            )
        kernel["enable_flashinfer_autotune"] = False
    print("[MAESTRO/GRPO] Disabled vLLM FlashInfer autotune in all optimization presets", flush=True)


def build_grpo_kwargs(cfg: dict) -> dict:
    """Pure constructor argument mapper; tested without importing CUDA libraries."""
    plan = batch_plan(cfg)
    rollout, train, paths = cfg["rollout"], cfg["training"], cfg["paths"]
    result = {
        "output_dir": str(paths["output_dir"]),
        "per_device_train_batch_size": plan["per_device_train_batch_size"],
        "gradient_accumulation_steps": plan["gradient_accumulation_steps"],
        "generation_batch_size": plan["generation_batch_size"],
        "num_generations": plan["num_generations"],
        "num_iterations": 1,
        "max_completion_length": rollout["max_completion_tokens"],
        "temperature": rollout["temperature"],
        "top_p": rollout["top_p"],
        "learning_rate": train["learning_rate"],
        "weight_decay": train["weight_decay"],
        # Transformers 5 accepts a fraction of training steps here.
        "warmup_steps": train["warmup_ratio"],
        "lr_scheduler_type": "cosine",
        "optim": train["optim"],
        "bf16": True,
        "fp16": False,
        "num_train_epochs": train["epochs"],
        "max_steps": train["max_steps"],
        "beta": train["kl_coef"],
        "epsilon": train["clip_range"],
        "loss_type": train["loss_type"],
        "scale_rewards": "group",
        "max_grad_norm": train["max_grad_norm"],
        "save_steps": train["save_steps"],
        "logging_steps": train["logging_steps"],
        "seed": train["seed"],
        "save_total_limit": 2,
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # Unsloth attaches its shared vLLM engine when fast_inference=True.
        "use_vllm": True,
        "vllm_mode": cfg["distributed"]["vllm_mode"],
        "vllm_gpu_memory_utilization": rollout["gpu_memory_utilization"],
        "vllm_max_model_length": cfg["model"]["max_seq_length"],
        "vllm_enable_sleep_mode": cfg["distributed"]["vllm_mode"] == "colocate",
        "vllm_tensor_parallel_size": 1,
        "ddp_find_unused_parameters": False,
        # TRL's optional train/inference IS correction is NOT part of O1;
        # disable it for closer parity with original O1/verl sampling.
        "vllm_importance_sampling_correction": False,
    }
    if cfg["distributed"]["vllm_mode"] == "server":
        result.update({
            "vllm_server_host": cfg["distributed"]["server_host"],
            "vllm_server_port": cfg["distributed"]["server_port"],
        })
    return result

