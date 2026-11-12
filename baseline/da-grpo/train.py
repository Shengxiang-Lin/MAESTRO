#!/usr/bin/env python3
"""Single-GPU MAESTRO GRPO using Unsloth, TRL GRPOTrainer and colocated vLLM.

All policy-gradient, advantage, KL, optimization and checkpoint logic belongs
to TRL/Unsloth. This entrypoint only validates MAESTRO data, configures the
frameworks and supplies the original Orchestra-o1 four-dimensional reward.
"""
from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "config.yaml"

# The dataset/tokenization package lives at the repository root. A path-launched
# script has baseline/da-grpo in sys.path[0], not necessarily MAESTRO root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def repo_path(name: str | Path) -> Path:
    value = Path(name).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    source = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("version") != 2:
        raise ValueError("DA-GRPO requires the new Unsloth/TRL config version: 2")
    if set(cfg) != {"version", "paths", "judge", "model", "rollout", "training"}:
        raise ValueError("Invalid top-level configuration sections")
    for name in ("paths", "judge", "model", "rollout", "training"):
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
    expected = {"prompts_per_step", "num_generations", "max_prompt_tokens",
                "max_completion_tokens", "temperature", "top_p",
                "gpu_memory_utilization", "max_num_seqs", "kv_cache_dtype",
                "disable_flashinfer", "enforce_eager", "standby"}
    if set(rollout) != expected:
        raise ValueError("Invalid rollout configuration")
    for key in ("prompts_per_step", "num_generations", "max_prompt_tokens",
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
    if type(rollout["enforce_eager"]) is not bool or type(rollout["standby"]) is not bool:
        raise ValueError("Invalid rollout boolean")

    train = cfg["training"]
    required = {"epochs", "max_steps", "learning_rate", "weight_decay", "warmup_ratio",
                "clip_range", "kl_coef", "loss_type", "optim",
                "max_grad_norm", "save_steps", "logging_steps", "seed"}
    if set(train) != required:
        raise ValueError("Invalid training configuration")
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


def batch_plan(cfg: dict) -> dict[str, int]:
    rollout = cfg["rollout"]
    # In TRL, per_device_train_batch_size counts completions (NOT prompts).
    # One-completion backward plus 32 gradient accumulation steps:
    # 4 different prompts x 8 generations = 32 completions per optimizer step.
    generation_batch = rollout["prompts_per_step"] * rollout["num_generations"]
    return {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": generation_batch,
        "generation_batch_size": generation_batch,
        "num_generations": rollout["num_generations"],
        "prompts_per_step": rollout["prompts_per_step"],
    }


def environment_checks(cfg: dict) -> dict[str, bool]:
    result = {name: importlib.util.find_spec(name) is not None for name in
              ("unsloth", "trl", "vllm", "datasets", "torch", "pyarrow", "bitsandbytes")}
    result["model"] = cfg["paths"]["model"].is_dir()
    result["train_parquet"] = cfg["paths"]["train_parquet"].is_file()
    return result


def show_config(cfg: dict) -> None:
    checks = environment_checks(cfg)
    print(json.dumps({
        "backend": "Unsloth + TRL GRPOTrainer + colocated vLLM",
        "config": str(cfg["_source"]),
        "model": str(cfg["paths"]["model"]),
        "train_parquet": str(cfg["paths"]["train_parquet"]),
        "output_dir": str(cfg["paths"]["output_dir"]),
        "batch_plan": batch_plan(cfg),
        "epochs": cfg["training"]["epochs"],
        "loss_type": cfg["training"]["loss_type"],
        "beta": cfg["training"]["kl_coef"],
        "disable_flashinfer": cfg["rollout"]["disable_flashinfer"],
        "kv_cache_dtype": cfg["rollout"]["kv_cache_dtype"],
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
    print(f"[da-grpo] Judge {cfg_j['model']} HTTP OK", flush=True)


def load_dataset_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError("Empty DA-GRPO Parquet")
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
    os.environ["UNSLOTH_VLLM_STANDBY"] = "1" if cfg["rollout"]["standby"] else "0"


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
    print("[da-grpo] Disabled vLLM FlashInfer autotune in all optimization presets", flush=True)


def build_grpo_kwargs(cfg: dict) -> dict:
    """Pure constructor argument mapper; tested without importing CUDA libraries."""
    plan = batch_plan(cfg)
    rollout, train, paths = cfg["rollout"], cfg["training"], cfg["paths"]
    return {
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
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": rollout["gpu_memory_utilization"],
        "vllm_max_model_length": cfg["model"]["max_seq_length"],
        "vllm_enable_sleep_mode": True,
        "vllm_tensor_parallel_size": 1,
        # TRL's optional train/inference IS correction is NOT part of O1;
        # disable it for closer parity with original O1/verl sampling.
        "vllm_importance_sampling_correction": False,
    }


def train(cfg: dict, *, resume: str | None = None) -> None:
    configure_runtime(cfg)
    print("[da-grpo] Unsloth FlashInfer disabled for SM120; using vLLM native compatible attention/sampler", flush=True)
    import torch
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Expected a single NVIDIA CUDA GPU with BF16 support")

    # IMPORTANT: Unsloth MUST be imported before TRL so it can patch GRPOTrainer.
    from unsloth import FastLanguageModel
    # Must run after Unsloth initializes its patches, but before it creates
    # the vLLM EngineArgs/VllmConfig for FastLanguageModel.from_pretrained().
    disable_vllm_flashinfer_autotune()
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    from reward_adapter import make_reward_function

    rows = load_dataset_rows(cfg["paths"]["train_parquet"])
    print(f"[da-grpo] {len(rows)} offline O1 Main-Agent turns; batch plan: {batch_plan(cfg)}", flush=True)
    model_cfg, rollout = cfg["model"], cfg["rollout"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(cfg["paths"]["model"]),
        max_seq_length=model_cfg["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=model_cfg["load_in_4bit"],
        fast_inference=True,
        max_lora_rank=model_cfg["lora_rank"],
        gpu_memory_utilization=rollout["gpu_memory_utilization"],
        max_num_seqs=rollout["max_num_seqs"],
        # FlashAttention on SM120 rejects FP8 KV Cache. Keep the native
        # model dtype (BF16) by explicitly disabling float8 KV in Unsloth.
        # Attention selection is handled by UNSLOTH_VLLM_NO_FLASHINFER.
        float8_kv_cache=False,
        enforce_eager=rollout["enforce_eager"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    check_prompt_lengths(rows, tokenizer, rollout["max_prompt_tokens"])

    model = FastLanguageModel.get_peft_model(
        model,
        r=model_cfg["lora_rank"],
        lora_alpha=model_cfg["lora_alpha"],
        lora_dropout=model_cfg["lora_dropout"],
        target_modules=model_cfg["lora_target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["training"]["seed"],
    )
    args = GRPOConfig(**build_grpo_kwargs(cfg))
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(rows),
        reward_funcs=[make_reward_function(cfg["judge"]["workers"])],
        args=args,
    )
    if not getattr(trainer, "use_vllm", False):
        raise RuntimeError("Unsloth/TRL did not activate vLLM rollout")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(cfg["paths"]["output_dir"] / "final_adapter"))
    tokenizer.save_pretrained(str(cfg["paths"]["output_dir"] / "final_adapter"))
    print("[da-grpo] Saved final LoRA adapter", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Inspect config without GPU or network")
    parser.add_argument("--check", action="store_true", help="Check required packages and files")
    parser.add_argument("--test-judge", action="store_true", help="Probe API without loading a GPU model")
    parser.add_argument("--max-steps", type=int, help="Override training steps for smoke test")
    parser.add_argument("--max-completion-tokens", type=int, help="Override generated response cap")
    parser.add_argument("--output-dir", type=Path, help="Independent experiment output")
    parser.add_argument("--resume", type=Path, help="TRL checkpoint path (old custom checkpoints unsupported)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.max_steps is not None:
        if args.max_steps < 1:
            parser.error("--max-steps must be positive")
        cfg["training"]["max_steps"] = args.max_steps
    if args.max_completion_tokens is not None:
        if not 1 <= args.max_completion_tokens <= cfg["model"]["max_seq_length"]:
            parser.error("--max-completion-tokens out of range")
        cfg["rollout"]["max_completion_tokens"] = args.max_completion_tokens
        if cfg["rollout"]["max_prompt_tokens"] + args.max_completion_tokens > cfg["model"]["max_seq_length"]:
            parser.error("Prompt + completion exceeds model context")
    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = repo_path(args.output_dir)
    if args.dry_run:
        show_config(cfg)
        return 0
    checks = environment_checks(cfg)
    if args.check:
        show_config(cfg)
        if not all(checks.values()):
            raise RuntimeError("Missing required packages/files: " +
                               ", ".join(k for k, ok in checks.items() if not ok))
        return 0
    if not args.test_judge and not all(checks.values()):
        raise RuntimeError("Install the new Unsloth/TRL requirements and verify local model/data first: " +
                           ", ".join(k for k, ok in checks.items() if not ok))
    key = judge_key(cfg["judge"]["api_key_env"])
    setup_judge(cfg, key)
    probe_judge(cfg, key)
    if args.test_judge:
        return 0
    train(cfg, resume=str(repo_path(args.resume)) if args.resume else None)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[da-grpo] ERROR: {exc}", file=sys.stderr)
        # Unsloth sometimes wraps worker exceptions and prints only their
        # short message; full Python traceback is useful for debugging.
        if os.environ.get("MAESTRO_GRPO_TRACEBACK") == "1":
            import traceback
            traceback.print_exc()
        raise SystemExit(1)
