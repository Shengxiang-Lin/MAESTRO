#!/usr/bin/env python3
"""One-GPU MAESTRO SFT (Unsloth + TRL); offline O1 decision imitation.

No vLLM, judge, reference model, online rollouts, or policy-gradient loop.
The TRL conversational prompt-completion dataset supervises ONLY completions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "config.yaml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def repo_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("version") != 1:
        raise ValueError("DA-SFT expects config version 1")
    if set(cfg) != {"version", "paths", "model", "training"}:
        raise ValueError("Unexpected config sections")
    if any(not isinstance(cfg[k], dict) for k in ("paths", "model", "training")):
        raise ValueError("paths/model/training must be mappings")
    paths = cfg["paths"]
    if set(paths) != {"model", "train_parquet", "eval_parquet", "output_dir"}:
        raise ValueError("Invalid paths")
    for key in paths:
        paths[key] = repo_path(paths[key])
    model = cfg["model"]
    if set(model) != {
        "dtype", "max_seq_length", "max_prompt_tokens", "load_in_4bit",
        "lora_rank", "lora_alpha", "lora_dropout", "lora_target_modules",
        "eos_token",
    }:
        raise ValueError("Invalid model keys")
    if model["dtype"] != "bfloat16" or type(model["load_in_4bit"]) is not bool:
        raise ValueError("Expected BF16 and boolean load_in_4bit")
    if type(model["max_seq_length"]) is not int or model["max_seq_length"] <= 0:
        raise ValueError("Invalid max_seq_length")
    if not 0 < model["max_prompt_tokens"] < model["max_seq_length"]:
        raise ValueError("max_prompt_tokens must be below full context")
    if any(type(model[k]) is not int or model[k] <= 0 for k in ("lora_rank", "lora_alpha")):
        raise ValueError("Invalid LoRA rank/alpha")
    if not 0 <= model["lora_dropout"] < 1 or not model["lora_target_modules"]:
        raise ValueError("Invalid LoRA modules/dropout")
    if model["eos_token"] != "<|im_end|>":
        raise ValueError("Qwen3 O1 SFT must terminate assistant with <|im_end|>")
    t = cfg["training"]
    if set(t) != {
        "epochs", "max_steps", "learning_rate", "weight_decay", "warmup_ratio",
        "per_device_train_batch_size", "gradient_accumulation_steps", "optim",
        "gradient_checkpointing", "packing", "bf16",
        "save_steps", "logging_steps", "seed",
    }:
        raise ValueError("Invalid training keys")
    for key in ("epochs", "per_device_train_batch_size", "gradient_accumulation_steps", "save_steps", "logging_steps"):
        if type(t[key]) is not int or t[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if type(t["max_steps"]) is not int or (t["max_steps"] != -1 and t["max_steps"] <= 0):
        raise ValueError("Invalid max_steps")
    if t["optim"] != "adamw_8bit":
        raise ValueError("Expected 8-bit AdamW")
    if any(t[k] is not True for k in ("gradient_checkpointing", "bf16")):
        raise ValueError("gradient_checkpointing and BF16 must be on")
    if type(t["packing"]) is not bool:
        raise ValueError("packing must be boolean")
    if t["packing"]:
        raise ValueError("Packing needs a separate completion-mask/attention preflight; keep false")
    if t["learning_rate"] <= 0 or t["weight_decay"] < 0 or not 0 <= t["warmup_ratio"] < 1:
        raise ValueError("Invalid learning rate / decay / warmup")
    cfg["_source"] = str(Path(path).resolve())
    return cfg


def load_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(f"Missing SFT split: {path}")
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError(f"Empty SFT split: {path}")
    for i, row in enumerate(rows):
        if set(row) != {"prompt", "completion"}:
            raise ValueError(f"{path}:{i} must have only prompt and completion")
        prompt, completion = row["prompt"], row["completion"]
        if (
            not isinstance(prompt, list) or len(prompt) != 1
            or prompt[0].get("role") != "user"
            or not isinstance(prompt[0].get("content"), str)
            or not isinstance(completion, list) or len(completion) != 1
            or completion[0].get("role") != "assistant"
            or not isinstance(completion[0].get("content"), str)
        ):
            raise ValueError(f"{path}:{i}: invalid conversational prompt/completion")
        try:
            output = json.loads(completion[0]["content"])
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{i}: expert label is not JSON") from error
        if not isinstance(output, dict) or set(output) != {"action", "reasoning", "params"}:
            raise ValueError(f"{path}:{i}: incomplete original O1 action")
    return rows


def check_lengths(rows: list[dict], tokenizer: Any, max_prompt: int, max_total: int, split: str) -> dict:
    """Abort rather than silently truncating original history or expert JSON."""
    from dataset.tokenization import extract_token_ids

    prompts, totals, failures = [], [], []
    for index, row in enumerate(rows):
        prompt = extract_token_ids(tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True,
        ))
        full = extract_token_ids(tokenizer.apply_chat_template(
            row["prompt"] + row["completion"], tokenize=True, add_generation_prompt=False,
        ))
        prompts.append(len(prompt))
        totals.append(len(full))
        if len(prompt) > max_prompt or len(full) > max_total or len(full) <= len(prompt):
            failures.append((index, len(prompt), len(full)))
    if failures:
        raise ValueError(
            f"{split}: {len(failures)} invalid/overlong SFT sequences "
            f"(prompt_limit={max_prompt}, total_limit={max_total}); "
            f"examples={failures[:8]}"
        )
    return {
        "samples": len(rows),
        "max_prompt_tokens": max(prompts),
        "max_total_tokens": max(totals),
        "mean_prompt_tokens": round(sum(prompts) / len(prompts), 1),
        "mean_total_tokens": round(sum(totals) / len(totals), 1),
    }


def sft_kwargs(cfg: dict) -> dict:
    t = cfg["training"]
    return {
        "output_dir": str(cfg["paths"]["output_dir"]),
        "max_length": cfg["model"]["max_seq_length"],
        "eos_token": cfg["model"]["eos_token"],
        "completion_only_loss": True,
        "assistant_only_loss": False,
        "packing": False,
        "eval_packing": False,
        "per_device_train_batch_size": t["per_device_train_batch_size"],
        "gradient_accumulation_steps": t["gradient_accumulation_steps"],
        "per_device_eval_batch_size": 1,
        "learning_rate": t["learning_rate"],
        "weight_decay": t["weight_decay"],
        "warmup_ratio": t["warmup_ratio"],
        "lr_scheduler_type": "cosine",
        "num_train_epochs": t["epochs"],
        "max_steps": t["max_steps"],
        "optim": t["optim"],
        "gradient_checkpointing": t["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": t["bf16"],
        "fp16": False,
        "eval_strategy": "epoch",
        "save_strategy": "steps",
        "save_steps": t["save_steps"],
        "save_total_limit": 2,
        "logging_steps": t["logging_steps"],
        "seed": t["seed"],
        "report_to": "none",
        "dataloader_num_workers": 0,
        "remove_unused_columns": True,
    }


def check_mask(trainer: Any) -> dict:
    """Inspect TRL's ACTUAL tokenized dataset for prompt masking."""
    inspected = min(3, len(trainer.train_dataset))
    observed = []
    for i in range(inspected):
        example = trainer.train_dataset[i]
        mask = example.get("completion_mask")
        if mask is not None:
            mask = list(mask)
            if not mask or not any(x == 0 for x in mask) or not any(x == 1 for x in mask):
                raise RuntimeError(f"TRL completion mask missing prompt/answer split: {i}")
            observed.append({"prompt_masked": mask.count(0), "supervised": mask.count(1)})
        elif "labels" in example:
            labels = list(example["labels"])
            if not any(x == -100 for x in labels) or not any(x != -100 for x in labels):
                raise RuntimeError(f"TRL labels do not mask the user prompt: {i}")
            observed.append({
                "prompt_masked": labels.count(-100),
                "supervised": sum(x != -100 for x in labels),
            })
        else:
            raise RuntimeError("TRL did not expose labels or completion_mask; aborting unsafe SFT")
    return {"checked": inspected, "samples": observed}


def run(cfg: dict, *, resume: str | None = None, check_only: bool = False) -> None:
    rows_train = load_rows(cfg["paths"]["train_parquet"])
    rows_eval = load_rows(cfg["paths"]["eval_parquet"])
    # Dataset and loss-mask checks do NOT instantiate vLLM or a Judge.
    if check_only:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(cfg["paths"]["model"]), trust_remote_code=False)
        stats = {
            "train": check_lengths(rows_train, tokenizer, cfg["model"]["max_prompt_tokens"], cfg["model"]["max_seq_length"], "train"),
            "eval": check_lengths(rows_eval, tokenizer, cfg["model"]["max_prompt_tokens"], cfg["model"]["max_seq_length"], "eval"),
        }
        print(json.dumps(stats, indent=2))
        return

    import torch
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("DA-SFT expects one NVIDIA GPU supporting BF16")
    # Import Unsloth BEFORE TRL so SFT patching takes effect.
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    model_cfg = cfg["model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(cfg["paths"]["model"]),
        max_seq_length=model_cfg["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=model_cfg["load_in_4bit"],
        fast_inference=False,  # no colocated vLLM and no GRPO rollouts
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_tokens_to_ids(model_cfg["eos_token"]) is None:
        raise RuntimeError("Qwen3 <|im_end|> EOS token not found")
    stats = {
        "train": check_lengths(rows_train, tokenizer, model_cfg["max_prompt_tokens"], model_cfg["max_seq_length"], "train"),
        "eval": check_lengths(rows_eval, tokenizer, model_cfg["max_prompt_tokens"], model_cfg["max_seq_length"], "eval"),
    }
    print("[da-sft] token preflight: " + json.dumps(stats), flush=True)
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
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=SFTConfig(**sft_kwargs(cfg)),
        train_dataset=Dataset.from_list(rows_train),
        eval_dataset=Dataset.from_list(rows_eval),
    )
    print("[da-sft] actual TRL completion mask: " + json.dumps(check_mask(trainer)), flush=True)
    trainer.train(resume_from_checkpoint=resume)
    adapter_path = cfg["paths"]["output_dir"] / "final_adapter"
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"[da-sft] saved LoRA adapter: {adapter_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="Validate YAML and print plan only")
    parser.add_argument("--check", action="store_true", help="CPU-only data/tokenizer preflight")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--load-in-4bit", action="store_true", help="Switch BF16 LoRA to QLoRA")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.max_steps is not None:
        if args.max_steps <= 0:
            parser.error("--max-steps must be positive")
        cfg["training"]["max_steps"] = args.max_steps
    if args.output_dir is not None:
        cfg["paths"]["output_dir"] = repo_path(args.output_dir)
    if args.load_in_4bit:
        cfg["model"]["load_in_4bit"] = True
    if args.dry_run:
        print(json.dumps({
            "framework": "Unsloth + TRL SFTTrainer",
            "config": cfg["_source"],
            "model": str(cfg["paths"]["model"]),
            "train_parquet": str(cfg["paths"]["train_parquet"]),
            "eval_parquet": str(cfg["paths"]["eval_parquet"]),
            "output_dir": str(cfg["paths"]["output_dir"]),
            "load_in_4bit": cfg["model"]["load_in_4bit"],
            "trainer_args": sft_kwargs(cfg),
            "requirements": {
                pkg: importlib.util.find_spec(pkg) is not None
                for pkg in ("torch", "unsloth", "trl", "datasets", "pyarrow", "bitsandbytes")
            },
        }, indent=2, default=str))
        return 0
    run(cfg, resume=args.resume, check_only=args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
