#!/usr/bin/env python3
"""Single-GPU Qwen3-8B LoRA GRPO for MAESTRO's O1-equivalent Main turns.

Only policy GRPO + KL + the original four-dimension O1 reward. No verl,
future latents, expert CE, CHORD, consequence heads or external Sub-Agents.
One observation -> n candidates -> group-normalized advantages -> streamed
single-candidate backward -> ONE optimizer update.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CONFIG = HERE / "config.yaml"


def repo_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("Empty path")
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration not found: {source}")
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    sections = {"version", "paths", "judge", "model", "rollout", "training"}
    if not isinstance(cfg, dict) or set(cfg) != sections or cfg["version"] != 1:
        raise ValueError("Invalid single-GPU config sections/version")
    for name in sections - {"version"}:
        if not isinstance(cfg[name], dict):
            raise ValueError(f"{name} must be a mapping")
    if set(cfg["paths"]) != {"model", "train_parquet", "output_dir"}:
        raise ValueError("paths must specify model, train_parquet, output_dir")
    for key in cfg["paths"]:
        cfg["paths"][key] = repo_path(cfg["paths"][key])

    judge = cfg["judge"]
    if set(judge) != {"registry", "model", "api_key_env", "enable_thinking", "workers"}:
        raise ValueError("Invalid judge configuration keys")
    judge["registry"] = repo_path(judge["registry"])
    if not judge["registry"].is_file():
        raise FileNotFoundError("Shared model registry not found")
    if not isinstance(judge["enable_thinking"], bool):
        raise ValueError("judge.enable_thinking must be a boolean")
    if not isinstance(judge["workers"], int) or not 1 <= judge["workers"] <= 16:
        raise ValueError("judge.workers must be within 1..16")
    if not isinstance(judge["api_key_env"], str) or not judge["api_key_env"].isidentifier():
        raise ValueError("Invalid Judge API key environment variable")
    registry = yaml.safe_load(judge["registry"].read_text(encoding="utf-8"))
    selection = registry.get("models", {}).get(judge["model"])
    if not isinstance(selection, dict) or selection.get("backend") != "api":
        raise ValueError("Judge must reference a remote model in config/model_config.yaml")
    if selection.get("api_type") != "openai":
        raise ValueError("Judge must use an OpenAI-compatible provider")
    judge["base_url"] = selection["base_url"]
    if not judge["base_url"].startswith("https://"):
        raise ValueError("Judge base URL must use HTTPS")

    model = cfg["model"]
    model_fields = {"dtype", "gradient_checkpointing", "lora_rank", "lora_alpha",
                    "lora_dropout", "lora_target_modules", "logits_chunk_size"}
    if set(model) != model_fields:
        raise ValueError("Invalid model configuration keys")
    if model["dtype"] != "bfloat16" or model["gradient_checkpointing"] is not True:
        raise ValueError("Single 96GB profile requires BF16 and gradient checkpointing")
    if not isinstance(model["lora_target_modules"], list) or not model["lora_target_modules"]:
        raise ValueError("Nonempty LoRA module list required")
    for name in ("lora_rank", "lora_alpha", "logits_chunk_size"):
        if type(model[name]) is not int or model[name] <= 0:
            raise ValueError(f"model.{name} must be positive int")
    if type(model["lora_dropout"]) not in (float, int) or not 0 <= model["lora_dropout"] < 1:
        raise ValueError("model.lora_dropout outside [0,1)")

    rollout = cfg["rollout"]
    required_rollout = {"backend", "group_size", "temperature", "top_p",
                        "max_prompt_tokens", "max_response_tokens", "max_model_len",
                        "gpu_memory_utilization", "kv_cache_dtype", "enable_sleep_mode",
                        "enable_prefix_caching", "max_num_seqs", "sync_dir"}
    if set(rollout) != required_rollout:
        raise ValueError("Invalid rollout configuration keys")
    if rollout["backend"] != "vllm" or rollout["enable_sleep_mode"] is not True:
        raise ValueError("Single-card profile requires vLLM sleep mode")
    if rollout["temperature"] != 1.0 or rollout["top_p"] != 1.0:
        raise ValueError("GRPO importance ratios require temperature=top_p=1")
    for name in ("group_size", "max_prompt_tokens", "max_response_tokens",
                 "max_model_len", "max_num_seqs"):
        if type(rollout[name]) is not int or rollout[name] < (2 if name == "group_size" else 1):
            raise ValueError(f"rollout.{name} invalid")
    if rollout["max_prompt_tokens"] + rollout["max_response_tokens"] > rollout["max_model_len"]:
        raise ValueError("Prompt plus response limit exceeds vLLM max_model_len")
    if not 0 < float(rollout["gpu_memory_utilization"]) <= 1:
        raise ValueError("rollout.gpu_memory_utilization invalid")
    rollout["sync_dir"] = repo_path(rollout["sync_dir"])

    train = cfg["training"]
    required_train = {"epochs", "max_updates", "learning_rate", "weight_decay",
                      "clip_range", "kl_coef", "micro_batch_size", "max_tokens_per_gpu",
                      "max_grad_norm", "warmup_ratio", "save_every", "log_every",
                      "seed", "min_gpu_memory_gib"}
    if set(train) != required_train:
        raise ValueError("Invalid training configuration keys")
    for name in ("epochs", "micro_batch_size", "max_tokens_per_gpu",
                 "save_every", "log_every"):
        if type(train[name]) is not int or train[name] < 1:
            raise ValueError(f"training.{name} must be positive integer")
    for name in ("learning_rate", "max_grad_norm", "min_gpu_memory_gib"):
        if type(train[name]) not in (int, float) or train[name] <= 0:
            raise ValueError(f"training.{name} must be positive")
    if train["micro_batch_size"] != 1:
        raise ValueError("Exact streamed candidate backward requires micro_batch_size: 1")
    if train["max_updates"] is not None and (type(train["max_updates"]) is not int
                                             or train["max_updates"] < 1):
        raise ValueError("training.max_updates must be null or positive integer")
    if rollout["max_prompt_tokens"] + rollout["max_response_tokens"] > train["max_tokens_per_gpu"]:
        raise ValueError("Prompt + response exceeds backward token budget")
    if not 0 <= train["clip_range"] < 1 or train["kl_coef"] < 0:
        raise ValueError("GRPO clip_range or KL coefficient invalid")
    if not 0 <= train["warmup_ratio"] < 1:
        raise ValueError("warmup_ratio invalid")
    cfg["_source"] = source
    return cfg


def local_checks(cfg: dict) -> dict:
    import importlib.util
    paths = cfg["paths"]
    checks = {
        "model": paths["model"].is_dir(),
        "train.parquet": paths["train_parquet"].is_file(),
        "python_torch": importlib.util.find_spec("torch") is not None,
        "python_peft": importlib.util.find_spec("peft") is not None,
        "python_transformers": importlib.util.find_spec("transformers") is not None,
        "python_vllm": importlib.util.find_spec("vllm") is not None,
        "python_pyarrow": importlib.util.find_spec("pyarrow") is not None,
    }
    info: dict[str, Any] = {"count": None, "memory_gib": None}
    if checks["python_torch"]:
        import torch
        count = torch.cuda.device_count()
        info["count"] = count
        if count > 0:
            info["memory_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )
        checks["one_cuda_gpu"] = count == 1
        checks["bf16_supported"] = count > 0 and torch.cuda.is_bf16_supported()
        checks["gpu_memory"] = (
            count == 1 and info["memory_gib"] >= cfg["training"]["min_gpu_memory_gib"]
        )
    else:
        checks.update(one_cuda_gpu=False, bf16_supported=False, gpu_memory=False)
    return {"checks": checks, "hardware": info}


def show_config(cfg: dict) -> None:
    state = local_checks(cfg)
    print(json.dumps({
        "config": str(cfg["_source"]),
        "model": str(cfg["paths"]["model"]),
        "train_parquet": str(cfg["paths"]["train_parquet"]),
        "output_dir": str(cfg["paths"]["output_dir"]),
        "judge": {
            "model": cfg["judge"]["model"],
            "api_base": cfg["judge"]["base_url"],
            "api_key_available": bool(os.environ.get(cfg["judge"]["api_key_env"])),
            "enable_thinking": cfg["judge"]["enable_thinking"],
        },
        "mode": "single_gpu_lora_grpo",
        "backend": "torch_peft_vllm_no_verl",
        "group_size": cfg["rollout"]["group_size"],
        "max_prompt_tokens": cfg["rollout"]["max_prompt_tokens"],
        "max_response_tokens": cfg["rollout"]["max_response_tokens"],
        "hardware": state["hardware"],
        "checks": state["checks"],
    }, indent=2, ensure_ascii=False))


def key_from_tty(name: str) -> str:
    key = (os.environ.get(name) or "").strip()
    if key:
        return key
    if not sys.stdin.isatty():
        raise RuntimeError(f"Set {name} securely or use an interactive terminal")
    key = getpass.getpass(f"Enter {name} (hidden, not saved): ").strip()
    if not key:
        raise ValueError("API key cannot be empty")
    return key


def setup_judge_env(cfg: dict, key: str) -> None:
    judge = cfg["judge"]
    os.environ["OPENAI_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = judge["base_url"]
    os.environ["LLM_JUDGE_MODEL"] = judge["model"]
    os.environ["LLM_JUDGE_ENABLE_THINKING"] = str(judge["enable_thinking"]).lower()


def probe_judge(cfg: dict, key: str) -> None:
    """Fail fast on HTTP 401 before GPU allocation and paid training."""
    import requests
    judge = cfg["judge"]
    url = judge["base_url"].rstrip("/") + "/chat/completions"
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": judge["model"],
                "messages": [{"role": "user", "content": "Reply only: OK"}],
                "temperature": 0,
                "max_tokens": 32,
                "enable_thinking": judge["enable_thinking"],
            },
            timeout=45,
        )
        response.raise_for_status()
        value = response.json()["choices"][0]["message"]["content"]
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError("Judge returned an empty response")
        print(f"[da-grpo] Judge {judge['model']} HTTP OK")
    except requests.HTTPError as exc:
        code = exc.response.status_code if exc.response is not None else "unknown"
        raise RuntimeError(
            f"Judge HTTP {code}. Verify DASHSCOPE_API_KEY and the region of "
            f"{judge['base_url']}; training has NOT started."
        ) from exc


def read_examples(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError("Empty training Parquet")
    for index, row in enumerate(rows):
        msg = row.get("prompt")
        info = row.get("extra_info")
        if not isinstance(msg, list) or not msg or not isinstance(info, dict):
            raise ValueError(f"Malformed dataset prompt/extra_info row {index}")
        if any(entry.get("role") != "user" for entry in msg):
            raise ValueError("Only O1 Main-Agent user prompts may be model-visible")
        if not info.get("question") or not info.get("has_expert"):
            raise ValueError(f"Missing O1 Judge metadata in row {index}")
        if not isinstance(row.get("reward_model"), dict):
            raise ValueError(f"Missing ground-truth metadata in row {index}")
    return rows


def score_group(candidates: list[dict], record: dict, workers: int) -> list[dict]:
    from reward_fn import compute_score

    def score(candidate):
        return compute_score(
            record["data_source"],
            candidate["text"],
            record["reward_model"].get("ground_truth", ""),
            record["extra_info"],
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        scores = list(pool.map(score, candidates))
    if any(not 0 <= float(item["score"]) <= 1 for item in scores):
        raise ValueError("Original O1 reward produced an out-of-range value")
    return scores


def learning_rate_schedule(optimizer, total_updates: int, warmup_ratio: float):
    import torch
    warmup = min(total_updates, int(total_updates * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total_updates - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


def save_state(
    output: Path, model, tokenizer, optimizer, scheduler,
    *, step: int, epoch: int, next_position: int,
) -> None:
    """Adapter + optimizer/scheduler, to resume at exactly the next observation."""
    import torch
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"checkpoint-{step:06d}"
    temp = output / f".checkpoint-{step:06d}.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    model.save_pretrained(temp)
    tokenizer.save_pretrained(temp)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "next_position": next_position,
        "total_updates": scheduler.total_steps if hasattr(scheduler, "total_steps") else None,
    }, temp / "train_state.pt")
    (temp / "metadata.json").write_text(
        json.dumps({"step": step, "epoch": epoch, "next_position": next_position}, indent=2),
        encoding="utf-8",
    )
    if target.exists():
        shutil.rmtree(target)
    temp.rename(target)
    print(f"[da-grpo] saved adapter checkpoint: {target}", flush=True)


def train(cfg: dict, args) -> None:
    import torch
    from policy import action_log_probs, load_policy, prompt_ids, reference_log_probs
    from losses import clipped_grpo_loss, group_advantages, low_var_kl
    from rollout import ColocatedRollout

    rows = read_examples(cfg["paths"]["train_parquet"])
    model_cfg = cfg["model"]
    rollout_cfg = cfg["rollout"]
    tr = cfg["training"]
    random.seed(tr["seed"])
    torch.manual_seed(tr["seed"])
    torch.cuda.manual_seed_all(tr["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    resume = repo_path(args.resume) if args.resume else None
    model, tokenizer = load_policy(
        str(cfg["paths"]["model"]), model_cfg,
        resume_adapter=str(resume) if resume else None,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        params, lr=tr["learning_rate"], weight_decay=tr["weight_decay"],
    )
    update_limit = (
        args.max_updates if args.max_updates is not None
        else tr["max_updates"] or len(rows) * tr["epochs"]
    )
    if update_limit < 1:
        raise ValueError("max_updates must be positive")
    scheduler = learning_rate_schedule(optimizer, update_limit, tr["warmup_ratio"])

    step = 0
    start_epoch = 0
    start_position = 0
    if resume:
        # Only load self-generated local trusted checkpoint files.
        state = torch.load(resume / "train_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = int(state["step"])
        start_epoch = int(state["epoch"])
        start_position = int(state["next_position"])

    vllm_cfg = {**rollout_cfg, "lora_rank": model_cfg["lora_rank"]}
    engine = ColocatedRollout(
        str(cfg["paths"]["model"]), tokenizer, vllm_cfg, rollout_cfg["sync_dir"],
    )
    output_dir = cfg["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    skipped = 0
    last_position = start_position
    last_epoch = start_epoch

    try:
        for epoch in range(start_epoch, tr["epochs"]):
            indexes = list(range(len(rows)))
            random.Random(tr["seed"] + epoch).shuffle(indexes)
            first = start_position if epoch == start_epoch else 0
            for position in range(first, len(indexes)):
                if step >= update_limit:
                    break
                record = rows[indexes[position]]
                ids = prompt_ids(tokenizer, record["prompt"], rollout_cfg["max_prompt_tokens"])
                response_cap = (
                    args.max_response_tokens if args.max_response_tokens is not None
                    else rollout_cfg["max_response_tokens"]
                )
                if len(ids) + response_cap > rollout_cfg["max_model_len"]:
                    raise ValueError("Prompt + response exceeds vLLM context")
                if len(ids) + response_cap > tr["max_tokens_per_gpu"]:
                    raise ValueError("Prompt + response exceeds training token budget")

                started = time.monotonic()
                model.eval()
                candidates = engine.sample(
                    model, ids,
                    n=rollout_cfg["group_size"],
                    max_tokens=response_cap,
                    temperature=rollout_cfg["temperature"],
                    top_p=rollout_cfg["top_p"],
                )
                rollout_seconds = time.monotonic() - started
                scores = score_group(candidates, record, cfg["judge"]["workers"])
                utilities = [float(value["score"]) for value in scores]
                advantages = group_advantages(utilities)
                sample_id = record["extra_info"].get("sample_id", "")
                last_position, last_epoch = position + 1, epoch
                if not any(abs(value) > 1e-6 for value in advantages):
                    skipped += 1
                    print(json.dumps({
                        "skip_uninformative_group": skipped,
                        "sample_id": sample_id,
                        "rewards": utilities,
                    }), flush=True)
                    continue

                # The GROUP denominator is common to every streamed candidate.
                # No optimizer step (nor scheduler step) until all G backprops end.
                global_action_tokens = sum(len(x["action_input_ids"]) for x in candidates)
                optimizer.zero_grad(set_to_none=True)
                model.train()
                loss_total = 0.0
                kl_total = 0.0
                logprob_mae = 0.0
                for candidate, advantage in zip(candidates, advantages):
                    actions = candidate["action_input_ids"]
                    if len(ids) + len(actions) > tr["max_tokens_per_gpu"]:
                        raise ValueError("Generated candidate exceeds backward token budget")
                    ref = reference_log_probs(
                        model, ids, actions,
                        logits_chunk_size=model_cfg["logits_chunk_size"],
                    )
                    current = action_log_probs(
                        model, ids, actions,
                        logits_chunk_size=model_cfg["logits_chunk_size"],
                    )
                    old = torch.tensor(
                        candidate["old_log_probs"], dtype=torch.float32,
                        device=current.device,
                    )
                    if current.shape != old.shape or ref.shape != current.shape:
                        raise ValueError("Candidate/new/reference logprobs misaligned")
                    surrogate = clipped_grpo_loss(
                        current, old, advantage,
                        clip_range=tr["clip_range"],
                        normalization_tokens=global_action_tokens,
                    )
                    penalty = low_var_kl(
                        current, ref,
                        normalization_tokens=global_action_tokens,
                    )
                    loss = surrogate + tr["kl_coef"] * penalty
                    loss.backward()
                    loss_total += float(loss.detach())
                    kl_total += float(penalty.detach())
                    logprob_mae += float((current.detach() - old).abs().sum())
                    del ref, current, old, surrogate, penalty, loss
                torch.nn.utils.clip_grad_norm_(params, tr["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                step += 1
                result = {
                    "step": step, "epoch": epoch,
                    "sample_id": sample_id, "loss": loss_total,
                    "kl": kl_total, "mean_reward": sum(utilities) / len(utilities),
                    "min_reward": min(utilities), "max_reward": max(utilities),
                    "group_size": len(candidates),
                    "response_tokens_mean": global_action_tokens / len(candidates),
                    "json_completed_rate": sum(bool(c["json_complete"]) for c in candidates) / len(candidates),
                    "old_new_logprob_mae": logprob_mae / global_action_tokens,
                    "rollout_seconds": round(rollout_seconds, 2),
                    "total_seconds": round(time.monotonic() - started, 2),
                    "lr": scheduler.get_last_lr()[0],
                    "skipped_uninformative_groups": skipped,
                }
                with metrics_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(result, ensure_ascii=False) + "\n")
                if step % tr["log_every"] == 0:
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                torch.cuda.empty_cache()
                if step % tr["save_every"] == 0:
                    save_state(
                        output_dir, model, tokenizer, optimizer, scheduler,
                        step=step, epoch=epoch, next_position=position + 1,
                    )
            if step >= update_limit:
                break
            start_position = 0
    finally:
        engine.shutdown()
        # The last update must be persisted even if its scheduled save
        # interval has not been reached.
        if step > 0:
            save_state(
                output_dir, model, tokenizer, optimizer, scheduler,
                step=step, epoch=last_epoch, next_position=last_position,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--test-judge", action="store_true")
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--max-response-tokens", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.max_updates is not None and args.max_updates < 1:
        parser.error("--max-updates must be >=1")
    if args.max_response_tokens is not None and args.max_response_tokens < 1:
        parser.error("--max-response-tokens must be >=1")
    cfg = load_config(args.config)
    if args.dry_run:
        show_config(cfg)
        return 0
    if args.check or not args.test_judge:
        state = local_checks(cfg)
        show_config(cfg)
        missing = [key for key, ok in state["checks"].items() if not ok]
        if missing:
            raise RuntimeError("Single GPU preflight failed: " + ", ".join(missing))
        if args.check:
            print("[da-grpo] local single-GPU checks passed (no training started)")
            return 0
    key = key_from_tty(cfg["judge"]["api_key_env"])
    setup_judge_env(cfg, key)
    probe_judge(cfg, key)
    if args.test_judge:
        return 0
    train(cfg, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[da-grpo] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
