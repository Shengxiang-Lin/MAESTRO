"""Legacy reference trainer: risk-conditioned KV + LoRA-only grouped policy optimization.

Default reward is a deterministic CURRENT-ROUND orchestration score using
private, pre-action Stage-4 DAG/evidence state. It does not judge teacher
text similarity or require a future environment rollout. External reward
hooks remain available for alternate research conditions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from train_maestro_8b.risk.model import WindowRiskHead
from train_maestro_8b.policy.model import RiskAwarePolicy
from train_maestro_8b.data.contracts import iter_jsonl
from train_maestro_8b.policy.rewards import resolve_reward


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--risk_checkpoint", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--reward_callable", help=(
        "override deterministic default with module.path:function(candidate_text, record)"
    ))
    p.add_argument("--reward_debug_jsonl", help=(
        "optional private diagnostics of sampled actions and reward components"
    ))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--max_prompts", type=int, default=0)
    p.add_argument("--max_length", type=int, default=16384)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--steer_last_n", type=int, default=8)
    p.add_argument("--steer_strength", type=float, default=0.1)
    p.add_argument("--steer_threshold", type=float, default=0.5)
    p.add_argument("--clip_epsilon", type=float, default=0.2)
    p.add_argument("--sft_warmup", type=int, default=0,
                   help="teacher imitation passes before GRPO (only LoRA trained)")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def build_engine(args):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    cfg = json.loads((Path(args.risk_checkpoint) / "risk_config.json").read_text())
    if cfg.get("schema") != "maestro.frozen_window_risk.v1":
        raise ValueError("Stage1 risk checkpoint incompatible; train the frozen-head version")
    if cfg["base_model"] != args.model:
        raise ValueError("Risk Head and policy MUST use identical base model identifiers")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = {"dtype": dtype, "attn_implementation": "sdpa"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise ValueError("4-bit LoRA requires CUDA")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": "cuda:0"}
    base = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if not args.load_in_4bit:
        base.to("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        base = prepare_model_for_kbit_training(base)
    base.config.use_cache = True
    policy = get_peft_model(base, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    if not any("lora_" in name and p.requires_grad for name, p in policy.named_parameters()):
        raise AssertionError("no trainable LoRA")
    if any("lora_" not in name and p.requires_grad for name, p in policy.named_parameters()):
        raise AssertionError("non-LoRA policy weights must remain frozen")
    head = WindowRiskHead(cfg["hidden_size"], cfg["head_size"], cfg["top_k"])
    head.load_state_dict(torch.load(
        Path(args.risk_checkpoint) / "risk_head.pt",
        map_location="cpu", weights_only=True,
    ))
    head.to(next(policy.parameters()).device).eval().requires_grad_(False)
    return RiskAwarePolicy(
        policy, tok, head, cfg, max_length=args.max_length,
        last_n_layers=args.steer_last_n, strength=args.steer_strength,
        threshold=args.steer_threshold,
    )


def reference_ids(engine, record):
    target = json.dumps(record["reference_action"], ensure_ascii=False,
                        separators=(",", ":"))
    if engine.tokenizer.eos_token:
        target += engine.tokenizer.eos_token
    return torch.tensor(engine.tokenizer.encode(
        target, add_special_tokens=False,
    ), dtype=torch.long, device=engine.device)


def group_advantages(rewards):
    values = torch.tensor(rewards, dtype=torch.float32)
    if values.numel() < 2:
        raise ValueError("GRPO needs two or more candidates per question")
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-4)


def main():
    args = parse_args()
    if (args.epochs < 1 or args.num_generations < 2 or
            args.temperature <= 0 or args.max_new_tokens < 1 or
            not 0 < args.clip_epsilon < 1 or args.sft_warmup < 0):
        raise ValueError("invalid training parameters")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = build_engine(args)
    reward_fn = resolve_reward(args.reward_callable)
    data = list(iter_jsonl(args.train_file))
    if args.max_prompts:
        data = data[:args.max_prompts]
    if not data:
        raise ValueError("empty GRPO dataset")
    trainable = [p for p in engine.policy.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr)
    engine.policy.train()
    for epoch in range(args.sft_warmup):
        random.shuffle(data)
        for row in data:
            prefix = engine.prefill(row["prompt"], row["reports"])
            target = reference_ids(engine, row)
            optimizer.zero_grad(set_to_none=True)
            loss = -engine.log_probs(prefix, target).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        print(f"sft_warmup_epoch={epoch+1} complete", flush=True)
    updates = 0
    debug_path = Path(args.reward_debug_jsonl) if args.reward_debug_jsonl else None
    if debug_path:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
    from environment.untrusted_orchestration.orchestration_reward import score_decision
    skipped_groups = 0
    for epoch in range(args.epochs):
        random.shuffle(data)
        for row in data:
            prefix = engine.prefill(row["prompt"], row["reports"])
            samples, rewards, details = [], [], []
            for _ in range(args.num_generations):
                ids, old = engine.sample(
                    prefix, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                candidate = engine.decode(ids)
                info = (
                    score_decision(candidate, row["decision_state"])
                    if args.reward_callable is None else None
                )
                reward = float(
                    info["reward"] if info is not None
                    else reward_fn(candidate, row)
                )
                if not (-1e8 < reward < 1e8):
                    raise ValueError("invalid reward; must be finite and bounded")
                samples.append((ids, old))
                rewards.append(reward)
                if info is not None:
                    details.append(info)
                    if debug_path:
                        with debug_path.open("a", encoding="utf-8") as debug:
                            debug.write(json.dumps({
                                "sample_id": row["sample_id"],
                                "epoch": epoch + 1,
                                "score": info,
                                "candidate": candidate,
                            }, ensure_ascii=False) + "\n")
            advantage = group_advantages(rewards).to(engine.device)
            if advantage.abs().max().item() < 1e-6:
                skipped_groups += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i, (ids, old) in enumerate(samples):
                new = engine.log_probs(prefix, ids, temperature=args.temperature)
                ratio = (new - old).clamp(-20, 20).exp()
                pg = -torch.minimum(
                    ratio * advantage[i],
                    ratio.clamp(1 - args.clip_epsilon,
                                1 + args.clip_epsilon) * advantage[i],
                ).mean()
                (pg / args.num_generations).backward()
                losses.append(float(pg.detach()))
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            updates += 1
            if updates % args.log_every == 0:
                msg = (f"epoch={epoch+1} updates={updates} "
                       f"reward={sum(rewards)/len(rewards):.4f} "
                       f"pg_loss={sum(losses)/len(losses):.4f} "
                       f"zero_adv_groups={skipped_groups}")
                if details:
                    for metric in ("coverage", "priority", "parallel", "nonredundancy"):
                        msg += f" {metric}={sum(x[metric] for x in details)/len(details):.3f}"
                    msg += f" invalid={sum(not x['valid'] for x in details)}/{len(details)}"
                print(msg, flush=True)
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    engine.policy.save_pretrained(str(dest))
    engine.tokenizer.save_pretrained(str(dest))
    (dest / "risk_policy_config.json").write_text(json.dumps({
        "schema": "maestro.risk_steered_lora.v1",
        "risk_checkpoint": str(Path(args.risk_checkpoint).resolve()),
        "steer_last_n": args.steer_last_n,
        "steer_strength": args.steer_strength,
        "steer_threshold": args.steer_threshold,
        "anchor_ids": engine.anchor_ids,
        "reward_type": "custom" if args.reward_callable else "deterministic_orchestration",
        "grpo_updates": updates,
        "zero_advantage_groups": skipped_groups,
    }, indent=2), encoding="utf-8")
    print(f"saved LoRA adapter to {dest}", flush=True)


if __name__ == "__main__":
    main()
