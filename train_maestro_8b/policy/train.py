"""Unsloth + TRL Stage2 launcher for KV-steered, LoRA-only GRPO.

This intentionally uses Unsloth FastLanguageModel and TRL GRPOConfig with
HF Trainer/Accelerate optimization hooks, NOT stock TRL GRPOTrainer rollout:
the stock rollout/logprob path cannot apply the same risk-edited KV cache.
Optional vLLM attention kernels consume the same edited KV during sampling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--risk_checkpoint", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--backend", choices=("unsloth", "hf"), default="unsloth",
                   help="Unsloth QLoRA default; HF baseline for comparison/debugging")
    p.add_argument("--rollout_attention", choices=("sdpa", "vllm"), default="sdpa",
                   help="Attention kernel for RiskKV sampling; replay uses Transformers")
    p.add_argument("--reward_callable", help="optional module.path:function(candidate, record)")
    p.add_argument("--reward_debug_jsonl", default=None)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_generations", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
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
    p.add_argument("--sft_warmup", type=int, default=0)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--preflight_max_logprob_drift", type=float, default=0.01)
    p.add_argument("--skip_preflight", action="store_true",
                   help="disables risk-cache replay sanity check; not recommended")
    p.add_argument("--preflight_only", action="store_true",
                   help="load the GPU model and verify cache/logprob, then exit")
    p.add_argument("--seed", type=int, default=17)
    return p.parse_args(argv)


def validate_args(args):
    if (args.epochs <= 0 or args.num_generations < 2 or args.max_length < 1
            or args.max_new_tokens < 1 or args.lora_rank < 1
            or args.temperature <= 0 or not 0 < args.clip_epsilon < 1):
        raise ValueError("invalid GRPO/LoRA hyperparameters")
    if args.gradient_accumulation_steps < args.num_generations or (
        args.gradient_accumulation_steps % args.num_generations
    ):
        raise ValueError(
            "TRL GRPOConfig effective batch must be divisible by "
            "num_generations: set --gradient_accumulation_steps to "
            "num_generations or its multiple"
        )
    if not 0 <= args.steer_threshold <= 1 or args.steer_strength < 0:
        raise ValueError("invalid risk-steering settings")
    if args.backend == "unsloth" and not args.load_in_4bit:
        raise ValueError("initial Unsloth integration is NF4-only; remove --no_4bit")
    if args.sft_warmup < 0 or args.save_steps < 1 or args.logging_steps < 1:
        raise ValueError("invalid warmup/save/logging settings")
    if not 0 < args.preflight_max_logprob_drift < 1:
        raise ValueError("preflight log-probability drift bound must be in (0, 1)")
    if args.max_steps == 0:
        raise ValueError("--max_steps must be -1 (epoch) or positive")


def _sft_warmup(engine, dataset, args):
    """Optional reference-action warmup, always with the same risk KV path."""
    import torch
    from torch.optim import AdamW
    params = [p for p in engine.policy.parameters() if p.requires_grad]
    optim = AdamW(params, lr=args.lr)
    engine.policy.train()
    for epoch in range(args.sft_warmup):
        indices = list(range(len(dataset)))
        random.Random(args.seed + epoch).shuffle(indices)
        for j in indices:
            row = dataset[j]
            action = row.get("reference_action")
            if not isinstance(action, dict):
                raise ValueError("SFT warmup requires reference_action; GRPO does not")
            text = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
            if engine.tokenizer.eos_token:
                text += engine.tokenizer.eos_token
            target = torch.tensor(
                engine.tokenizer.encode(text, add_special_tokens=False),
                device=engine.device, dtype=torch.long,
            )
            if len(target) > args.max_new_tokens:
                # The warmup target must not be silently truncated.
                raise ValueError(
                    "SFT teacher target exceeds --max_new_tokens; "
                    "increase the limit or disable --sft_warmup"
                )
            prefix = engine.prefill(row["prompt"], row["reports"])
            optim.zero_grad(set_to_none=True)
            loss = -engine.log_probs(prefix, target).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
        print(f"sft_warmup_epoch={epoch+1} complete", flush=True)


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    # Import Unsloth FIRST, before importing transformers/trl/peft.
    # For the HF reference backend the module is never imported.
    if args.backend == "unsloth":
        from train_maestro_8b.policy.backend import load_risk_engine, check_risk_cache_contract
        engine = load_risk_engine(args)
    else:
        from train_maestro_8b.policy.reference import build_engine
        from train_maestro_8b.policy.backend import check_risk_cache_contract
        engine = build_engine(args)

    if args.rollout_attention == "vllm":
        from train_maestro_8b.policy.vllm_attention import register_vllm_attention
        engine.rollout_attention = register_vllm_attention()

    import torch
    from trl import GRPOConfig
    from train_maestro_8b.data.contracts import iter_jsonl
    from train_maestro_8b.policy.rewards import resolve_reward
    from train_maestro_8b.policy.trainer import MainDecisionDataset, make_trainer_class

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    records = list(iter_jsonl(args.train_file))
    if args.max_prompts > 0:
        records = records[:args.max_prompts]
    dataset = MainDecisionDataset(records)
    if not args.skip_preflight:
        probe = check_risk_cache_contract(
            engine, next((row for row in records if row["reports"]), records[0]),
            temperature=args.temperature,
            max_logprob_drift=args.preflight_max_logprob_drift,
        )
        print("risk_steering_preflight="+json.dumps(probe), flush=True)
    if args.preflight_only:
        return

    if args.sft_warmup:
        if args.resume_from_checkpoint:
            raise ValueError("cannot combine SFT warmup with Trainer checkpoint resume")
        _sft_warmup(engine, dataset, args)

    # TRL's configuration and HF Trainer's lifecycle are shared, but the
    # default GRPOTrainer rollout/loss are deliberately bypassed.
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        beta=0.0,
        loss_type="grpo",
        use_vllm=False,
        bf16=args.dtype == "bf16",
        fp16=args.dtype == "fp16",
        remove_unused_columns=False,
        gradient_checkpointing=False,
        use_cache=True,
        warmup_steps=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=0,
        report_to="none",
        seed=args.seed,
    )
    # The current environment reward is scored by the private evaluator
    # within trainer.training_step; no private state is appended to the prompt.
    reward = resolve_reward(args.reward_callable) if args.reward_callable else None
    trainer = make_trainer_class()(
        engine=engine,
        reward_fn=reward,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        clip_epsilon=args.clip_epsilon,
        debug_path=args.reward_debug_jsonl,
        args=cfg,
        train_dataset=dataset,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    engine.tokenizer.save_pretrained(args.output_dir)
    dest = Path(args.output_dir)
    (dest / "risk_policy_config.json").write_text(json.dumps({
        "schema": "maestro.risk_steered_lora.v2",
        "backend": args.backend,
        "trainer": "trl_grpo_config_hf_trainer_custom_risk_rollout",
        "vllm": False,  # TRL/vLLM engine, separate from the rollout attention kernel.
        "rollout_attention": args.rollout_attention,
        "gradient_checkpointing": False,
        "risk_checkpoint": str(Path(args.risk_checkpoint).resolve()),
        "steer_last_n": args.steer_last_n,
        "steer_strength": args.steer_strength,
        "steer_threshold": args.steer_threshold,
        "anchor_ids": engine.anchor_ids,
        "reward_type": args.reward_callable or "deterministic_orchestration",
        "grpo_training_steps": trainer.state.global_step,
        "zero_advantage_groups": trainer.zero_adv_groups,
        "total_groups": trainer.seen_groups,
        "num_generations": args.num_generations,
    }, indent=2), encoding="utf-8")
    print(f"Saved Unsloth+TRL risk-aware LoRA adapter to {dest}", flush=True)


if __name__ == "__main__":
    main()
