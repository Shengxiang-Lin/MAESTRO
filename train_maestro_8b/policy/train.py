"""Unsloth + TRL Stage2 launcher for KV-steered, LoRA-only GRPO.

This intentionally uses Unsloth FastLanguageModel and TRL GRPOConfig with
HF Trainer/Accelerate optimization hooks, NOT stock TRL GRPOTrainer rollout:
the stock rollout/logprob path cannot apply the same risk-edited KV cache.
Optional vLLM attention kernels consume the same edited KV during sampling.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
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
    p.add_argument("--vllm_engine", action="store_true", help="Full vLLM RiskKV rollout")
    p.add_argument("--vllm_memory_utilization", type=float, default=0.70)
    p.add_argument("--reward_callable", help="optional module.path:function(candidate, record)")
    p.add_argument("--reward_debug_jsonl", default=None)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_generations", type=int, default=8,
                   help="GRPO completions PER Main prompt (DA-GRPO default: 8)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="MAIN PROMPT groups accumulated per optimizer step, not completions; DA parity is 1")
    p.add_argument("--max_prompts", type=int, default=0)
    p.add_argument("--max_length", type=int, default=12288,
                   help="maximum raw O1 Main prompt tokens, no truncation (DA cap: 12288)")
    p.add_argument("--max_new_tokens", type=int, default=4096,
                   help="maximum generated Action tokens (DA cap: 4096)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--optim", choices=("adamw_8bit", "adamw_torch"),
                   default="adamw_8bit")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--steer_last_n", type=int, default=8)
    p.add_argument("--steer_strength", type=float, default=0.1)
    p.add_argument("--steer_threshold", type=float, default=0.5)
    p.add_argument("--clip_epsilon", type=float, default=0.2)
    p.add_argument("--zero_adv_strategy", choices=("skip", "reference_sft"),
                   default="skip",
                   help="When every completion scores equally, use an independently "
                        "verified reference Action as auxiliary SFT, never fake GRPO advantage")
    p.add_argument("--zero_adv_sft_weight", type=float, default=0.25,
                   help="Weight for reference SFT fallback on no-preference groups")
    p.add_argument("--sft_warmup", type=int, default=0,
                   help="Optional reference-Action supervised warmup epochs before GRPO")
    p.add_argument("--sft_warmup_samples", type=int, default=0,
                   help="Select up to N legal reference Actions from FULL train set; 0 means all")
    p.add_argument("--fail_on_zero_adv", action="store_true",
                   help="Fail Pilot only when all candidate advantages tie AND no legal supervised fallback is enabled")
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--load_in_4bit", dest="load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--preflight_max_logprob_drift", type=float, default=None,
                   help="Optional STRICT absolute logprob equality gate. Default: "
                        "enforce RiskKV causal/gradient invariants plus the existing "
                        "GRPO clip_epsilon importance-ratio interval.")
    p.add_argument("--skip_preflight", action="store_true",
                   help="disables risk-cache replay sanity check; not recommended")
    p.add_argument("--preflight_only", action="store_true",
                   help="load the GPU model and verify cache/logprob, then exit")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def validate_args(args):
    if args.vllm_engine and (args.backend != "unsloth" or args.rollout_attention != "sdpa"):
        raise ValueError("--vllm_engine requires Unsloth and replaces --rollout_attention")
    if not 0 < args.vllm_memory_utilization < 1:
        raise ValueError("--vllm_memory_utilization must be in (0, 1)")
    if (args.epochs <= 0 or args.num_generations < 2 or args.max_length < 1
            or args.max_new_tokens < 1 or args.lora_rank < 1
            or args.temperature <= 0 or not 0 < args.clip_epsilon < 1
            or not 0 < args.zero_adv_sft_weight <= 1
            or args.lora_alpha < 1 or args.lr <= 0
            or args.weight_decay < 0 or args.max_grad_norm <= 0):
        raise ValueError("invalid GRPO/LoRA hyperparameters")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation counts FULL prompt groups and must be >=1")
    if args.max_length + args.max_new_tokens > 16384:
        raise ValueError(
            "DA-compatible token budget requires prompt + completion <= 16384; "
            "the RiskKV decision anchor is separately reserved by the model loader"
        )
    if not 0 <= args.steer_threshold <= 1 or args.steer_strength < 0:
        raise ValueError("invalid risk-steering settings")
    if args.backend == "unsloth" and not args.load_in_4bit:
        raise ValueError("initial Unsloth integration is NF4-only; remove --no_4bit")
    if (args.sft_warmup < 0 or args.sft_warmup_samples < 0
            or args.save_steps < 1 or args.logging_steps < 1):
        raise ValueError("invalid warmup/save/logging settings")
    # Legacy strict absolute-logp gate stays available as an explicit
    # diagnostic override. By default we validate hard RiskKV invariants
    # AND require the sampler/trainer importance ratio to remain inside
    # the current GRPO clip_epsilon band. NF4 GEMV and GEMM need not be
    # numerically identical.
    if (args.preflight_max_logprob_drift is not None
            and not 0 < args.preflight_max_logprob_drift < 1):
        raise ValueError(
            "explicit strict preflight log-probability drift bound must be in (0,1)"
        )
    if args.max_steps == 0:
        raise ValueError("--max_steps must be -1 (epoch) or positive")


def select_warmup_records(records, *, seed, limit):
    """Cold-start only from legal reference Actions; never teach invalid goals.

    Full private reward state is passed ONLY to this offline selector and the
    deterministic validator. The student still sees only the original O1
    prompt with RiskKV and the target JSON Action.
    """
    from collections import Counter
    from environment.untrusted_orchestration.orchestration_reward import score_decision
    selected = []
    rejected = Counter()
    for row in records:
        action = row.get("reference_action")
        if not isinstance(action, dict):
            rejected["missing_reference_action"] += 1
            continue
        state = row.get("decision_state")
        result = score_decision(action, state)
        if result["valid"]:
            selected.append(row)
        else:
            rejected[result["reason"]] += 1
    available = len(selected)
    rng = random.Random(seed)
    rng.shuffle(selected)
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError(
            "No legal reference Actions available for SFT cold-start; "
            f"rejection reasons: {dict(rejected)}"
        )
    print("risk_sft_warmup_plan=" + json.dumps({
        "eligible_reference_actions": available,
        "selected_samples": len(selected),
        "reference_rejections": dict(rejected),
        "selection_seed": seed,
    }), flush=True)
    return selected


def _sft_warmup(engine, full_records, args):
    """Optional label-grounded RiskKV LoRA warmup for untrained JSON policy.

    The GRPO pilot is often one Main prompt, but warmup MUST sample legal
    references from the complete 257-row dataset, not the pilot subset.
    """
    import torch
    from torch.optim import AdamW
    from train_maestro_8b.policy.trainer import reference_action_ids
    selected = select_warmup_records(
        full_records, seed=args.seed, limit=args.sft_warmup_samples,
    )
    params = [p for p in engine.policy.parameters() if p.requires_grad]
    optim = AdamW(params, lr=args.lr)
    engine.policy.train()
    for epoch in range(args.sft_warmup):
        # Keep legal samples reproducible while changing order by epoch.
        indices = list(range(len(selected)))
        random.Random(args.seed + epoch).shuffle(indices)
        total_loss = 0.0
        for j in indices:
            row = selected[j]
            target = reference_action_ids(
                engine.tokenizer, row["reference_action"], device=engine.device,
                max_new_tokens=args.max_new_tokens,
            )
            prefix = engine.prefill(row["prompt"], row["reports"])
            optim.zero_grad(set_to_none=True)
            loss = -engine.log_probs(prefix, target).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            total_loss += float(loss.detach().item())
        print(
            "risk_sft_warmup_epoch=" + json.dumps({
                "epoch": epoch + 1, "samples": len(indices),
                "mean_token_nll": round(total_loss / len(indices), 5),
            }), flush=True,
        )
    engine.policy.zero_grad(set_to_none=True)


def main(argv=None):
    args = parse_args(argv)
    validate_args(args)
    with ExitStack() as stack:
        run(args, stack)


def run(args, stack):
    # Import Unsloth FIRST, before importing transformers/trl/peft.
    # For the HF reference backend the module is never imported.
    if args.backend == "unsloth":
        from train_maestro_8b.policy.backend import load_risk_engine, check_risk_cache_contract
        engine = load_risk_engine(args)
    else:
        from train_maestro_8b.policy.reference import build_engine
        from train_maestro_8b.policy.backend import check_risk_cache_contract
        engine = build_engine(args)

    if args.vllm_engine:
        from train_maestro_8b.policy.vllm_engine import VLLMRiskRollout
        engine.rollout_engine = VLLMRiskRollout(engine.policy, args)
        stack.callback(engine.rollout_engine.close)
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
    full_records = list(iter_jsonl(args.train_file))
    records = full_records[:args.max_prompts] if args.max_prompts > 0 else full_records
    dataset = MainDecisionDataset(records)
    # One policy/training microstep owns a WHOLE GRPO group of completions.
    # Unlike stock TRL GRPOTrainer, gradient_accumulation_steps therefore
    # counts PROMPTS, never generated completions.
    print("risk_grpo_batch_plan=" + json.dumps({
        "distinct_prompts_per_optimizer_step": args.gradient_accumulation_steps,
        "num_generations_per_prompt": args.num_generations,
        "zero_adv_strategy": args.zero_adv_strategy,
        "completions_per_optimizer_step":
            args.gradient_accumulation_steps * args.num_generations,
        "max_prompt_tokens": args.max_length,
        "max_completion_tokens": args.max_new_tokens,
        "raw_o1_prompt": True,
        "model_backend": "NF4 RiskKV + separate vLLM" if args.vllm_engine
                         else "NF4 RiskKV + Transformers",
    }), flush=True)
    if not args.skip_preflight:
        probe = check_risk_cache_contract(
            engine, next((row for row in records if row["reports"]), records[0]),
            temperature=args.temperature,
            max_logprob_drift=args.preflight_max_logprob_drift,
            clip_epsilon=args.clip_epsilon,
        )
        print("risk_steering_preflight="+json.dumps(probe), flush=True)
    if args.preflight_only:
        return

    if args.reward_callable and args.zero_adv_strategy == "reference_sft":
        raise ValueError(
            "Custom reward plugin requires --zero_adv_strategy=skip; "
            "otherwise its reward semantics may not match private legal references"
        )
    if args.sft_warmup:
        if args.resume_from_checkpoint:
            raise ValueError("cannot combine SFT warmup with Trainer checkpoint resume")
        _sft_warmup(engine, full_records, args)

    # TRL's configuration and HF Trainer's lifecycle are shared, but the
    # default GRPOTrainer rollout/loss are deliberately bypassed.
    cfg = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # TRL GRPOConfig validates that a *generation* batch contains a
        # multiple of G completions. Our custom Trainer creates its own
        # per-prompt G completions, so this is configuration metadata only.
        generation_batch_size=(
            args.gradient_accumulation_steps * args.num_generations
        ),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
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
        fail_on_zero_adv=args.fail_on_zero_adv,
        zero_adv_strategy=args.zero_adv_strategy,
        zero_adv_sft_weight=args.zero_adv_sft_weight,
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
        "vllm": args.vllm_engine,
        "preflight_max_logprob_drift": args.preflight_max_logprob_drift,
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
        "grpo_gradient_groups": trainer.grpo_gradient_groups,
        "reference_sft_fallback_groups": trainer.reference_sft_fallback_groups,
        "zero_adv_strategy": args.zero_adv_strategy,
        "zero_adv_sft_weight": args.zero_adv_sft_weight,
        "total_groups": trainer.seen_groups,
        "num_generations": args.num_generations,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "optim": args.optim,
        "max_grad_norm": args.max_grad_norm,
        "sft_warmup_epochs": args.sft_warmup,
        "sft_warmup_samples": args.sft_warmup_samples,
        "prompts_per_optimizer_step": args.gradient_accumulation_steps,
        "max_prompt_tokens": args.max_length,
        "max_completion_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }, indent=2), encoding="utf-8")
    print(f"Saved Unsloth+TRL risk-aware LoRA adapter to {dest}", flush=True)


if __name__ == "__main__":
    main()
