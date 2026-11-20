"""Stage 2 native DA-style GRPO backend: no Stage-1 Risk Head or RiskKV.

This is an independent Stage-2-owned implementation of the validated DA
training mechanics. No baseline/da-grpo source is imported:
- Unsloth FastLanguageModel BF16 LoRA + TRL external vLLM Server Mode
- Qwen3 Chat Template and TRL GRPOTrainer, DAPO + 0.01 KL
- Global Evidence & Recovery LLM Judge (default): graph+visible facts+recovery
- Legacy MAESTRO hybrid 0.6/0.4 and DA 4D Judge kept as explicit ablations.
Private decision states are loaded by the reward only, never by the policy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


from train_maestro_8b.policy import grpo_runtime as runtime

DEFAULT_CONFIG = runtime.DEFAULT_CONFIG


def build_stage2_config(args):
    cfg = runtime.load_config(args.config)
    cfg["paths"]["model"] = Path(args.model).expanduser().resolve()
    cfg["paths"]["train_parquet"] = Path(args.train_file).expanduser().resolve()
    cfg["paths"]["output_dir"] = Path(args.output_dir).expanduser().resolve()
    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["max_steps"] = args.max_steps
    cfg["training"]["learning_rate"] = args.lr
    cfg["training"]["weight_decay"] = args.weight_decay
    cfg["training"]["max_grad_norm"] = args.max_grad_norm
    cfg["training"]["optim"] = args.optim
    cfg["training"]["kl_coef"] = args.kl_coef
    cfg["training"]["clip_range"] = args.clip_epsilon
    cfg["training"]["warmup_ratio"] = args.warmup_ratio
    cfg["training"]["seed"] = args.seed
    cfg["model"]["lora_rank"] = args.lora_rank
    cfg["model"]["lora_alpha"] = args.lora_alpha
    cfg["training"]["save_steps"] = args.save_steps
    cfg["training"]["logging_steps"] = args.logging_steps
    cfg["rollout"]["temperature"] = args.temperature
    cfg["rollout"]["num_generations"] = args.num_generations
    cfg["training"]["per_device_train_batch_size"] = getattr(args, "micro_batch_size", 1)
    cfg["training"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    cfg["rollout"]["max_prompt_tokens"] = args.max_length
    cfg["rollout"]["max_completion_tokens"] = args.max_new_tokens
    cfg["rollout"]["max_num_seqs"] = max(
        cfg["rollout"]["max_num_seqs"], args.num_generations,
    )
    cfg["rollout"]["gpu_memory_utilization"] = args.vllm_memory_utilization
    cfg["judge"]["workers"] = args.judge_workers
    cfg["judge"]["model"] = args.judge_model
    cfg["judge"]["api_key_env"] = args.judge_api_key_env
    cfg["judge"]["registry"] = str(Path(args.judge_registry).expanduser().resolve())
    if args.judge_model != "qwen3.7-flash" or args.judge_registry != "config/model_config.yaml":
        import yaml
        models = yaml.safe_load(Path(cfg["judge"]["registry"]).read_text(encoding="utf-8"))["models"]
        info = models.get(args.judge_model, {})
        if info.get("backend") != "api" or info.get("api_type") != "openai":
            raise ValueError("DA Stage2 Judge must be an OpenAI-compatible API")
        cfg["judge"]["base_url"] = info["base_url"]
    if not cfg["judge"]["base_url"].startswith("https://"):
        raise ValueError("Judge endpoint must use HTTPS")
    if (args.max_length + args.max_new_tokens > cfg["model"]["max_seq_length"]
            or args.max_length < 1 or args.max_new_tokens < 1):
        raise ValueError("Stage2 prompt+generation exceeds Qwen3 context")
    if args.num_generations < 2 or args.gradient_accumulation_steps < 1 or getattr(args, "micro_batch_size", 1) < 1:
        raise ValueError("DA GRPO requires >=2 candidates and >=1 prompt group")
    if (args.epochs <= 0 or args.max_steps == 0 or args.lr <= 0
            or args.max_grad_norm <= 0 or args.weight_decay < 0
            or args.judge_workers < 1 or not 0 < args.vllm_memory_utilization < 1
            or args.lora_rank < 1 or args.lora_alpha < 1
            or args.optim != "adamw_8bit" or args.temperature != 1.0
            or not 0 <= args.clip_epsilon < 1 or args.kl_coef < 0
            or not 0 <= args.warmup_ratio < 1
            or args.save_steps < 1 or args.logging_steps < 1):
        raise ValueError("Invalid Stage2 DA training settings")
    runtime.batch_plan(cfg)  # Validate global completions divisible by G.
    return cfg


def train_stage2(cfg: dict, *, resume: str | None, args):
    """Native Stage2 GRPO. Keep DA's trainer mechanics; replace only reward.

    Constructs and trains the policy here. The original DA reward is an
    explicit ablation; hybrid uses private Environment + semantic Judge.
    """
    runtime.configure_runtime(cfg)
    print("[MAESTRO/Stage2] native DA GRPO; no Stage1 Risk Head/KV steering", flush=True)
    import torch
    topology = cfg["distributed"]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size != topology["train_gpus"]:
        raise RuntimeError(
            f"Stage2 expects {topology['train_gpus']} training processes, "
            f"but WORLD_SIZE={world_size}; start via pipeline.sh pilot/grpo"
        )
    if not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError("Stage2 LOCAL_RANK has no visible training GPU")
    torch.cuda.set_device(local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage2 requires BF16-capable training GPUs")

    # Unsloth must patch before importing TRL, but standalone vLLM is
    # launched by a separate process on the rollout-only GPU.
    from unsloth import FastLanguageModel
    if topology["vllm_mode"] == "colocate":
        runtime.disable_vllm_flashinfer_autotune()
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    from train_maestro_8b.policy.da_reward_adapter import make_reward_function as make_da_reference_reward

    rows = runtime.load_dataset_rows(cfg["paths"]["train_parquet"])
    private_states = None
    if args.reward_mode != "da":
        from train_maestro_8b.policy.trl_reward_adapter import (
            load_private_states, select_scorable_rows, prompt_lookup,
            make_stage2_reward_function,
        )
        private_states = load_private_states(args.decision_states)
        rows, selected_report = select_scorable_rows(rows, private_states)
        print("[MAESTRO/Stage2] state-aligned rows: " +
              json.dumps(selected_report, ensure_ascii=False), flush=True)
    if args.max_train_rows:
        if args.max_train_rows > len(rows):
            raise ValueError(
                f"pilot asks for {args.max_train_rows} scorable rows, "
                f"only {len(rows)} available"
            )
        rows = rows[:args.max_train_rows]
    if not rows:
        raise ValueError("Empty Stage2 training set")
    if args.reward_mode == "da":
        reward_function = make_da_reference_reward(
            cfg["judge"]["workers"],
            diagnostics_path=runtime.ranked_path(cfg["paths"]["output_dir"] / "reward_diagnostics.jsonl"),
            num_generations=cfg["rollout"]["num_generations"],
        )
    else:
        global_contexts = None
        if args.reward_mode == "global":
            from train_maestro_8b.policy.global_judge_context import load_global_contexts
            global_contexts, audit = load_global_contexts(
                scenes_path=args.scenes_file,
                claim_states_path=args.claim_states,
                private_states=private_states, rows=rows,
            )
            print("[MAESTRO/Stage2] global evidence context: " +
                  json.dumps(audit, ensure_ascii=False), flush=True)
        judge = (
            make_global_judge_instance(args, cfg)
            if args.reward_mode == "global" else
            make_semantic_judge(args, cfg)
            if args.reward_mode != "deterministic" else None
        )
        reward_function = make_stage2_reward_function(
            private_states=private_states, prompts=prompt_lookup(rows),
            mode=args.reward_mode, judge=judge, judge_weight=args.judge_weight,
            num_generations=cfg["rollout"]["num_generations"],
            diagnostics_path=runtime.ranked_path(cfg["paths"]["output_dir"] / "reward_diagnostics.jsonl"),
            global_contexts=global_contexts,
        )
    if rank == 0:
        print(f"[MAESTRO/Stage2] {len(rows)} reward-compatible Main decisions; "
              f"batch_plan={runtime.batch_plan(cfg)}", flush=True)
    model_cfg, rollout = cfg["model"], cfg["rollout"]
    model_kwargs = {
        "model_name": str(cfg["paths"]["model"]),
        "max_seq_length": model_cfg["max_seq_length"],
        "dtype": torch.bfloat16,
        "load_in_4bit": model_cfg["load_in_4bit"],
        "fast_inference": topology["vllm_mode"] == "colocate",
    }
    if topology["vllm_mode"] == "colocate":
        model_kwargs.update({
            "max_lora_rank": model_cfg["lora_rank"],
            "gpu_memory_utilization": rollout["gpu_memory_utilization"],
            "max_num_seqs": rollout["max_num_seqs"],
            "float8_kv_cache": False,
            "enforce_eager": rollout["enforce_eager"],
        })
    policy, tokenizer = FastLanguageModel.from_pretrained(**model_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # DA uses Qwen3 role/chat templates, not raw continuation+newline.
    runtime.check_prompt_lengths(rows, tokenizer, rollout["max_prompt_tokens"])
    policy = FastLanguageModel.get_peft_model(
        policy, r=model_cfg["lora_rank"], lora_alpha=model_cfg["lora_alpha"],
        lora_dropout=model_cfg["lora_dropout"],
        target_modules=model_cfg["lora_target_modules"],
        bias="none", use_gradient_checkpointing="unsloth",
        random_state=cfg["training"]["seed"],
    )
    stage1_warmstart = None
    if args.stage1_belief_adapter:
        from train_maestro_8b.policy.belief_init import load_stage1_belief_adapter
        stage1_warmstart = load_stage1_belief_adapter(
            policy, args.stage1_belief_adapter,
            stage2_model=args.model,
            rank=model_cfg["lora_rank"],
            alpha=model_cfg["lora_alpha"],
        )
        print("[MAESTRO/Stage2] Stage1 belief warm-start: " +
              json.dumps(stage1_warmstart, ensure_ascii=False), flush=True)
    training_config = GRPOConfig(**runtime.build_grpo_kwargs(cfg))
    trainer = GRPOTrainer(
        model=policy,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(rows),
        reward_funcs=[reward_function],
        args=training_config,
    )
    if not getattr(trainer, "use_vllm", False):
        raise RuntimeError("Stage2 DA vLLM rollout is not enabled")
    trainer.train(resume_from_checkpoint=resume)
    dest = cfg["paths"]["output_dir"] / "final_adapter"
    trainer.save_model(str(dest))
    if rank == 0:
        tokenizer.save_pretrained(str(dest))


def make_semantic_judge(args, cfg):
    """Create the previously defined semantic Judge without DA/O1 prompt changes."""
    from train_maestro_8b.policy.semantic_judge import SemanticJudge
    cache = args.judge_cache_jsonl or str(
        cfg["paths"]["output_dir"] / "semantic_judge_cache.jsonl"
    )
    cache = str(runtime.ranked_path(cache))
    return SemanticJudge.from_registry(
        registry_path=cfg["judge"]["registry"],
        model=cfg["judge"]["model"],
        api_key_env=cfg["judge"]["api_key_env"],
        workers=cfg["judge"]["workers"],
        timeout=args.judge_timeout, retries=args.judge_retries,
        cache_path=cache, enable_thinking=False,
    )


def make_global_judge_instance(args, cfg):
    """Version-separated reward-only full graph / factual evidence Judge."""
    from train_maestro_8b.policy.global_judge import make_global_judge
    cache = args.judge_cache_jsonl or str(
        cfg["paths"]["output_dir"] / "global_judge_cache.jsonl"
    )
    cache = str(runtime.ranked_path(cache))
    return make_global_judge(
        registry_path=cfg["judge"]["registry"],
        model=cfg["judge"]["model"],
        api_key_env=cfg["judge"]["api_key_env"],
        workers=cfg["judge"]["workers"],
        timeout=args.judge_timeout, retries=args.judge_retries,
        cache_path=cache,
    )


def verify_private_reward_data(args, cfg):
    from train_maestro_8b.policy.trl_reward_adapter import (
        load_private_states, select_scorable_rows,
    )
    states = load_private_states(args.decision_states)
    rows = runtime.load_dataset_rows(cfg["paths"]["train_parquet"])
    rows, summary = select_scorable_rows(rows, states)
    print("[MAESTRO/Stage2] private reward alignment: " +
          json.dumps(summary, ensure_ascii=False), flush=True)
    if args.reward_mode == "global":
        from train_maestro_8b.policy.global_judge_context import load_global_contexts
        _, audit = load_global_contexts(
            scenes_path=args.scenes_file, claim_states_path=args.claim_states,
            private_states=states, rows=rows,
        )
        print("[MAESTRO/Stage2] global preflight: " +
              json.dumps(audit, ensure_ascii=False), flush=True)


def run(args):
    cfg = build_stage2_config(args)
    if args.max_train_rows < 0 or not 0 <= args.judge_weight <= 1:
        raise ValueError("Invalid pilot size / semantic Judge weight")
    if args.judge_timeout <= 0 or args.judge_retries < 0:
        raise ValueError("Invalid semantic Judge timeout/retries")
    if args.dry_run:
        runtime.show_config(cfg)
        print("[MAESTRO/Stage2] reward_mode=" + args.reward_mode +
              " decision_states=" + str(args.decision_states))
        return
    checks = runtime.environment_checks(cfg)
    if args.check:
        runtime.show_config(cfg)
        if not all(checks.values()):
            raise RuntimeError("Stage2 missing requirements: " + str(checks))
        if args.reward_mode != "da":
            verify_private_reward_data(args, cfg)
        return
    if not args.test_judge and not all(checks.values()):
        raise RuntimeError("Stage2 preflight failed: " + str(checks))
    if args.reward_mode == "da":
        key = runtime.judge_key(cfg["judge"]["api_key_env"])
        runtime.setup_judge(cfg, key)
        runtime.probe_judge(cfg, key)
    elif args.reward_mode in ("global", "hybrid", "judge_only"):
        judge = (make_global_judge_instance(args, cfg)
                 if args.reward_mode == "global" else make_semantic_judge(args, cfg))
        if args.test_judge:
            # Probe exact rubric and schema without GPU, not a training rollout.
            visible_prompt = "User asks for project status. No reports have returned yet."
            if args.reward_mode == "global":
                from train_maestro_8b.policy.global_judge import build_global_judge_prompt
                from train_maestro_8b.policy.global_judge_context import SCHEMA
                visible_prompt = build_global_judge_prompt(visible_prompt, {
                    "schema": SCHEMA, "decision_round": 1,
                    "task_graph": {"tasks": [{"task_id": "A", "depends_on": []}]},
                    "scheduling_reference": {"max_parallel_tasks": 2,
                                             "currently_feasible_goals": [{"task_id": "A"}]},
                    "observed_evidence": {"claim_nodes": [], "claim_states": []},
                    "recovery_reference": {"observable_issues": []},
                    "evaluator_only_reference": {},
                })
            result = judge._request(
                visible_prompt,
                '{"action":"delegate_task","reasoning":"Find missing evidence",'
                '"params":{"tasks":[{"task_instruction":"Verify missing'
                ' evidence","model":"model_1","tools":["GoogleSearchAction"]}]}}',
            )
            print("[MAESTRO/Stage2] " + args.reward_mode + " Judge HTTP OK: " +
                  json.dumps(result, ensure_ascii=False), flush=True)
    elif args.test_judge:
        print("[MAESTRO/Stage2] deterministic reward requires no LLM Judge")
    if args.test_judge:
        return
    print("[MAESTRO/Stage2] backend=Unsloth+TRL/vLLM; risk_head=off; "
          f"reward={args.reward_mode}; formatter=Qwen3_chat_template", flush=True)
    train_stage2(
        cfg, resume=args.resume_from_checkpoint, args=args,
    )
    if int(os.environ.get("RANK", "0")) != 0:
        return
    destination = cfg["paths"]["output_dir"] / "stage2_da_training.json"
    destination.write_text(json.dumps({
        "schema": "maestro.stage2_da_training.v1",
        "trainer": "trl.GRPOTrainer",
        "rollout": "trl_vllm_" + cfg["distributed"]["vllm_mode"],
        "distributed": cfg["distributed"],
        "batch_plan": runtime.batch_plan(cfg),
        "stage1_risk_head": False,
        "riskkv": False,
        "stage1_belief_adapter": str(args.stage1_belief_adapter) if args.stage1_belief_adapter else None,
        "reward": args.reward_mode,
        "reward_formula": ("0.6*environment+0.4*semantic" if
                           args.reward_mode == "hybrid" and args.judge_weight == 0.4
                           else f"environment_weight={1-args.judge_weight};semantic_weight={args.judge_weight}" if args.reward_mode == "hybrid"
                           else args.reward_mode),
        "decision_states": str(args.decision_states) if args.reward_mode != "da" else None,
        "global_scene_file": str(args.scenes_file) if args.reward_mode == "global" else None,
        "global_claim_states": str(args.claim_states) if args.reward_mode == "global" else None,
        "data_source": str(cfg["paths"]["train_parquet"]),
        "num_generations": cfg["rollout"]["num_generations"],
        "max_prompt_tokens": cfg["rollout"]["max_prompt_tokens"],
        "max_completion_tokens": cfg["rollout"]["max_completion_tokens"],
    }, indent=2), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--model", required=True)
    p.add_argument("--train_file", required=True,
                   help="Stage2-specific train.parquet prepared from unified Environment decisions")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_train_rows", type=int, default=0)
    p.add_argument("--stage1_belief_adapter", default=None,
                   help="Optional Stage1 PEFT adapter directory; loads weights into the existing LoRA")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8,
                   help="Actual optimizer gradient accumulation steps, per training GPU")
    p.add_argument("--micro_batch_size", type=int, default=1,
                   help="Completion micro-batch per training GPU")
    p.add_argument("--max_length", type=int, default=12288)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--vllm_memory_utilization", type=float, default=0.70)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--optim", default="adamw_8bit")
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--clip_epsilon", type=float, default=0.2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--judge_registry", default="config/model_config.yaml")
    p.add_argument("--judge_model", default="qwen3.7-flash")
    p.add_argument("--judge_api_key_env", default="DASHSCOPE_API_KEY")
    p.add_argument("--judge_workers", type=int, default=4)
    p.add_argument("--reward_mode",
                   choices=("global", "hybrid", "deterministic", "judge_only", "da"),
                   default="global",
                   help="global=full evidence+recovery LLM Judge; hybrid=old strict local gate")
    p.add_argument("--scenes_file",
                   default="environment/data/stage1/questions/scenes.jsonl")
    p.add_argument("--claim_states",
                   default="environment/data/stage4/private/claim_state_supervision.jsonl")
    p.add_argument("--decision_states",
                   default="workspace/maestro_8b/stage2_da/private/decision_reward_states.jsonl",
                   help="Local private pre-action state sidecar, never sent to policy/Judge")
    p.add_argument("--judge_weight", type=float, default=0.4)
    p.add_argument("--judge_timeout", type=float, default=90.0)
    p.add_argument("--judge_retries", type=int, default=2)
    p.add_argument("--judge_cache_jsonl", default=None)
    p.add_argument("--check", action="store_true")
    p.add_argument("--test_judge", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume_from_checkpoint")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
