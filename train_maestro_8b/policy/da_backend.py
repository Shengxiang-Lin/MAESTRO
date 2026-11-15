"""Stage 2 native DA-style GRPO backend: no Stage-1 Risk Head or RiskKV.

This is a Stage-2-owned entrypoint. For the first stable GRPO baseline, use
the already tested DA implementation as a library without modifying it:
- Unsloth FastLanguageModel BF16 LoRA + colocated vLLM
- Qwen3 Chat Template and TRL GRPOTrainer, DAPO + 0.01 KL
- Original 4D O1 LLM Judge prompt/weights and concurrent adapter
The private expert/gold columns are accessible to the reward only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "baseline" / "da-grpo"
DEFAULT_CONFIG = BASELINE / "config.yaml"


def get_da_baseline():
    """Load the reference implementation by file path, not as a new fork."""
    if str(BASELINE) not in sys.path:
        sys.path.insert(0, str(BASELINE))
    spec = importlib.util.spec_from_file_location(
        "maestro_stage2_da_reference", BASELINE / "train.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import DA-GRPO reference trainer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_stage2_config(args, baseline):
    cfg = baseline.load_config(args.config)
    cfg["paths"]["model"] = Path(args.model).expanduser().resolve()
    cfg["paths"]["train_parquet"] = Path(args.train_file).expanduser().resolve()
    cfg["paths"]["output_dir"] = Path(args.output_dir).expanduser().resolve()
    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["max_steps"] = args.max_steps
    cfg["training"]["learning_rate"] = args.lr
    cfg["training"]["weight_decay"] = args.weight_decay
    cfg["training"]["max_grad_norm"] = args.max_grad_norm
    cfg["training"]["save_steps"] = args.save_steps
    cfg["training"]["logging_steps"] = args.logging_steps
    cfg["rollout"]["num_generations"] = args.num_generations
    cfg["rollout"]["prompts_per_step"] = args.gradient_accumulation_steps
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
    if args.num_generations < 2 or args.gradient_accumulation_steps < 1:
        raise ValueError("DA GRPO requires >=2 candidates and >=1 prompt group")
    if (args.epochs <= 0 or args.max_steps == 0 or args.lr <= 0
            or args.max_grad_norm <= 0 or args.weight_decay < 0
            or args.judge_workers < 1 or not 0 < args.vllm_memory_utilization < 1):
        raise ValueError("Invalid Stage2 DA training settings")
    return cfg


def subset_for_pilot(cfg, max_train_rows: int):
    """Materialize an isolated sample without mutating source parquet."""
    if max_train_rows <= 0:
        return
    import pandas as pd
    from tempfile import TemporaryDirectory
    # This is called within run()'s temporary-directory lifetime instead.
    raise RuntimeError("subset_for_pilot must be invoked with a destination path")


def make_subset(cfg: dict, max_train_rows: int, dest: Path):
    import pandas as pd
    rows = pd.read_parquet(cfg["paths"]["train_parquet"])
    if rows.empty:
        raise ValueError("Empty Stage2 training Parquet")
    if max_train_rows < 1 or max_train_rows > len(rows):
        raise ValueError(f"Requested {max_train_rows} rows, available {len(rows)}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows.head(max_train_rows).to_parquet(dest, engine="pyarrow", index=False)
    cfg["paths"]["train_parquet"] = dest


def run(args):
    baseline = get_da_baseline()
    cfg = build_stage2_config(args, baseline)
    if args.max_train_rows < 0:
        raise ValueError("max_train_rows must be >=0")
    if args.dry_run:
        baseline.show_config(cfg)
        return
    if args.max_train_rows:
        make_subset(
            cfg, args.max_train_rows,
            cfg["paths"]["output_dir"] / "stage2_pilot_dataset.parquet",
        )
    checks = baseline.environment_checks(cfg)
    if args.check:
        baseline.show_config(cfg)
        if not all(checks.values()):
            raise RuntimeError("Stage2 DA missing requirements: " + str(checks))
        return
    if not args.test_judge and not all(checks.values()):
        raise RuntimeError("Stage2 DA preflight failed: " + str(checks))
    key = baseline.judge_key(cfg["judge"]["api_key_env"])
    baseline.setup_judge(cfg, key)
    baseline.probe_judge(cfg, key)
    if args.test_judge:
        return
    print("[MAESTRO/Stage2] backend=DA/Unsloth+TRL/vLLM; risk_head=off; "
          "reward=original_DA_4D_Judge; formatter=Qwen3_chat_template", flush=True)
    baseline.train(cfg, resume=args.resume_from_checkpoint)
    destination = cfg["paths"]["output_dir"] / "stage2_da_training.json"
    destination.write_text(json.dumps({
        "schema": "maestro.stage2_da_training.v1",
        "trainer": "trl.GRPOTrainer",
        "rollout": "unsloth_colocated_vllm",
        "stage1_risk_head": False,
        "riskkv": False,
        "reward": "baseline_da_four_dimension_llm_judge",
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
                   help="Stage2-owned Stage4 DA-format Parquet")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_train_rows", type=int, default=0)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Distinct Main prompts per optimizer step, not completions")
    p.add_argument("--max_length", type=int, default=12288)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--vllm_memory_utilization", type=float, default=0.70)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--judge_registry", default="config/model_config.yaml")
    p.add_argument("--judge_model", default="qwen3.7-flash")
    p.add_argument("--judge_api_key_env", default="DASHSCOPE_API_KEY")
    p.add_argument("--judge_workers", type=int, default=4)
    p.add_argument("--check", action="store_true")
    p.add_argument("--test_judge", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume_from_checkpoint")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
