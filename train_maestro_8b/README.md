# MAESTRO 8B — risk-aware Main-Agent training

This package trains the Main orchestrator on **exact Stage4 O1 histories**.
The pipeline has two learned stages:

- **Stage 1:** Frozen full Qwen3-8B -> report-window features -> train only a
  persistent Risk Head with masked MIL supervision. Keep full history (up to
  `MAX_LENGTH`); do not put private risk labels in the Main prompt.
- **Stage 2:** Frozen 8B prefill + frozen Risk Head -> risk-weighted historical
  Value KV -> decision anchor + **trainable LoRA** suffix. Optimize current
  Main decisions with deterministic evidence-coverage / urgency / legal
  parallelism / nonredundancy reward from private Environment state.

Default backend: **Unsloth QLoRA + TRL GRPOConfig / HF Trainer** with
a custom risk-cache-preserving rollout and logprob implementation. It does
not use an unmodified `trl.GRPOTrainer` or vLLM rollout. One CUDA process is
supported, and full Qwen3-8B 16K throughput must be measured on your GPU.

## Run (from Environment outputs to training)

    cd /root/autodl-tmp/MAESTRO
    bash train_maestro_8b/run/pipeline.sh all

The script uses the Qwen3-8B model in `base_models/Qwen3-8B` and saves new
features/checkpoints to `workspace/maestro_8b/` by default. No new
Environment generation is triggered; Stage4 data must already exist.
View every subcommand and adjustable environment variable in
[run/README.md](run/README.md).

    bash train_maestro_8b/run/pipeline.sh status
    bash train_maestro_8b/run/pipeline.sh export
    bash train_maestro_8b/run/pipeline.sh features
    bash train_maestro_8b/run/pipeline.sh risk
    bash train_maestro_8b/run/pipeline.sh prepare-grpo
    bash train_maestro_8b/run/pipeline.sh preflight
    bash train_maestro_8b/run/pipeline.sh pilot
    bash train_maestro_8b/run/pipeline.sh grpo

The recommended NF4 feature cache **must** match the default Unsloth NF4
training backbone. Preflight validates sampling/recomputed logprobs, cache
continuation, and LoRA-only gradients. If compatibility checks fail, do
not disable them before training.

## Code structure

    train_maestro_8b/
    ├── run/
    │   ├── pipeline.sh          # One-command reproducible orchestration
    │   ├── config.env.example  # Optional local run parameters
    │   └── README.md
    ├── data/
    │   ├── contracts.py         # Risk labels, JSONL, span/window alignment
    │   ├── dataset.py           # Full-prompt risk examples and collator
    │   ├── features.py          # Frozen feature caching/loading
    │   ├── export_risk.py       # Stage4 -> risk Train/Val
    │   ├── backfill_rewards.py  # No-LLM State4 audit replay if needed
    │   ├── export_policy.py     # Risk examples + private rewards -> GRPO
    │   └── audit_rewards.py     # Quantitative reward / binding diagnostics
    ├── risk/
    │   ├── model.py             # Window-level Risk Head
    │   ├── extract.py           # Frozen Qwen3 feature extraction CLI
    │   └── train.py             # Risk Head-only training CLI
    ├── policy/
    │   ├── cache.py             # Risk-conditioned Value KV steering
    │   ├── model.py             # Shared rollout/logprob semantics
    │   ├── backend.py           # Unsloth FastLanguageModel + NF4 loader
    │   ├── trainer.py           # TRL-configured risk GRPO training step
    │   ├── train.py             # Canonical Stage2 training CLI
    │   ├── reference.py         # Original PyTorch reference trainer
    │   └── rewards.py           # Optional legacy teacher-proxy hook
    ├── tests/                  # Unit + tiny-Qwen3 integration tests
    ├── requirements-unsloth.txt
    └── README.md

Previous command modules (`export_reliability`, `extract_features`,
`train_reliability`, `export_grpo`, `backfill_reward_states`,
`inspect_rewards`, `train_grpo`) are retained as tiny import/CLI
compatibility shims. New code imports from the organized packages.

## Artifacts and privacy

| Step | Main output |
|---|---|
| Stage4 private risk export | `environment/data/stage4/private/window_risk/{train,val}.jsonl` |
| Frozen NF4 feature caching | `workspace/maestro_8b/features/{train,val}-nf4/` |
| Stage1 Risk Head | `workspace/maestro_8b/checkpoints/risk-head/` |
| Private Stage2 reward export | `environment/data/stage4/private/risk_grpo/{train,val}.jsonl` |
| GRPO score audit | `workspace/maestro_8b/reward_reference_audit.jsonl` |
| Pilot/final LoRA | `workspace/maestro_8b/checkpoints/grpo-{pilot}/`, `.../grpo/` |

The scorer can see current global Macro DAG and execution state, but Main
sees only Question and previous Sub-Agent feedback. No Oracle/Gold labels or
private future observations enter the policy prompt. Evaluator scores are
**current-round decisions**, not end-to-end environment recovery success.

Run tests with:

    python -m compileall -q train_maestro_8b
    python -m unittest discover -s train_maestro_8b/tests -p "test_*.py" -v

The complete test suite needs PyTorch, TRL/Transformers/PEFT, and the
Environment's dependencies. GitHub CI separates lightweight tests, frozen
Risk Head checks and tiny Qwen3/TRL optimizer integration tests.
