# MAESTRO 8B — one-command run

Stage4 already generated? Run the complete Qwen3-8B training pipeline:

    cd /root/autodl-tmp/MAESTRO
    bash train_maestro_8b/run/pipeline.sh all

This runs, in order: (1) export Stage4 risk data, (2) cache train/val
frozen-8B NF4 features, (3) train Risk Head only, (4) build GRPO private
reward states / training records and audit scores, (5) Unsloth RiskKV
compatibility preflight, (6) small GRPO pilot, (7) full Unsloth+TRL GRPO.

Run independently to resume at a stage:

    bash train_maestro_8b/run/pipeline.sh export
    bash train_maestro_8b/run/pipeline.sh features
    bash train_maestro_8b/run/pipeline.sh risk
    bash train_maestro_8b/run/pipeline.sh prepare-grpo
    bash train_maestro_8b/run/pipeline.sh preflight
    bash train_maestro_8b/run/pipeline.sh pilot
    bash train_maestro_8b/run/pipeline.sh grpo
    bash train_maestro_8b/run/pipeline.sh status

Defaults expect the Qwen3-8B model in <repo>/base_models/Qwen3-8B,
Stage4 in <repo>/environment/data/stage4, and put all new feature caches,
Risk Head weights and GRPO adapters under <repo>/workspace/maestro_8b.

Override the settings through exported environment variables or copy
config.env.example to config.env (local-only; never commit secrets/paths).
For example:

    MODEL=/models/Qwen3-8B RUN_ROOT=/tmp/maestro-pilot \
      bash train_maestro_8b/run/pipeline.sh all

Features are explicitly NF4 to match the Unsloth Stage2 backbone. The
launcher checks cached feature metadata and Risk Head quantization,
prevents silently reusing incomplete shard caches, and skips complete
outputs by default. Use a NEW RUN_ROOT for new experiments.

For historical Stage4 journals without decision_reward_states.jsonl,
prepare-grpo runs deterministic, fail-closed, no-LLM backfill. A high
invalid-binding fraction in the reward audit is a signal to investigate
the evaluator before spending time on GRPO.

Stage2 Unsloth uses a single GPU and a customized GRPO rollout/loss, not
stock TRL GRPOTrainer; vLLM is not enabled. Run the mandatory preflight on
the target CUDA host. Environment's data-generation LLM is not required
by these steps. No changes to Environment generation counts occur.

Install compatible Unsloth / TRL for Stage2 separately from a running
vLLM server:

    python -m pip install -r train_maestro_8b/requirements-unsloth.txt
