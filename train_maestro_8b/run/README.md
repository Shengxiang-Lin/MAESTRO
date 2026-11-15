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
    bash train_maestro_8b/run/pipeline.sh inspect-io
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
stock TRL GRPOTrainer; the full vLLM engine is optional (GRPO_VLLM_ENGINE=1).
Run the mandatory preflight on
the target CUDA host. Environment's data-generation LLM is not required
by these steps. No changes to Environment generation counts occur.

Install compatible Unsloth / TRL for Stage2 separately from a running
vLLM server:

    python -m pip install -r train_maestro_8b/requirements-unsloth.txt


## DA-GRPO comparison profile (default)

The production policy budget matches the **count/length** in
`baseline/da-grpo/config.yaml`, not its underlying model execution:

| Setting | DA-GRPO | MAESTRO RiskKV |
|---|---|---|
| Distinct O1 Main prompts / optimizer step | 1 | 1 |
| Candidates per prompt | 8 | 8 |
| Generated completions / update | 8 | 8 |
| Prompt cap | 12,288 tokens | 12,288 *raw O1* tokens |
| Completion cap (full run) | 4,096 tokens | 4,096 tokens |
| Completion cap (one-step smoke) | 512 | 512 |
| Sampling temperature | 1.0 | 1.0 |
| Trainer accumulation field | 8 | **1** |

The final row differs **by design**. DA-GRPO's TRL GRPOTrainer takes one
completion per microstep, so 8 accumulation steps means one O1 prompt
and its eight variations per update. MAESTRO's custom
`RiskSteeredGRPOTrainer.training_step` already generates, scores and
backpropagates an **entire 8-completion group in a single microstep**.
Therefore `GRPO_GRAD_ACCUM=1` is the equivalent one-prompt batch.
Setting it to 8 would mean EIGHT distinct prompts / 64 completions per
optimizer update, not DA parity.

Run an actual data audit **before** loading Qwen3-8B onto GPU:

    bash train_maestro_8b/run/pipeline.sh inspect-io

This prints per-sample counts, raw prompt token quantiles, the analogous
Qwen chat-template-wrapped token counts used by DA, reference JSON Action
token quantiles, Action type distribution and small previews of the exact
Main-visible prompt/reference Action. Full report:
`workspace/maestro_8b/io_inspection.json`.
Any prompt over 12,288 tokens FAILS CLOSED. Neither the O1 history nor
the original private evidence state is silently truncated or added to
the model-visible context.

**Not full model/runtime parity:** DA uses BF16 LoRA r=32,
Q/K/V/O+MLP, stock TRL DAPO and 0.01 KL, and Unsloth colocated shared
vLLM. MAESTRO uses NF4 QLoRA r=16, Q/K/V/O only, Risk Head + custom
Value-KV cache + reward + clipped loss with no KL. Optional MAESTRO
full vLLM is a SECOND inference model and synchronizes its frozen weights,
LoRA snapshots and edited paged KV per group. The same batch token budget
therefore does not imply similar VRAM or tokens/s. Run identical cases and
compare measured times and external `nvidia-smi` peak usage.

The GRPO trainer now emits `risk_grpo/prompt_tokens`,
`risk_grpo/completion_tokens`, `risk_grpo/prefill_s`,
`risk_grpo/rollout_and_score_s`, `risk_grpo/backward_s`,
`risk_grpo/reward_unique` and `risk_grpo/zero_adv_groups`, plus
CUDA allocated/reserved peaks. These CUDA allocator peaks do not count a
separate vLLM worker's GPU allocations; use `nvidia-smi` for total GPU
consumption. If `reward_unique=1`, the GRPO advantage is zero and that
prompt group does not produce a policy gradient.

To test the actual FULL vLLM backend rather than only its FlashAttention:

    GRPO_VLLM_ENGINE=1 GRPO_ROLLOUT_ATTENTION=sdpa \
      GRPO_VLLM_MEMORY_UTILIZATION=0.35 \
      bash train_maestro_8b/run/pipeline.sh preflight

Use a fresh `PILOT_CKPT` for each independent comparison; a completed
pilot is intentionally reused when `SKIP_COMPLETED=1`.

## NF4 one-token versus three-token numerical parity

If preflight reports negligible `sample_vs_incremental` and
`batched_eval_vs_train`, but nonzero `incremental_vs_batched_eval`
that persists under Eager attention, inspect the additional field
`nf4_projection_shape_probe`.

The probe captures a real activation from Qwen3 layer 0 while evaluating
the same 3-token RiskKV suffix. It then feeds the identical first
activation vector through selected **quantized linear base layers** at
sequence sizes M=1 and M=3. Results include each layer's maximum and
mean absolute output discrepancy. This isolates 4-bit matrix
multiplication numerical differences from attention, RoPE, and KV
caching. An empty array indicates that the loader exposes no matching
quantized Qwen3 projections; it does *not* certify equality.

No model weights, prompts, rewards, or trainable gradients are changed.
The 8B CUDA diagnostics showed NF4 linear projections can differ
between one-token and three-token shapes (up_proj: max 0.0078125).
Default preflight therefore separates STRICT model/cache invariants
from bounded TRAINING-INFERENCE numerical mismatch as described next.

## RiskKV preflight: structural correctness vs numerical mismatch

RTX 6000D BF16/NF4 Qwen3-8B diagnostics found:

- Local sampled logp == forced incremental logp exactly
- Batch eval logp == batch train logp exactly
- No future-token influence detected by causal-mask perturbation
- Single-token vs 3-token batch logp differed by 0.124283 max; Eager attention
  still differed by 0.124559
- Quantized layer-0 up_proj differed by 0.0078125 max at M=1 vs M=3

Numerical differences in 4-bit execution do NOT automatically mean a bad
KV cache. The ACTUAL sampled logprobs remain the behavior denominator in
the existing clipped_policy_loss; they are not replaced with teacher-forced
logprobs just to fabricate matching ratios.

The default preflight now strictly checks three numerical invariants:
(1) local sampling vs forced one-token replay <= 0.01 logp;
(2) batched eval vs differentiable batched train <= 0.01 logp;
(3) perturbing a future input does not influence earlier logits <= 0.01.
It also requires the initial sampled-to-training per-token importance ratios
to fall within the configured PPO clip interval (default [0.8, 1.2]), and
requires nonzero LoRA-only gradients without modifying frozen weights.

Outputs now include risk_steering_preflight_contract with
initial_importance_ratio_min, initial_importance_ratio_max,
initial_importance_ratio_clipped_fraction and probability_contract.
The real GRPO Pilot also logs risk_grpo/importance_ratio_min,
risk_grpo/importance_ratio_max and
risk_grpo/importance_ratio_clipped_fraction over ALL generated tokens.
The 3-token preflight cannot guarantee that 512/4096-token completions
never leave that interval. Review Pilot metrics before full training.

Default bounded-ratio check:

    GRPO_VLLM_ENGINE=0 GRPO_ROLLOUT_ATTENTION=sdpa bash train_maestro_8b/run/pipeline.sh preflight

Optional LEGACY strict logprob equality check:

    GRPO_PREFLIGHT_MAX_LOGPROB_DRIFT=0.01 GRPO_VLLM_ENGINE=0 GRPO_ROLLOUT_ATTENTION=sdpa bash train_maestro_8b/run/pipeline.sh preflight

Do not silently raise PPO clipping merely to pass a failing contract.
