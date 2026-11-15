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

## Pilot all-invalid / zero-gradient cold start (8/8 invalid)

If the Pilot reports G=8, all 1,536-token generations reached the cap,
reward=-1 and reward_unique=1, the optimization did NOT happen:
zero_adv_groups=1, grad_norm=0 and backward_s=0. The 10-minute time
comes from 8 sequential Transformers generations, not LoRA backward.

Check the ACTUAL most recent generated texts and fail reasons first.
This uses CPU only, and does not expose private reference/reward state:

    bash train_maestro_8b/run/pipeline.sh audit-pilot

You may also run the underlying tool and specify the max token count
for this particular run:

    python -m train_maestro_8b.data.audit_rollouts --input workspace/maestro_8b/reward_pilot.jsonl --tail 8 --token_cap 1536

The audit distinguishes invalid_json, malformed/empty delegate tasks,
premature complete, unbound evidence source/field, and other deterministic
binding failures. Do NOT assume invalid_json until the audit confirms it.

The default model receives the exact RAW O1 prompt without chat template
(as required for consistency with the existing frozen Risk Head). This is
not a guarantee that an initially untrained LoRA policy will emit valid JSON
or immediately name the correct evidence sources, fields, tools and models.

An optional **supervised cold start** exists in the same RiskKV policy:
it selects only legal existing Stage4 reference Actions using
score_decision (private state used ONLY for label filtering), computes
supervised NLL against complete reference JSON+EOS, and updates LoRA
before the first GRPO rollout. It does not alter the prompt, risk feature
distribution, Risk Head, GRPO reward, or PPO clipping. Although the Pilot
uses only ONE prompt for GRPO, the warmup selects references from the
FULL 257-row file rather than that one prompt.

    GRPO_VLLM_ENGINE=0 GRPO_SFT_WARMUP=1 GRPO_SFT_WARMUP_SAMPLES=32 \
    PILOT_MAX_NEW_TOKENS=1536 \
    PILOT_CKPT="$PWD/workspace/maestro_8b/checkpoints/pilot-sft32-$(date +%Y%m%d-%H%M%S)" \
    bash train_maestro_8b/run/pipeline.sh pilot

The warmup prints risk_sft_warmup_plan (eligible/rejected reference
count), then risk_sft_warmup_epoch (samples and supervised NLL).
If all references are invalid, it fails rather than teaching the model
bad actions. If a complete legal reference Action exceeds the Pilot
max_new_tokens, it fails rather than silently truncating supervision.
Consider a shorter first diagnostic with G=2 / 256 output tokens
ONLY to inspect formatting, not to compare the full 8-way objective.

Pilot now defaults to PILOT_REQUIRE_GRAD=1, so an all-identical-reward
group raises a diagnostic RuntimeError with the failure reason histogram
and does NOT produce a misleading successfully-trained checkpoint.
For normal GRPO training this strict per-group check is off: some
individual groups can legitimately have zero advantage.
When SFT warmup is enabled for full training, the same options apply to
all legally valid reference Actions (cap of 32 by default); training
itself retains 1 prompt x 8 candidates per optimizer update.

Important: even with warmup, prompt-to-action semantics must be checked
using both the deterministic reward audit and a real Pilot. The sample
count and validity distribution may vary. The starter dataset contains
reference supervision, but GRPO's private scorer is still independently
evaluated and never receives teacher text similarity as a reward.
