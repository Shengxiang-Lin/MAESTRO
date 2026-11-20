## Required canonical Environment dataset for GRPO

`prepare-grpo` consumes the immutable `environment/data/unified` output
of `python -m environment.data_export`. It no longer reads raw Stage4
turns to regenerate training prompts. Generate the unified base first:

```bash
python -m environment.data_export \
  --scenes environment/data/stage1/questions/scenes.jsonl \
  --clean-episodes environment/data/stage3/episodes.jsonl \
  --clean-turns environment/data/stage3/main_turns.jsonl \
  --stage4-dir environment/data/stage4 \
  --output-dir environment/data/unified
```

Adjust the Stage3 paths to your actual Environment run. The unified
exporter checks their shared source/version consistency; do not mix runs.

Next run `bash train_maestro_8b/run/pipeline.sh prepare-grpo`.
Stage2 applies reviewed-only selection and creates Scene-separated
`train.parquet`, `val.parquet`, `test.parquet` plus split and quality
reports. The default split is 80/10/10 (seed 17). Use `DA_SPLIT_MAP`
to reuse a frozen baseline `splits.json`, `DA_INCLUDE_FALLBACK=1` to
opt in to non-reviewed rows, or `SKIP_COMPLETED=0` to explicitly rebuild.

For `global`, `hybrid`, `deterministic` and `judge_only`, only Stage4
Main decisions with scorable canonical reward states are trainable.
The original DA four-dimensional reward also supports clean Stage3
decisions. The private-state sidecar comes from the unified labels;
Global Judge's Stage4 claim snapshots and Stage1 Scenes are checked
against the unified source and are reward-only, never student input.

---
## Stage 2 Default: Global Evidence & Recovery LLM Judge

The native Unsloth + colocated vLLM + TRL GRPOTrainer / Qwen3
chat-template pipeline remains unchanged. The **new default** is
`DA_REWARD_MODE=global` (independent per-completion reward, not groupwise).

Unlike DA's teacher-action comparison or the old local hard-binding hybrid,
the LLM Judge sees a reward-only context built from THREE views:

- **Global DAG:** every Macro task, its source/field goals, dependencies,
  and current admissible work/parallel limit.
- **Actually returned factual evidence:** Stage4 claim nodes with concrete
  subject/attribute/value, source/provenance, revision, claim relations,
  conflicts, missing fields and affected dependencies. Exact branch and
  pre-action round are enforced.
- **Evaluator-only reference:** simulated source observations for sources
  already reported, plus the final reference answer. These may contain
  facts not actually observed by Main. The rubric explicitly FORBIDS
  penalizing Main for failing to divine hidden corruption or future facts.

Only the unchanged O1 prompt reaches the 8B student. The reference context
goes to the external **reward Judge only**; it must not appear in the policy
prompt. The Judge individually grades every structurally valid Action on
0..10 task-tool alignment, evidence grounding, recovery quality and global
orchestration quality. Current scoring weights are:

```text
Invalid O1 JSON or required structural fields  => -1
Valid O1 Action => 0.10 format credit
                 + 0.15 task/tool alignment
                 + 0.15 evidence grounding
                 + 0.25 recovery quality
                 + 0.35 global orchestration quality
```

All dimension ratings are LLM judged and normalized to [0,1].
`context` and `tools` remain OPTIONAL as defined by Orchestra-o1.
Wrong tool/field/parallel strategy is not automatically -1 from an
exact-string matcher; the earlier deterministic scorer's error is logged
as `legacy_env_reason` **for diagnosis only**.

Reward states are derived from unified private labels into
`DA_DATA/private/decision_reward_states.jsonl`. Global Judge also
requires the matching Stage4 `private/claim_state_supervision.jsonl`.
The new context builder also reads the exact Stage1 `questions/scenes.jsonl`.
Preflight rejects missing or future-timestamped claim states. If legacy
journals lack pre-action claim audits, regenerate the Stage4 sidecar; do not
fill it with inferred future observations.

```bash
git pull origin main
bash train_maestro_8b/run/pipeline.sh prepare-grpo
bash train_maestro_8b/run/pipeline.sh preflight
export DASHSCOPE_API_KEY='YOUR_KEY'
bash train_maestro_8b/run/pipeline.sh judge
bash train_maestro_8b/run/pipeline.sh pilot
GRPO_MAX_STEPS=100 bash train_maestro_8b/run/pipeline.sh grpo
```

Outputs are isolated in `stage2-global-pilot` and
`stage2-global-grpo` directories. The global Judge uses its own
versioned cache; it must not reuse the old Semantic Judge's v1 cache.
Monitor `reward_diagnostics.jsonl`: per Action type, Judge dimension
scores, reward, and `legacy_env_reason`. No raw candidate text or global
reference records are written into diagnostics. Judge API failures abort
rather than fabricating rewards.

Ablations: `DA_REWARD_MODE=hybrid` (legacy exact-binding Environment +
Semantic Judge), `deterministic` (local only), `judge_only` (old
semantic model behind the local gate), `da` (DA four-dimension expert/gold
Judge). Do not resume an optimizer checkpoint from a different reward mode.
For the experimental Stage1 RiskKV path, use `STAGE2_TRAINER=riskkv`.

## Legacy ablation: MAESTRO Hybrid (DA_REWARD_MODE=hybrid)

`STAGE2_TRAINER=da` continues to use the working Unsloth/TRL/vLLM
pipeline, while `DA_REWARD_MODE=hybrid` selects the previously
defined **0.6 local Environment + 0.4 semantic Judge** reward.
Invalid Action = -1; legal Action gets the mixed score. Training
and rollout implementations are untouched.

```bash
bash train_maestro_8b/run/pipeline.sh prepare-grpo
bash train_maestro_8b/run/pipeline.sh preflight
export DASHSCOPE_API_KEY='your-key'
bash train_maestro_8b/run/pipeline.sh judge
bash train_maestro_8b/run/pipeline.sh pilot
GRPO_MAX_STEPS=100 bash train_maestro_8b/run/pipeline.sh grpo
```

The private reward snapshot file at
`environment/data/stage4/private/decision_reward_states.jsonl`
must exist. State rows are joined by the exact Stage4 candidate+round,
and any unscorable/missing rows are explicitly reported/excluded. For
legacy Stage4 data that lacks snapshots, regenerate/backfill those
states before training; there is no implicit teacher imitation
fallback. The original DA Parquet is reused unchanged.

Check `workspace/maestro_8b/checkpoints/stage2-hybrid-pilot/reward_diagnostics.jsonl`
for `valid`, `reason`, `env_reward`, `judge_reward`,
`judge_dimensions` and `reward`; it never stores raw prompt or Action.
The training source may contain gold/expert metadata for DA compatibility,
but the hybrid Judge sees **only** the Main-visible Prompt and Action.
The hybrid-mode paths are separate from the global and DA runs.

For ablations: `DA_REWARD_MODE=deterministic` or `judge_only`.
To reproduce the previous four-dimensional DA Judge exactly, use
`DA_REWARD_MODE=da`. To enable the earlier edited-KV approach, use
`STAGE2_TRAINER=riskkv` separately.

## Stage2 DA-first: recommended current training path

```bash
# after publishing Stage4 reviewed Main-Agent turns
STAGE2_TRAINER=da bash train_maestro_8b/run/pipeline.sh prepare-grpo
STAGE2_TRAINER=da bash train_maestro_8b/run/pipeline.sh preflight
export DASHSCOPE_API_KEY='...'
STAGE2_TRAINER=da bash train_maestro_8b/run/pipeline.sh judge
STAGE2_TRAINER=da bash train_maestro_8b/run/pipeline.sh pilot
GRPO_MAX_STEPS=100 STAGE2_TRAINER=da bash train_maestro_8b/run/pipeline.sh grpo
```

The default already is `STAGE2_TRAINER=da`. Data conversion, trainer
selection and run outputs are all owned by `train_maestro_8b/`.
The Stage2 backend constructs its own Unsloth + TRL GRPOTrainer, while
using only MAESTRO-owned GRPO runtime helpers and four-dimension DA
ablation reward modules. No code or configuration from
`baseline/da-grpo/` is imported or read. No edited-KV RiskKV prefill
and no Risk Head are loaded.
Training uses native Qwen3 chat template, Unsloth+TRL GRPOTrainer,
colocated vLLM, DAPO+KL, and the Global Evidence/Recovery Judge by
default. The optional `DA_REWARD_MODE=da` ablation uses the legacy
four-dimension O1 Judge (expert/gold stay **reward-only**).

To compare RiskKV, set `STAGE2_TRAINER=riskkv` explicitly. The older
instructions below concern that **experimental** mode.

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
stock TRL GRPOTrainer; the separate vLLM RiskKV rollout engine is enabled by default (GRPO_VLLM_ENGINE=1).
Run the mandatory preflight on
the target CUDA host. Environment's data-generation LLM is not required
by these steps. No changes to Environment generation counts occur.

Install compatible Unsloth / TRL for Stage2 separately from a running
vLLM server:

    python -m pip install -r train_maestro_8b/requirements-unsloth.txt


## DA-GRPO-aligned 83 GiB training profile (default)

The standalone default config is `train_maestro_8b/policy/grpo_config.yaml`.
Current MAESTRO Stage2 defaults match DA's training and generation
hyperparameters where the existing NF4 RiskKV training architecture allows.

| Setting | DA-GRPO | MAESTRO RiskKV |
|---|---|---|
| Qwen3 model and dtype | 8B BF16 LoRA | 8B NF4 + BF16 compute |
| LoRA rank / alpha | 32 / 64 | 32 / 64 |
| LoRA projection modules | Q/K/V/O + gate/up/down | Q/K/V/O + gate/up/down |
| Prompts per optimizer update | 1 | 1 |
| Generations per prompt | 8 | 8 |
| Prompt/output tokens | 12,288 / 4,096 | 12,288 / 4,096 |
| Temperature / top-p | 1.0 / 1.0 | 1.0 / full-vocab |
| Learning rate | 5e-5 | 5e-5 |
| Optimizer / weight decay | 8-bit AdamW / 0.01 | 8-bit AdamW / 0.01 |
| Max grad norm | 1.0 | 1.0 |
| Epochs / save / log | 3 / 50 / 1 | 3 / 50 / 1 |
| PPO epsilon | 0.2 | 0.2 |
| vLLM memory utilization | 0.70 | 0.70 (independent worker) |
| Trainer gradient accumulation field | 8 | **1 (same effective batch)** |
| Reward | DA's external judge | Strict deterministic current-round scorer |
| Training loss | TRL DAPO + KL 0.01 | RiskKV custom clipped GRPO, no KL |

The trainer accumulation field **must differ**: DA-GRPO's stock TRL
backprops one candidate per microstep and accumulates 8. MAESTRO's
custom `training_step` backprops all G=8 candidates for one prompt inside
each microstep. Setting `GRPO_GRAD_ACCUM=8` would create eight prompts /
64 candidates per optimizer step, and is NOT the same effective batch.

**Do not switch MAESTRO to BF16 with a flag.** Its frozen Stage1 Risk Head
was trained from NF4 backbone features; a BF16 prefill would change
those features. Similarly, MAESTRO's risk-edited cache cannot be sent
through DA's unmodified colocated vLLM path. MAESTRO instead loads a
second, BF16 materialized vLLM model for sampling. 0.70 is the baseline
requested reservation, but it is **not proven safe** for all MAESTRO
training activations even with 83 GiB: if the full-engine preflight or
backward hits OOM, try
`GRPO_VLLM_MEMORY_UTILIZATION=0.35` to leave more GPU memory to LoRA
backpropagation. The PyTorch allocator peak excludes the vLLM worker;
check total GPU memory with `nvidia-smi`.

Run the actual data audit without a GPU:

    bash train_maestro_8b/run/pipeline.sh inspect-io

The 257 reference Actions were all legal in the user's audit; the
longest required about 1051 tokens. However 8/8 and 2/2 untrained
generations hit the 1536-token pilot cap, and every output was invalid
JSON. The cap may truncate an in-progress Action; it is not the sole
reason: some outputs begin with free-form analysis, and 4096 does NOT
guarantee valid JSON or an EOS. Sampling remains unconstrained; the
original O1 prompt already specifies a JSON Action. Invalid outputs
continue to score exactly -1.

A 1-step smoke now uses the full 4096-token completion budget by
default. If every sampled reward ties, `PILOT_REQUIRE_GRAD=1` fails
with a useful reason histogram instead of falsely claiming a policy
update. In normal `grpo` training the strict `skip` strategy
does not fabricate an advantage for such a group. Warmup with legal
SFT references remains **optional** and off by default; no reward
shaping, JSON prefix/repair, or automatic SFT fallback is enabled.

The trainer reports token counts, reward uniqueness, strict validity,
zero-advantage group count, prefill/rollout/backward times and initial
sampling-vs-training importance ratio clipping. Check actual total
VRAM externally and don't equate a clean 3-token preflight with
long-context training success.

To test native full vLLM on the actual 83-GiB GPU:

    GRPO_VLLM_ENGINE=1 GRPO_VLLM_MEMORY_UTILIZATION=0.70 \
      bash train_maestro_8b/run/pipeline.sh preflight

Use a fresh `PILOT_CKPT` for each experiment; a completed checkpoint
is reused with `SKIP_COMPLETED=1`.

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


## Optional hybrid LLM Judge reward

The default `GRPO_REWARD_MODE=deterministic` makes **no external API calls**.
To enable mixed reward (valid Actions only), export an API key and run:

    export DASHSCOPE_API_KEY=your_key_in_shell
    GRPO_REWARD_MODE=hybrid GRPO_JUDGE_MODEL=qwen3.7-flash \
      GRPO_JUDGE_WORKERS=4 GRPO_JUDGE_WEIGHT=0.4 \
      bash train_maestro_8b/run/pipeline.sh pilot

The remote model name and endpoint are resolved through
`config/model_config.yaml`; the API key itself is **never** committed.
You can also use `GRPO_REWARD_MODE=judge_only` for ablation. In all modes,
the local Environment hard-validity check runs first: invalid JSON, illegal
delegations, broken dependencies, and premature completion always score -1.
For valid Actions, hybrid reward is 0.6 * Environment reward + 0.4 *
semantic reward by default. The Judge scores visible evidence grounding
(30%), recovery strategy (30%), subtask instruction quality (20%) and
reasoning quality (20%) on a 0..4 integer scale.

The Judge sees **only** the original Main-visible O1 prompt and each valid
candidate Action. Teacher actions, fault origin, future observations, Gold
answers and `decision_state` are never sent. **The visible Main prompt
itself is sent to the external API**: enable only if your report/question
data are permitted to leave your environment. API failures, invalid judge
JSON or invalid score ranges stop the training step rather than silently
creating scores; valid candidates are submitted concurrently with
`GRPO_JUDGE_WORKERS`. Identical inputs are cached by SHA-256 (not raw
prompt text) in `GRPO_JUDGE_CACHE_JSONL`, so resumed training can reuse
scores. The cache schema includes a rubric version: change it when the
rubric changes.

`GRPO_ZERO_ADV_STRATEGY=skip` remains required for hybrid and judge-only
experiments. A group of entirely invalid Actions still has zero advantage;
adding a Judge does not repair JSON-format cold start. Reward diagnostics in
`reward_grpo.jsonl` include Environment and Judge scores for each candidate,
while Trainer logs add `risk_grpo/judge_reward`,
`risk_grpo/judge_scored_fraction` and four normalized rubric scores.
Unlike the baseline API Judge, this mode does not ask the external LLM to
override deterministic action validity.

## Strict reward and zero-gradient handling

The strict deterministic scorer remains unchanged. It parses the
**entire output** as one JSON Action. Any invalid JSON or illegal Action
scores exactly `-1`; an otherwise plausible JSON fragment surrounded
by prose remains invalid. The generator is not forcibly constrained to
produce JSON and does not stop at a closing brace; it stops naturally
on EOS or the completion-token cap. Full vLLM is configured to honor
EOS instead of ignoring it.

Historical observations on the RTX 6000D:
- G=8 with 1536 cap: 8/8 invalid_json, all 1536 tokens, 611s rollout
  via slow Transformers decoding, zero GRPO gradient;
- SFT warmup (32 legal references, one epoch) then G=2 with 1536 cap:
  both invalid_json and zero GRPO gradient.

The current default `GRPO_ZERO_ADV_STRATEGY=skip` leaves tied-reward
GRPO groups without gradients; this is the mathematically correct
behavior for a strictly uniform reward. The Pilot uses
`PILOT_REQUIRE_GRAD=1` to fail clearly if that happens. It does not
imply the model/optimizer is broken.

Inspect the last candidate texts and exact failure reasons (CPU only):

    bash train_maestro_8b/run/pipeline.sh audit-pilot

For a STRICT GRPO-only 83GiB pilot with the new full-length defaults:

    GRPO_VLLM_ENGINE=1 GRPO_VLLM_MEMORY_UTILIZATION=0.70 \
      GRPO_GENERATIONS=8 GRPO_GRAD_ACCUM=1 \
      PILOT_MAX_NEW_TOKENS=4096 \
      PILOT_CKPT="$PWD/workspace/maestro_8b/checkpoints/strict-g8-$(date +%Y%m%d-%H%M%S)" \
      bash train_maestro_8b/run/pipeline.sh pilot

If a strict group repeatedly has no signal, it cannot learn from GRPO
until some candidates pass strict validation. One explicit experiment
is SFT warmup on the full 257 legal reference Actions (this is supervised
learning, NOT part of GRPO reward), then rerun strict GRPO:

    GRPO_SFT_WARMUP=1 GRPO_SFT_WARMUP_SAMPLES=0 \
      GRPO_VLLM_ENGINE=1 \
      PILOT_MAX_NEW_TOKENS=4096 \
      PILOT_CKPT="$PWD/workspace/maestro_8b/checkpoints/sft-all-strict-$(date +%Y%m%d-%H%M%S)" \
      bash train_maestro_8b/run/pipeline.sh pilot

This remains opt-in. The observed 32-reference warmup alone was
insufficient, and even full-dataset SFT is not guaranteed to make every
free-form sample strictly valid. A legal reference is never used in
a GRPO reward or leaked into the Main-visible prompt. The optional
zero-advantage `reference_sft` fallback is also disabled by default;
if enabled for an ablation it is counted as supervised training,
not GRPO policy gradient.
