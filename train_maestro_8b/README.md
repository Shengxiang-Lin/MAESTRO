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
a custom risk-cache-preserving rollout and logprob implementation. One CUDA
process is supported. The optional `GRPO_ROLLOUT_ATTENTION=vllm` uses the CUDA
FlashAttention kernel from the installed vLLM package for rollout attention,
over the same edited Transformers cache. The Transformers forward owns
prefix prefill, token sampling, cache updates and differentiable replay.

For full engine rollout on vLLM 0.32, use `GRPO_VLLM_ENGINE=1` with
`GRPO_ROLLOUT_ATTENTION=sdpa`. MAESTRO still computes the frozen prefix and
Risk Head scores. A KV connector writes the edited prefix into vLLM's paged
KV cache, including an incomplete final page. vLLM decodes from the decision
anchor using the current LoRA snapshot and returns temperature-adjusted
sampled log probabilities. The training forward and loss run in Transformers
with the same RiskKV prefix.

The engine uses a temporary 16-bit materialization of the frozen NF4 weights,
so it keeps a second model copy on the same GPU. Allow roughly 16 GiB of
temporary disk space next to the checkpoint as well. The memory budget for
vLLM is configured through `GRPO_VLLM_MEMORY_UTILIZATION` (default 0.45).

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
    bash train_maestro_8b/run/pipeline.sh inspect-io
    bash train_maestro_8b/run/pipeline.sh preflight
    bash train_maestro_8b/run/pipeline.sh pilot
    bash train_maestro_8b/run/pipeline.sh grpo

### GRPO cold-start training: progressive reward and legal fallback


The GRPO trainer now supports free-form JSON learning: the
GRPO_REWARD_MODE=curriculum training reward grades unsuccessful output
structure and reserves stronger feedback for strict legal Actions,
without altering the deterministic evaluation scorer. When an entire
group has no reward variance, the default
GRPO_ZERO_ADV_STRATEGY=reference_sft performs a separately counted
supervised LoRA update using the independently verified legal Action
for that record. This is NOT reported as a GRPO gradient. Use
GRPO_ZERO_ADV_STRATEGY=skip and GRPO_REWARD_MODE=strict to restore
strict GRPO-only behavior. Never equate a nonzero supervised fallback
gradient with successful policy optimization. No JSON prefix,
constrained decoding, or output repair is introduced.


The exact O1 prompt/return format is not guaranteed to be learned
from the initial Qwen3-8B weights alone. In particular, an 8/8 invalid
candidate group (all reward=-1, all completions reach token cap)
returns zero GRPO gradient and no effective update. Run
`bash train_maestro_8b/run/pipeline.sh audit-pilot` to inspect last
candidate previews and reason counts without a GPU.

Optional `GRPO_SFT_WARMUP=1 GRPO_SFT_WARMUP_SAMPLES=32` first
supervises LoRA using *only evaluator-legal* reference Actions from
the full dataset (even for a one-prompt Pilot), with the same frozen
RiskKV context. Use `PILOT_MAX_NEW_TOKENS=1536` for legal Actions
whose JSON target exceeds the 512-token smoke cap. An all-zero-reward
Pilot now trains using the explicitly logged reference_sft fallback by
default; with `GRPO_ZERO_ADV_STRATEGY=skip` and
`PILOT_REQUIRE_GRAD=1` it fails instead of silently skipping gradients.
See [run/README.md](run/README.md) for commands.

The recommended NF4 feature cache **must** match the default Unsloth NF4
training backbone. Preflight checks cache continuation, causal isolation,
local incremental-replay integrity and LoRA-only gradients. Since NF4
single-token and multi-token kernels differ numerically on the target GPU,
the default also REQUIRES initial sampled-to-training importance ratios
to remain inside GRPO's existing [1-clip_epsilon,1+clip_epsilon] interval,
without changing the loss or weakening structural invariants.
GRPO_PREFLIGHT_MAX_LOGPROB_DRIFT=0.01 remains an optional strict absolute
logprob override. See run/README.md for measured mismatch and real Pilot
importance-ratio metrics.

## Batch and token budget aligned with DA-GRPO

The standard MAESTRO Stage2 settings now match DA-GRPO's **visible**
rollout budget: one distinct O1 Main prompt per optimizer step, eight
candidate Actions, 12,288 raw Main-prompt tokens, 4,096 output tokens,
and temperature 1.0. The one-step pilot uses 512 output tokens, as
the DA smoke test does. The Risk Head feature cache still uses its
original 16,384-token maximum, so Stage1 data/weights do not need to
be regenerated just for this Stage2 cap.

**Important difference in accumulation:** this custom Trainer already
runs G completions/backwards inside one per-prompt training_step. Set
`GRPO_GRAD_ACCUM=1` to match DA's one-prompt / eight-candidate update;
DA's stock GRPOTrainer instead accumulates eight completion microsteps,
so it uses `gradient_accumulation_steps=8`. Copying the raw number 8
to MAESTRO would train eight prompts (64 candidates) before one step.

MAESTRO uses the exact original O1 prompt STRING, while DA supplies a
one-user-message list rendered through Qwen's chat template. To
measure both lengths and inspect actual prompt tails + reference Action
examples without loading the 8B model, use:

    bash train_maestro_8b/run/pipeline.sh inspect-io

This command is strict on overlong prompts and writes a token-length
report to `workspace/maestro_8b/io_inspection.json`.

Matching 12K/4K/G8 does **not** equalize hardware cost or learning
objective: DA uses BF16 LoRA r32 (QKVO+MLP), native TRL DAPO+KL and
shared colocated vLLM; MAESTRO uses NF4 LoRA r16 (QKVO), a frozen
Risk Head, steered Value-KV, custom offline decision reward and
custom GRPO without reference KL. Its optional full vLLM engine
requires a separate BF16 inference model and per-group KV/LoRA
synchronization, unlike Unsloth's shared-weight colocated vLLM.
See `run/README.md` for profiling and fair comparisons.

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
