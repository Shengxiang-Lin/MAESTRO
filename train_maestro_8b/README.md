## Main Stage 2 training (default): DA-style Unsloth + TRL GRPO

The current main Stage2 training path is **`train_maestro_8b`**, not
`baseline/da-grpo`. After Stage4 has published accepted Main turns, run:

```bash
git pull origin main
bash train_maestro_8b/run/pipeline.sh prepare-grpo
bash train_maestro_8b/run/pipeline.sh preflight
export DASHSCOPE_API_KEY='your-key'
bash train_maestro_8b/run/pipeline.sh judge
bash train_maestro_8b/run/pipeline.sh pilot
GRPO_MAX_STEPS=100 bash train_maestro_8b/run/pipeline.sh grpo
```

The pipeline defaults to `STAGE2_TRAINER=da`, with Stage2-owned
`data/export_da_grpo.py`, `policy/da_backend.py` and
`policy/train.py da ...`. **No Stage1 Risk Head/features/checkpoint is
required for this run.** The original baseline source remains unchanged.
The main Stage2 backend **constructs and runs** BF16 LoRA + Unsloth/vLLM
with TRL GRPOTrainer and DAPO+KL 0.01 inside `train_maestro_8b/policy/da_backend.py`.
It reuses the unmodified DA baseline configuration/tokenization helpers
and the exact four-dimension LLM Judge prompt/weights
(format .1, action .1, tool .2, decision .6).
It sends public task/history plus the private gold/expert reference to
the external Judge **for reward only**, never to the Main policy prompt.

The Stage2-specific outputs are `workspace/maestro_8b/stage2_da/train.parquet`
and `workspace/maestro_8b/checkpoints/stage2-da-{pilot,grpo}/`.
The default pilot runs 3 updates with 16 training rows and 1536 generated
tokens per candidate, then checks if any group had nonzero reward variance.
This is an early training diagnostic, not an end-to-end GPU validation.
The remote Judge requires credentials; do not commit API keys.

**Legacy RiskKV mode**: `STAGE2_TRAINER=riskkv bash
train_maestro_8b/run/pipeline.sh all`. That experimental mode still
trains the Stage1 Risk Head, uses edited KV caches and the private
deterministic/hybrid rewards. It is retained for controlled Stage1
ablations, not silently mixed into this DA-first Stage2 run.

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
vLLM is configured through `GRPO_VLLM_MEMORY_UTILIZATION` (default 0.70).
This is a separate vLLM engine (not DA's colocated shared engine), so
reduce to 0.35 if the 83-GiB host runs out of training headroom.

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

### Optional semantic LLM Judge (hybrid or judge-only)

GRPO now optionally combines the existing deterministic private-state reward
with a remote LLM Judge. The default remains strict deterministic scoring,
with **no API calls**. Set `GRPO_REWARD_MODE=hybrid` and export
`DASHSCOPE_API_KEY` for concurrent scoring of valid generated Actions.
Illegal Actions remain exactly -1; only Main-visible prompt/Action text goes
to the API. See [run/README.md](run/README.md#optional-hybrid-llm-judge-reward)
for the scoring rubric, cache, privacy constraints and pilot command.

### Strict GRPO reward and optional supervised warmup

The default GRPO reward is the ORIGINAL deterministic
`score_decision`: invalid JSON or illegal Main-Agent Actions score
exactly **-1**. No partial JSON/prose reward, generation repair, fixed
JSON prefix, or forced decoding constraint is applied. Every output is
freely sampled until EOS or the token cap. The O1 prompt itself already
contains the required JSON format.

The previous Pilot used 1536 output tokens: all eight candidates hit
the cap and received -1; a 32-example / 1-epoch SFT warmup still left
2/2 candidates invalid. This is partly consistent with truncation,
but several generations begin with free-form prose, so raising the
cap to the DA-GRPO default 4096 does NOT guarantee valid JSON.
Pilot defaults to 4096 tokens now.

An all-invalid group has ZERO GRPO advantage and therefore cannot
update policy parameters. The strict default
`GRPO_ZERO_ADV_STRATEGY=skip` never manufactures a learning signal.
`PILOT_REQUIRE_GRAD=1` makes the one-step Pilot stop with the
failure reasons when no update can occur. Optional
`GRPO_SFT_WARMUP=1 GRPO_SFT_WARMUP_SAMPLES=0` uses ALL validated
reference Actions before GRPO, but that is a separate supervised
learning experiment, not reward shaping. No SFT is enabled by default.

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
and temperature 1.0. The one-step pilot uses the same 4,096 output-token cap; the
model may terminate earlier upon generating EOS. The Risk Head feature cache still uses its
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
objective: DA uses BF16 LoRA r32 (Q/K/V/O + MLP), native TRL DAPO+KL
and shared colocated vLLM. MAESTRO now uses the SAME r32/alpha64/seven
projection LoRA settings and 5e-5 / 8-bit AdamW hyperparameters, but
must retain NF4 training (the frozen Risk Head uses NF4 features),
steered Value-KV, a strict private Action scorer and custom GRPO loss
without reference KL. Its default full vLLM requires a SECOND BF16
inference model and per-group KV/LoRA synchronization, unlike
Unsloth's shared-weight colocated vLLM.
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
