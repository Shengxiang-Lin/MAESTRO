# DA-GRPO — Unsloth + TRL GRPOTrainer + vLLM

This directory is the **new** MAESTRO/Orchestra-o1 Main-Agent policy trainer.
The old hand-written Torch+PEFT GRPO loop, losses, reference log-prob
code and hand-managed vLLM LoRA engine were **deleted**. There is no verl
dependency and no fall-back to the old trainer.

## Overview

- **Model**: local `base_models/Qwen3-8B`, BF16 LoRA r=32 / alpha=64.
- **Framework**: `unsloth.FastLanguageModel(fast_inference=True)` +
  `trl.GRPOTrainer`; vLLM uses Unsloth's shared-weight integration,
  not a separately created second engine.
- **Context**: 16,384 tokens total = 12,288-token prompt cap +
  4,096-token completion cap. Long original Main-Agent history is
  **not silently truncated**: the preflight tokenizer check aborts if a
  dataset prompt exceeds 12,288 tokens. A CLI test that lowers the
  completion cap to 512 still uses the same prompt validation limit.
- **Batch**: 1 distinct prompt × 8 completions = **8 completions**
  in one generation batch. TRL's `per_device_train_batch_size=1`,
  `gradient_accumulation_steps=8`, and
  `generation_batch_size=8` implement streamed backward and **one
  optimizer update** per one-prompt batch. These values count completions,
  not prompts. vLLM `max_num_seqs=8` bounds concurrent generation
  requests; it does not change GRPO's eight-candidate group size.
- **Loss**: TRL `loss_type="dapo"` with 0.01 KL, 0.2 clip,
  `scale_rewards="group"`, one policy iteration per generated batch.
  The DAPO loss is token-normalized like the former O1-style training,
  but it is **not bitwise identical** to the original custom loss.
  vLLM importance-sampling correction is disabled for closer baseline
  comparison; TRL/Unsloth still own the actual objective.
- **Judge**: original local `reward_fn.compute_score`, weighted
  0.10 FORMAT + 0.10 ACTION + 0.20 TOOL + 0.60 DECISION. TRL passes
  one completion per reward-row. 4 Judge requests run in parallel by
  default. Judge uses Alibaba `qwen3.7-flash` with thinking disabled;
  policy Qwen3 thinking stays at its default (not forcibly disabled).
- **Dataset**: method-local Parquet `baseline/da-grpo/data/train.parquet` prepared from unified base JSONL.
  Policy sees exactly one original Orchestra-o1 Main-Agent user prompt.
  Ground truth, Stage 3/4 reference action, and other expert metadata
  are separate dataset columns passed to the **reward only**.

## Prepare DA-GRPO data from the environment base

`environment.data_export` now emits generic `cases.jsonl`,
`decisions.jsonl` and `private/decision_labels.jsonl` only. It does
not produce DA-GRPO or DA-SFT Parquet and does not split data.

From the repository root, after running the unified exporter:

```bash
python baseline/da-grpo/prepare_data.py \
  --source-dir environment/data/unified \
  --output-dir baseline/da-grpo/data
python baseline/da-grpo/train.py --check
```

DA-GRPO's converter owns filtering, Scene-level 80/10/10 splits, and reward
metadata. It produces `train.parquet`, `val.parquet`, `test.parquet`,
`splits.json`, `quality_report.json`, `summary.json` **under
`baseline/da-grpo/data/`**. The policy input remains the single O1
user prompt; `ground_truth`, `expert_params`, `subtask_history` and
other Judge metadata exist in reward-only columns, not in the prompt.

To exactly reuse DA-SFT's frozen Scene assignment, pass
`--split-map baseline/da-sft/data/splits.json`. Default reviewed-only
filter excludes template and fallback actions, with explicit
`--include-fallback` available for ablation. Untrusted recovery variants
remain grouped with their originating Scene. Inspect the baseline-local
quality report before training; never silently deduplicate or truncate.

## Install (new environment recommended)

Use an **unused** Python 3.11+ environment, not the environment used
by a currently running experiment. Stop any other GPU training before
starting this 8B run.

```bash
cd /root/autodl-tmp/MAESTRO
python3 -m venv /root/autodl-tmp/maestro-unsloth-venv
source /root/autodl-tmp/maestro-unsloth-venv/bin/activate
python -m pip install -U pip
python -m pip install -r baseline/da-grpo/requirements.txt
python -m unittest discover -s baseline/da-grpo -p 'test_*.py'
python baseline/da-grpo/train.py --dry-run
python baseline/da-grpo/train.py --check
```

Dependency solvers can upgrade CUDA/PyTorch/vLLM; record the versions
from `python -m pip freeze` before running full training. **This
migration has not yet been smoke-tested on RTX 6000D (SM120)**. The
previous environment hit FlashInfer JIT issues. The Unsloth 2026.11.4
runtime was observed to **override the sampler setting** and select
FLASHINFER even though the config requested TRITON_ATTN; additionally,
vLLM 0.32.0 reported `VLLM_ATTENTION_BACKEND` as an unknown environment
variable. The current loader instead uses Unsloth's own
`UNSLOTH_VLLM_NO_FLASHINFER=1` switch BEFORE importing Unsloth.
This disables both the FlashInfer sampler and attention backend. Unsloth
selects its compatible `FLASH_ATTN` fallback on supported NVIDIA GPUs.
BF16 KV cache (`kv_cache_dtype: auto`) and `enforce_eager: true` are retained.

If you still see `SM 12.x requires CUDA >= 12.9`, confirm that the
installed `unsloth_zoo.vllm_utils` contains
`UNSLOTH_VLLM_NO_FLASHINFER` and that the new training command is
running in a fresh Python process. A CUDA 13.0 toolkit is already shown
in the failing log, so the error can be an incompatible FlashInfer JIT
architecture probe rather than an actually missing toolkit version.
Avoid installing FlashInfer as a fix when the feature is not needed.

## Start

Export the key without putting it in Git:

```bash
export DASHSCOPE_API_KEY='your-key'
python baseline/da-grpo/train.py --test-judge
```

One optimizer-step smoke test, with isolated output directory:

```bash
python baseline/da-grpo/train.py \
  --max-steps 1 --max-completion-tokens 512 \
  --output-dir baseline/da-grpo/output_unsloth_smoke
```

Full run (default: 3 epochs, 1 prompt × 8 candidates, 4096 output cap):

```bash
bash baseline/da-grpo/run.sh
```

The run script is **foreground**, not `nohup`. Use a separate output
directory for independent experiments. TRL checkpoint directories
(`checkpoint-XX`) are saved automatically. Resume only from
**new TRL checkpoints**:

```bash
python baseline/da-grpo/train.py \
  --resume baseline/da-grpo/output/checkpoint-50
```

Custom PyTorch DA-GRPO checkpoints created before this migration are
**not compatible** with TRL optimizer or trainer state. The LoRA weights
may be reusable for a separate warm start, but exact resume is not.

## Training semantics and limitations

TRL repeats each source prompt to create 8 completions per update.
Advantages are grouped within the same prompt (8 completions).
The trainer takes 8 sequential 1-completion gradient micro-steps before
an optimizer step. The vLLM engine's `max_num_seqs=8` is a concurrency
**cap**, not a guarantee all 8 will decode simultaneously.

Existing MAESTRO training remains *offline*: the generated Main-Agent
decision does not trigger a real Sub-Agent/tool execution during GRPO.
The Judge compares each candidate with the reference action and
ground-truth answer in reward-only metadata.

TRL typically performs optimization steps even on all-equal-score
groups (with zero policy advantage but possible KL); it does **not**
automatically reproduce the old trainer's `skip_uninformative_group`
behavior. Similarly, its generation truncation, temperature, reference
KL normalization and completion filtering should be checked before
claiming an exact numerical reproduction of upstream Orchestra-o1.

We deliberately do **not** enable 4-bit QLoRA in the first migration:
keep BF16 LoRA to compare model quality against the old trainer.
Turning on 4-bit requires a separate experiment and revalidating
attention kernels, LoRA weight sharing, reference log probabilities
and reward curves.

## Files

- `config.yaml`: one configuration, version 2.
- `train.py`: configuration and dataset validation; TRL setup, CLI.
- `reward_adapter.py`: TRL completion-to-Orchestra-o1 Judge bridge.
- `reward_fn.py`: original four-dimensional LLM Judge implementation.
- `run.sh`: thin foreground entrypoint.
- `test_grpo.py`: CPU-only configuration/reward contract tests.
- `requirements.txt`: framework dependencies for a fresh environment.

Official references:
- https://huggingface.co/docs/trl/main/grpo_trainer
- https://huggingface.co/docs/trl/vllm_integration
- https://github.com/unslothai/notebooks (Qwen3-8B FP8 GRPO recipe)

## RTX 6000D FlashInfer sampler regression (2026-11-12)

Symptoms: the Judge/Unsloth checks succeed, model weights load, but
initialization fails with `FlashInfer top-p/top-k sampling unavailable:
SM 12.x requires CUDA >= 12.9`. The root cause is that Unsloth resets
`VLLM_USE_FLASHINFER_SAMPLER=1` after MAESTRO previously set it to 0.

The fix in `train.py:configure_runtime` sets the stronger
`UNSLOTH_VLLM_NO_FLASHINFER=1` knob and clears unsupported
`VLLM_ATTENTION_BACKEND` before the first Unsloth import.

```bash
git pull origin main
python -m unittest discover -s baseline/da-grpo -p 'test_*.py'
python baseline/da-grpo/train.py --dry-run
python baseline/da-grpo/train.py --max-steps 1 --max-completion-tokens 512 \
  --output-dir baseline/da-grpo/output_unsloth_smoke_retry
```

After restart, the vLLM backend should NOT say
`Using FLASHINFER attention backend`. The Unsloth fallback typically
selects `FLASH_ATTN`; another compatible native kernel may be used
depending on its build. Confirm from the startup logs. The single-step
training path has not been GPU-validated by the repository authoring
environment.

## SM120 FLASH_ATTN / FP8 KV incompatibility (2026-11-12)

If vLLM aborts with `Selected backend AttentionBackendEnum.FLASH_ATTN is
not valid for this configuration` and `kv_cache_dtype not supported`, the
previous FlashInfer opt-out worked; this is a **new** attention/KV-cache
compatibility error, not success.

For RTX 6000D, `config.yaml` now uses `kv_cache_dtype: auto`. Unsloth
receives `float8_kv_cache=False`, and BF16 base weights use BF16 KV
Cache. The loader rejects FP8 explicitly to prevent repeat failures.
Do not revert to FP8 merely to reduce memory without changing and
validating the attention kernel and hardware compatibility first.

BF16 KV Cache uses more memory than FP8. Unsloth Standby may also
increase `gpu_memory_utilization` beyond the raw config percentage;
observe the actual vLLM startup logs and memory. If there is a new OOM,
test fewer concurrent sequences and/or a shorter maximum context before
changing algorithmic group size or reinstalling CUDA.

```bash
git pull origin main
python -m unittest discover -s baseline/da-grpo -p 'test_*.py'
python baseline/da-grpo/train.py --dry-run
python baseline/da-grpo/train.py \
  --max-steps 1 --max-completion-tokens 512 \
  --output-dir baseline/da-grpo/output_unsloth_smoke_bf16_kv
```

## FlashInfer Autotuner failure after KV initialization (2026-11-12)

A follow-up error occurs **after** the BF16 KV Cache has initialized and
vLLM reports `FlashInfer top-p/top-k sampling disabled`:

```text
No module named 'flashinfer.autotuner'; 'flashinfer' is not a package
```

This is not an Attention or Tokenizer error. The Unsloth opt-out hides
FlashInfer from Python import selection; vLLM 0.32's default **O2**
optimization preset nevertheless sets
`kernel_config.enable_flashinfer_autotune=True`. Its startup cache
synchronization then tries to import `flashinfer.autotuner` despite the
mask. The `enforce_eager` flag does not disable that autotuner.

**Fix:** `train.py:disable_vllm_flashinfer_autotune` changes vLLM's
in-memory `OPTIMIZATION_LEVEL_TO_CONFIG` presets to opt out of this
autotune flag **before vLLM constructs the engine**. This is a
targeted compatibility workaround for Unsloth 2026.11.4 +
vLLM 0.32.0. It neither edits installed packages nor introduces a
fake FlashInfer module. A regression test checks the flag for each
optimization level; if the vLLM internal preset shape changes,
initialization fails early with a clear message.

```bash
git pull origin main
python -m unittest discover -s baseline/da-grpo -p 'test_*.py'
python baseline/da-grpo/train.py \
  --max-steps 1 --max-completion-tokens 512 \
  --output-dir baseline/da-grpo/output_unsloth_smoke_no_fi_autotune
```

The startup log should contain
`[da-grpo] Disabled vLLM FlashInfer autotune in all optimization presets`
and no attempt to import `flashinfer.autotuner` in the worker.
This is not yet an end-to-end GPU validation: inspect the subsequent
TRL rollout/gradient logs to verify an optimizer update completes.

## RTX 6000D first-update CUDA OOM (2026-11-12)

Observed: 4 prompts × 8 candidate actions finished vLLM rollout, but the
first backward needed 5.48 GiB with only ~1.70 GiB GPU free. The first
training step did **not** finish. vLLM had silently replaced
`gpu_memory_utilization: 0.35` with ~`0.87875` because Standby
automatically sizes the KV cache. This is partly an inference-memory
budget issue and partly long-context BF16 LoRA activation pressure;
a complete fix is not established until a GRPO optimizer step succeeds.

Unsloth has an official override `UNSLOTH_VLLM_STANDBY_UTIL_OVERRIDE=1`.
The new default sets it *before* importing Unsloth and uses
`gpu_memory_utilization: 0.70`, `max_num_seqs: 16`. This gives the
policy's long-context backward more room and reduces rollout concurrency
while preserving **4 prompt groups × 8 candidates each** and
`gradient_accumulation_steps: 32`. Fewer concurrently scheduled
rollouts may be slower but the group semantics remain unchanged.

```bash
git pull origin main
python -m unittest discover -s baseline/da-grpo -p 'test_*.py'
python baseline/da-grpo/train.py --dry-run
MAESTRO_GRPO_TRACEBACK=1 python baseline/da-grpo/train.py \
  --max-steps 1 --max-completion-tokens 512 \
  --output-dir baseline/da-grpo/output_unsloth_vram70
```

Check that `Unsloth: Standby mode is enabled. Changing
gpu_memory_utilization to 0.87875` is **absent** and the printed vLLM
utilization reflects ~0.70 (possibly adjusted down by Unsloth's planner).
If one optimization step still OOMs, run another isolated experiment
with `--vllm-memory-utilization 0.60 --max-num-seqs 8`, and inspect
the longest prompt lengths; reducing `max_num_seqs` alone will not
reduce per-completion backward activations.

Do **not** blindly apply the CUDA error's suggestion
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: colocated vLLM
Sleep/Standby uses the CuMemAllocator, and expandable CUDA allocator
segments may conflict with its memory pool. The
`CUDA Error: invalid argument` after the primary OOM may be a
secondary failure during GPU/allocator teardown. The tokenizer
`bos_token_id: None` alignment message is also not an OOM cause.

If CUDA is in a bad state after an OOM, exit the failed Python process
and check `nvidia-smi` before restarting. The unchanged MAESTRO dataset
and 4D Judge should remain the benchmark reference.

## Why Prompt Batch 1 is not the per-device training micro-batch

Here Prompt Batch means **one distinct Main-Agent decision state per
optimizer update**. TRL generates eight candidate completions and runs
eight gradient accumulation micro-steps with
`per_device_train_batch_size=1`. The vLLM generation batch is eight
candidate requests, capped at eight active sequences.
The original four-prompt mode used 32 completions/update; the current
one-prompt mode uses 8 completions/update. The GRPO advantage is still
computed from eight competing candidates **within one prompt**.

## 16K context after first-update OOM

The default max context was reduced from 32,768 to 16,384 tokens to
reduce long-sequence activation memory on a single 83 GiB GPU.
The current 1-prompt × 8-candidate GRPO algorithm and BF16 LoRA
remain unchanged by the 16K context setting.
`max_prompt_tokens=12,288` and `max_completion_tokens=4,096`
must fit within the 16K total. Do not silently discard earlier
Sub-Agent observations just to make an overlong prompt fit.

If training fails with a preflight `prompts exceed 12288 tokens`
message, the dataset contains inputs longer than this revised budget.
Inspect those sample IDs and lengths; explicitly decide whether to
filter, summarize, or restore the 32K context with additional memory
optimization. A successful `--dry-run` is only config validation; it
does not inspect real token lengths or execute GPU backward.

## One-prompt GRPO batch default (2026-11-12)

Prompt Batch was reduced from four to one following the first backward
OOM. Each optimizer update now groups **one** Main-Agent prompt with
eight vLLM completions. TRL uses
`generation_batch_size=8`, `num_generations=8`,
`gradient_accumulation_steps=8`, and
`per_device_train_batch_size=1`; the loss is still computed by TRL
with per-prompt GRPO advantages. vLLM `max_num_seqs=8` matches the
maximum number of candidates in each rollout. This is NOT equivalent to
eight prompt examples per training step.

A smaller Prompt Batch reduces rollout work and some buffered state
per optimizer step, but does NOT guarantee a smaller maximum activation
allocation for an individual long completion. If a single 16K example
still OOMs, investigate its actual input+output token length and the
model's backward peak; reducing Prompt Batch further is not possible
without changing the GRPO group structure.

```bash
git pull origin main
python -m unittest discover -s baseline/da-grpo -p 'test_*.py'
python baseline/da-grpo/train.py --dry-run
MAESTRO_GRPO_TRACEBACK=1 python baseline/da-grpo/train.py \
  --max-steps 1 --max-completion-tokens 512 \
  --output-dir baseline/da-grpo/output_prompt1_test
```
