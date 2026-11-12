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
- **Batch**: 4 distinct prompts × 8 completions = **32 completions**
  in one generation batch. TRL's `per_device_train_batch_size=1`,
  `gradient_accumulation_steps=32`, and
  `generation_batch_size=32` implement streamed backward and **one
  optimizer update** per 4-prompt batch. These values count completions,
  not prompts.
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
- **Dataset**: unchanged offline Parquet `dataset/data/da-grpo/train.parquet`.
  Policy sees exactly one original Orchestra-o1 Main-Agent user prompt.
  Ground truth, Stage 3/4 reference action, and other expert metadata
  are separate dataset columns passed to the **reward only**.

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
previous environment hit FlashInfer JIT issues; the new config
explicitly forces vLLM `TRITON_ATTN`, keeps FP8 KV cache and starts
with `enforce_eager: true`. If the JIT still reports
`SM 12.x requires CUDA >= 12.9`, check `nvcc --version` versus
`torch.version.cuda`. `nvidia-smi` displays the driver's supported
CUDA version, not the compiler version.

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

Full run (default: 3 epochs, 4 prompts × 8 candidates, 4096 output cap):

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

TRL repeats the 4 source prompts to create 32 completions per update.
Advantages are grouped by original prompt (8 completions each). The
trainer takes 32 sequential 1-completion gradient micro-steps before an
optimizer step. The vLLM engine's `max_num_seqs=32` is a concurrency
**cap**, not a guarantee all 32 will decode simultaneously.

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
