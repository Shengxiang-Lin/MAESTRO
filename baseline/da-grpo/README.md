# DA-GRPO: Orchestra-o1 Qwen3-8B reproduction

This directory reproduces **the original Orchestra-o1 small Main-Agent
training method**; it does not add MAESTRO-specific rewards, alternate
actions, online Sub-Agent execution, or an SFT pretraining stage.

The name `da-grpo` follows the source paper's terminology. The released
launcher actually specifies `algorithm.adv_estimator=grpo` in verl.
Do not assume an additional DA-specific algorithm is implemented here.

## Files

- `train_grpo_qwen3_8b.sh`: full-parameter Qwen3-8B GRPO launcher (verl,
  FSDP2, vLLM rollout, 8 × H20 96 GB setup).
- `reward_fn.py`: copied from `Shengxiang-Lin/orchestra-o1`
  `train_qwen3_8b/grpo/reward_fn.py`, preserving the original four-term
  LLM-as-judge reward and `compute_score` callback.
- `ds_config.json`: original DeepSpeed reference config; **not consumed**
  by the current FSDP2 launcher.

## Prerequisites

Install/clone a compatible [verl](https://github.com/volcengine/verl)
checkout separately. You also need PyTorch, vLLM, Ray, the corresponding
GPU CUDA stack, `requests` (reward HTTP calls), and optionally WandB.
Do not hardcode API credentials; the reward uses the OpenAI-compatible
`OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `LLM_JUDGE_MODEL`
(default `claude-haiku-4-5`) environment variables.

## 1. Export the dataset

```bash
python -m dataset.prepare_da_grpo --output-dir dataset/data/da-grpo
```

The exporter creates `train.parquet` and `expert_decisions.jsonl`, and
keeps gold/expert references only in `extra_info`. Optional Stage 4 turns
are provided with `--untrusted-turns`; see `dataset/README.md`.

## 2. Launch the original GRPO setup

```bash
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export OPENAI_API_KEY=...           # never check secrets into Git
export LLM_JUDGE_MODEL=claude-haiku-4-5

VERL_DIR=/path/to/verl \
MODEL_PATH=/path/to/Qwen3-8B \
DATA_DIR="$(pwd)/dataset/data/da-grpo" \
bash baseline/da-grpo/train_grpo_qwen3_8b.sh
```

The script has the same training settings as the upstream O1 training
launcher (GRPO, 8 rollouts per prompt, full-parameter FSDP2, KL loss,
LLM-judge reward). The default data path is already absolute, based on
this repo, so specifying `DATA_DIR` is optional.

Defaults: 8 GPUs, batch size 24, PPO mini-batch 12, 5 epochs,
learning rate 5e-6, prompt 24576 tokens, response 4096 tokens,
vLLM tensor parallel size 2. Overrides use the upstream env variables
`TRAIN_BSZ`, `MINI_BSZ`, `ROLLOUT_N`, `EPOCHS`, `TP_SIZE`, etc.

All additional `verl.trainer.main_ppo` overrides can be passed as
positional arguments after the shell script. Outputs/logs are written under
`baseline/da-grpo/output/` and `baseline/da-grpo/logs/`.

## Fidelity and verification

The original training repository does not publish its data-preparation
scripts or Parquet, so this exporter is a **callback-compatible** data
adapter, not a byte-for-byte reconstruction of the authors' training data.
The original reward is LLM-only: it does not execute the candidate
`delegate_task` in an environment. It also retains original grading
thresholds and fallback behavior for method fidelity.

Before a GPU run, check:

```bash
python -m unittest dataset.test_prepare_da_grpo
python -m dataset.prepare_da_grpo --dry-run
bash -n baseline/da-grpo/train_grpo_qwen3_8b.sh
```

The absence of actual generated `environment/data` and GPUs in the
repository means end-to-end training must be run on your workstation.
