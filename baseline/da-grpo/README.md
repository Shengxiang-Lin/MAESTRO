# DA-GRPO — Reproduce Orchestra-o1 Qwen3-8B with verl

This baseline reuses the **original** Orchestra-o1 per-decision GRPO
algorithm, vLLM rollout + FSDP2 and the four-dimensional LLM-as-Judge reward.
The only provider-specific addition is configurable DashScope
`enable_thinking`, **without changing the scoring weights or reward rubric**.

## Which training code runs?

- `train.py` reads `config.yaml`, resolves the existing
  `config/model_config.yaml` registry, gets the secret key from
  `DASHSCOPE_API_KEY` or hidden terminal input, and checks files/GPUs.
- `train_grpo_qwen3_8b.sh` runs `python -m verl.trainer.main_ppo`
  with the original GRPO hyperparameters.
- `reward_fn.py` is the original Orchestra-o1 Judge reward, with an
  opt-in provider compatibility parameter (`enable_thinking`).
- **The neural-network training loop lives in the external
  [verl](https://github.com/verl-project/verl) framework.** It is not in
  `train_maestro_8b/` (MAESTRO's separate Accelerate/SFT work), or in
  `baseline/orchestra-o1/` (the preserved baseline scripts).

## 0. Install verl from source — NOT `pip install verl`

The official installation instructions recommend a Git checkout. From the
*same Python/Conda environment* used to run MAESTRO:

```bash
cd /root/autodl-tmp
git clone https://github.com/verl-project/verl.git
cd verl
python -m pip install -e ".[vllm]"
python -c "import verl; print(verl.__file__)"
```

Before installing, check your installed PyTorch / CUDA / Python versions
against the selected verl revision. Modern verl may require a newer CUDA
stack and a compatible vLLM, Ray, Transformers, and PyTorch combination.
Use a dedicated environment or the official Docker image if the existing
AutoDL environment cannot support those dependencies. This repository
does not automatically install or silently upgrade GPU packages.

The default YAML points to `../verl` relative to the MAESTRO root
(e.g. `/root/autodl-tmp/verl`). Adjust that path if the checkout lives elsewhere.

## 1. Configure once

Edit **`baseline/da-grpo/config.yaml`**:

```yaml
paths:
  verl_dir: ../verl
  model_path: base_models/Qwen3-8B
  data_dir: dataset/data/da-grpo
judge:
  registry: config/model_config.yaml
  model: qwen3.7-flash
  api_key_env: DASHSCOPE_API_KEY
  enable_thinking: false
```

The Judge endpoint comes from the existing
`config/model_config.yaml` registration for `qwen3.7-flash`:
`https://dashscope.aliyuncs.com/compatible-mode/v1` (Beijing).
Use a key from the same geographic region. Do not store keys in YAML.

The original 8 × H20 96GB hyperparameters are under `training:`.
For a different GPU instance, configure `n_gpus_per_node`, TP,
batches and memory conservatively; lowering GPU count alone may be
insufficient to fit full-parameter Qwen3-8B GRPO.

## 2. Export per-round MAESTRO observations

```bash
python -m dataset.prepare_da_grpo \
  --untrusted-turns environment/data/maestro_untrusted_rounds_v9/main_turns.jsonl
```

This creates `dataset/data/da-grpo/train.parquet` (one current Main
decision per row) and `expert_decisions.jsonl`.
No gold/expert fields enter the model's `prompt`; they stay in reward
`extra_info`.

## 3. Inspect, test provider, train

From the repository root, using the same active Python environment:

```bash
python baseline/da-grpo/train.py --dry-run
python baseline/da-grpo/train.py --check
python baseline/da-grpo/train.py --test-judge
python baseline/da-grpo/train.py
```

Alternatively run `bash baseline/da-grpo/run.sh`.
You can pass extra Hydra/verl CLI overrides after the command:
`python baseline/da-grpo/train.py trainer.total_epochs=1`.

The launcher prompts `Enter DASHSCOPE_API_KEY` with hidden terminal input
when it is not set; the key is used **only in memory** and passed to the
training subprocess via its environment. The launcher does not print or
write the secret. For non-interactive runs, configure
`DASHSCOPE_API_KEY` securely in the environment before starting.

`--dry-run` is offline and does not need credentials/GPU/verl; it reports
which prerequisites are present. `--check` verifies local prerequisites
without an API request. `--test-judge` sends one small Qwen API request
without starting training; this incurs a small provider charge.
The default loggers are `[console, file]`; set
`logging.backends: [console, wandb, file]` to restore original logging
destinations. Main training outputs are written under
`baseline/da-grpo/output/` (ignored by Git).

## Reproduction scope

The original Orchestra-o1 repository provides a launcher and reward but
not the original training Parquet generator or compiled data. Our
`dataset/prepare_da_grpo.py` is callback-compatible, not a byte-for-byte
copy of that unpublished preprocessing. The algorithm remains
`algorithm.adv_estimator=grpo` (despite the DA-GRPO experiment name).
No MAESTRO-specific macro critic, SFT pre-stage, or online Sub-Agent
rollout is introduced.

If `import verl` fails, install verl from its Git checkout; it is
not a module provided by either `baseline/orchestra-o1/` or
`train_maestro_8b/`.


## Single-GPU (96 GB-class) option: same GRPO and judge, LoRA updates

Orchestra-o2's [single-card long-context profile](https://github.com/Shengxiang-Lin/orchestra-o2/blob/main/orchestra-o3/model/configs/qwen3_8b_bf16_lora_1x96g_long.yaml)
uses **Qwen3-8B LoRA rank 32**, shared-GPU vLLM and streamed candidate
backpropagation. That run is **not** an 8B full-parameter Orchestra-o1
GRPO run; it additionally implements future latents and composite losses.

For a simplified single-GPU variant we reuse **verl's official LoRA
support** (not o2's custom trainer). O1's GRPO estimator, eight candidate
rollouts, original four-dimensional reward and KL settings remain unchanged.
**Trainable weights differ:** this is a resource-adapted reproduction,
**not** exact full-parameter O1 training.

The separate `config.single-gpu.yaml` assumes **one 80–96 GB GPU**
(96 GB recommended), sets LoRA rank 32, TP 1, FSDP2 CPU offload,
small prompt/mini-batches, and vLLM memory budget 0.35. Hardware preflight
checks both GPU count and device memory. Training fit and vLLM compatibility
still require a real test; the 24K prompt / 4K response upper bounds can be
expensive even on 96 GB.

```bash
python baseline/da-grpo/train.py \
  --config baseline/da-grpo/config.single-gpu.yaml --dry-run
python baseline/da-grpo/train.py \
  --config baseline/da-grpo/config.single-gpu.yaml --check
# Only after resolving DashScope HTTP 401:
python baseline/da-grpo/train.py \
  --config baseline/da-grpo/config.single-gpu.yaml --test-judge
# First: limit to one optimizer update, with shorter decoding for smoke test.
python baseline/da-grpo/train.py \
  --config baseline/da-grpo/config.single-gpu.yaml \
  trainer.total_training_steps=1 data.max_response_length=512
```

After smoke testing, omit the final two Hydra overrides to restore the
profile's 4096 response-token ceiling and full five-epoch schedule.
For the strict O1 **full-parameter** reproduction, use `config.yaml`
with its eight-GPU hardware rather than this LoRA profile.
