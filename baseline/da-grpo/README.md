# DA-GRPO: single-GPU Qwen3-8B Main-Agent training

This is the **only maintained training path** under `baseline/da-grpo/`.
It reproduces Orchestra-o1's **per-turn GRPO objective, group-normalized
advantages, KL penalty, and four-dimensional LLM-as-Judge reward**, while
using the single-card memory engineering from Orchestra-o2/o3.

**No verl, 8-card FSDP configuration, DeepSpeed, Future Latent, separate
critic head, teacher CE, CHORD, SFT pre-stage or custom outcome reward.**

## Differences from O2/O3 and strict O1

- **Identical task / reward**: one current Main-Agent observation sampled
  into `group_size=8` proposed `delegate_task` or `complete` actions.
  `reward_fn.py::compute_score` is the original O1 four-term Judge reward
  (format 10%, action validity 10%, tool reasonableness 20%, decision quality
  60%). External Judge uses the shared DashScope model registry.
- **Original O1-style objective**: clipped token-level GRPO surrogate with
  group-relative advantages, plus low-variance KL against frozen base Qwen.
  The reference is computed by temporarily disabling LoRA; **no second 8B
  reference model** is kept resident.
- **O2/O3 resource technique**: Qwen3-8B BF16, rank-32 LoRA on seven
  attention/MLP projections, gradient checkpointing, single-card vLLM,
  `kv_cache_dtype=fp8`, sleep/wake between rollout and training, one
  candidate backward at a time and one optimizer step per entire GRPO group.
- **Not exact full-parameter O1**: trainable parameters are LoRA adapters
  rather than all 8B weights, and the train optimizer/learning rate follows
  the practical single-GPU profile. This is an *algorithm/reward*
  reproduction with a different parameterization, not a claim of identical
  full-parameter dynamics.

## Layout

```text
baseline/da-grpo/
├── config.yaml       # ONLY YAML profile (single 96GB-class GPU)
├── train.py          # Dataset, auth, orchestration, checkpoint/resume
├── policy.py         # Plain Qwen3-8B LoRA; chunked selected-token logprobs
├── rollout.py        # vLLM + LoRA adapter sync + FP8 KV cache + sleep mode
├── losses.py         # Group GRPO, clipping, KL
├── reward_fn.py      # Original Orchestra-o1 Judge rubric/score weights
├── run.sh            # Thin wrapper calling train.py
├── test_train.py     # CPU-only contract checks
└── README.md
```

There is no dependency on `/root/autodl-tmp/verl`; uninstalling it
is unnecessary and is not part of this change.
`train_maestro_8b/` and `baseline/orchestra-o1/` remain separate.

## 1. Prepare dependencies

Use the existing Qwen3-8B / LoRA / vLLM environment, with compatible CUDA,
PyTorch, Transformers and vLLM versions. Consult the versions already used
in Orchestra-o2/o3's working
[1×96GB profile](https://github.com/Shengxiang-Lin/orchestra-o2/blob/main/orchestra-o3/model/configs/qwen3_8b_bf16_lora_1x96g_long.yaml).

```bash
python -m pip install pyyaml pandas pyarrow requests peft transformers accelerate
# Install a compatible vLLM build for your CUDA/PyTorch environment.
python -c "import torch, peft, transformers, vllm, pyarrow; print('imports OK')"
```

A single 96GB GPU is the reference hardware. A configurable
`min_gpu_memory_gib: 80` preflight guard prevents accidental attempts on
24GB hardware; this does not prove that any long-context rollout fits.

## 2. Training data

The input is the previously exported per-turn Parquet, NOT full episodes
or macro-task state.

```bash
python -m dataset.prepare_da_grpo \
  --untrusted-turns environment/data/maestro_untrusted_rounds_v9/main_turns.jsonl
```

Each `prompt` contains only the O1 Main Agent's view. The expert action,
ground truth and Sub-Agent history in `extra_info` are visible only to
the Reward callback, not as policy target tokens. No live Sub-Agent
execution occurs during GRPO.

## 3. Shared Aliyun Judge settings

`config.yaml` points to `config/model_config.yaml` and selects
`qwen3.7-flash` (with `enable_thinking: false`). No secret is stored
in config. The launcher securely prompts for `DASHSCOPE_API_KEY`
if missing. `--test-judge` sends one request and stops on HTTP 401
before allocating the training model.

**If HTTP 401 persists**, verify that the API key belongs to the same
DashScope region as the registered Base URL. GPU settings cannot solve
an API authentication error.

## 4. Commands from MAESTRO root

```bash
git pull origin main
python -m unittest discover -s baseline/da-grpo -p 'test_*.py'

# Offline configuration and availability inspection
python baseline/da-grpo/train.py --dry-run
python baseline/da-grpo/train.py --check

# Verify the Judge request (can incur a minimal API charge)
python baseline/da-grpo/train.py --test-judge

# One real Qwen update with reduced output cap: smoke test only
python baseline/da-grpo/train.py --max-updates 1 --max-response-tokens 512

# Full configured single-GPU GRPO
python baseline/da-grpo/train.py
# Equivalent: bash baseline/da-grpo/run.sh
```

`--max-updates` and `--max-response-tokens` are optional CLI
debug overrides; the default `config.yaml` keeps G=8, max prompt 24576,
max response 4096 and epochs 5. Smoke-test runs initialize their own
optimizer and should be treated as separate experiments, not resumed
as if they used the full schedule.

**Important:** One group is one **optimizer update** only if candidate
rewards differ enough to give nonzero group advantages. Equal-reward groups
are skipped rather than silently moving the policy. API errors must be
addressed before running paid GRPO.

## 5. Checkpoints and monitoring

`baseline/da-grpo/output/` (Git-ignored) contains:

- `metrics.jsonl`: per-optimizer-step policy/KL loss, Judge reward range,
  JSON completion fraction, vLLM generation duration and policy/logprob drift.
- `checkpoint-000050/` etc: PEFT LoRA adapter and tokenizer, with
  `train_state.pt` holding optimizer/scheduler state for resuming.
- `vllm_sync/current_adapter/`: temporary adapter snapshot for vLLM.

Resume *only from a trusted checkpoint generated by this trainer*:

```bash
python baseline/da-grpo/train.py \
  --resume baseline/da-grpo/output/checkpoint-000050
```

Since optimizer checkpoints use PyTorch serialization, do not load
untrusted arbitrary `train_state.pt` files.

## Validation status

Source code and unit tests have been added to GitHub. A full GPU run,
vLLM sleep/wake cycling, and actual Qwen Judge calls have **not**
been executed from this environment. Verify `--check`, `--test-judge`
and a one-update smoke test on the 96GB AutoDL instance before assuming
the memory profile fits end-to-end.

## RTX 6000D / SM120 (vLLM 0.32) initialization

On the user's RTX 6000D (83.05 GiB), FlashInfer JIT was selected despite
an incompatible CUDA toolkit and reported `FlashInfer requires GPUs with sm75
or higher`. A separate CUDA Graph profiling/capture step exhausted device
memory. The sole single-GPU config therefore uses:

```yaml
rollout:
  attention_backend: TRITON_ATTN
  enforce_eager: true
  gpu_memory_utilization: 0.35
  kv_cache_dtype: fp8
```

`TRITON_ATTN` is available in vLLM 0.32 and supports FP8 KV cache.
`enforce_eager` avoids both CUDA Graph capture and torch.compile; expect
slower rollout startup/throughput in exchange for compatibility. It does
NOT disable vLLM sleep mode or modify GRPO / the O1 reward.

```bash
python -m unittest discover -s baseline/da-grpo -p 'test_*.py'
python baseline/da-grpo/train.py --dry-run
python baseline/da-grpo/train.py --max-updates 1 --max-response-tokens 512
```

If there are still architecture failures, distinguish `nvidia-smi`'s
**driver** CUDA version from the **runtime/toolkit** CUDA versions:

```bash
python -c 'import torch; print("PyTorch CUDA:", torch.version.cuda)'
nvcc --version
```

Avoid reinstalling CUDA, PyTorch or vLLM just to resolve a backend selection
problem unless this conservative configuration also fails.
