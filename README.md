# MAESTRO: Multi-Agent Evidence-aware Scheduling for Trustworthy Recovery and Orchestration

This repository contains MAESTRO and the Orchestra-o1 baseline used for comparison.

## Repository layout

```text
MAESTRO/
├── orchestra/                         # canonical shared runtime + MAESTRO Main-Agent
│   ├── base_main_agent.py             # Main-Agent mechanics shared by MAESTRO/O1
│   ├── subagents/                     # shared SubAgent implementation
│   ├── tools/                         # shared delegation/completion tools
│   ├── common/                        # shared orchestration utilities
│   ├── runners/                       # shared runner + MAESTRO specialization
│   ├── main_agent.py                  # MAESTRO-specific Main-Agent policy
│   ├── prospective.py                 # MAESTRO-specific logic
│   ├── routing.py
│   └── state.py
├── baseline/
│   └── orchestra-o1/
│       ├── orchestra_o1/
│       │   ├── main_agent.py          # O1 policy wrapper only
│       │   └── prompts/               # O1-specific Main-Agent prompt
│       ├── bench_orchestra_o1_omnigaia.py
│       ├── bench_qwen/                # local-Qwen O1 evaluation/training helpers
│       ├── train_qwen3_8b/            # O1 Main-Agent training
│       └── config/benchmarks/         # O1 experiment configs
├── base/                              # lower-level agent/LLM primitives
├── benchmark/                         # shared OmniGAIA environment/tooling
├── config/                            # shared model registry + MAESTRO configs
└── bench_orchestra_omnigaia.py        # MAESTRO entrypoint
```

The important dependency direction is:

```text
                orchestra/ shared runtime
                   ↑                ↑
                   │                │
        Orchestra-o1 policy     MAESTRO policy
        baseline/orchestra-o1   orchestra/main_agent.py
```

Orchestra-o1 does **not** carry its own SubAgent, runner, tools, or common runtime. It reuses the canonical implementation under `orchestra/` and only keeps the Main-Agent behavior that differs from MAESTRO.

## Model configuration

`config/model_config.yaml` is the shared model registry. Registry keys are real provider model identifiers used at call time; do not create entries such as `gpt-*` or `claude-*` that silently map to a different backend model.

Experiment YAML files choose models by role. For example:

```yaml
main_model: "qwen3.8-max"
sub_models:
  - "qwen3.8-max"

tool_models:
  image_analysis: "qwen3.7-flash"
  audio_analysis: "qwen3.5-omni-flash"
  video_vision: "qwen3.7-flash"
  video_audio: "qwen3.5-omni-flash"
  video_summary: "qwen3.7-flash"

evaluation_model: "qwen3.7-plus"
```

`tool_models` controls the models used internally by the shared OmniGAIA image/audio/video tools. MAESTRO and Orchestra-o1 use the same role names, so changing a tool backend only requires editing the benchmark YAML. Each run also writes the resolved model roles into its trajectory/CSV output and a `model_roles_<timestamp>.json` manifest.

## Local models with vLLM

Main-Agent and Sub-Agent can use a local model without changing their implementation. Add an experiment-local model definition with `backend: vllm`; the benchmark launcher starts (or reuses) an OpenAI-compatible vLLM server before the agents are created.

```yaml
main_model: "local-qwen-27b"
sub_models:
  - "local-qwen-27b"

models:
  local-qwen-27b:
    backend: vllm
    model_path: "/data/models/Qwen-27B"
    served_model_name: "local-qwen-27b"
    host: "127.0.0.1"
    port: 8804

    # Multi-GPU: CUDA_VISIBLE_DEVICES=0,1 and vLLM tensor parallelism = 2.
    gpus: [0, 1]
    tensor_parallel_size: 2
    dtype: "bfloat16"
    gpu_memory_utilization: 0.90
    max_model_len: 32768
    max_num_seqs: 8
    max_num_batched_tokens: 8192
    reasoning_parser: "qwen3"
    enforce_eager: true

    reuse_existing: true
    auto_start: true
    startup_timeout: 1200
    shutdown_on_exit: false

    extra_body:
      chat_template_kwargs:
        enable_thinking: true
```

If `tensor_parallel_size` is omitted and `gpus` is a list, it defaults to `len(gpus)`. `extra_args` can be a list of raw vLLM CLI arguments or a mapping for options not represented explicitly. `python_executable`, `env`, `log_file`, `quantization`, `kv_cache_dtype`, `pipeline_parallel_size`, and other common vLLM settings can also be configured in YAML.

The launcher checks `/v1/models` before starting a process. A matching vLLM service is reused; an unrelated process on the configured port is never killed or replaced. Services started by MAESTRO are kept alive by default so repeated experiments do not reload the model; set `shutdown_on_exit: true` if you want benchmark-owned services stopped automatically.

A complete two-GPU 27B example is provided at:

```text
config/benchmarks/orchestra_omnigaia_local_vllm.example.yaml
```

This requires vLLM to be installed in the Python environment selected by `python_executable` (or the current Python environment when it is omitted).

## Run MAESTRO

```bash
python bench_orchestra_omnigaia.py \
  --config config/benchmarks/orchestra_omnigaia.yaml
```

For a local model, the command stays the same; only the YAML changes:

```bash
python bench_orchestra_omnigaia.py \
  --config config/benchmarks/orchestra_omnigaia_local_vllm.example.yaml
```

## Run Orchestra-o1 baseline

```bash
python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia.yaml
```

The O1 launcher uses the shared root runner, shared SubAgent, and shared tools, but injects `baseline/orchestra-o1/orchestra_o1/main_agent.py` and the O1 prompt.

## Run Orchestra-o1 with local Qwen3-8B

The Qwen3-8B config now starts/reuses vLLM itself. Set `QWEN3_8B_MODEL_PATH` (or put the absolute path directly in YAML), then run the normal O1 entrypoint:

```bash
python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia_qwen3_8b_base.yaml
```

The GRPO checkpoint config uses the same mechanism with `QWEN3_8B_GRPO_MODEL_PATH`:

```bash
python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/bench_qwen/orchestra_o1_omnigaia_qwen_grpo.yaml
```

The older `baseline/orchestra-o1/bench_qwen/bench_qwen_omnigaia.py` launcher remains available for compatibility, but it is no longer required just to start a local vLLM model.

## Train the Orchestra-o1 Qwen3-8B Main Agent

```bash
bash baseline/orchestra-o1/train_qwen3_8b/grpo/train_grpo_qwen3_8b.sh
```

Set `VERL_DIR`, `MODEL_PATH`, and optionally `DATA_DIR` before training. The training code remains with the O1 baseline because it trains the O1 Main-Agent policy.

## Scope

This migration covers Orchestra-o1 and the former Orchestra-o2/MAESTRO implementation only. Orchestra-o3/data-generation code is intentionally not included.
