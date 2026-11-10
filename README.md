# MAESTRO: Multi-Agent Evidence-aware Scheduling for Trustworthy Recovery and Orchestration

This repository contains MAESTRO and the Orchestra-o1 baseline used for comparison.

## Environment setup

The reference environment uses **Python 3.11**. `requirements.txt` pins its complete Python package snapshot, including vLLM 0.27.1, PyTorch 2.13.0, and Transformers 5.15.1.

Run these commands from the MAESTRO repository root (on the current server: `/root/autodl-tmp/MAESTRO`):

```bash
conda create --prefix ./X python=3.11
conda activate ./X
pip install -r requirements.txt
```

The reference platform is Linux x86_64, with PyTorch's CUDA 13.0 build (`2.13.0+cu130`). GPU runs require an NVIDIA driver compatible with that build; the Python interpreter and system dependencies are managed separately from `requirements.txt`.

## MAESTRO dataset workflows

Three trusted workflows and an independent round-local untrusted orchestration workflow share frozen Scene and evidence contracts.
All paths, model endpoints, worker counts and output directories live in
[environment/configs](environment/configs/README.md).

```bash
python -m environment generate-questions --config environment/configs/questions.yaml
python -m environment naturalize-subtasks --config environment/configs/subagent_reports.yaml
python -m environment generate-main-turns --config environment/configs/base_orchestration.yaml
python -m environment generate-untrusted-turns --config environment/configs/untrusted_orchestration.yaml
```

The first workflow constructs validated Questions and Macro DAGs from a fixed
candidate pool; the second materializes exact evidence traces and reviewed
Sub-Agent responses; the third generates trusted, cumulative-history
Main-Agent decision trajectories. The fourth workflow branches from an accepted
Main decision round, injects a last-batch primitive evidence fault, and
executes dynamic recovery and no-repair counterfactuals. Rejected cases are
not silently accepted or replaced to hit a quota.

Implementation details: [environment/README.md](environment/README.md).

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

`config/model_config.yaml` is the canonical shared model/deployment registry. It describes **how a model is called or served**: API endpoint, local model path, vLLM GPU allocation, tensor parallelism, context length, memory utilization, and related runtime options.

Benchmark YAML files describe **which model each experiment role uses**. For example:

```yaml
main_model: "local-qwen-27b"
sub_models:
  - "local-qwen-27b"

tool_models:
  image_analysis: "qwen3.7-flash"
  audio_analysis: "qwen3.5-omni-flash"
  video_vision: "qwen3.7-flash"
  video_audio: "qwen3.5-omni-flash"
  video_summary: "qwen3.7-flash"

evaluation_model: "qwen3.7-plus"
```

This separation lets the same benchmark config switch between API and local models by changing only the role names. `tool_models` controls the models used internally by the shared OmniGAIA image/audio/video tools. Each run records the resolved model roles and local runtime metadata in its result manifest.

Do not create misleading registry aliases such as a `gpt-*` key that silently calls a Qwen backend. API registry keys should reflect the real provider model identifier; local deployment names such as `qwen3-8b-local` and `local-qwen-27b` should clearly indicate that they are local services.

## Local models with vLLM

Main-Agent and Sub-Agent use the same LLM client for API and local models. A registry entry with `backend: vllm` is automatically exposed through a local OpenAI-compatible vLLM server and then consumed through the existing `AsyncLLM` path.

Local model definitions belong in `config/model_config.yaml`. For example, the repository contains a two-GPU 27B entry equivalent to:

```yaml
models:
  local-qwen-27b:
    backend: vllm
    model_path: "${QWEN3_27B_MODEL_PATH}"
    served_model_name: "local-qwen-27b"
    host: "127.0.0.1"
    port: 8804

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
    log_file: "logs/vllm/local-qwen-27b.log"

    extra_body:
      chat_template_kwargs:
        enable_thinking: true
```

Before running that model:

```bash
export QWEN3_27B_MODEL_PATH=/data/models/Qwen3-27B
```

Then the experiment YAML only selects it:

```yaml
main_model: "local-qwen-27b"
sub_models:
  - "local-qwen-27b"
```

If `tensor_parallel_size` is omitted and `gpus` is a list, it defaults to `len(gpus)`. `extra_args` can be a list of raw vLLM CLI arguments or a mapping for options not represented explicitly. `python_executable`, `env`, `log_file`, `quantization`, `kv_cache_dtype`, `pipeline_parallel_size`, and other common vLLM settings are also configurable in the model registry.

The launcher checks `/v1/models` before starting a process. A matching vLLM service is reused; an unrelated process on the configured port is never killed or replaced. Services started by MAESTRO are kept alive by default so repeated experiments do not reload the model; set `shutdown_on_exit: true` if you want benchmark-owned services stopped automatically.

### Experiment-local model overrides

Benchmark YAML may still contain an optional `models:` section for one-off deployment changes. These entries are **recursive partial overrides** on top of `config/model_config.yaml`, not replacement model definitions.

For example, to run the registered 27B model on GPUs 4 and 5 with another port:

```yaml
main_model: "local-qwen-27b"
sub_models:
  - "local-qwen-27b"

models:
  local-qwen-27b:
    gpus: [4, 5]
    port: 8810
```

All unspecified fields such as `model_path`, `dtype`, `tensor_parallel_size`, `gpu_memory_utilization`, and `extra_body` continue to inherit from the shared registry.

A role-only local 27B benchmark example is provided at:

```text
config/benchmarks/orchestra_omnigaia_local_vllm.example.yaml
```

This requires vLLM to be installed in the Python environment selected by `python_executable`, or in the current Python environment when `python_executable` is omitted.

## Run MAESTRO

```bash
python bench_orchestra_omnigaia.py \
  --config config/benchmarks/orchestra_omnigaia.yaml
```

For the registered local 27B model:

```bash
export QWEN3_27B_MODEL_PATH=/data/models/Qwen3-27B
python bench_orchestra_omnigaia.py \
  --config config/benchmarks/orchestra_omnigaia_local_vllm.example.yaml
```

## Run Orchestra-o1 baseline

```bash
python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia.yaml
```

The O1 launcher uses the shared root runner, shared SubAgent, shared model registry, and shared tools, but injects `baseline/orchestra-o1/orchestra_o1/main_agent.py` and the O1 prompt.

## Run Orchestra-o1 with local Qwen3-8B

The base Qwen3-8B and GRPO deployment definitions are also stored in `config/model_config.yaml`.

Base model:

```bash
export QWEN3_8B_MODEL_PATH=/data/models/Qwen3-8B
python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia_qwen3_8b_base.yaml
```

GRPO checkpoint:

```bash
export QWEN3_8B_GRPO_MODEL_PATH=/data/models/qwen3-8b-grpo
python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/bench_qwen/orchestra_o1_omnigaia_qwen_grpo.yaml
```

The older `baseline/orchestra-o1/bench_qwen/bench_qwen_omnigaia.py` launcher remains available for compatibility, but it now reuses the same root `config/model_config.yaml`; there is no second local-Qwen model registry to maintain.

## Train the Orchestra-o1 Qwen3-8B Main Agent

```bash
bash baseline/orchestra-o1/train_qwen3_8b/grpo/train_grpo_qwen3_8b.sh
```

Set `VERL_DIR`, `MODEL_PATH`, and optionally `DATA_DIR` before training. The training code remains with the O1 baseline because it trains the O1 Main-Agent policy.

## Scope

This migration covers Orchestra-o1 and the former Orchestra-o2/MAESTRO implementation only. Orchestra-o3/data-generation code is intentionally not included.


## 数据集接入

MAESTRO 和 Orchestra-o1 支持通过 `dataset_type: OmniGAIA` 或
`dataset_type: MM-BrowseComp` 选择数据集。`dataset_path` 指向原始 JSON/JSONL；
启动时自动生成统一格式 JSONL，并处理 MM-BrowseComp 题目前缀和本地图片附件。
省略类别时保持原有 OmniGAIA 行为。完整配置与运行示例见
[数据集配置说明](docs/datasets.md)。
