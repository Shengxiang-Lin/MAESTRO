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
│       ├── bench_qwen/                # local-Qwen O1 evaluation
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

## Run MAESTRO

```bash
python bench_orchestra_omnigaia.py \
  --config config/benchmarks/orchestra_omnigaia.yaml
```

## Run Orchestra-o1 baseline

```bash
python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia.yaml
```

The O1 launcher uses the shared root runner, shared SubAgent, and shared tools, but injects `baseline/orchestra-o1/orchestra_o1/main_agent.py` and the O1 prompt.

## Run Orchestra-o1 with local Qwen

```bash
python baseline/orchestra-o1/bench_qwen/bench_qwen_omnigaia.py \
  --config baseline/orchestra-o1/bench_qwen/orchestra_o1_omnigaia_qwen_grpo.yaml
```

## Train the Orchestra-o1 Qwen3-8B Main Agent

```bash
bash baseline/orchestra-o1/train_qwen3_8b/grpo/train_grpo_qwen3_8b.sh
```

Set `VERL_DIR`, `MODEL_PATH`, and optionally `DATA_DIR` before training. The training code remains with the O1 baseline because it trains the O1 Main-Agent policy.

## Scope

This migration covers Orchestra-o1 and the former Orchestra-o2/MAESTRO implementation only. Orchestra-o3/data-generation code is intentionally not included.
