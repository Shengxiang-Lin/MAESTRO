# MAESTRO code layout

The repository uses MAESTRO's runtime as the canonical implementation. Orchestra-o1 is a Main-Agent baseline layered on top of that runtime rather than a second copy of the framework.

```text
MAESTRO/
├── base/                         # low-level agent / LLM primitives
├── benchmark/                    # shared OmniGAIA environment and benchmark tools
├── orchestra/                    # canonical shared runtime + MAESTRO policy
│   ├── base_main_agent.py        # behavior shared by both Main-Agents
│   ├── common/
│   ├── subagents/
│   ├── tools/
│   ├── runners/
│   │   ├── base_omnigaia_runner.py
│   │   └── omnigaia_runner.py    # MAESTRO specialization
│   ├── main_agent.py             # MAESTRO-only Main-Agent logic
│   ├── prompts/
│   ├── prospective.py
│   ├── routing.py
│   └── state.py
├── baseline/
│   └── orchestra-o1/
│       ├── bench_orchestra_o1_omnigaia.py
│       ├── orchestra_o1/
│       │   ├── main_agent.py     # O1-only policy wrapper
│       │   └── prompts/          # O1 Main-Agent prompt
│       ├── config/benchmarks/
│       ├── bench_qwen/
│       └── train_qwen3_8b/
├── config/
│   ├── model_config.yaml
│   └── benchmarks/
└── bench_orchestra_omnigaia.py
```

## Dependency rule

The shared implementation always points outward from `orchestra/`:

```text
orchestra/base + subagents + tools + runner
             ↑                 ↑
             │                 │
baseline/orchestra-o1      MAESTRO Main-Agent
```

Do not put shared SubAgent, runner, delegation tool, or common utility code under `baseline/orchestra-o1/`. If O1 and MAESTRO use the same behavior, the implementation belongs under `orchestra/` (or the lower-level `base/` / `benchmark/` packages when appropriate).

The O1 baseline should only contain code genuinely specific to O1: its Main-Agent policy/prompt, experiment configuration, local-Qwen evaluation helpers, and Main-Agent training code.

## MAESTRO

```bash
python bench_orchestra_omnigaia.py \
  --config config/benchmarks/orchestra_omnigaia.yaml
```

## Orchestra-o1

```bash
python baseline/orchestra-o1/bench_orchestra_o1_omnigaia.py \
  --config baseline/orchestra-o1/config/benchmarks/orchestra_o1_omnigaia.yaml
```

The launcher reuses `orchestra.runners.base_omnigaia_runner.BaseOmniGAIARunner`, the root SubAgent, and the root delegation/completion tools; it only injects the O1 Main-Agent and O1 prompt.

## O1 small-model evaluation and training

```bash
python baseline/orchestra-o1/bench_qwen/bench_qwen_omnigaia.py
bash baseline/orchestra-o1/bench_qwen/run_qwen3_8b_base.sh
bash baseline/orchestra-o1/bench_qwen/run_qwen3_8b_grpo.sh
bash baseline/orchestra-o1/train_qwen3_8b/grpo/train_grpo_qwen3_8b.sh
```

## Secrets and paths

- Provider secrets must be supplied through environment variables such as `DASHSCOPE_API_KEY`.
- Runtime scratch files use `ORCHESTRA_TMP_DIR`; the default is `<repo>/tmp`.
- Machine-specific absolute paths should not be committed.

## Scope

This layout refactor covers Orchestra-o1 and the former Orchestra-o2/MAESTRO code only. Orchestra-o3/data-generation work is intentionally outside this migration.
