# Orchestra-o1 Qwen3-8B training

This directory owns the small-model training artifacts used by the Orchestra-o1 baseline. It is intentionally separated from the main `orchestra/` implementation.

## Layout

- `ds_config.json`: DeepSpeed runtime configuration.
- `grpo/`: GRPO-specific trainer/reward code and launchers.
- `outputs/`: local training outputs (not committed).

## Path convention

All launchers should derive paths from this directory and the MAESTRO repository root. Do not hard-code a workstation path such as `/home/<user>/code/orchestra-o2/...`.

Runtime scratch files should use `ORCHESTRA_TMP_DIR` (default: `<MAESTRO>/tmp`). Provider credentials must come from environment variables such as `DASHSCOPE_API_KEY`; never commit API keys into YAML or shell scripts.

## Evaluation

The trained/exported model is evaluated through:

```bash
bash baseline/orchestra-o1/bench_qwen/run_qwen3_8b_grpo.sh
```

or by setting `MODEL_PATH` explicitly.

## Migration note

The directory boundary is part of the O1 baseline: training/evaluation assets belong here, while `base/` and `benchmark/` remain shared with MAESTRO. When changing the GRPO reward/trainer itself, treat that as a separate experiment change rather than a repository-layout refactor.
