#!/usr/bin/env bash
# Launch a real multi-round Orchestra-o1 evaluation with DA-GRPO LoRA.
# GPU is configured as gpus: [2] in config/model_config.yaml; change it there
# or use models: overrides in a benchmark config for another machine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINE_ROOT}/../.." && pwd)"
export QWEN3_8B_MODEL_PATH="${MODEL_PATH:-${QWEN3_8B_MODEL_PATH:-${REPO_ROOT}/base_models/Qwen3-8B}}"
export QWEN3_8B_DA_GRPO_ADAPTER_PATH="${ADAPTER_PATH:-${QWEN3_8B_DA_GRPO_ADAPTER_PATH:-${REPO_ROOT}/baseline/da-grpo/output/final_adapter}}"
CONFIG_PATH="${CONFIG_PATH:-${BASELINE_ROOT}/config/omnigaia/pretest/orchestra_o1_da_grpo_lora.yaml}"

cd "${REPO_ROOT}"
exec python "${BASELINE_ROOT}/bench_orchestra_o1_omnigaia.py" \
  --config "${CONFIG_PATH}" \
  "$@"
