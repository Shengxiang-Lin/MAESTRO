#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINE_ROOT}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${QWEN3_8B_GRPO_MODEL_PATH:-${BASELINE_ROOT}/train_qwen3_8b/outputs/qwen3-8b-grpo}}"
if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "GRPO model not found: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH=/path/to/exported/model or QWEN3_8B_GRPO_MODEL_PATH in the environment." >&2
  exit 2
fi

# vLLM startup/GPU/port/runtime settings now live in the benchmark YAML.
export QWEN3_8B_GRPO_MODEL_PATH="${MODEL_PATH}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/orchestra_o1_omnigaia_qwen_grpo.yaml}"

cd "${REPO_ROOT}"
exec python "${BASELINE_ROOT}/bench_orchestra_o1_omnigaia.py" \
  --config "${CONFIG_PATH}" \
  "$@"
