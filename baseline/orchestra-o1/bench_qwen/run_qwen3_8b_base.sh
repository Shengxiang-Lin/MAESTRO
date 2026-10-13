#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINE_ROOT}/../.." && pwd)"

# vLLM startup/GPU/port/runtime settings live in config/model_config.yaml.
# MODEL_PATH can be a Hugging Face model id or a local directory.
export QWEN3_8B_MODEL_PATH="${MODEL_PATH:-${QWEN3_8B_MODEL_PATH:-Qwen/Qwen3-8B}}"
CONFIG_PATH="${CONFIG_PATH:-${BASELINE_ROOT}/config/benchmarks/orchestra_o1_omnigaia_qwen3_8b_base.yaml}"

cd "${REPO_ROOT}"
exec python "${BASELINE_ROOT}/bench_orchestra_o1_omnigaia.py" \
  --config "${CONFIG_PATH}" \
  "$@"
