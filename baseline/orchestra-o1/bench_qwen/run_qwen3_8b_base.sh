#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINE_ROOT}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
VLLM_PORT="${VLLM_PORT:-8803}"
CONFIG_PATH="${CONFIG_PATH:-${BASELINE_ROOT}/config/benchmarks/orchestra_o1_omnigaia_qwen3_8b_base.yaml}"
export ORCHESTRA_TMP_DIR="${ORCHESTRA_TMP_DIR:-${REPO_ROOT}/tmp}"
mkdir -p "${ORCHESTRA_TMP_DIR}"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name qwen3-8b \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  > "${SCRIPT_DIR}/vllm_qwen3_8b_base.log" 2>&1 &
VLLM_PID=$!

for _ in $(seq 1 120); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited before becoming ready. See ${SCRIPT_DIR}/vllm_qwen3_8b_base.log" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

if ! curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "Timed out waiting for vLLM on port ${VLLM_PORT}" >&2
  exit 1
fi

python "${SCRIPT_DIR}/bench_qwen_omnigaia.py" --config "${CONFIG_PATH}" "$@"
