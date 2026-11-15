#!/usr/bin/env bash
# Phase-0 MAESTRO: proven DA-GRPO path, without Stage-1 Risk Head or RiskKV.
# No custom rollout, no new Reward/Judge prompt; training calls DA train.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SCENES="${DA_STAGE4_SCENES:-environment/data/stage1/questions/scenes.jsonl}"
TURNS="${DA_STAGE4_TURNS:-environment/data/stage4/main_turns.jsonl}"
DATA="${DA_STAGE4_DATA:-dataset/data/da-grpo-stage4}"
MODEL="${DA_STAGE4_MODEL:-base_models/Qwen3-8B}"
OUTPUT="${DA_STAGE4_OUTPUT:-baseline/da-grpo/output_stage4}"
SMOKE_OUTPUT="${DA_STAGE4_SMOKE_OUTPUT:-baseline/da-grpo/output_stage4_smoke}"
SMOKE_STEPS="${DA_SMOKE_STEPS:-3}"
SMOKE_ROWS="${DA_SMOKE_ROWS:-16}"
SMOKE_COMPLETION="${DA_SMOKE_COMPLETION_TOKENS:-1536}"
PROMPT_CAP="${DA_STAGE4_PROMPT_CAP:-12288}"
REVIEWED_ONLY="${DA_STAGE4_REVIEWED_ONLY:-0}"

usage() {
  cat <<'HELP'
DA-first MAESTRO Stage4 GRPO (uses baseline/da-grpo training and reward unchanged)
  bash baseline/da-grpo/run_stage4.sh prepare    # Stage4 -> DA Parquet
  bash baseline/da-grpo/run_stage4.sh check      # preflight, no GPU or API
  bash baseline/da-grpo/run_stage4.sh judge      # test qwen3.7-flash API only
  bash baseline/da-grpo/run_stage4.sh smoke      # 3 optimizer steps, 16 rows, 1536 tokens
  bash baseline/da-grpo/run_stage4.sh train      # full GRPO, 3 epochs by default
  bash baseline/da-grpo/run_stage4.sh help

Environment:
  DA_STAGE4_SCENES, DA_STAGE4_TURNS, DA_STAGE4_DATA, DA_STAGE4_MODEL
  DA_STAGE4_OUTPUT, DA_STAGE4_SMOKE_OUTPUT
  DA_STAGE4_REVIEWED_ONLY=1 (optional strictly LLM-reviewed dataset)
  DA_SMOKE_STEPS=3, DA_SMOKE_ROWS=16, DA_SMOKE_COMPLETION_TOKENS=1536
  DA_MAX_STEPS=100 (optional cap for train command)
  DASHSCOPE_API_KEY (remote DA-GRPO Judge credentials)
Do not start concurrent RiskKV/DA-GRPO GPU jobs. No Stage-1 Risk Head is loaded.
HELP
}

need_data() {
  if [[ ! -f "$DATA/train.parquet" ]]; then
    echo "Missing $DATA/train.parquet; run: bash baseline/da-grpo/run_stage4.sh prepare" >&2
    exit 2
  fi
}

train_args=(--train-parquet "$DATA/train.parquet")

case "${1:-help}" in
  help|-h|--help) usage ;;
  prepare)
    for file in "$SCENES" "$TURNS"; do
      if [[ ! -f "$file" ]]; then echo "Missing input: $file" >&2; exit 2; fi
    done
    args=(--stage4-only --scenes "$SCENES" --untrusted-turns "$TURNS"
          --output-dir "$DATA" --tokenizer "$MODEL"
          --max-prompt-tokens "$PROMPT_CAP")
    if [[ "$REVIEWED_ONLY" == "1" ]]; then args+=(--require-llm-expert); fi
    python -m dataset.prepare_da_grpo "${args[@]}"
    ;;
  check)
    need_data
    python baseline/da-grpo/train.py "${train_args[@]}" --check
    ;;
  judge)
    need_data
    python baseline/da-grpo/train.py "${train_args[@]}" --test-judge
    ;;
  smoke)
    need_data
    if [[ -e "$SMOKE_OUTPUT/final_adapter" ]]; then
      echo "Smoke output contains final_adapter: $SMOKE_OUTPUT; select a fresh DA_STAGE4_SMOKE_OUTPUT" >&2
      exit 2
    fi
    python baseline/da-grpo/train.py "${train_args[@]}" \
      --max-steps "$SMOKE_STEPS" --max-train-rows "$SMOKE_ROWS" \
      --max-completion-tokens "$SMOKE_COMPLETION" --output-dir "$SMOKE_OUTPUT"
    ;;
  train)
    need_data
    args=(--output-dir "$OUTPUT")
    if [[ -n "${DA_MAX_STEPS:-}" ]]; then args+=(--max-steps "$DA_MAX_STEPS"); fi
    python baseline/da-grpo/train.py "${train_args[@]}" "${args[@]}"
    ;;
  *) echo "Unknown command: $1" >&2; usage >&2; exit 2 ;;
esac
