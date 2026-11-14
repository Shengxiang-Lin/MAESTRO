#!/usr/bin/env bash
# MAESTRO Stage4 -> NF4 Risk Head -> private-reward Unsloth GRPO.
set -Eeuo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$RUN_DIR/../.." && pwd)"
if [[ -f "$RUN_DIR/config.env" ]]; then source "$RUN_DIR/config.env"; fi
cd "$ROOT"

MODEL="${MODEL:-$ROOT/base_models/Qwen3-8B}"
STAGE1_DIR="${STAGE1_DIR:-$ROOT/environment/data/stage1}"
STAGE4_DIR="${STAGE4_DIR:-$ROOT/environment/data/stage4}"
RUN_ROOT="${RUN_ROOT:-$ROOT/workspace/maestro_8b}"
RISK_DATA="${RISK_DATA:-$STAGE4_DIR/private/window_risk}"
REWARD_STATES="${REWARD_STATES:-$STAGE4_DIR/private/decision_reward_states.jsonl}"
GRPO_DATA="${GRPO_DATA:-$STAGE4_DIR/private/risk_grpo}"
FEATURES_TRAIN="${FEATURES_TRAIN:-$RUN_ROOT/features/train-nf4}"
FEATURES_VAL="${FEATURES_VAL:-$RUN_ROOT/features/val-nf4}"
RISK_CKPT="${RISK_CKPT:-$RUN_ROOT/checkpoints/risk-head}"
GRPO_CKPT="${GRPO_CKPT:-$RUN_ROOT/checkpoints/grpo}"
PILOT_CKPT="${PILOT_CKPT:-$RUN_ROOT/checkpoints/grpo-pilot}"

MAX_LENGTH="${MAX_LENGTH:-16384}"
WINDOW_SIZE="${WINDOW_SIZE:-128}"
WINDOW_STRIDE="${WINDOW_STRIDE:-96}"
RISK_EPOCHS="${RISK_EPOCHS:-3}"
RISK_BATCH_SIZE="${RISK_BATCH_SIZE:-8}"
GRPO_EPOCHS="${GRPO_EPOCHS:-3}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:--1}"
GRPO_GENERATIONS="${GRPO_GENERATIONS:-4}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-4}"
GRPO_MAX_NEW_TOKENS="${GRPO_MAX_NEW_TOKENS:-256}"
GRPO_LR="${GRPO_LR:-1e-5}"
GRPO_STEER_LAYERS="${GRPO_STEER_LAYERS:-8}"
GRPO_STEER_STRENGTH="${GRPO_STEER_STRENGTH:-0.1}"
GRPO_STEER_THRESHOLD="${GRPO_STEER_THRESHOLD:-0.5}"
PILOT_PROMPTS="${PILOT_PROMPTS:-20}"
PILOT_STEPS="${PILOT_STEPS:-2}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
GRPO_RESUME_FROM="${GRPO_RESUME_FROM:-}"

say() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }
need() { [[ -s "$1" ]] || die "Missing/empty: $1"; }
need_model() { [[ -d "$MODEL" ]] || die "Set MODEL to an existing Qwen3-8B directory: $MODEL"; }

usage() {
  cat <<'EOF'
Usage: bash train_maestro_8b/run/pipeline.sh COMMAND

  all           export -> features -> risk -> prepare-grpo -> preflight -> pilot -> grpo
  export        Stage4 episodes/claim states -> train/val risk rows
  features      Frozen Qwen3-8B NF4 train/val window-feature cache
  risk          Train Risk Head only, using cached NF4 features
  prepare-grpo  Backfill missing private states, export and audit GRPO rows
  preflight     Check Unsloth KV/logprob/LoRA gradient compatibility
  pilot         Short GRPO run (20 prompts, 2 optimizer steps by default)
  grpo          Full Unsloth+TRL GRPO training
  status        Show generated data and checkpoint files
  help          This help

Override settings via environment variables or run/config.env:
  MODEL, STAGE1_DIR, STAGE4_DIR, RUN_ROOT, MAX_LENGTH,
  RISK_EPOCHS, RISK_BATCH_SIZE, GRPO_EPOCHS, GRPO_GENERATIONS,
  GRPO_GRAD_ACCUM, GRPO_MAX_STEPS, SKIP_COMPLETED, GRPO_RESUME_FROM

Run on a NEW RUN_ROOT to avoid mixing stale caches or experiments.
This script DOES NOT generate Environment data or launch a local LLM.
EOF
}
verify_cache() {
  local folder="$1" source="$2"
  need "$folder/manifest.json"
  need "$source"
  need "$folder/.input_sha256"
  local expected saved
  expected="$(sha256sum "$source" | cut -d' ' -f1)"
  saved="$(cat "$folder/.input_sha256")"
  [[ "$saved" == "$expected" ]] || die "Frozen feature cache input changed: $folder (use a new RUN_ROOT)"
  python - "$folder/manifest.json" "$MODEL" "$MAX_LENGTH" "$WINDOW_SIZE" "$WINDOW_STRIDE" <<'PY'
import json,sys
from pathlib import Path
manifest, model, maximum, window, stride = sys.argv[1:]
c=json.loads(Path(manifest).read_text())
assert c["schema"]=="maestro.risk_features.v1"
assert c["model"]==model, "model mismatch"
assert c["feature_model_quantization"]=="nf4", "NOT NF4"
assert c["max_length"]==int(maximum), "max_length mismatch"
assert c["window_size"]==int(window), "window mismatch"
assert c["window_stride"]==int(stride), "stride mismatch"
assert c["examples"]>0 and c["shards"], "empty feature cache"
assert all((Path(manifest).parent/p).is_file() for p,_ in c["shards"]), "missing shards"
print(f"Verified NF4 features: {c['examples']} examples")
PY
}
verify_risk() {
  need "$RISK_CKPT/risk_head.pt"
  need "$RISK_CKPT/risk_config.json"
  python - "$RISK_CKPT/risk_config.json" "$MODEL" "$WINDOW_SIZE" "$WINDOW_STRIDE" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
assert c["schema"]=="maestro.frozen_window_risk.v1"
assert c["base_model"]==sys.argv[2], "Risk Head backbone mismatch"
assert c["window_size"]==int(sys.argv[3]) and c["window_stride"]==int(sys.argv[4])
assert c["feature_model_quantization"]=="nf4", "Risk Head not NF4 compatible"
print("Verified frozen Risk Head checkpoint")
PY
}
export_data() {
  need "$STAGE1_DIR/questions/scenes.jsonl"
  need "$STAGE4_DIR/episodes.jsonl"
  need "$STAGE4_DIR/private/claim_state_supervision.jsonl"
  if [[ "$SKIP_COMPLETED" == "1" && -s "$RISK_DATA/train.jsonl" && -f "$RISK_DATA/summary.json" ]]; then
    say "Reusing risk samples: $RISK_DATA"; return
  fi
  say "Export risk samples"
  python -m train_maestro_8b.data.export_risk \
    --scenes "$STAGE1_DIR/questions/scenes.jsonl" \
    --episodes "$STAGE4_DIR/episodes.jsonl" \
    --states "$STAGE4_DIR/private/claim_state_supervision.jsonl" \
    --output_dir "$RISK_DATA"
  need "$RISK_DATA/train.jsonl"
}
features_one() {
  local split="$1" folder="$2"
  need "$RISK_DATA/$split.jsonl"
  if [[ -f "$folder/manifest.json" ]]; then
    verify_cache "$folder" "$RISK_DATA/$split.jsonl"
    [[ "$SKIP_COMPLETED" == "1" ]] && return
    die "Refusing to overwrite existing features. Use another RUN_ROOT."
  fi
  if [[ -d "$folder" && -n "$(ls -A "$folder")" ]]; then
    die "Incomplete feature cache $folder; move it aside or use a new RUN_ROOT."
  fi
  say "NF4 frozen feature extraction: $split"
  python -m train_maestro_8b.risk.extract \
    --model "$MODEL" --input "$RISK_DATA/$split.jsonl" \
    --output_dir "$folder" --load_in_4bit \
    --max_length "$MAX_LENGTH" \
    --window_size "$WINDOW_SIZE" --window_stride "$WINDOW_STRIDE"
  sha256sum "$RISK_DATA/$split.jsonl" | cut -d' ' -f1 > "$folder/.input_sha256"
  verify_cache "$folder" "$RISK_DATA/$split.jsonl"
}
extract_features() {
  need_model
  features_one train "$FEATURES_TRAIN"
  features_one val "$FEATURES_VAL"
}
train_risk() {
  verify_cache "$FEATURES_TRAIN" "$RISK_DATA/train.jsonl"
  verify_cache "$FEATURES_VAL" "$RISK_DATA/val.jsonl"
  if [[ -f "$RISK_CKPT/risk_config.json" ]]; then
    verify_risk
    [[ "$SKIP_COMPLETED" == "1" ]] && { say "Reusing Risk Head"; return; }
    die "Existing Risk Head checkpoint; choose a new RUN_ROOT."
  fi
  say "Train only Risk Head"
  python -m train_maestro_8b.risk.train \
    --model "$MODEL" --features_dir "$FEATURES_TRAIN" \
    --eval_features_dir "$FEATURES_VAL" --output_dir "$RISK_CKPT" \
    --epochs "$RISK_EPOCHS" --batch_size "$RISK_BATCH_SIZE" \
    --max_length "$MAX_LENGTH" --window_size "$WINDOW_SIZE" \
    --window_stride "$WINDOW_STRIDE"
  verify_risk
}
prepare_grpo() {
  need "$RISK_DATA/train.jsonl"
  need "$STAGE4_DIR/main_turns.jsonl"
  if [[ ! -s "$REWARD_STATES" ]]; then
    say "Backfill missing private reward state from Stage4 journals (no LLM)"
    python -m train_maestro_8b.data.backfill_rewards \
      --stage4 "$STAGE4_DIR" \
      --scenes "$STAGE1_DIR/questions/scenes.jsonl" \
      --output "$REWARD_STATES" \
      --max_parallel_tasks 4 --max_retries 1 --model_alias model_1
  fi
  need "$REWARD_STATES"
  if [[ "$SKIP_COMPLETED" == "1" && -s "$GRPO_DATA/train.jsonl" && -f "$GRPO_DATA/summary.json" ]]; then
    say "Reusing GRPO data"
  else
    say "Export GRPO private-state scoring examples"
    python -m train_maestro_8b.data.export_policy \
      --risk_dir "$RISK_DATA" --main_turns "$STAGE4_DIR/main_turns.jsonl" \
      --decision_states "$REWARD_STATES" --output_dir "$GRPO_DATA"
  fi
  need "$GRPO_DATA/train.jsonl"
  say "Deterministic reward sanity audit"
  mkdir -p "$RUN_ROOT"
  python -m train_maestro_8b.data.audit_rewards \
    --train_file "$GRPO_DATA/train.jsonl" \
    --output_jsonl "$RUN_ROOT/reward_reference_audit.jsonl"
}
common_args=()
init_grpo_args() {
  common_args=(
    --model "$MODEL" --risk_checkpoint "$RISK_CKPT"
    --train_file "$GRPO_DATA/train.jsonl"
    --backend unsloth --load_in_4bit --max_length "$MAX_LENGTH"
    --max_new_tokens "$GRPO_MAX_NEW_TOKENS"
    --num_generations "$GRPO_GENERATIONS"
    --gradient_accumulation_steps "$GRPO_GRAD_ACCUM"
    --steer_last_n "$GRPO_STEER_LAYERS"
    --steer_strength "$GRPO_STEER_STRENGTH"
    --steer_threshold "$GRPO_STEER_THRESHOLD" --lr "$GRPO_LR"
  )
}
preflight() {
  need_model; verify_risk; need "$GRPO_DATA/train.jsonl"
  say "Unsloth RiskKV preflight: sampling/logprob/LoRA gradients"
  python -m train_maestro_8b.policy.train \
    --model "$MODEL" --risk_checkpoint "$RISK_CKPT" \
    --train_file "$GRPO_DATA/train.jsonl" \
    --output_dir "$GRPO_CKPT" --backend unsloth \
    --load_in_4bit --max_length "$MAX_LENGTH" \
    --max_new_tokens "$GRPO_MAX_NEW_TOKENS" \
    --steer_last_n "$GRPO_STEER_LAYERS" \
    --steer_strength "$GRPO_STEER_STRENGTH" \
    --steer_threshold "$GRPO_STEER_THRESHOLD" --preflight_only
}
pilot() {
  need_model; verify_risk; need "$GRPO_DATA/train.jsonl"
  if [[ "$SKIP_COMPLETED" == "1" && -f "$PILOT_CKPT/risk_policy_config.json" ]]; then
    say "Reusing completed GRPO pilot"; return
  fi
  if [[ -d "$PILOT_CKPT" && -n "$(ls -A "$PILOT_CKPT")" ]]; then
    die "Incomplete GRPO pilot at $PILOT_CKPT; use a new PILOT_CKPT or move old files aside."
  fi
  init_grpo_args
  say "Pilot GRPO"
  python -m train_maestro_8b.policy.train "${common_args[@]}" \
    --output_dir "$PILOT_CKPT" \
    --max_prompts "$PILOT_PROMPTS" --max_steps "$PILOT_STEPS" \
    --logging_steps 1 --reward_debug_jsonl "$RUN_ROOT/reward_pilot.jsonl"
}
grpo() {
  need_model; verify_risk; need "$GRPO_DATA/train.jsonl"
  if [[ "$SKIP_COMPLETED" == "1" && -f "$GRPO_CKPT/risk_policy_config.json" ]]; then
    say "Reusing completed GRPO checkpoint"; return
  fi
  if [[ -d "$GRPO_CKPT" && -n "$(ls -A "$GRPO_CKPT")" && -z "$GRPO_RESUME_FROM" ]]; then
    die "Existing GRPO output $GRPO_CKPT; set GRPO_RESUME_FROM or a new RUN_ROOT."
  fi
  init_grpo_args
  local resume=()
  [[ -n "$GRPO_RESUME_FROM" ]] && resume=(--resume_from_checkpoint "$GRPO_RESUME_FROM")
  say "Full Unsloth + TRL risk-aware GRPO"
  python -m train_maestro_8b.policy.train "${common_args[@]}" \
    --output_dir "$GRPO_CKPT" --epochs "$GRPO_EPOCHS" \
    --max_steps "$GRPO_MAX_STEPS" --logging_steps 5 --save_steps 100 \
    --reward_debug_jsonl "$RUN_ROOT/reward_grpo.jsonl" "${resume[@]}"
}
status() {
  say "MODEL=$MODEL | RUN_ROOT=$RUN_ROOT"
  for file in "$STAGE4_DIR/main_turns.jsonl" "$RISK_DATA/train.jsonl" \
              "$RISK_DATA/val.jsonl" "$GRPO_DATA/train.jsonl"; do
    [[ -f "$file" ]] && printf '%7s rows %s\n' "$(wc -l < "$file")" "$file" ||
                       printf 'MISSING      %s\n' "$file"
  done
  for file in "$REWARD_STATES" "$FEATURES_TRAIN/manifest.json" \
              "$FEATURES_VAL/manifest.json" "$RISK_CKPT/risk_config.json" \
              "$GRPO_CKPT/risk_policy_config.json"; do
    [[ -f "$file" ]] && printf 'OK           %s\n' "$file" ||
                       printf 'MISSING      %s\n' "$file"
  done
}
case "${1:-help}" in
  all) export_data; extract_features; train_risk; prepare_grpo; preflight; pilot; grpo; status ;;
  export) export_data ;;
  features) extract_features ;;
  risk) train_risk ;;
  prepare-grpo) prepare_grpo ;;
  preflight) preflight ;;
  pilot) pilot ;;
  grpo) grpo ;;
  status) status ;;
  help|-h|--help) usage ;;
  *) usage >&2; die "Unknown pipeline command: $1" ;;
esac
