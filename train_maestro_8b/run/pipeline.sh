#!/usr/bin/env bash
# MAESTRO Stage2: default DA/TRL GRPO; optional legacy Stage1 RiskKV route.
set -Eeuo pipefail

RUN_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$RUN_DIR/../.." && pwd)"
if [[ -f "$RUN_DIR/config.env" ]]; then source "$RUN_DIR/config.env"; fi
cd "$ROOT"

MODEL="${MODEL:-$ROOT/base_models/Qwen3-8B}"
STAGE1_DIR="${STAGE1_DIR:-$ROOT/environment/data/stage1}"
STAGE4_DIR="${STAGE4_DIR:-$ROOT/environment/data/stage4}"
RUN_ROOT="${RUN_ROOT:-$ROOT/workspace/maestro_8b}"
STAGE2_TRAINER="${STAGE2_TRAINER:-da}"     # da (default) or riskkv
DA_REWARD_MODE="${DA_REWARD_MODE:-hybrid}"  # hybrid | deterministic | judge_only | da
DA_DATA="${DA_DATA:-$RUN_ROOT/stage2_da}"
DA_GRPO_CKPT="${DA_GRPO_CKPT:-$RUN_ROOT/checkpoints/stage2-${DA_REWARD_MODE}-grpo}"
DA_PILOT_CKPT="${DA_PILOT_CKPT:-$RUN_ROOT/checkpoints/stage2-${DA_REWARD_MODE}-pilot}"
DA_PILOT_STEPS="${DA_PILOT_STEPS:-3}"
DA_PILOT_ROWS="${DA_PILOT_ROWS:-16}"
DA_PILOT_NEW_TOKENS="${DA_PILOT_NEW_TOKENS:-1536}"
DA_REVIEWED_ONLY="${DA_REVIEWED_ONLY:-0}"
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
GRPO_GENERATIONS="${GRPO_GENERATIONS:-8}"
GRPO_GRAD_ACCUM="${GRPO_GRAD_ACCUM:-1}"
GRPO_MAX_NEW_TOKENS="${GRPO_MAX_NEW_TOKENS:-4096}"
GRPO_MAX_PROMPT_TOKENS="${GRPO_MAX_PROMPT_TOKENS:-12288}"
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-1.0}"
PILOT_MAX_NEW_TOKENS="${PILOT_MAX_NEW_TOKENS:-4096}"
GRPO_LR="${GRPO_LR:-5e-5}"
GRPO_WEIGHT_DECAY="${GRPO_WEIGHT_DECAY:-0.01}"
GRPO_OPTIM="${GRPO_OPTIM:-adamw_8bit}"
GRPO_MAX_GRAD_NORM="${GRPO_MAX_GRAD_NORM:-1.0}"
GRPO_LORA_RANK="${GRPO_LORA_RANK:-32}"
GRPO_LORA_ALPHA="${GRPO_LORA_ALPHA:-64}"
GRPO_LOGGING_STEPS="${GRPO_LOGGING_STEPS:-1}"
GRPO_SAVE_STEPS="${GRPO_SAVE_STEPS:-50}"
GRPO_ZERO_ADV_STRATEGY="${GRPO_ZERO_ADV_STRATEGY:-skip}"
GRPO_REWARD_MODE="${GRPO_REWARD_MODE:-deterministic}"
GRPO_JUDGE_REGISTRY="${GRPO_JUDGE_REGISTRY:-config/model_config.yaml}"
GRPO_JUDGE_MODEL="${GRPO_JUDGE_MODEL:-qwen3.7-flash}"
GRPO_JUDGE_API_KEY_ENV="${GRPO_JUDGE_API_KEY_ENV:-DASHSCOPE_API_KEY}"
GRPO_JUDGE_WORKERS="${GRPO_JUDGE_WORKERS:-4}"
GRPO_JUDGE_TIMEOUT="${GRPO_JUDGE_TIMEOUT:-90}"
GRPO_JUDGE_RETRIES="${GRPO_JUDGE_RETRIES:-2}"
GRPO_JUDGE_WEIGHT="${GRPO_JUDGE_WEIGHT:-0.4}"
GRPO_JUDGE_CACHE_JSONL="${GRPO_JUDGE_CACHE_JSONL:-$RUN_ROOT/semantic_judge_cache.jsonl}"
GRPO_ZERO_ADV_SFT_WEIGHT="${GRPO_ZERO_ADV_SFT_WEIGHT:-0.25}"
GRPO_SFT_WARMUP="${GRPO_SFT_WARMUP:-0}"
GRPO_SFT_WARMUP_SAMPLES="${GRPO_SFT_WARMUP_SAMPLES:-0}"
PILOT_REQUIRE_GRAD="${PILOT_REQUIRE_GRAD:-1}"
GRPO_ROLLOUT_ATTENTION="${GRPO_ROLLOUT_ATTENTION:-sdpa}"
GRPO_VLLM_ENGINE="${GRPO_VLLM_ENGINE:-1}"
GRPO_VLLM_MEMORY_UTILIZATION="${GRPO_VLLM_MEMORY_UTILIZATION:-0.70}"
GRPO_PREFLIGHT_MAX_LOGPROB_DRIFT="${GRPO_PREFLIGHT_MAX_LOGPROB_DRIFT:-}"
GRPO_STEER_LAYERS="${GRPO_STEER_LAYERS:-8}"
GRPO_STEER_STRENGTH="${GRPO_STEER_STRENGTH:-0.1}"
GRPO_STEER_THRESHOLD="${GRPO_STEER_THRESHOLD:-0.5}"
PILOT_PROMPTS="${PILOT_PROMPTS:-1}"
PILOT_STEPS="${PILOT_STEPS:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
GRPO_RESUME_FROM="${GRPO_RESUME_FROM:-}"

say() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }
need() { [[ -s "$1" ]] || die "Missing/empty: $1"; }
need_model() { [[ -d "$MODEL" ]] || die "Set MODEL to an existing Qwen3-8B directory: $MODEL"; }

usage() {
  cat <<'EOF'
Usage: bash train_maestro_8b/run/pipeline.sh COMMAND

  all           Default DA: prepare-grpo -> preflight -> pilot -> grpo
                With STAGE2_TRAINER=riskkv: old Stage1 Risk Head -> RiskKV
  export        Stage4 episodes/claim states -> train/val risk rows
  features      Frozen Qwen3-8B NF4 train/val window-feature cache
  risk          Train Risk Head only, using cached NF4 features
  prepare-grpo  DA: export Stage4 -> Parquet; RiskKV: private-state reward rows
  preflight     DA: baseline package/input checks; RiskKV: cache/logprob
  pilot         DA: 3-step smoke; hybrid: local validity + LLM semantic Judge
                RiskKV: previous strict Action-reward pilot
  audit-pilot   DA: review Judge reward diagnostics; RiskKV: invalid reasons
  grpo          Main Stage2 GRPO, defaults to native DA/TRL training
  inspect-io    DA: export preflight; RiskKV: native prompt/reference audit
  judge         Stage2: check semantic Judge API (or DA reference API) without GPU
  status        Show generated data and checkpoint files
  help          This help

Override settings via environment variables or run/config.env:
  STAGE2_TRAINER (da|riskkv), DA_REWARD_MODE (hybrid|deterministic|judge_only|da),
  DA_DATA, DA_GRPO_CKPT, DA_PILOT_CKPT,
  DA_PILOT_STEPS, DA_PILOT_ROWS, DA_PILOT_NEW_TOKENS, DA_REVIEWED_ONLY,
  MODEL, STAGE1_DIR, STAGE4_DIR, RUN_ROOT, MAX_LENGTH,
  RISK_EPOCHS, RISK_BATCH_SIZE, GRPO_EPOCHS, GRPO_GENERATIONS,
  GRPO_GRAD_ACCUM, GRPO_MAX_PROMPT_TOKENS, GRPO_MAX_NEW_TOKENS,
  GRPO_TEMPERATURE, PILOT_MAX_NEW_TOKENS, GRPO_MAX_STEPS,
  GRPO_SFT_WARMUP, GRPO_SFT_WARMUP_SAMPLES, PILOT_REQUIRE_GRAD,
  GRPO_LR, GRPO_WEIGHT_DECAY, GRPO_OPTIM, GRPO_MAX_GRAD_NORM,
  GRPO_LORA_RANK, GRPO_LORA_ALPHA, GRPO_LOGGING_STEPS, GRPO_SAVE_STEPS,
  GRPO_ZERO_ADV_STRATEGY, GRPO_ZERO_ADV_SFT_WEIGHT,
  GRPO_REWARD_MODE, GRPO_JUDGE_MODEL, GRPO_JUDGE_API_KEY_ENV,
  GRPO_JUDGE_REGISTRY, GRPO_JUDGE_WORKERS, GRPO_JUDGE_TIMEOUT,
  GRPO_JUDGE_RETRIES, GRPO_JUDGE_WEIGHT, GRPO_JUDGE_CACHE_JSONL,
  SKIP_COMPLETED, GRPO_RESUME_FROM,
  GRPO_VLLM_ENGINE, GRPO_VLLM_MEMORY_UTILIZATION, GRPO_PREFLIGHT_MAX_LOGPROB_DRIFT

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
  if [[ "$STAGE2_TRAINER" == "da" ]]; then
    need "$STAGE1_DIR/questions/scenes.jsonl"
    need "$STAGE4_DIR/main_turns.jsonl"
    if [[ "$DA_REWARD_MODE" != "da" ]]; then
      need "$REWARD_STATES"
    fi
    if [[ "$SKIP_COMPLETED" == "1" && -s "$DA_DATA/train.parquet" && -f "$DA_DATA/summary.json" ]] &&
      python - "$DA_DATA/summary.json" "$STAGE1_DIR/questions/scenes.jsonl" "$STAGE4_DIR/main_turns.jsonl" <<'PY'
import hashlib,json,sys
from pathlib import Path
def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()
s=json.loads(Path(sys.argv[1]).read_text())
if (s.get("schema") != "maestro.stage2_da_grpo.v1"
        or s.get("scenes_sha256") != digest(sys.argv[2])
        or s.get("turns_sha256") != digest(sys.argv[3])):
    raise SystemExit(1)
PY
    then
      say "Reusing verified Stage2 DA Parquet: $DA_DATA/train.parquet"
      return
    fi
    say "Stage2 DA export: exact O1 prompts + reward-only expert metadata"
    local extra=()
    [[ "$DA_REVIEWED_ONLY" == "1" ]] && extra+=(--reviewed_only)
    python -m train_maestro_8b.data.export_da_grpo \
      --scenes "$STAGE1_DIR/questions/scenes.jsonl" \
      --turns "$STAGE4_DIR/main_turns.jsonl" \
      --output_dir "$DA_DATA" --tokenizer "$MODEL" \
      --max_prompt_tokens "$GRPO_MAX_PROMPT_TOKENS" "${extra[@]}"
    need "$DA_DATA/train.parquet"
    return
  fi
  [[ "$STAGE2_TRAINER" == "riskkv" ]] || die "STAGE2_TRAINER must be da or riskkv"
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
stage2_da_args=()
init_stage2_da_args() {
  stage2_da_args=(
    --model "$MODEL" --train_file "$DA_DATA/train.parquet"
    --max_length "$GRPO_MAX_PROMPT_TOKENS"
    --max_new_tokens "$GRPO_MAX_NEW_TOKENS"
    --num_generations "$GRPO_GENERATIONS"
    --gradient_accumulation_steps "$GRPO_GRAD_ACCUM"
    --vllm_memory_utilization "$GRPO_VLLM_MEMORY_UTILIZATION"
    --lr "$GRPO_LR" --weight_decay "$GRPO_WEIGHT_DECAY"
    --temperature "$GRPO_TEMPERATURE"
    --lora_rank "$GRPO_LORA_RANK" --lora_alpha "$GRPO_LORA_ALPHA"
    --optim "$GRPO_OPTIM"
    --max_grad_norm "$GRPO_MAX_GRAD_NORM"
    --judge_registry "$GRPO_JUDGE_REGISTRY"
    --judge_model "$GRPO_JUDGE_MODEL"
    --judge_api_key_env "$GRPO_JUDGE_API_KEY_ENV"
    --judge_workers "$GRPO_JUDGE_WORKERS"
    --reward_mode "$DA_REWARD_MODE"
    --decision_states "$REWARD_STATES"
    --judge_weight "$GRPO_JUDGE_WEIGHT"
    --judge_timeout "$GRPO_JUDGE_TIMEOUT"
    --judge_retries "$GRPO_JUDGE_RETRIES"
    --judge_cache_jsonl "$GRPO_JUDGE_CACHE_JSONL"
    --logging_steps "$GRPO_LOGGING_STEPS"
    --save_steps "$GRPO_SAVE_STEPS"
  )
}
stage2_da_check() {
  need_model; need "$DA_DATA/train.parquet"
  init_stage2_da_args
  python -m train_maestro_8b.policy.train da "${stage2_da_args[@]}" \
    --output_dir "$DA_GRPO_CKPT" --check
}
stage2_da_judge() {
  need "$DA_DATA/train.parquet"
  init_stage2_da_args
  python -m train_maestro_8b.policy.train da "${stage2_da_args[@]}" \
    --output_dir "$DA_GRPO_CKPT" --test_judge
}
stage2_da_pilot() {
  need_model; need "$DA_DATA/train.parquet"
  if [[ "$SKIP_COMPLETED" == "1" && -f "$DA_PILOT_CKPT/stage2_da_training.json" ]]; then
    say "Reusing completed Stage2 DA Pilot"; return
  fi
  if [[ -d "$DA_PILOT_CKPT" && -n "$(ls -A "$DA_PILOT_CKPT")" ]]; then
    die "Stage2 DA pilot output exists: $DA_PILOT_CKPT; choose fresh DA_PILOT_CKPT"
  fi
  init_stage2_da_args
  say "Stage2 native TRL Pilot: reward=$DA_REWARD_MODE; risk head OFF"
  python -m train_maestro_8b.policy.train da "${stage2_da_args[@]}" \
    --output_dir "$DA_PILOT_CKPT" --max_steps "$DA_PILOT_STEPS" \
    --max_train_rows "$DA_PILOT_ROWS" --max_new_tokens "$DA_PILOT_NEW_TOKENS"
  need "$DA_PILOT_CKPT/reward_diagnostics.jsonl"
  python - "$DA_PILOT_CKPT/reward_diagnostics.jsonl" <<'PY'
import json,sys
from collections import defaultdict
groups=defaultdict(list)
for line in open(sys.argv[1], encoding="utf-8"):
    x=json.loads(line)
    groups[(x["batch_id"],x["group_index"])].append(float(x["reward"]))
nonzero=sum(max(vals)-min(vals)>1e-7 for vals in groups.values())
valid=judged=0
for line in open(sys.argv[1], encoding="utf-8"):
    row=json.loads(line)
    valid+=int(row.get("valid") is True)
    judged+=int(row.get("judge_reward") is not None)
print(f"Stage2 Pilot: {len(groups)} groups, {nonzero} nonzero-variance, {valid} valid Actions, {judged} semantic Judge scores",flush=True)
if not groups or not nonzero:
    raise SystemExit("No GRPO preference signal; inspect action validity and JSON normalization before full training")
if valid == 0 and judged == 0 and any(
    "valid" in json.loads(line)
    for line in open(sys.argv[1], encoding="utf-8")
):
    raise SystemExit("Stage2 Hybrid Pilot has no valid Action; stop before full GRPO")
PY
}
stage2_da_grpo() {
  need_model; need "$DA_DATA/train.parquet"
  if [[ "$SKIP_COMPLETED" == "1" && -f "$DA_GRPO_CKPT/stage2_da_training.json" ]]; then
    say "Reusing completed Stage2 DA GRPO"; return
  fi
  if [[ -d "$DA_GRPO_CKPT" && -n "$(ls -A "$DA_GRPO_CKPT")" && -z "$GRPO_RESUME_FROM" ]]; then
    die "Stage2 DA output exists: $DA_GRPO_CKPT; use a fresh DA_GRPO_CKPT or GRPO_RESUME_FROM"
  fi
  init_stage2_da_args
  local resume=()
  [[ -n "$GRPO_RESUME_FROM" ]] && resume=(--resume_from_checkpoint "$GRPO_RESUME_FROM")
  say "Main Stage2 training: Unsloth + TRL GRPOTrainer, reward=$DA_REWARD_MODE, Risk Head OFF"
  python -m train_maestro_8b.policy.train da "${stage2_da_args[@]}" \
    --output_dir "$DA_GRPO_CKPT" --epochs "$GRPO_EPOCHS" \
    --max_steps "$GRPO_MAX_STEPS" "${resume[@]}"
}
common_args=()
init_grpo_args() {
  common_args=(
    --model "$MODEL" --risk_checkpoint "$RISK_CKPT"
    --train_file "$GRPO_DATA/train.jsonl"
    --backend unsloth --load_in_4bit --max_length "$GRPO_MAX_PROMPT_TOKENS"
    --rollout_attention "$GRPO_ROLLOUT_ATTENTION"
    --max_new_tokens "$GRPO_MAX_NEW_TOKENS" --temperature "$GRPO_TEMPERATURE"
    --num_generations "$GRPO_GENERATIONS"
    --gradient_accumulation_steps "$GRPO_GRAD_ACCUM"
    --steer_last_n "$GRPO_STEER_LAYERS"
    --steer_strength "$GRPO_STEER_STRENGTH"
    --steer_threshold "$GRPO_STEER_THRESHOLD" --lr "$GRPO_LR"
    --weight_decay "$GRPO_WEIGHT_DECAY" --optim "$GRPO_OPTIM"
    --max_grad_norm "$GRPO_MAX_GRAD_NORM"
    --lora_rank "$GRPO_LORA_RANK" --lora_alpha "$GRPO_LORA_ALPHA"
    --zero_adv_strategy "$GRPO_ZERO_ADV_STRATEGY"
    --reward_mode "$GRPO_REWARD_MODE"
    --judge_registry "$GRPO_JUDGE_REGISTRY"
    --judge_model "$GRPO_JUDGE_MODEL"
    --judge_api_key_env "$GRPO_JUDGE_API_KEY_ENV"
    --judge_workers "$GRPO_JUDGE_WORKERS"
    --judge_timeout "$GRPO_JUDGE_TIMEOUT"
    --judge_retries "$GRPO_JUDGE_RETRIES"
    --judge_weight "$GRPO_JUDGE_WEIGHT"
    --judge_cache_jsonl "$GRPO_JUDGE_CACHE_JSONL"
    --zero_adv_sft_weight "$GRPO_ZERO_ADV_SFT_WEIGHT"
  )
  if [[ "$GRPO_VLLM_ENGINE" == "1" ]]; then
    common_args+=(--vllm_engine --vllm_memory_utilization "$GRPO_VLLM_MEMORY_UTILIZATION")
  fi
  if [[ -n "$GRPO_PREFLIGHT_MAX_LOGPROB_DRIFT" ]]; then
    common_args+=(--preflight_max_logprob_drift "$GRPO_PREFLIGHT_MAX_LOGPROB_DRIFT")
  fi
}
inspect_io() {
  if [[ "$STAGE2_TRAINER" == "da" ]]; then
    need_model
    python -m train_maestro_8b.data.export_da_grpo \
      --scenes "$STAGE1_DIR/questions/scenes.jsonl" \
      --turns "$STAGE4_DIR/main_turns.jsonl" --output_dir "$DA_DATA" \
      --tokenizer "$MODEL" --max_prompt_tokens "$GRPO_MAX_PROMPT_TOKENS" --dry_run
    return
  fi
  need_model
  need "$GRPO_DATA/train.jsonl"
  say "Inspect actual Main prompts and reference Action budgets (no GPU)"
  python -m train_maestro_8b.data.inspect_io \
    --model "$MODEL" --data "$GRPO_DATA/train.jsonl" \
    --max_prompt_tokens "$GRPO_MAX_PROMPT_TOKENS" \
    --max_completion_tokens "$GRPO_MAX_NEW_TOKENS" \
    --num_generations "$GRPO_GENERATIONS" \
    --out "$RUN_ROOT/io_inspection.json"
}
preflight() {
  if [[ "$STAGE2_TRAINER" == "da" ]]; then stage2_da_check; return; fi
  need_model; verify_risk; need "$GRPO_DATA/train.jsonl"
  say "Unsloth RiskKV preflight: sampling/logprob/LoRA gradients"
  init_grpo_args
  python -m train_maestro_8b.policy.train "${common_args[@]}" \
    --output_dir "$GRPO_CKPT" --preflight_only
}
pilot() {
  if [[ "$STAGE2_TRAINER" == "da" ]]; then stage2_da_pilot; return; fi
  need_model; verify_risk; need "$GRPO_DATA/train.jsonl"
  if [[ "$SKIP_COMPLETED" == "1" && -f "$PILOT_CKPT/risk_policy_config.json" ]]; then
    say "Reusing completed GRPO pilot"; return
  fi
  if [[ -d "$PILOT_CKPT" && -n "$(ls -A "$PILOT_CKPT")" ]]; then
    die "Incomplete GRPO pilot at $PILOT_CKPT; use a new PILOT_CKPT or move old files aside."
  fi
  init_grpo_args
  say "Pilot GRPO"
  local pilot_extra=(--sft_warmup "$GRPO_SFT_WARMUP" \
                     --sft_warmup_samples "$GRPO_SFT_WARMUP_SAMPLES")
  if [[ "$PILOT_REQUIRE_GRAD" == "1" ]]; then
    pilot_extra+=(--fail_on_zero_adv)
  fi
  python -m train_maestro_8b.policy.train "${common_args[@]}" \
    "${pilot_extra[@]}" \
    --max_new_tokens "$PILOT_MAX_NEW_TOKENS" \
    --output_dir "$PILOT_CKPT" \
    --max_prompts "$PILOT_PROMPTS" --max_steps "$PILOT_STEPS" \
    --warmup_ratio 0 --logging_steps 1 \
    --reward_debug_jsonl "$RUN_ROOT/reward_pilot.jsonl"
}
grpo() {
  if [[ "$STAGE2_TRAINER" == "da" ]]; then stage2_da_grpo; return; fi
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
    --sft_warmup "$GRPO_SFT_WARMUP" \
    --sft_warmup_samples "$GRPO_SFT_WARMUP_SAMPLES" \
    --output_dir "$GRPO_CKPT" --epochs "$GRPO_EPOCHS" \
    --max_steps "$GRPO_MAX_STEPS" --logging_steps "$GRPO_LOGGING_STEPS" \
    --save_steps "$GRPO_SAVE_STEPS" \
    --reward_debug_jsonl "$RUN_ROOT/reward_grpo.jsonl" "${resume[@]}"
}
audit_pilot() {
  if [[ "$STAGE2_TRAINER" == "da" ]]; then
    need "$DA_PILOT_CKPT/reward_diagnostics.jsonl"
    tail -n "$GRPO_GENERATIONS" "$DA_PILOT_CKPT/reward_diagnostics.jsonl"
    return
  fi
  need "$RUN_ROOT/reward_pilot.jsonl"
  say "Inspect recently sampled Actions and deterministic invalid reasons (no GPU)"
  python -m train_maestro_8b.data.audit_rollouts \
    --input "$RUN_ROOT/reward_pilot.jsonl" \
    --tail "$GRPO_GENERATIONS"
}
status() {
  say "MODEL=$MODEL | RUN_ROOT=$RUN_ROOT | STAGE2_TRAINER=$STAGE2_TRAINER"
  [[ -f "$DA_DATA/summary.json" ]] && cat "$DA_DATA/summary.json" || true
  [[ -f "$DA_PILOT_CKPT/stage2_da_training.json" ]] && echo "OK Stage2 DA Pilot" || true
  [[ -f "$DA_GRPO_CKPT/stage2_da_training.json" ]] && echo "OK Stage2 DA GRPO" || true
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
  all)
    if [[ "$STAGE2_TRAINER" == "da" ]]; then
      prepare_grpo; preflight; pilot; grpo; status
    else
      export_data; extract_features; train_risk; prepare_grpo; inspect_io; preflight; pilot; grpo; status
    fi
    ;;
  export) export_data ;;
  features) extract_features ;;
  risk) train_risk ;;
  prepare-grpo) prepare_grpo ;;
  inspect-io) inspect_io ;;
  preflight) preflight ;;
  pilot) pilot ;;
  audit-pilot) audit_pilot ;;
  judge) [[ "$STAGE2_TRAINER" == "da" ]] || die "judge is only a DA Stage2 command"; stage2_da_judge ;;
  grpo) grpo ;;
  status) status ;;
  help|-h|--help) usage ;;
  *) usage >&2; die "Unknown pipeline command: $1" ;;
esac
