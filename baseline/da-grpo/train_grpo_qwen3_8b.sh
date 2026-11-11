#!/usr/bin/env bash
# GRPO training for the Orchestra-o1 Main Agent (Qwen3-8B).
# Run from anywhere; paths are resolved relative to the MAESTRO repository.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERL_DIR="${VERL_DIR:-/path/to/verl}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-8B}"

DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/dataset/data/da-grpo}"
TRAIN_PARQUET="${DATA_DIR}/train.parquet"
EXPERT_JSONL="${DATA_DIR}/expert_decisions.jsonl"
GRPO_DIR="${SCRIPT_DIR}"
REWARD_FN="${GRPO_DIR}/reward_fn.py"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="qwen3_8b_main_agent_grpo_${TIMESTAMP}"
PROJECT_NAME="grpo_orchestra_o1_qwen3_8b"
CKPT_DIR="${GRPO_DIR}/output/${EXP_NAME}"
LOG_DIR="${GRPO_DIR}/logs"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
export VERL_FILE_LOGGER_PATH="${CKPT_DIR}/metrics.jsonl"
export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_NAME="${EXP_NAME}"

NNODES=${NNODES:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TRAIN_BSZ=${TRAIN_BSZ:-24}
MINI_BSZ=${MINI_BSZ:-12}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-24576}
MAX_RESP_LEN=${MAX_RESP_LEN:-4096}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-32768}
ROLLOUT_N=${ROLLOUT_N:-8}
LR=${LR:-5e-6}
EPOCHS=${EPOCHS:-5}
SAVE_FREQ=${SAVE_FREQ:-54}
TP_SIZE=${TP_SIZE:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.35}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}

if [[ ! -f "${EXPERT_JSONL}" ]]; then
    echo "[train] ERROR: expert decisions not found: ${EXPERT_JSONL}"
    exit 1
fi
if [[ ! -f "${TRAIN_PARQUET}" ]]; then
    echo "[train] ERROR: train parquet not found: ${TRAIN_PARQUET}"
    exit 1
fi
if ! head -n 1 "${EXPERT_JSONL}" | grep -q '"expert_steps"'; then
    echo "[train] ERROR: old expert_decisions.jsonl schema (no 'expert_steps' field)."
    exit 1
fi

echo "[train] repo:    ${PROJECT_DIR}"
echo "[train] expert:  ${EXPERT_JSONL}"
echo "[train] parquet: ${TRAIN_PARQUET}"

cd "${VERL_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${VERL_DIR}:${PYTHONPATH:-}"
export RAY_DEDUP_LOGS=0
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export TOKENIZERS_PARALLELISM=false
export RAY_worker_register_timeout_seconds=${RAY_worker_register_timeout_seconds:-600}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}
export NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

TRAIN_LOG="${LOG_DIR}/train_${TIMESTAMP}.log"
echo "[train] log -> ${TRAIN_LOG}"
echo "[train] ckpt -> ${CKPT_DIR}"

set -x
set +e
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${TRAIN_PARQUET}" \
    data.train_batch_size=${TRAIN_BSZ} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESP_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU} \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path="${REWARD_FN}" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb","file"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.default_local_dir="${CKPT_DIR}" \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","hf_model","optimizer","extra"]' \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=-1 \
    trainer.total_epochs=${EPOCHS} \
    ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS} \
    "$@" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set +x

if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
    echo "[train] Training completed successfully"
else
    echo "[train] Training exited abnormally (exit code ${TRAIN_EXIT_CODE})"
fi
exit ${TRAIN_EXIT_CODE}
