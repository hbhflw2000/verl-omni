#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-python3}
CONFIG_ONLY=${CONFIG_ONLY:-0}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}

: "${MODEL_PATH:?Set MODEL_PATH to a Qwen3-Omni checkpoint path or HF model id.}"
: "${TRAIN_FILES:?Set TRAIN_FILES to the GSM8K train parquet path.}"
: "${VAL_FILES:?Set VAL_FILES to the GSM8K validation parquet path.}"
STAGE_CONFIG=${STAGE_CONFIG:-${SCRIPT_DIR}/qwen3_omni_thinker_only_tp4_fastdev.yaml}

TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
EP=${EP:-4}
ETP=${ETP:-1}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-256}
MAX_POSITION_EMBEDDINGS=${MAX_POSITION_EMBEDDINGS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}

LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}

ACTOR_LR=${ACTOR_LR:-3e-6}
ACTOR_LR_WARMUP_STEPS=${ACTOR_LR_WARMUP_STEPS:-0}
ACTOR_WEIGHT_DECAY=${ACTOR_WEIGHT_DECAY:-0.0}
ACTOR_CLIP_GRAD=${ACTOR_CLIP_GRAD:-1.0}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
POLICY_LOSS_MODE=${POLICY_LOSS_MODE:-vanilla}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}
USE_KL_LOSS=${USE_KL_LOSS:-True}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-low_var_kl}

ROLLOUT_N=${ROLLOUT_N:-1}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.25}
ROLLOUT_CALCULATE_LOG_PROBS=${ROLLOUT_CALCULATE_LOG_PROBS:-False}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-1024}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-4}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-1024}
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}

REWARD_MANAGER=${REWARD_MANAGER:-naive}
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-False}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-0}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}
OVERLONG_BUFFER_LOG=${OVERLONG_BUFFER_LOG:-False}

EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_lora_4gpu_smoke}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/megatron_lora_4gpu}
CKPTS_DIR=${CKPTS_DIR:-${REPO_ROOT}/ckpt/${EXP_NAME}}
CACHE_ROOT=${CACHE_ROOT:-${REPO_ROOT}/.cache/${EXP_NAME}}
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/vray4_${USER}}
mkdir -p "${LOG_DIR}" "${CKPTS_DIR}" "${CACHE_ROOT}" "${RAY_TMPDIR}"

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export VERL_PPO_LOGGING_LEVEL=${VERL_PPO_LOGGING_LEVEL:-INFO}
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export HF_HOME=${HF_HOME:-${CACHE_ROOT}/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export TORCH_HOME=${TORCH_HOME:-${CACHE_ROOT}/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}

DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.truncation=error
    data.filter_overlong_prompts=True
    data.shuffle=False
    data.dataloader_num_workers=0
    data.return_raw_chat=True
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.use_fused_kernels=False
    +actor_rollout_ref.model.override_config.max_position_embeddings="${MAX_POSITION_EMBEDDINGS}"
    actor_rollout_ref.model.lora.rank="${LORA_RANK}"
    actor_rollout_ref.model.lora.alpha="${LORA_ALPHA}"
    actor_rollout_ref.model.lora.lora_A_init_method=kaiming
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.optim.lr_warmup_steps="${ACTOR_LR_WARMUP_STEPS}"
    actor_rollout_ref.actor.optim.weight_decay="${ACTOR_WEIGHT_DECAY}"
    actor_rollout_ref.actor.optim.clip_grad="${ACTOR_CLIP_GRAD}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.actor.policy_loss.loss_mode="${POLICY_LOSS_MODE}"
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}"
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}"
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}"
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.megatron.use_remove_padding=False
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}"
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type="${KL_LOSS_TYPE}"
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TP}"
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${PP}"
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${EP}"
    actor_rollout_ref.actor.megatron.context_parallel_size="${CP}"
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="${ETP}"
    actor_rollout_ref.actor.megatron.sequence_parallel=True
    actor_rollout_ref.actor.megatron.param_offload="${ALL_OFFLOAD}"
    actor_rollout_ref.actor.megatron.optimizer_offload="${ALL_OFFLOAD}"
    actor_rollout_ref.actor.megatron.grad_offload="${ALL_OFFLOAD}"
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=local
    ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm_omni
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.data_parallel_size=1
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    actor_rollout_ref.rollout.calculate_log_probs="${ROLLOUT_CALCULATE_LOG_PROBS}"
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}"
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}"
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.agent.num_workers=4
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}"
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_init_timeout=1200
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${TP}"
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${PP}"
    actor_rollout_ref.ref.megatron.expert_model_parallel_size="${EP}"
    actor_rollout_ref.ref.megatron.context_parallel_size="${CP}"
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size="${ETP}"
    actor_rollout_ref.ref.megatron.sequence_parallel=True
    actor_rollout_ref.ref.megatron.param_offload="${ALL_OFFLOAD}"
    actor_rollout_ref.ref.megatron.use_remove_padding=False
    ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend=local
    ++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion=False
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

REWARD=(
    reward.reward_manager.name="${REWARD_MANAGER}"
)

if [[ "${REWARD_MANAGER}" == "dapo" ]]; then
    REWARD+=(
        +reward.reward_kwargs.overlong_buffer_cfg.enable="${ENABLE_OVERLONG_BUFFER}"
        +reward.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}"
        +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor="${OVERLONG_PENALTY_FACTOR}"
        +reward.reward_kwargs.overlong_buffer_cfg.log="${OVERLONG_BUFFER_LOG}"
        +reward.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}"
    )
fi

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=qwen3_omni_megatron_lora_4gpu
    trainer.experiment_name="${EXP_NAME}"
    trainer.n_gpus_per_node=4
    trainer.nnodes=1
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.val_before_train=False
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.max_actor_ckpt_to_keep=1
)

RAY=(
    ray_kwargs.ray_init.num_cpus=16
    +ray_kwargs.ray_init.num_gpus=4
    +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}"
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}"
)

CMD=(
    "${PYTHON}" -m verl.trainer.main_ppo
    --config-path=config
    --config-name=ppo_megatron_trainer.yaml
    "${DATA[@]}"
    "${ALGORITHM[@]}"
    "${MODEL[@]}"
    "${ROLLOUT[@]}"
    "${ACTOR[@]}"
    "${REF[@]}"
    "${REWARD[@]}"
    "${TRAINER[@]}"
    "${RAY[@]}"
)

if [[ "${CONFIG_ONLY}" == "1" ]]; then
    CMD+=(--cfg job --resolve)
fi

LOG_PATH="${LOG_DIR}/${EXP_NAME}_$(date +'%Y%m%d_%H%M%S')"
if [[ "${CONFIG_ONLY}" == "1" ]]; then
    LOG_PATH="${LOG_PATH}_config_only.log"
else
    LOG_PATH="${LOG_PATH}.log"
fi

echo "[info] REPO_ROOT=${REPO_ROOT}"
echo "[info] PYTHON=${PYTHON}"
echo "[info] MODEL_PATH=${MODEL_PATH}"
echo "[info] TRAIN_FILES=${TRAIN_FILES}"
echo "[info] VAL_FILES=${VAL_FILES}"
echo "[info] STAGE_CONFIG=${STAGE_CONFIG}"
echo "[info] parallel tp/pp/cp/ep/etp=${TP}/${PP}/${CP}/${EP}/${ETP}"
echo "[info] TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
echo "[info] train_batch/rollout_n/max_prompt/max_response=${TRAIN_BATCH_SIZE}/${ROLLOUT_N}/${MAX_PROMPT_LENGTH}/${MAX_RESPONSE_LENGTH}"
echo "[info] reward_manager=${REWARD_MANAGER}"
echo "[info] CONFIG_ONLY=${CONFIG_ONLY}"
echo "[info] LOG_PATH=${LOG_PATH}"

"${CMD[@]}" "$@" 2>&1 | tee "${LOG_PATH}"
