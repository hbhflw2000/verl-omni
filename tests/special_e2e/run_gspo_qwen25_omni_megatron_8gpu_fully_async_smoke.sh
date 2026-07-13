#!/usr/bin/env bash
# 8-GPU single-node Qwen2.5-Omni-3B text-only fully-async RL smoke.
#
# Submit this as one k8s pod/job with 8 visible GPUs.  This wrapper only pins
# the Qwen2.5/single-node resource shape; it deliberately delegates to the
# 32-GPU launcher for the Ray/k8s hard parts: node-rank detection, head-port
# negotiation, per-job port pools, flock-based port locks, Ray autostart, and
# CONFIG_ONLY/RAY_ONLY preflight modes.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ASYNC_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)/.."

export CONDA_ENV=${CONDA_ENV:-/nfs/ml-training-ssd/users/liuwei/verl_mega_async}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

export MODEL_PATH=${MODEL_PATH:-/nfs/ml-training-ssd/users/liuwei/models/Qwen2.5-Omni-3B}
export TRAIN_FILES=${TRAIN_FILES:-/nfs/ml-training-ssd/users/liuwei/data/gsm8k_verl_prompt/train.parquet}
export VAL_FILES=${VAL_FILES:-/nfs/ml-training-ssd/users/liuwei/data/gsm8k_verl_prompt/test.parquet}
export STAGE_CONFIG=${STAGE_CONFIG:-${SCRIPT_DIR}/qwen25_omni_thinker_only_tp4_async_raw_logprobs.yaml}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${ASYNC_ROOT}/outputs/qwen25_omni}
export CACHE_ROOT=${CACHE_ROOT:-/nfs/ml-training-ssd/users/liuwei/verl_mega_async_qwen25_cache}

# Force the cluster launcher into one-node mode even if the k8s platform leaves
# multi-node role/index envs in the container.  Set
# QWEN25_OMNI_SINGLE_NODE_FORCE_RANK0=0 only when intentionally reusing this
# wrapper outside a one-pod 8-GPU submit.
export QWEN25_OMNI_SINGLE_NODE_FORCE_RANK0=${QWEN25_OMNI_SINGLE_NODE_FORCE_RANK0:-1}
if [[ "${QWEN25_OMNI_SINGLE_NODE_FORCE_RANK0}" == "1" ]]; then
  export NODE_RANK=0
  export RAY_NODE_RANK=0
  export RAY_NODE_ROLE=head
else
  export NODE_RANK=${NODE_RANK:-0}
  export RAY_NODE_RANK=${RAY_NODE_RANK:-0}
  export RAY_NODE_ROLE=${RAY_NODE_ROLE:-head}
fi
export NNODES=${NNODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export TRAIN_NNODES=${TRAIN_NNODES:-1}
export TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-4}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
export ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-4}
export ROLLOUT_TP=${ROLLOUT_TP:-4}
export ROLLOUT_DP=${ROLLOUT_DP:-1}

export ACTOR_TP=${ACTOR_TP:-1}
export ACTOR_PP=${ACTOR_PP:-1}
export ACTOR_CP=${ACTOR_CP:-1}
export ACTOR_EP=${ACTOR_EP:-1}
export ACTOR_ETP=${ACTOR_ETP:-1}
export REF_TP=${REF_TP:-${ACTOR_TP}}
export REF_PP=${REF_PP:-${ACTOR_PP}}
export REF_CP=${REF_CP:-${ACTOR_CP}}
export REF_EP=${REF_EP:-${ACTOR_EP}}
export REF_ETP=${REF_ETP:-${ACTOR_ETP}}

export MODEL_EXTERNAL_LIB=${MODEL_EXTERNAL_LIB:-null}
export VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES:-verl_omni}
export MEGATRON_BRIDGE_REPO=${MEGATRON_BRIDGE_REPO:-${ASYNC_ROOT}/megatron-bridge}

export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}
export ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
export REQUIRE_BATCHES=${REQUIRE_BATCHES:-1}
export TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-4}

export USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
export OFFLOAD=${OFFLOAD:-False}
export SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-False}
export ATTENTION_BACKEND=${ATTENTION_BACKEND:-local}
export MASKED_SOFTMAX_FUSION=${MASKED_SOFTMAX_FUSION:-False}
export MOE_PERMUTE_FUSION=${MOE_PERMUTE_FUSION:-False}
export GRADIENT_ACCUMULATION_FUSION=${GRADIENT_ACCUMULATION_FUSION:-False}

export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-8}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}
export ROLLOUT_CALCULATE_LOG_PROBS=${ROLLOUT_CALCULATE_LOG_PROBS:-True}
export ROLLOUT_LOGPROBS_MODE=${ROLLOUT_LOGPROBS_MODE:-raw_logprobs}
export VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS=${VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS:-1}
export VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS=${VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS:-0}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.8}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.9}
export ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}

export PROJECT_NAME=${PROJECT_NAME:-qwen25_omni_megatron_async_smoke}
RUN_ID_SOURCE=${RUN_ID:-${LUBAN_JOB_ID:-${AIP_JOB_ID:-${VC_JOB_ID:-${JOB_ID:-${APP_ID:-${K8S_APP_ID:-qwen25_8gpu_$(date +%Y%m%d_%H%M%S)}}}}}}}
RUN_ID_SAFE="$(printf '%s' "${RUN_ID_SOURCE}" | tr -c 'A-Za-z0-9_.-' '_')"
export RUN_ID=${RUN_ID_SAFE}
RUN_ID_SHORT="${RUN_ID_SAFE:0:24}"
RUN_ID_HASH="$(printf '%s' "${RUN_ID_SAFE}" | cksum | awk '{print $1}')"
export EXP_NAME=${EXP_NAME:-qwen25_omni_megatron_8gpu_fully_async_${RUN_ID}}
export PROFILE_LABEL=${PROFILE_LABEL:-8-GPU Qwen2.5-Omni text-only fully-async smoke}

# Keep all Ray ports job-seeded and locally probed.  The parent launcher owns
# the actual allocation/validation; these defaults make single-node k8s submits
# avoid stale /tmp state and cross-job port collisions.  Keep RAY_TMPDIR short:
# Ray appends session_*/sockets/plasma_store and AF_UNIX caps paths at 107 bytes.
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rq25_${RUN_ID_SHORT}_${RUN_ID_HASH}}
export RAY_AUTOSTART=${RAY_AUTOSTART:-1}
export RAY_PORT_SEED=${RAY_PORT_SEED:-${RUN_ID}}
export RAY_WORKER_PORT_SEED=${RAY_WORKER_PORT_SEED:-${RAY_PORT_SEED}}
export RAY_INCLUDE_DASHBOARD=${RAY_INCLUDE_DASHBOARD:-0}
export RAY_HEAD_PORT_NEGOTIATE=${RAY_HEAD_PORT_NEGOTIATE:-1}
export RAY_WORKER_PORT_LOCAL_PROBE=${RAY_WORKER_PORT_LOCAL_PROBE:-1}
export RAY_COMPONENT_PORT_LOCAL_PROBE=${RAY_COMPONENT_PORT_LOCAL_PROBE:-1}
export RAY_WORKER_PORT_SPAN=${RAY_WORKER_PORT_SPAN:-256}
export RAY_NODE_CPUS=${RAY_NODE_CPUS:-32}
export VERL_FULLY_ASYNC_TRAINER_NUM_CPUS=${VERL_FULLY_ASYNC_TRAINER_NUM_CPUS:-1}
export VERL_FULLY_ASYNC_ROLLOUTER_NUM_CPUS=${VERL_FULLY_ASYNC_ROLLOUTER_NUM_CPUS:-1}

export VERL_OMNI_FORCE_STANDALONE_ROLLOUT=${VERL_OMNI_FORCE_STANDALONE_ROLLOUT:-1}
export VERL_OMNI_RESOURCE_SPLIT_IMPL=${VERL_OMNI_RESOURCE_SPLIT_IMPL:-standalone_rollout}
export VERL_OMNI_SKIP_INITIAL_ROLLOUT_RESUME=${VERL_OMNI_SKIP_INITIAL_ROLLOUT_RESUME:-1}
export VERL_OMNI_SKIP_INITIAL_ROLLOUT_SLEEP=${VERL_OMNI_SKIP_INITIAL_ROLLOUT_SLEEP:-1}
export VERL_OMNI_WAKE_TAGS=${VERL_OMNI_WAKE_TAGS:-kv_cache,weights}
export VERL_OMNI_SLEEP_LEVEL=${VERL_OMNI_SLEEP_LEVEL:-1}

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fully_async.sh" "$@"
