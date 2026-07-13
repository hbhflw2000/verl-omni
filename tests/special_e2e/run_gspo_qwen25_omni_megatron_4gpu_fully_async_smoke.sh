#!/usr/bin/env bash
# 4-GPU single-node Qwen2.5-Omni-3B text-only smoke:
# Megatron actor/ref on 2 GPUs + standalone vLLM-Omni rollout on 2 GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NNODES=${NNODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export TRAIN_NNODES=${TRAIN_NNODES:-1}
export TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-2}
export ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
export ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE:-2}
export ROLLOUT_TP=${ROLLOUT_TP:-2}
export ROLLOUT_DP=${ROLLOUT_DP:-1}

export STAGE_CONFIG=${STAGE_CONFIG:-${SCRIPT_DIR}/qwen25_omni_thinker_only_tp2_async_raw_logprobs.yaml}
export RUN_ID=${RUN_ID:-qwen25_4gpu_$(date +%Y%m%d_%H%M%S)}
export EXP_NAME=${EXP_NAME:-qwen25_omni_megatron_4gpu_fully_async_${RUN_ID}}
export PROFILE_LABEL=${PROFILE_LABEL:-4-GPU Qwen2.5-Omni text-only fully-async smoke}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/verl_ray_${USER}_qwen25_omni_async_4gpu}

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

export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
export ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-1}
export TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-2}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-4}

exec "${SCRIPT_DIR}/run_gspo_qwen25_omni_megatron_8gpu_fully_async_smoke.sh" "$@"
