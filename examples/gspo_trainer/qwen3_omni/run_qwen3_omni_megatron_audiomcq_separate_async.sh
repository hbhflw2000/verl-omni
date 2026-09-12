#!/usr/bin/env bash
# Run on the head of an allocated Ray cluster; resource counts are Hydra overrides.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to a Qwen3-Omni checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to prepared AudioMCQ train.parquet}"
: "${VAL_FILE:?Set VAL_FILE to prepared AudioMCQ validation.parquet}"
OUTPUT_DIR=${OUTPUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/audiomcq-run.XXXXXX")}
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)
# Each invocation gets a separate log and resolved configuration.
RUN_DIR=$(mktemp -d "${OUTPUT_DIR}/run.XXXXXX")
echo "AudioMCQ artifacts: ${RUN_DIR}"
export VERL_USE_EXTERNAL_MODULES=verl_omni
export TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"

args=(
  --config-path "${REPO_ROOT}/examples/gspo_trainer/qwen3_omni/config"
  --config-name audiomcq_megatron_separate_async
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  "trainer.default_local_dir=${RUN_DIR}/checkpoints"
  "$@"
)
printf '%q ' "${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" > "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"
git rev-parse HEAD > "${RUN_DIR}/commit.txt"
VLLM_LOGGING_STREAM=ext://sys.stderr "${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" \
  --cfg job --resolve > "${RUN_DIR}/config.yaml" 2> >(tee "${RUN_DIR}/config.log" >&2)
"${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
