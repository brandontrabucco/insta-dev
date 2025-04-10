#!/bin/bash

DOCKER_IMAGE=${DOCKER_IMAGE:-"hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2-verl0.3.0.post1"}
VERL_COMMAND=${VERL_COMMAND:-"bash verl/train_grpo_qwen2.5-1.5b.sh"}

MODEL_PATH=${MODEL_PATH:-"./qwen-1.5b-sft"}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"./qwen-1.5b-grpo-n0"}

PROJECT_NAME=${PROJECT_NAME:-"verl_qwen_grpo"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen2.5_1.5b_grpo_n0_lr1e-5-large-batch"}

TRAIN_FILES=${TRAIN_FILES:-"./verl/insta-150k-v2-grpo-n0.parquet"}
VAL_FILES=${VAL_FILES:-"./verl/insta-150k-v2-grpo-n0.parquet"}

VERL_LOG=${VERL_LOG:-"verl/trainer.log"}

if [ -z "${WANDB_API_KEY}" ]; then
    echo "Please set the WANDB_API_KEY environment variable."
    exit 1
fi

if [ -z "${DOCKER_IMAGE}" ]; then
    echo "Please set the DOCKER_IMAGE environment variable."
    exit 1
fi

if [ -z "${VERL_COMMAND}" ]; then
    echo "Please set the VERL_COMMAND environment variable."
    exit 1
fi

docker pull ${DOCKER_IMAGE}

ENVIRONMENT_ARGS=(
    -e=WANDB_API_KEY=${WANDB_API_KEY}
    -e=MODEL_PATH=${MODEL_PATH}
    -e=DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR}
    -e=PROJECT_NAME=${PROJECT_NAME}
    -e=EXPERIMENT_NAME=${EXPERIMENT_NAME}
    -e=TRAIN_FILES=${TRAIN_FILES}
    -e=VAL_FILES=${VAL_FILES}
    -e=VERL_LOG=${VERL_LOG}
)

DOCKER_ARGS=(
    --mount=type=bind,src=.,dst=/insta-dev
    --runtime=nvidia -it --rm 
    --shm-size="10g"
    --cap-add=SYS_ADMIN
    ${ENVIRONMENT_ARGS[@]}
)

DOCKER_COMMAND="cd /insta-dev && pip install -e . && ${VERL_COMMAND}"

docker run ${DOCKER_ARGS[@]} \
    ${DOCKER_IMAGE} \
    bash -c "${DOCKER_COMMAND}"