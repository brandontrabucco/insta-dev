#!/bin/bash

DOCKER_IMAGE=${DOCKER_IMAGE:-"hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2-verl0.3.0.post1"}
VERL_COMMAND=${VERL_COMMAND:-"bash verl/train_grpo_qwen2.5-1.5b.sh"}

MODEL_PATH=${MODEL_PATH:-"./qwen-1.5b-grpo-n0"}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"./qwen-1.5b-grpo-n1"}

PROJECT_NAME=${PROJECT_NAME:-"verl_qwen_grpo"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen2.5_1.5b_grpo_n1_lr1e-5"}

ROLLOUT_DIRS=${ROLLOUT_DIRS:-"./qwen-1.5b-grpo-n0-rollouts/*"}
DATASET_OUTPUT_FILE=${DATASET_OUTPUT_FILE:-"./verl/insta-150k-v2-grpo-n0.parquet"}

TRAIN_FILES=${TRAIN_FILES:-$DATASET_OUTPUT_FILE}
VAL_FILES=${VAL_FILES:-$DATASET_OUTPUT_FILE}

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

DATASET_ARGS=(
    --data_dirs ${ROLLOUTS_DIR}
    --dataset_output_file ${DATASET_OUTPUT_FILE}
)

python verl/create_verl_dataset.py ${DATASET_ARGS[@]}

DOCKER_COMMAND="cd /insta-dev && pip install -e . && ${VERL_COMMAND}"

docker run ${DOCKER_ARGS[@]} \
    ${DOCKER_IMAGE} \
    bash -c "${DOCKER_COMMAND}"

CKPT_DIR=$(
    ls -d ${DEFAULT_LOCAL_DIR}/global_step*/ 
    | sort -V | tail -n 1
)

MERGE_ARGS=(
    --local_dir ${CKPT_DIR}/actor/
    --target_dir ${CKPT_DIR}
    --hf_model_path ${CKPT_DIR}
)

cp ${CKPT_DIR}/actor/huggingface/* ${CKPT_DIR}/

python verl/model_merger.py \
    --backend fsdp ${MERGE_ARGS[@]}

rm -rf ${DEFAULT_LOCAL_DIR}/global_step*
