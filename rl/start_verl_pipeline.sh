#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

DOCKER_IMAGE=${DOCKER_IMAGE:-"hiyouga/verl:ngc-th2.6.0-cu120-vllm0.8.2-verl0.3.0.post1"}
VERL_COMMAND=${VERL_COMMAND:-"bash rl/train_grpo_qwen2.5-1.5b.sh"}

MODEL_PATH=${MODEL_PATH:-"./qwen-1.5b-grpo-n0"}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"./qwen-1.5b-grpo-n1"}

PROJECT_NAME=${PROJECT_NAME:-"verl_qwen_grpo"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen2.5_1.5b_grpo_n1_lr1e-5"}

ROLLOUT_DIRS=${ROLLOUT_DIRS:-"./qwen-1.5b-grpo-n0-rollouts/*"}
DATASET_OUTPUT_FILE=${DATASET_OUTPUT_FILE:-"./rl/insta-150k-v2-grpo-n1.parquet"}

TRAIN_FILES=${TRAIN_FILES:-$DATASET_OUTPUT_FILE}
VAL_FILES=${VAL_FILES:-$DATASET_OUTPUT_FILE}

VERL_LOG=${VERL_LOG:-"rl/verl.log"}

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
    --data_dirs ${ROLLOUT_DIRS}
    --dataset_output_file ${DATASET_OUTPUT_FILE}
)

python rl/create_verl_dataset.py ${DATASET_ARGS[@]}

docker run ${DOCKER_ARGS[@]} ${DOCKER_IMAGE} \
    bash -c "cd /insta-dev && pip install -e . && ${VERL_COMMAND}"

LAST_CKPT_DIR=$(ls -d ${DEFAULT_LOCAL_DIR}/global_step* | sort -V | tail -n 1)

echo "Using Last Checkpoint: ${LAST_CKPT_DIR}"

mv ${LAST_CKPT_DIR}/actor/huggingface/* ${LAST_CKPT_DIR}/

CHECKPOINT_ARGS=(
    --local_dir ${LAST_CKPT_DIR}/actor/
    --target_dir ${LAST_CKPT_DIR}
    --hf_model_path ${LAST_CKPT_DIR}
)

python rl/verl_to_huggingface.py \
    --backend fsdp ${CHECKPOINT_ARGS[@]}
