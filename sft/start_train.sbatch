#!/bin/bash
#SBATCH --job-name=insta
#SBATCH --output=logs/sft-%A-%a-%N.out
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H100:8
#SBATCH --mem=256G
#SBATCH --partition=flame-earlybirds
#SBATCH --qos=earlybird_qos

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta
cd ~/insta-dev

NFS_DIR=${NFS_DIR:-"/data/matrix/projects/rsalakhugroup"}

DATA_PREFIX=${DATA_PREFIX:-"${NFS_DIR}/btrabucc/insta-150k-v2-sft-qwen3-235b"}
MODEL_PREFIX=${MODEL_PREFIX:-"${NFS_DIR}/btrabucc/qwen3-1.7b"}

DATA_SUFFIX=${DATA_SUFFIX:-"-behaviors"}
MODEL_SUFFIX=${MODEL_SUFFIX:-"-behaviors"}

export HF_HOME=${NFS_DIR}/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"sft/accelerate_config.yaml"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-1.7B"}

ALL_JUDGE_NAMES=${ALL_JUDGE_NAMES:-"judgments judgments-qwen-235b"}
ALL_MAX_NUM_SAMPLES=${ALL_MAX_NUM_SAMPLES:-"2000 5000 10000 20000"}
ALL_SUCCESS_THRESHOLDS=${ALL_SUCCESS_THRESHOLDS:-"0.5"}

for JUDGE_NAME in ${ALL_JUDGE_NAMES}; do
for MAX_NUM_SAMPLES in ${ALL_MAX_NUM_SAMPLES}; do
for SUCCESS_THRESHOLD in ${ALL_SUCCESS_THRESHOLDS}; do

EXPERIMENT_IDENTIFIER="-${MAX_NUM_SAMPLES}x-${SUCCESS_THRESHOLD}s-${JUDGE_NAME}"
DATASET_PATH="${DATA_PREFIX}${EXPERIMENT_IDENTIFIER}${DATA_SUFFIX}"
FINAL_MODEL_DIR="${MODEL_PREFIX}${EXPERIMENT_IDENTIFIER}${MODEL_SUFFIX}"

TRAIN_ARGS=(
    --model_name ${MODEL_NAME}
    --dataset_path ${DATASET_PATH}
    --final_model_dir ${FINAL_MODEL_DIR}
    --max_seq_length 16384
    --use_bf16
)

LAUNCH_ARGS=(
    --config_file ${ACCELERATE_CONFIG}
    sft/train_sft.py
    ${TRAIN_ARGS[@]}
)

echo "Starting SFT with args: ${LAUNCH_ARGS[@]}"

accelerate launch ${LAUNCH_ARGS[@]}

done
done
done
