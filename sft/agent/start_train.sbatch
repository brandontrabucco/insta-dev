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

NFS_DIR=${NFS_DIR:-"/data/matrix/projects/rsalakhugroup"}

SFT_DATASET_PREFIX=${SFT_DATASET_PREFIX:-"${NFS_DIR}/btrabucc/neurips_feedback_experiment_sft/gemini-2.5-flash-step6-sft"}
SFT_MODEL_PREFIX=${SFT_MODEL_PREFIX:-"${NFS_DIR}/btrabucc/neurips_feedback_experiment_sft/qwen3-1.7b-step6-e5"}

export HF_HOME=${NFS_DIR}/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"sft/accelerate.yaml"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-1.7B"}

ALL_JUDGE_NAMES=${ALL_JUDGE_NAMES:-"gemini-2.5-flash-judge"}
ALL_MAX_NUM_SAMPLES=${ALL_MAX_NUM_SAMPLES:-"5000"}
ALL_SUCCESS_THRESHOLDS=${ALL_SUCCESS_THRESHOLDS:-"1.0"}
ALL_SECONDARY_THRESHOLDS=${ALL_SECONDARY_THRESHOLDS:-"0.0"}

for JUDGE_NAME in ${ALL_JUDGE_NAMES}; do
for MAX_NUM_SAMPLES in ${ALL_MAX_NUM_SAMPLES}; do
for SUCCESS_THRESHOLD in ${ALL_SUCCESS_THRESHOLDS}; do
for SECONDARY_THRESHOLD in ${ALL_SECONDARY_THRESHOLDS}; do

IDX="${MAX_NUM_SAMPLES}x-${SUCCESS_THRESHOLD}s-${SECONDARY_THRESHOLD}c"

SFT_DATASET_PATH="${SFT_DATASET_PREFIX}-${IDX}-${JUDGE_NAME}"
SFT_MODEL_PATH="${SFT_MODEL_PREFIX}-${IDX}-${JUDGE_NAME}"

TRAINING_ARGS=(
    --model_name ${MODEL_NAME}
    --sft_model_path ${SFT_MODEL_PATH}
    --sft_dataset_path ${SFT_DATASET_PATH}
    --max_seq_length 16384
    --num_epochs 5
    --use_bf16
)

SCRIPT_ARGS=(
    --config_file ${ACCELERATE_CONFIG}
    sft/train_sft.py
    ${TRAINING_ARGS[@]}
)

echo "SFT: ${SCRIPT_ARGS[@]}"
accelerate launch ${SCRIPT_ARGS[@]}

done
done
done
done
