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

SPLIT_DIR=${SPLIT_DIR:-"${NFS_DIR}/btrabucc/neurips_sft_task_proposer"}
MODEL_DIR=${MODEL_DIR:-"${NFS_DIR}/btrabucc/neurips_sft_task_proposer/models"}

export HF_HOME=${NFS_DIR}/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"sft/accelerate.yaml"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-1.7B"}

ALL_TASK_PROPOSER_NAMES=${ALL_TASK_PROPOSER_NAMES:-"gemini-2.5-flash-task-proposer"}
ALL_JUDGE_NAMES=${ALL_JUDGE_NAMES:-"qwen3-235b-judge"}
ALL_MAX_NUM_SAMPLES=${ALL_MAX_NUM_SAMPLES:-"2000 5000 10000"}

for TASK_PROPOSER_NAME in ${ALL_TASK_PROPOSER_NAMES}; do
for JUDGE_NAME in ${ALL_JUDGE_NAMES}; do
for MAX_NUM_SAMPLES in ${ALL_MAX_NUM_SAMPLES}; do

IDENTIFIER="${MAX_NUM_SAMPLES}x-${TASK_PROPOSER_NAME}-${JUDGE_NAME}"
TARGET_SPLIT="${SPLIT_DIR}/${IDENTIFIER}"
TARGET_MODEL="${MODEL_DIR}/qwen3-1.7b-${IDENTIFIER}"

LAUNCH_ARGS=(
    --config_file ${ACCELERATE_CONFIG}
    sft/train_sft.py
    --model_name ${MODEL_NAME}
    --dataset_path ${DATASET_PATH}
    --final_model_dir ${FINAL_MODEL_DIR}
    --max_seq_length 12288
    --use_bf16
)

echo "Starting SFT with: ${LAUNCH_ARGS[@]}"
accelerate launch ${LAUNCH_ARGS[@]}

done
done
done
