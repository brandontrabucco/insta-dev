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

export HF_HOME=${NFS_DIR}/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN

ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"sft/accelerate_config.yaml"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-1.7B"}

for MAX_NUM_SAMPLES in 2000 5000 10000 20000; do
for SUCCESS_THRESHOLD in 0.5; do

DATASET_PATH="${NFS_DIR}/btrabucc/insta-150k-v2-sft-qwen3-235b-${MAX_NUM_SAMPLES}x-${SUCCESS_THRESHOLD}s-qwen3-judge"
FINAL_MODEL_DIR="${NFS_DIR}/btrabucc/qwen3-1.7b-${MAX_NUM_SAMPLES}x-${SUCCESS_THRESHOLD}s-qwen3-judge"

TRAIN_ARGS=(
    --model_name ${MODEL_NAME}
    --dataset_path ${DATASET_PATH}
    --final_model_dir ${FINAL_MODEL_DIR}
    --max_seq_length 16384
    --use_bf16
)

echo "Starting SFT with args: ${TRAIN_ARGS[@]}"

accelerate launch --config_file ${ACCELERATE_CONFIG} \
    sft/train_sft.py ${TRAIN_ARGS[@]}

done
done
