#!/bin/bash
#SBATCH --job-name=insta
#SBATCH --exclude=shire-1-1,shire-1-6,shire-1-10,shire-2-5,shire-2-9
#SBATCH --output=logs/datagen-%A-%a-%N.out
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=general

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

NFS_DIR=${NFS_DIR:-"/data/matrix/projects/rsalakhugroup"}
INPUT_DATA_DIR=${INPUT_DATA_DIR:-"${NFS_DIR}/btrabucc/neurips_data_collection/qwen3-1.7b-10000x-0.9s-qwen3-235b-judge"}

SPLIT_DIR=${SPLIT_DIR:-"${NFS_DIR}/btrabucc/neurips_sft_judge"}

export HF_HOME=${NFS_DIR}/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN

ALL_JUDGE_NAMES=${ALL_JUDGE_NAMES:-"qwen3-235b-judge"}
ALL_MAX_NUM_SAMPLES=${ALL_MAX_NUM_SAMPLES:-"2000 5000 10000 20000 40000"}

for JUDGE_NAME in ${ALL_JUDGE_NAMES}; do
for MAX_NUM_SAMPLES in ${ALL_MAX_NUM_SAMPLES}; do

IDENTIFIER="${MAX_NUM_SAMPLES}x-${JUDGE_NAME}"
TARGET_SPLIT="${SPLIT_DIR}/${IDENTIFIER}"

DATASET_ARGS=(
    sft/judge/create_dataset.py
    --input_data_dir ${INPUT_DATA_DIR}
    --dataset_output_dir ${TARGET_SPLIT}
    --judge_name ${JUDGE_NAME}
    --max_num_samples ${MAX_NUM_SAMPLES}
)

echo "Data args: ${DATASET_ARGS[@]}"
python ${DATASET_ARGS[@]}
    
done
done
