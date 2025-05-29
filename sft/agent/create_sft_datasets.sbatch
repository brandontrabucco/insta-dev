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

INPUT_DATASET_PATH=${INPUT_DATASET_PATH:-"${NFS_DIR}/btrabucc/neurips_data_collection/gemini-2.5-flash-train"}
DATA_PREFIX=${DATA_PREFIX:-"${NFS_DIR}/btrabucc/neurips_sft/gemini-2.5-flash-train"}

export HF_HOME=${NFS_DIR}/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN

ALL_JUDGE_NAMES=${ALL_JUDGE_NAMES:-"gemini-2.5-flash-judge"}
ALL_MAX_NUM_SAMPLES=${ALL_MAX_NUM_SAMPLES:-"10000"}
ALL_SUCCESS_THRESHOLDS=${ALL_SUCCESS_THRESHOLDS:-"1.0"}

for JUDGE_NAME in ${ALL_JUDGE_NAMES}; do
for MAX_NUM_SAMPLES in ${ALL_MAX_NUM_SAMPLES}; do
for SUCCESS_THRESHOLD in ${ALL_SUCCESS_THRESHOLDS}; do

IDENTIFIER="${MAX_NUM_SAMPLES}x-${SUCCESS_THRESHOLD}s"
OUTPUT_DATASET_PATH="${DATA_PREFIX}-${IDENTIFIER}-${JUDGE_NAME}"

DATASET_ARGS=(
    sft/agent/create_sft_dataset.py
    --input_data_dir ${INPUT_DATASET_PATH}
    --dataset_output_dir ${OUTPUT_DATASET_PATH}
    --judge_name ${JUDGE_NAME}
    --max_num_samples ${MAX_NUM_SAMPLES}
    --success_threshold ${SUCCESS_THRESHOLD}
    --efficiency_threshold ${SUCCESS_THRESHOLD}
    --self_correction_threshold ${SUCCESS_THRESHOLD}
)

echo "Data args: ${DATASET_ARGS[@]}"
python ${DATASET_ARGS[@]}
    
done
done
done
