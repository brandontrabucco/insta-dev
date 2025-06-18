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

INPUT_DATA_DIR=${INPUT_DATA_DIR:-"${NFS_DIR}/btrabucc/neurips_feedback_experiment/gemini-2.5-flash-step6-sft"}
SFT_DATASET_PREFIX=${SFT_DATASET_PREFIX:-"${NFS_DIR}/btrabucc/neurips_feedback_experiment_sft/gemini-2.5-flash-step6-sft"}

export HF_HOME=${NFS_DIR}/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN

ALL_JUDGE_NAMES=${ALL_JUDGE_NAMES:-"gemini-2.5-flash-judge"}
ALL_MAX_NUM_SAMPLES=${ALL_MAX_NUM_SAMPLES:-"5000"}
ALL_SUCCESS_THRESHOLDS=${ALL_SUCCESS_THRESHOLDS:-"1.0"}
ALL_SECONDARY_THRESHOLDS=${ALL_SECONDARY_THRESHOLDS:-"0.0"}

JUDGE_NAME="gemini-2.5-flash-judge"

for MAX_NUM_SAMPLES in ${ALL_MAX_NUM_SAMPLES}; do
for SUCCESS_THRESHOLD in ${ALL_SUCCESS_THRESHOLDS}; do
for SECONDARY_THRESHOLD in ${ALL_SECONDARY_THRESHOLDS}; do

IDX="${MAX_NUM_SAMPLES}x-${SUCCESS_THRESHOLD}s-${SECONDARY_THRESHOLD}c"
SFT_DATASET_PATH="${SFT_DATASET_PREFIX}-${IDX}-${JUDGE_NAME}"

DATASET_ARGS=(
    --input_data_dir ${INPUT_DATA_DIR}
    --dataset_output_dir ${SFT_DATASET_PATH}
    --judge_names ${ALL_JUDGE_NAMES}
)

FILTERING_ARGS=(
    --max_num_samples ${MAX_NUM_SAMPLES}
    --success_threshold ${SUCCESS_THRESHOLD}
    --efficiency_threshold ${SECONDARY_THRESHOLD}
    --self_correction_threshold ${SECONDARY_THRESHOLD}
)

SCRIPT_ARGS=(
    sft/agent/create_sft_dataset.py
    ${DATASET_ARGS[@]}
    ${FILTERING_ARGS[@]}
)

echo "SFT: ${SCRIPT_ARGS[@]}"
python ${SCRIPT_ARGS[@]}
    
done
done
done
