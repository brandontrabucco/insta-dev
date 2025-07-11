#!/bin/bash
#SBATCH --job-name=insta
#SBATCH --exclude=shire-1-1,shire-1-6,shire-1-10,shire-2-5,shire-2-9,babel-2-29,babel-0-19
#SBATCH --output=logs/gemini-upload-%A-%a-%N.out
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=array
#SBATCH --array=0-0



# Arguments for data collection

export NFS_DIR=${NFS_DIR:-/data/matrix/projects/rsalakhugroup}

DATASET=${DATASET:-"btrabucco/refiner-step1"}
DATASET_SPLIT=${DATASET_SPLIT:-"train"}

INPUT_DATA_DIR=${INPUT_DATA_DIR:-"${NFS_DIR}/btrabucc/neurips_refiner_experiment/qwen3-235b-refiner-step0"}

JUDGE_NAME=${JUDGE_NAME:-"qwen3-235b-judge"}
TASK_PROPOSER_NAME=${TASK_PROPOSER_NAME:-"qwen3-235b-task-refiner"}

SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD:-2.0}



# Arguments for uploading

UPLOAD_ARGS=(
    scripts/upload_tasks.py
    --input_data_dir ${INPUT_DATA_DIR}
    --judge_name ${JUDGE_NAME}
    --task_proposer_name ${TASK_PROPOSER_NAME}
    --dataset ${DATASET}
    --dataset_split ${DATASET_SPLIT}
    --success_threshold ${SUCCESS_THRESHOLD}
)



# Upload tasks

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

export HF_HOME=${NFS_DIR}/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN

python -u ${UPLOAD_ARGS[@]}
