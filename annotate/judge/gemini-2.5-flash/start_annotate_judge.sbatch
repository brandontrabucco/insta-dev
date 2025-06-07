#!/bin/bash
#SBATCH --job-name=insta
#SBATCH --exclude=babel-6-13,babel-4-9,babel-4-29,babel-13-13,babel-4-17,babel-4-1,babel-13-1,babel-8-13,babel-13-29
#SBATCH --output=logs/judge-%A-%a-%N.out
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=array
#SBATCH --array=0-7



# Slurm arguments for parallelism

export NFS_DIR=${NFS_DIR:-/data/matrix/projects/rsalakhugroup}

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}

RANK=${RANK:-${SLURM_ARRAY_TASK_ID}}
WORLD_SIZE=${WORLD_SIZE:-${SLURM_ARRAY_TASK_COUNT}}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta



# Arguments for data collection

DATASET=${DATASET:-"data-for-agents/insta-150k-v3"}
DATASET_SPLIT=${DATASET_SPLIT:-"train"}

INPUT_DATA_DIR=${INPUT_DATA_DIR:-"${NFS_DIR}/btrabucc/neurips_data_collection/gemini-2.5-flash-train"}

SKIP_FINISHED=${SKIP_FINISHED:-"--skip_finished"}

JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-"gemini-2.5-flash-preview-05-20"}
JUDGE_LLM_ENDPOINT=${JUDGE_LLM_ENDPOINT:-"https://generativelanguage.googleapis.com/v1beta/openai/"}
JUDGE_API_KEY=${JUDGE_API_KEY:-${GOOGLE_API_KEY}}

JUDGE_NAME=${JUDGE_NAME:-"gemini-2.5-flash-judge"}

ADD_STEPS_TO_JUDGE=${ADD_STEPS_TO_JUDGE:-"--add_steps_to_judge"}
ADD_CRITERIA_TO_JUDGE=${ADD_CRITERIA_TO_JUDGE:-"--add_criteria_to_judge"}

JUDGE_DISABLE_THINKING_CHAT_TEMPLATE=${JUDGE_DISABLE_THINKING_CHAT_TEMPLATE:-""}
JUDGE_REASONING_EFFORT=${JUDGE_REASONING_EFFORT:-""}



# Array arguments for scripts

LLM_ARGS=(
    --judge_model_name ${JUDGE_MODEL_NAME}
    --judge_llm_endpoint ${JUDGE_LLM_ENDPOINT}
    --judge_api_key ${JUDGE_API_KEY}
    ${ADD_STEPS_TO_JUDGE}
    ${ADD_CRITERIA_TO_JUDGE}
)

SAMPLING_ARGS=(
    ${JUDGE_DISABLE_THINKING_CHAT_TEMPLATE}
    ${JUDGE_REASONING_EFFORT}
)

PIPELINE_ARGS=(
    --rank ${RANK}
    --world_size ${WORLD_SIZE}
    ${SKIP_FINISHED}
)

DATA_ARGS=(
    --input_data_dir ${INPUT_DATA_DIR}
    --judge_name ${JUDGE_NAME}
    --dataset ${DATASET}
    --dataset_split ${DATASET_SPLIT}
)

ANNOTATE_ARGS=(
    annotate/judge/query_judge.py
    ${LLM_ARGS[@]}
    ${SAMPLING_ARGS[@]}
    ${PIPELINE_ARGS[@]}
    ${DATA_ARGS[@]}
)



# Annotate trajectories using the judge

python -u ${ANNOTATE_ARGS[@]}