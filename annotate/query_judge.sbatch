#!/bin/bash
#SBATCH --job-name=insta
#SBATCH --exclude=babel-6-13,babel-4-9,babel-4-29,babel-13-13,babel-4-17,babel-4-1,babel-13-1,babel-8-13,babel-13-29
#SBATCH --output=logs/judge-%A-%a-%N.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=array
#SBATCH --array=0-7



# Slurm arguments for parallelism

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}

RANK=${RANK:-${SLURM_ARRAY_TASK_ID}}
WORLD_SIZE=${WORLD_SIZE:-${SLURM_ARRAY_TASK_COUNT}}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta



# Arguments for data collection

INPUT_DATA_DIR=${INPUT_DATA_DIR:-"${NFS_DIR}/btrabucc/neurips_data_collection/qwen3-1.7b-10000x-0.9s-qwen3-235b-judge"}

OVERWRITE=${OVERWRITE:-"--overwrite"}

JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-"gpt-4.1-nano"}
JUDGE_LLM_ENDPOINT=${JUDGE_LLM_ENDPOINT:-"https://api.openai.com/v1"}
JUDGE_API_KEY=${JUDGE_API_KEY:-${OPENAI_API_KEY}}

JUDGE_NAME=${JUDGE_NAME:-"gpt-4.1-nano-judge"}



# Array arguments for scripts

LLM_ARGS=(
    --judge_model_name ${JUDGE_MODEL_NAME}
    --judge_llm_endpoint ${JUDGE_LLM_ENDPOINT}
    --judge_api_key ${JUDGE_API_KEY}
)

PIPELINE_ARGS=(
    --input_data_dir ${INPUT_DATA_DIR}
    --judge_name ${JUDGE_NAME}
    --dataset data-for-agents/insta-150k-v2
    --dataset_split train
    --rank ${RANK}
    --world_size ${WORLD_SIZE}
    ${OVERWRITE}
)

ANNOTATE_ARGS=(
    ${LLM_ARGS[@]}
    ${PIPELINE_ARGS[@]}
)



# Annotate trajectories using the judge

python -u annotate/query_judge.py ${ANNOTATE_ARGS[@]}