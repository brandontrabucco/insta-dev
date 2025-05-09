#!/bin/bash
#SBATCH --job-name=insta
#SBATCH --exclude=shire-1-1,shire-1-6,shire-1-10,shire-2-5,shire-2-9,babel-2-29
#SBATCH --output=logs/llama-%A-%a-%N.out
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=array
#SBATCH --array=0-40%10



# Slurm arguments for parallelism

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}

NUM_AGENTS=${NUM_AGENTS:-8}
PLAYWRIGHT_WORKERS=${PLAYWRIGHT_WORKERS:-8}

PLAYWRIGHT_LOG="playwright-${SLURM_ARRAY_TASK_ID}.log"
PLAYWRIGHT_PORT=$(( 3000 + SLURM_ARRAY_TASK_ID * PLAYWRIGHT_WORKERS ))

export LLM_LOG="logs/llm-${SLURM_ARRAY_TASK_ID}.log"
AGENT_LLM_ENDPOINT_PORT=$(( 6000 + SLURM_ARRAY_TASK_ID * 2 ))



# Start the browser environment

SINGULARITY_ARGS=(
    --pwd /code/insta -w
    --env SERVER_LOG=${PLAYWRIGHT_LOG},SERVER_WORKERS=${PLAYWRIGHT_WORKERS},SERVER_BASE_PORT=${PLAYWRIGHT_PORT}
)

export NFS_DIR=${NFS_DIR:-/data/matrix/projects/rsalakhugroup}
singularity run ${SINGULARITY_ARGS[@]} -w \
    ${NFS_DIR}/btrabucc/insta-browser-environment.img & 

sleep ${WAIT_FOR_BROWSER:-90s}



# Arguments for data collection

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

AGENT_MODEL_NAME=${AGENT_MODEL_NAME:-"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"}
AGENT_LLM_ENDPOINT=${AGENT_LLM_ENDPOINT:-"https://api.together.xyz/v1"}
AGENT_API_KEY=${AGENT_API_KEY:-${TOGETHER_API_KEY}}

ROLLOUT_DIR=${ROLLOUT_DIR:-"${NFS_DIR}/btrabucc/insta-150k-v2-llama4-maverick-test"}

JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-"gpt-4.1-nano"}
JUDGE_LLM_ENDPOINT=${JUDGE_LLM_ENDPOINT:-"https://api.openai.com/v1"}
JUDGE_API_KEY=${JUDGE_API_KEY:-${OPENAI_API_KEY}}

RANK=${RANK:-${SLURM_ARRAY_TASK_ID}}
WORLD_SIZE=${WORLD_SIZE:-${SLURM_ARRAY_TASK_COUNT}}

SKIP_FINISHED=${SKIP_FINISHED:-"--skip_finished"}
PRUNE_OBSERVATIONS=${PRUNE_OBSERVATIONS:-"--prune_observations"}



# Array arguments for scripts

LLM_ARGS=(
    --agent_model_name ${AGENT_MODEL_NAME}
    --agent_llm_endpoint ${AGENT_LLM_ENDPOINT}
    --agent_api_key ${AGENT_API_KEY}
    --judge_model_name ${JUDGE_MODEL_NAME}
    --judge_llm_endpoint ${JUDGE_LLM_ENDPOINT}
    --judge_api_key ${JUDGE_API_KEY}
)

PIPELINE_ARGS=(
    --dataset data-for-agents/insta-150k-v2
    --dataset_split test
    --playwright_port ${PLAYWRIGHT_PORT}
    --playwright_workers ${PLAYWRIGHT_WORKERS}
    --num_agents ${NUM_AGENTS}
    --rank ${RANK}
    --world_size ${WORLD_SIZE}
    --action_parser json
    ${SKIP_FINISHED}
    ${PRUNE_OBSERVATIONS}
)

DATA_ARGS=(
    --observations_dir ${ROLLOUT_DIR}/observations
    --screenshot_dir ${ROLLOUT_DIR}/screenshots
    --actions_dir ${ROLLOUT_DIR}/actions
    --judgments_dir ${ROLLOUT_DIR}/gpt-4.1-nano-judge
)

INSTA_ARGS=(
    ${LLM_ARGS[@]}
    ${PIPELINE_ARGS[@]}
    ${DATA_ARGS[@]}
)



# Configure cuda and nvidia

export NCCL_P2P_DISABLE=1
unset LD_LIBRARY_PATH

export OUTLINES_CACHE_DIR=/scratch/.tmp-$RANDOM
export OMP_NUM_THREADS=8

export HF_HOME=${NFS_DIR}/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN



# Start InSTA pipeline

python -u datagen/insta_pipeline.py \
    ${INSTA_ARGS[@]} \
    > logs/agents-${RANK}.log 2>&1
