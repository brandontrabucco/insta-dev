#!/bin/bash
#SBATCH --job-name=insta
#SBATCH --exclude=babel-6-13,babel-4-9,babel-4-29,babel-13-13,babel-4-17,babel-4-1,babel-13-1,babel-8-13
#SBATCH --output=logs/datagen-%A-%a-%N.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=array
#SBATCH --gres=gpu:L40S:1
#SBATCH --array=0-319

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}

NUM_AGENTS=${NUM_AGENTS:-32}
PLAYWRIGHT_WORKERS=${PLAYWRIGHT_WORKERS:-8}

PLAYWRIGHT_LOG="playwright-${SLURM_ARRAY_TASK_ID}.log"
PLAYWRIGHT_PORT=$(( 3000 + SLURM_ARRAY_TASK_ID * PLAYWRIGHT_WORKERS ))

export VLLM_LOG="logs/vllm-${SLURM_ARRAY_TASK_ID}.log"
AGENT_LLM_ENDPOINT_PORT=$(( 6000 + SLURM_ARRAY_TASK_ID * 2 ))

SINGULARITY_ARGS=(
    --pwd /code/insta -w
    --env SERVER_LOG=${PLAYWRIGHT_LOG},SERVER_WORKERS=${PLAYWRIGHT_WORKERS},SERVER_BASE_PORT=${PLAYWRIGHT_PORT}
)

export NFS_DIR=${NFS_DIR:-/data/matrix/projects/rsalakhugroup}
singularity run ${SINGULARITY_ARGS[@]} -w \
    ${NFS_DIR}/btrabucc/insta-browser-environment.img & 

sleep ${WAIT_FOR_BROWSER:-90s}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

AGENT_MODEL_NAME=${AGENT_MODEL_NAME:-"btrabucco/Insta-Qwen2.5-1.5B-GRPO-n1"}
AGENT_LLM_ENDPOINT=${AGENT_LLM_ENDPOINT:-"http://localhost:${AGENT_LLM_ENDPOINT_PORT}/v1"}
AGENT_API_KEY=${AGENT_API_KEY:-"token-abc123"}

ROLLOUT_DIR=${ROLLOUT_DIR:-"${NFS_DIR}/btrabucc/insta-150k-v2-qwen-1.5b-grpo-n1-x3"}

JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-"gpt-4.1-nano"}
JUDGE_LLM_ENDPOINT=${JUDGE_LLM_ENDPOINT:-"https://api.openai.com/v1"}
JUDGE_API_KEY=${JUDGE_API_KEY:-${OPENAI_API_KEY}}

RANK=${RANK:-${SLURM_ARRAY_TASK_ID}}
WORLD_SIZE=${WORLD_SIZE:-${SLURM_ARRAY_TASK_COUNT}}

SKIP_FINISHED=${SKIP_FINISHED:-"--skip_finished"}
PRUNE_OBSERVATIONS=${PRUNE_OBSERVATIONS:-"--prune_observations"}

VLLM_ARGS=(
    --agent_model_name ${AGENT_MODEL_NAME}
    --agent_llm_endpoint ${AGENT_LLM_ENDPOINT}
    --agent_api_key ${AGENT_API_KEY}
    --judge_model_name ${JUDGE_MODEL_NAME}
    --judge_llm_endpoint ${JUDGE_LLM_ENDPOINT}
    --judge_api_key ${JUDGE_API_KEY}
)

PIPELINE_ARGS=(
    --dataset data-for-agents/insta-150k-v2
    --dataset_split train
    --playwright_port ${PLAYWRIGHT_PORT}
    --playwright_workers ${PLAYWRIGHT_WORKERS}
    --num_agents ${NUM_AGENTS}
    --rank ${RANK}
    --world_size ${WORLD_SIZE}
    --action_parser simplified_json
    ${SKIP_FINISHED}
    ${PRUNE_OBSERVATIONS}
)

export MODEL_NAME=${AGENT_MODEL_NAME}
export LLM_ENDPOINT_PORT=${AGENT_LLM_ENDPOINT_PORT}
bash rl/start_rollout_vllm.sh

DATA_ARGS=(
    --observations_dir ${ROLLOUT_DIR}/observations
    --screenshot_dir ${ROLLOUT_DIR}/screenshots
    --actions_dir ${ROLLOUT_DIR}/actions
    --judgments_dir ${ROLLOUT_DIR}/judgments
)

export HF_HOME=/scratch/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN

export NCCL_P2P_DISABLE=1
unset LD_LIBRARY_PATH

python -u rl/rollout_pipeline.py \
    ${PIPELINE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${VLLM_ARGS[@]} \
    > logs/agents-${RANK}.log 2>&1

screen -XS vllm quit
