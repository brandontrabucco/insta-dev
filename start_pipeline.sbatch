#!/bin/bash
#SBATCH --job-name=insta
#SBATCH --exclude=babel-4-9,babel-4-29,babel-13-13,babel-4-17,babel-4-1,babel-13-1,babel-8-13
#SBATCH --output=logs/slurm-%A-%a-%N.out
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --partition=array
#SBATCH --gres=gpu:L40S:8
#SBATCH --array=1-4

export NFS_DIR=${NFS_DIR:-/data/matrix/projects/rsalakhugroup}
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SLURM_ARRAY_TASK_COUNT=7

unset SSL_CERT_FILE
unset LD_LIBRARY_PATH
export NCCL_P2P_DISABLE=1

singularity run --pwd /code/insta --env SERVER_LOG="playwright-${SLURM_ARRAY_TASK_ID}.log" -w \
    ${NFS_DIR}/btrabucc/insta-browser-environment.img & 

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

export MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.3-70B-Instruct"}
NUM_AGENTS=${NUM_AGENTS:-32}
PLAYWRIGHT_WORKERS=${PLAYWRIGHT_WORKERS:-8}

SKIP_FINISHED=${SKIP_FINISHED:-"--skip_finished"}
PRUNE_OBSERVATIONS=${PRUNE_OBSERVATIONS:-"--prune_observations"}

export HF_HOME=/data/hf_cache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN

export VLLM_LOG="vllm-${SLURM_ARRAY_TASK_ID}.log"
bash start_vllm_server.sh

DATA_ARGS=(
    --observations_dir ${NFS_DIR}/datasets/insta-150k-v2-llama-3.3/observations
    --screenshot_dir ${NFS_DIR}/datasets/insta-150k-v2-llama-3.3/screenshots
    --actions_dir ${NFS_DIR}/datasets/insta-150k-v2-llama-3.3/actions
    --judgments_dir ${NFS_DIR}/datasets/insta-150k-v2-llama-3.3/judgments
)

PIPELINE_ARGS=(
    --dataset data-for-agents/insta-150k-v2
    --model_name ${MODEL_NAME}
    --num_agents ${NUM_AGENTS}
    --playwright_workers ${PLAYWRIGHT_WORKERS}
    --rank ${SLURM_ARRAY_TASK_ID}
    --world_size ${SLURM_ARRAY_TASK_COUNT}
    ${SKIP_FINISHED}
    ${PRUNE_OBSERVATIONS}
    ${DATA_ARGS[@]}
)

python -u run_pipeline.py ${PIPELINE_ARGS[@]}
