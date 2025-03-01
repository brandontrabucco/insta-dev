#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.3-70B-Instruct"}
NUM_AGENTS=${NUM_AGENTS:-32}
PLAYWRIGHT_WORKERS=${PLAYWRIGHT_WORKERS:-8}

RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}

SKIP_FINISHED=${SKIP_FINISHED:-"--skip_finished"}
PRUNE_OBSERVATIONS=${PRUNE_OBSERVATIONS:-"--prune_observations"}

unset LD_LIBRARY_PATH

PIPELINE_ARGS=(
    --model_name ${MODEL_NAME}
    --num_agents ${NUM_AGENTS}
    --playwright_workers ${PLAYWRIGHT_WORKERS}
    --rank ${RANK}
    --world_size ${WORLD_SIZE}
    ${SKIP_FINISHED}
    ${PRUNE_OBSERVATIONS}
)

python -u run_pipeline.py ${PIPELINE_ARGS[@]} >> agents.log 2>&1
