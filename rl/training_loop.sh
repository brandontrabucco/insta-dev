#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

export NUM_ITERATIONS=${NUM_ITERATIONS:-100}
export BEST_OF_N=${BEST_OF_N:-5}

for ((ITERATION = 1; ITERATION < NUM_ITERATIONS; ITERATION++)); do

export AGENT_MODEL_NAME="./qwen-1.5b-grpo-n${ITERATION}"

export RANK=${ITERATION}
export WORLD_SIZE=${NUM_ITERATIONS}

bash rl/start_rollout_pipeline.sh

NEXT_ITERATION=$(( ITERATION + 1 ))

export PROJECT_NAME="verl_qwen_grpo_pipeline"
export EXPERIMENT_NAME="qwen2.5_1.5b_grpo_n${NEXT_ITERATION}_lr1e-5"

export ROLLOUT_DIRS="./qwen-1.5b-grpo-n${ITERATION}-rollouts/*"
export DATASET_OUTPUT_FILE="./rl/insta-150k-v2-grpo-n${NEXT_ITERATION}.parquet"

export MODEL_PATH="./qwen-1.5b-grpo-n${ITERATION}"
export DEFAULT_LOCAL_DIR="./qwen-1.5b-grpo-n${NEXT_ITERATION}"

export VERL_LOG="rl/verl-${ITERATION}.log"

bash rl/start_verl_pipeline.sh

done