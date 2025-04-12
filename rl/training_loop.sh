#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

export NUM_ITERATIONS=${NUM_ITERATIONS:-100}
export BEST_OF_N=${BEST_OF_N:-8}

for ((STEP = 0; STEP < NUM_ITERATIONS; STEP++)); do

export AGENT_MODEL_NAME="./qwen-1.5b-grpo-n${STEP}"
bash rl/start_rollout_pipeline.sh

NEXT_STEP=$(( STEP + 1 ))

export PROJECT_NAME="verl_qwen_grpo_pipeline"
export EXPERIMENT_NAME="qwen2.5_1.5b_grpo_n${NEXT_STEP}_lr1e-5"

export ROLLOUT_DIRS="./qwen-1.5b-grpo-n${STEP}-rollouts/*"
export DATASET_OUTPUT_FILE="./rl/insta-150k-v2-grpo-n${NEXT_STEP}.parquet"

export MODEL_PATH="./qwen-1.5b-grpo-n${STEP}"
export DEFAULT_LOCAL_DIR="./qwen-1.5b-grpo-n${NEXT_STEP}"

export VERL_LOG="rl/verl-${STEP}.log"

bash rl/start_verl_pipeline.sh

done