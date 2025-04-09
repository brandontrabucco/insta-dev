#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

AGENT_MODEL_NAME=${AGENT_MODEL_NAME:-"./qwen-1.5b-grpo-n0"}
JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-"meta-llama/Llama-3.3-70B-Instruct"}

AGENT_LLM_ENDPOINT=${AGENT_LLM_ENDPOINT:-"http://localhost:8000/v1"}
JUDGE_LLM_ENDPOINT=${JUDGE_LLM_ENDPOINT:-"http://localhost:8001/v1"}

NUM_AGENTS=${NUM_AGENTS:-32}
PLAYWRIGHT_WORKERS=${PLAYWRIGHT_WORKERS:-8}

RANK=${RANK:-1}
WORLD_SIZE=${WORLD_SIZE:-150}

SKIP_FINISHED=${SKIP_FINISHED:-"--skip_finished"}
PRUNE_OBSERVATIONS=${PRUNE_OBSERVATIONS:-"--prune_observations"}

VLLM_ARGS=(
    --agent_model_name ${AGENT_MODEL_NAME}
    --agent_llm_endpoint ${AGENT_LLM_ENDPOINT}
    --judge_model_name ${JUDGE_MODEL_NAME}
    --judge_llm_endpoint ${JUDGE_LLM_ENDPOINT}
)

PIPELINE_ARGS=(
    --dataset data-for-agents/insta-150k-v2
    --dataset_split train
    --num_agents ${NUM_AGENTS}
    --playwright_workers ${PLAYWRIGHT_WORKERS}
    --rank ${RANK}
    --world_size ${WORLD_SIZE}
    --action_parser simplified_json
    ${SKIP_FINISHED}
    ${PRUNE_OBSERVATIONS}
)

unset LD_LIBRARY_PATH

for ITERATION in 0 1 2 3 4 5 6 7; do 

DATA_ARGS=(
    --observations_dir qwen-1.5b-grpo-n0-rollouts/x${ITERATION}/observations
    --screenshot_dir qwen-1.5b-grpo-n0-rollouts/x${ITERATION}/screenshots
    --actions_dir qwen-1.5b-grpo-n0-rollouts/x${ITERATION}/actions
    --judgments_dir qwen-1.5b-grpo-n0-rollouts/x${ITERATION}/judgments
)

python -u run_pipeline.py \
    ${PIPELINE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${VLLM_ARGS[@]} \
    > agents-${ITERATION}.log 2>&1

done
