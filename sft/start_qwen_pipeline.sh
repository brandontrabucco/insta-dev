#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

AGENT_MODEL_NAME=${AGENT_MODEL_NAME:-"./qwen-1.5b-sft"}
AGENT_LLM_ENDPOINT=${AGENT_LLM_ENDPOINT:-"http://localhost:8000/v1"}
AGENT_API_KEY=${AGENT_API_KEY:-"token-abc123"}

JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-"gpt-4o-mini"}
JUDGE_LLM_ENDPOINT=${JUDGE_LLM_ENDPOINT:-"https://api.openai.com/v1"}
JUDGE_API_KEY=${JUDGE_API_KEY:-${OPENAI_API_KEY}}

NUM_AGENTS=${NUM_AGENTS:-128}
PLAYWRIGHT_WORKERS=${PLAYWRIGHT_WORKERS:-32}

RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}

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
    --dataset_split test
    --num_agents ${NUM_AGENTS}
    --playwright_workers ${PLAYWRIGHT_WORKERS}
    --rank ${RANK}
    --world_size ${WORLD_SIZE}
    --action_parser simplified_json
    ${SKIP_FINISHED}
    ${PRUNE_OBSERVATIONS}
)

unset LD_LIBRARY_PATH

DATA_ARGS=(
    --observations_dir qwen-1.5b-sft-rollouts/test/observations
    --screenshot_dir qwen-1.5b-sft-rollouts/test/screenshots
    --actions_dir qwen-1.5b-sft-rollouts/test/actions
    --judgments_dir qwen-1.5b-sft-rollouts/test/judgments
)

python -u sft/run_qwen_pipeline.py \
    ${PIPELINE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${VLLM_ARGS[@]} \
    > agents-test.log 2>&1
