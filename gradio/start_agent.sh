#!/bin/bash

export SERVER_SCRIPT=${SERVER_SCRIPT:-"javascript/server/src/index.js"}
export SERVER_LOG=${SERVER_LOG:-"gradio/playwright.log"}
export VLLM_LOG=${VLLM_LOG:-"gradio/vllm.log"}

export SERVER_BASE_PORT=${SERVER_BASE_PORT:-3000}
export SERVER_WORKERS=${SERVER_WORKERS:-8}
export MAX_ERRORS=${MAX_ERRORS:-1000}

AGENT_LLM_ENDPOINT_PORT=8000

AGENT_MODEL_NAME=${AGENT_MODEL_NAME:-"btrabucco/Insta-Qwen2.5-1.5B-SFT"}
AGENT_LLM_ENDPOINT=${AGENT_LLM_ENDPOINT:-"http://localhost:${AGENT_LLM_ENDPOINT_PORT}/v1"}
AGENT_API_KEY=${AGENT_API_KEY:-"token-abc123"}

JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-"gpt-4.1-nano"}
JUDGE_LLM_ENDPOINT=${JUDGE_LLM_ENDPOINT:-"https://api.openai.com/v1"}
JUDGE_API_KEY=${JUDGE_API_KEY:-${OPENAI_API_KEY}}

NUM_SAMPLES=${NUM_SAMPLES:-8}
NUM_AGENTS=${NUM_AGENTS:-8}

LAST_OBS=${LAST_OBS:-3}
MAX_ACTIONS=${MAX_ACTIONS:-30}

source /miniconda3/bin/activate
conda activate insta

rm ${SERVER_LOG}
touch ${SERVER_LOG}
tail -f ${SERVER_LOG} &

rm ${VLLM_LOG}
touch ${VLLM_LOG}
tail -f ${VLLM_LOG} &

bash start_playwright_server.sh

VLLM_ARGS=(
    MODEL_NAME=${AGENT_MODEL_NAME}
    LLM_ENDPOINT_PORT=${AGENT_LLM_ENDPOINT_PORT}
)

export ${VLLM_ARGS[@]}
bash gradio/start_agent_vllm.sh

LLM_ARGS=(
    --agent_model_name ${AGENT_MODEL_NAME}
    --agent_llm_endpoint ${AGENT_LLM_ENDPOINT}
    --agent_api_key ${AGENT_API_KEY}
    --judge_model_name ${JUDGE_MODEL_NAME}
    --judge_llm_endpoint ${JUDGE_LLM_ENDPOINT}
    --judge_api_key ${JUDGE_API_KEY}
)

PLAYWRIGHT_ARGS=(
    --playwright_url "http://localhost:{port}"
    --playwright_port ${SERVER_BASE_PORT}
    --playwright_workers ${SERVER_WORKERS}
)

AGENT_ARGS=(
    --num_samples ${NUM_SAMPLES}
    --num_agents ${NUM_AGENTS}
    --last_obs ${LAST_OBS}
    --max_actions ${MAX_ACTIONS}
)

python -u gradio/demo_agent.py \
    ${LLM_ARGS[@]} ${PLAYWRIGHT_ARGS[@]} ${AGENT_ARGS[@]}