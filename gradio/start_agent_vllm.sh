#!/bin/bash

source /miniconda/etc/profile.d/conda.sh
conda activate insta

MODEL_NAME=${MODEL_NAME:-"btrabucco/Insta-Qwen2.5-1.5B-SFT"}
API_KEY=${API_KEY:-"token-abc123"}
DTYPE=${DTYPE:-"bfloat16"}

LLM_ENDPOINT_PORT=${LLM_ENDPOINT_PORT:-8000}
LLM_ENDPOINT=${LLM_ENDPOINT:-"http://localhost:${LLM_ENDPOINT_PORT}/v1"}

TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-1}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

CHUNKED_PREFILL=${CHUNKED_PREFILL:-"--enable-chunked-prefill"}
PREFIX_CACHING=${PREFIX_CACHING:-"--enable-prefix-caching"}

NUM_BATCHED_TOKENS=${NUM_BATCHED_TOKENS:-32768}
MAX_ERRORS=${MAX_ERRORS:-1000}

VLLM_LOG=${VLLM_LOG:-"gradio/vllm.log"}

VLLM_ARGS=(
    --port ${LLM_ENDPOINT_PORT}
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE
    --data-parallel-size $DATA_PARALLEL_SIZE
    --max-model-len $MAX_MODEL_LEN
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION
    ${CHUNKED_PREFILL} ${PREFIX_CACHING}
    --max-num-batched-tokens $NUM_BATCHED_TOKENS
    --api-key $API_KEY
    --dtype $DTYPE
)

read -r -d '' VLLM_COMMAND << END_OF_SCRIPT

source /miniconda/etc/profile.d/conda.sh
conda activate insta

for IDX in {1..${MAX_ERRORS}}; do

vllm serve $MODEL_NAME ${VLLM_ARGS[@]} >> ${VLLM_LOG} 2>&1

done

END_OF_SCRIPT

export NCCL_P2P_DISABLE=1
unset LD_LIBRARY_PATH

screen -S vllm -dm bash -c "${VLLM_COMMAND}"

WAIT_ARGS=(
    --model_name ${MODEL_NAME}
    --llm_endpoint ${LLM_ENDPOINT}
)

python wait_for_vllm.py ${WAIT_ARGS[@]}
    