#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

VLLM_LOG=${VLLM_LOG:-"vllm.log"}
MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.3-70B-Instruct"}
API_KEY=${API_KEY:-"token-abc123"}
MAX_ERRORS=${MAX_ERRORS:-1000}

TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

CHUNKED_PREFILL=${CHUNKED_PREFILL:-"--enable-chunked-prefill"}
PREFIX_CACHING=${PREFIX_CACHING:-"--enable-prefix-caching"}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}

VLLM_ARGS=(
    --dtype bfloat16
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE
    --max-model-len $MAX_MODEL_LEN
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION
    ${CHUNKED_PREFILL} ${PREFIX_CACHING}
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS
    --api-key $API_KEY
)

read -r -d '' VLLM_COMMAND << END_OF_SCRIPT

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

unset LD_LIBRARY_PATH

for IDX in {1..${MAX_ERRORS}}; do

vllm serve $MODEL_NAME ${VLLM_ARGS[@]} >> ${VLLM_LOG} 2>&1

done

END_OF_SCRIPT

# start a new vllm server in the background
screen -S vllm -dm bash -c "${VLLM_COMMAND}"
python wait_for_vllm.py \
    --model_name ${MODEL_NAME}
