#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

# Arguments to configure the LLM

MODEL_NAME=${MODEL_NAME:-"./qwen-1.5b-sft"}
API_KEY=${API_KEY:-"token-abc123"}
DTYPE=${DTYPE:-"--dtype bfloat16"}

LLM_ENDPOINT_PORT=${LLM_ENDPOINT_PORT:-8000}
LLM_ENDPOINT=${LLM_ENDPOINT:-"http://localhost:${LLM_ENDPOINT_PORT}/v1"}

LLM_LOG=${LLM_LOG:-"logs/vllm.log"}

TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-4}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-2}
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-1}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}

CHUNKED_PREFILL=${CHUNKED_PREFILL:-"--enable-chunked-prefill"}
PREFIX_CACHING=${PREFIX_CACHING:-"--enable-prefix-caching"}

NUM_BATCHED_TOKENS=${NUM_BATCHED_TOKENS:-32768}
MAX_ERRORS=${MAX_ERRORS:-1000}

# Script array arguments

LLM_ARGS=(
    --port ${LLM_ENDPOINT_PORT}
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE
    --pipeline-parallel-size $PIPELINE_PARALLEL_SIZE
    --data-parallel-size $DATA_PARALLEL_SIZE
    --max-model-len $MAX_MODEL_LEN
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION
    ${CHUNKED_PREFILL} ${PREFIX_CACHING}
    --max-num-batched-tokens $NUM_BATCHED_TOKENS
    --api-key $API_KEY ${DTYPE}
)

WAIT_ARGS=(
    --llm_endpoint ${LLM_ENDPOINT}
    --api_key ${API_KEY}
    --model_name ${MODEL_NAME}
)

# Command to run LLM

read -r -d '' LLM_COMMAND << END_OF_SCRIPT

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

rm ${LLM_LOG}

for IDX in {1..${MAX_ERRORS}}; do

vllm serve $MODEL_NAME ${LLM_ARGS[@]} >> ${LLM_LOG} 2>&1

done

END_OF_SCRIPT

# Configure cuda and nvidia

export NCCL_P2P_DISABLE=1
unset LD_LIBRARY_PATH

export OUTLINES_CACHE_DIR=/scratch/.tmp-$RANDOM
export OMP_NUM_THREADS=8

export HF_HOME=/data/matrix/projects/rsalakhugroup/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN

# Start the LLM and wait

screen -S llm-server -dm bash -c "${LLM_COMMAND}"
python wait_for_llm.py ${WAIT_ARGS[@]}
