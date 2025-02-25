#!/bin/bash

export MODEL_NAME=${MODEL_NAME:-"meta-llama/Llama-3.3-70B-Instruct"}
export WORLD_SIZE=${WORLD_SIZE:-32}
export PLAYWRIGHT_WORKERS=${PLAYWRIGHT_WORKERS:-8}

read -r -d '' AGENT_COMMAND << END_OF_SCRIPT

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta

unset LD_LIBRARY_PATH

python -u run_pipeline.py --rank {} \
    --world_size ${WORLD_SIZE} \
    --model_name ${MODEL_NAME} \
    --playwright_workers ${PLAYWRIGHT_WORKERS} \
    --skip_finished >> agents.log 2>&1

END_OF_SCRIPT

seq 0 $((WORLD_SIZE - 1)) | xargs -I {} -P $WORLD_SIZE \
    bash -c "${AGENT_COMMAND}"

