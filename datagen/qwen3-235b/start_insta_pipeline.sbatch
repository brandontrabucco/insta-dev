#!/bin/bash
#SBATCH --job-name=insta
#SBATCH --exclude=shire-1-1,shire-1-6,shire-1-10,shire-2-5,shire-2-9,babel-2-29
#SBATCH --output=logs/qwen-%A-%a-%N.out
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=array
#SBATCH --array=0-7



# Slurm arguments for parallelism

export NFS_DIR=${NFS_DIR:-/data/matrix/projects/rsalakhugroup}

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}

RANK=${RANK:-${SLURM_ARRAY_TASK_ID}}
WORLD_SIZE=${WORLD_SIZE:-${SLURM_ARRAY_TASK_COUNT}}

NUM_AGENTS=${NUM_AGENTS:-8}
PLAYWRIGHT_WORKERS=${PLAYWRIGHT_WORKERS:-8}

PLAYWRIGHT_LOG="playwright-${SLURM_ARRAY_TASK_ID}.log"
PLAYWRIGHT_PORT=$(( 3200 + SLURM_ARRAY_TASK_ID * PLAYWRIGHT_WORKERS ))

export LLM_LOG="logs/llm-${SLURM_ARRAY_TASK_ID}.log"
AGENT_LLM_ENDPOINT_PORT=$(( 6000 + SLURM_ARRAY_TASK_ID * 2 ))



# Start the browser environment

SINGULARITY_ARGS=(
    --pwd /code/insta -w
    --env SERVER_LOG=${PLAYWRIGHT_LOG},SERVER_WORKERS=${PLAYWRIGHT_WORKERS},SERVER_BASE_PORT=${PLAYWRIGHT_PORT}
)

singularity run ${SINGULARITY_ARGS[@]} -w \
    ${NFS_DIR}/btrabucc/insta-browser-environment.img & 

sleep ${WAIT_FOR_BROWSER:-90s}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta



# Arguments for data collection

DATASET=${DATASET:-"data-for-agents/insta-150k-v3"}
DATASET_SPLIT=${DATASET_SPLIT:-"train"}

SET_EXPLORATION_MODE=${SET_EXPLORATION_MODE:-""}
SET_ANNOTATE_JUDGE=${SET_ANNOTATE_JUDGE:-"--set_annotate_judge"}
SET_ANNOTATE_TASK_PROPOSER=${SET_ANNOTATE_TASK_PROPOSER:-"--set_annotate_task_proposer"}

INPUT_DATA_DIR=${INPUT_DATA_DIR:-"${NFS_DIR}/btrabucc/neurips_data_collection/qwen3-235b-test"}

SKIP_FINISHED=${SKIP_FINISHED:-"--skip_finished"}
PRUNE_OBSERVATIONS=${PRUNE_OBSERVATIONS:-"--prune_observations"}

AGENT_MODEL_NAME=${AGENT_MODEL_NAME:-"Qwen/Qwen3-235B-A22B-fp8-tput"}
AGENT_LLM_ENDPOINT=${AGENT_LLM_ENDPOINT:-"https://api.together.xyz/v1"}
AGENT_API_KEY=${AGENT_API_KEY:-${TOGETHER_API_KEY}}

JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-"Qwen/Qwen3-235B-A22B-fp8-tput"}
JUDGE_LLM_ENDPOINT=${JUDGE_LLM_ENDPOINT:-"https://api.together.xyz/v1"}
JUDGE_API_KEY=${JUDGE_API_KEY:-${TOGETHER_API_KEY}}

TASK_PROPOSER_MODEL_NAME=${TASK_PROPOSER_MODEL_NAME:-"Qwen/Qwen3-235B-A22B-fp8-tput"}
TASK_PROPOSER_LLM_ENDPOINT=${TASK_PROPOSER_LLM_ENDPOINT:-"https://api.together.xyz/v1"}
TASK_PROPOSER_API_KEY=${TASK_PROPOSER_API_KEY:-${TOGETHER_API_KEY}}

JUDGE_NAME=${JUDGE_NAME:-"qwen3-235b-judge"}
TASK_PROPOSER_NAME=${TASK_PROPOSER_NAME:-"qwen3-235b-task-refiner"}

ADD_STEPS_TO_AGENT=${ADD_STEPS_TO_AGENT:-"--add_steps_to_agent"}
ADD_CRITERIA_TO_AGENT=${ADD_CRITERIA_TO_AGENT:-"--add_criteria_to_agent"}

ADD_STEPS_TO_JUDGE=${ADD_STEPS_TO_JUDGE:-"--add_steps_to_judge"}
ADD_CRITERIA_TO_JUDGE=${ADD_CRITERIA_TO_JUDGE:-"--add_criteria_to_judge"}

ADD_STEPS_TO_TASK_PROPOSER=${ADD_STEPS_TO_TASK_PROPOSER:-"--add_steps_to_task_proposer"}
ADD_CRITERIA_TO_TASK_PROPOSER=${ADD_CRITERIA_TO_TASK_PROPOSER:-"--add_criteria_to_task_proposer"}

AGENT_DISABLE_THINKING_CHAT_TEMPLATE=${AGENT_DISABLE_THINKING_CHAT_TEMPLATE:-"--agent_disable_thinking_chat_template"}
JUDGE_DISABLE_THINKING_CHAT_TEMPLATE=${JUDGE_DISABLE_THINKING_CHAT_TEMPLATE:-"--judge_disable_thinking_chat_template"}
TASK_PROPOSER_DISABLE_THINKING_CHAT_TEMPLATE=${TASK_PROPOSER_DISABLE_THINKING_CHAT_TEMPLATE:-"--task_proposer_disable_thinking_chat_template"}

AGENT_REASONING_EFFORT=${AGENT_REASONING_EFFORT:-""}
JUDGE_REASONING_EFFORT=${JUDGE_REASONING_EFFORT:-""}
TASK_PROPOSER_REASONING_EFFORT=${TASK_PROPOSER_REASONING_EFFORT:-""}

AGENT_PROMPT=${AGENT_PROMPT:-"verbose"}
JUDGE_PROMPT=${JUDGE_PROMPT:-"verbose"}
TASK_PROPOSER_PROMPT=${TASK_PROPOSER_PROMPT:-"refiner"}



# Array arguments for scripts

LLM_ARGS=(
    --agent_model_name ${AGENT_MODEL_NAME}
    --agent_llm_endpoint ${AGENT_LLM_ENDPOINT}
    --agent_api_key ${AGENT_API_KEY}
    ${ADD_STEPS_TO_AGENT}
    ${ADD_CRITERIA_TO_AGENT}
    --judge_model_name ${JUDGE_MODEL_NAME}
    --judge_llm_endpoint ${JUDGE_LLM_ENDPOINT}
    --judge_api_key ${JUDGE_API_KEY}
    ${ADD_STEPS_TO_JUDGE}
    ${ADD_CRITERIA_TO_JUDGE}
    ${SET_ANNOTATE_JUDGE}
    --task_proposer_model_name ${TASK_PROPOSER_MODEL_NAME}
    --task_proposer_llm_endpoint ${TASK_PROPOSER_LLM_ENDPOINT}
    --task_proposer_api_key ${TASK_PROPOSER_API_KEY}
    ${ADD_STEPS_TO_TASK_PROPOSER}
    ${ADD_CRITERIA_TO_TASK_PROPOSER}
    ${SET_ANNOTATE_TASK_PROPOSER}
)

SAMPLING_ARGS=(
    ${AGENT_DISABLE_THINKING_CHAT_TEMPLATE}
    ${AGENT_REASONING_EFFORT}
    ${JUDGE_DISABLE_THINKING_CHAT_TEMPLATE}
    ${JUDGE_REASONING_EFFORT}
    ${TASK_PROPOSER_DISABLE_THINKING_CHAT_TEMPLATE}
    ${TASK_PROPOSER_REASONING_EFFORT}
)

PIPELINE_ARGS=(
    --playwright_port ${PLAYWRIGHT_PORT}
    --playwright_workers ${PLAYWRIGHT_WORKERS}
    --agent_prompt ${AGENT_PROMPT}
    --judge_prompt ${JUDGE_PROMPT}
    --task_proposer_prompt ${TASK_PROPOSER_PROMPT}
    --num_agents ${NUM_AGENTS}
    --rank ${RANK}
    --world_size ${WORLD_SIZE}
    ${SKIP_FINISHED}
    ${PRUNE_OBSERVATIONS}
)

DATA_ARGS=(
    --input_data_dir ${INPUT_DATA_DIR}
    --judge_name ${JUDGE_NAME}
    --task_proposer_name ${TASK_PROPOSER_NAME}
    --dataset ${DATASET}
    --dataset_split ${DATASET_SPLIT}
    ${SET_EXPLORATION_MODE}
)

INSTA_ARGS=(
    ${LLM_ARGS[@]}
    ${SAMPLING_ARGS[@]}
    ${PIPELINE_ARGS[@]}
    ${DATA_ARGS[@]}
)



# Configure cuda and nvidia

export NCCL_P2P_DISABLE=1
unset LD_LIBRARY_PATH

export OUTLINES_CACHE_DIR=/scratch/.tmp-$RANDOM
export OMP_NUM_THREADS=8

export HF_HOME=${NFS_DIR}/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN



# Start InSTA pipeline

echo "Starting pipeline with args ${INSTA_ARGS[@]}"

python -um insta.entry_points.insta_pipeline \
    ${INSTA_ARGS[@]} \
    > logs/agents-${RANK}.log 2>&1
