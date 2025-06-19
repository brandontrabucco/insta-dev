#!/bin/bash
#SBATCH --job-name=insta
#SBATCH --exclude=babel-6-13,babel-4-9,babel-4-29,babel-13-13,babel-4-17,babel-4-1,babel-13-1,babel-8-13,babel-13-29
#SBATCH --output=logs/insta-%A-%a-%N.out
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=array
#SBATCH --gres=gpu:L40S:1
#SBATCH --array=0-39



# Slurm arguments for parallelism

SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}

RANK=${RANK:-${SLURM_ARRAY_TASK_ID}}
WORLD_SIZE=${WORLD_SIZE:-${SLURM_ARRAY_TASK_COUNT}}

NUM_AGENTS=${NUM_AGENTS:-8}
PLAYWRIGHT_WORKERS=${PLAYWRIGHT_WORKERS:-8}

PLAYWRIGHT_LOG="playwright-${SLURM_ARRAY_TASK_ID}.log"
PLAYWRIGHT_PORT=$(( 3000 + SLURM_ARRAY_TASK_ID * PLAYWRIGHT_WORKERS ))

export LLM_LOG="logs/llm-${SLURM_ARRAY_TASK_ID}.log"
AGENT_LLM_ENDPOINT_PORT=$(( 6000 + SLURM_ARRAY_TASK_ID * 2 ))



# Start the browser environment

SINGULARITY_ARGS=(
    --pwd /code/insta -w
    --env SERVER_LOG=${PLAYWRIGHT_LOG},SERVER_WORKERS=${PLAYWRIGHT_WORKERS},SERVER_BASE_PORT=${PLAYWRIGHT_PORT}
)

export NFS_DIR=${NFS_DIR:-/data/matrix/projects/rsalakhugroup}
singularity run ${SINGULARITY_ARGS[@]} -w \
    ${NFS_DIR}/btrabucc/insta-browser-environment.img & 

sleep ${WAIT_FOR_BROWSER:-90s}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate insta



# Arguments for data collection

DATASET=${DATASET:-"btrabucco/gemini-2.5-flash-step6"}
DATASET_SPLIT=${DATASET_SPLIT:-"train"}

SET_EXPLORATION_MODE=${SET_EXPLORATION_MODE:-""}
SET_ANNOTATE_JUDGE=${SET_ANNOTATE_JUDGE:-"--set_annotate_judge"}

INPUT_DATA_DIR=${INPUT_DATA_DIR:-"${NFS_DIR}/btrabucc/neurips_feedback_experiment_sft/qwen3-1.7b-step6-e5-5000x-1.0s-0.0c-gemini-2.5-flash-judge"}

SKIP_FINISHED=${SKIP_FINISHED:-"--skip_finished"}
PRUNE_OBSERVATIONS=${PRUNE_OBSERVATIONS:-"--prune_observations"}

AGENT_MODEL_NAME=${AGENT_MODEL_NAME:-"${NFS_DIR}/btrabucc/neurips_feedback_experiment_sft/qwen3-1.7b-step6-e5-5000x-1.0s-0.0c-gemini-2.5-flash-judge"}
AGENT_LLM_ENDPOINT=${AGENT_LLM_ENDPOINT:-"http://localhost:${AGENT_LLM_ENDPOINT_PORT}/v1"}
AGENT_API_KEY=${AGENT_API_KEY:-"token-abc123"}

JUDGE_MODEL_NAME=${JUDGE_MODEL_NAME:-"gemini-2.5-flash"}
JUDGE_LLM_ENDPOINT=${JUDGE_LLM_ENDPOINT:-"https://generativelanguage.googleapis.com/v1beta/openai/"}
JUDGE_API_KEY=${JUDGE_API_KEY:-${GOOGLE_API_KEY}}

JUDGE_NAME=${JUDGE_NAME:-"gemini-2.5-flash-judge"}

ADD_STEPS_TO_AGENT=${ADD_STEPS_TO_AGENT:-""}
ADD_CRITERIA_TO_AGENT=${ADD_CRITERIA_TO_AGENT:-""}

ADD_STEPS_TO_JUDGE=${ADD_STEPS_TO_JUDGE:-"--add_steps_to_judge"}
ADD_CRITERIA_TO_JUDGE=${ADD_CRITERIA_TO_JUDGE:-"--add_criteria_to_judge"}

AGENT_DISABLE_THINKING_CHAT_TEMPLATE=${AGENT_DISABLE_THINKING_CHAT_TEMPLATE:-"--agent_disable_thinking_chat_template"}
JUDGE_DISABLE_THINKING_CHAT_TEMPLATE=${JUDGE_DISABLE_THINKING_CHAT_TEMPLATE:-""}

AGENT_REASONING_EFFORT=${AGENT_REASONING_EFFORT:-""}
JUDGE_REASONING_EFFORT=${JUDGE_REASONING_EFFORT:-"--judge_reasoning_effort none"}

AGENT_PROMPT=${AGENT_PROMPT:-"verbose"}
JUDGE_PROMPT=${JUDGE_PROMPT:-"verbose"}



# Arguments for sampling

AGENT_TEMPERATURE=${AGENT_TEMPERATURE:-0.5}
AGENT_TOP_P=${AGENT_TOP_P:-1}
AGENT_TOP_K=${AGENT_TOP_K:-20}



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
)

SAMPLING_ARGS=(
    --agent_temperature ${AGENT_TEMPERATURE}
    --agent_top_p ${AGENT_TOP_P}
    --agent_top_k ${AGENT_TOP_K}
    ${AGENT_DISABLE_THINKING_CHAT_TEMPLATE}
    ${AGENT_REASONING_EFFORT}
    ${JUDGE_DISABLE_THINKING_CHAT_TEMPLATE}
    ${JUDGE_REASONING_EFFORT}
)

PIPELINE_ARGS=(
    --playwright_port ${PLAYWRIGHT_PORT}
    --playwright_workers ${PLAYWRIGHT_WORKERS}
    --agent_prompt ${AGENT_PROMPT}
    --judge_prompt ${JUDGE_PROMPT}
    --num_agents ${NUM_AGENTS}
    --rank ${RANK}
    --world_size ${WORLD_SIZE}
    ${SKIP_FINISHED}
    ${PRUNE_OBSERVATIONS}
)

DATA_ARGS=(
    --observations_dir ${INPUT_DATA_DIR}/observations
    --screenshot_dir ${INPUT_DATA_DIR}/screenshots
    --actions_dir ${INPUT_DATA_DIR}/actions
    --judgments_dir ${INPUT_DATA_DIR}/${JUDGE_NAME}
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



# Start the LLM server

export MODEL_NAME=${AGENT_MODEL_NAME}
export LLM_ENDPOINT_PORT=${AGENT_LLM_ENDPOINT_PORT}
bash eval/start_llm_server.sh



# Configure cuda and nvidia

export NCCL_P2P_DISABLE=1
unset LD_LIBRARY_PATH

export OUTLINES_CACHE_DIR=/scratch/.tmp-$RANDOM
export OMP_NUM_THREADS=8

export HF_HOME=${NFS_DIR}/btrabucc/hfcache
huggingface-cli login --token $HUGGINGFACE_ACCESS_TOKEN



# Start InSTA pipeline

echo "Starting pipeline with args ${INSTA_ARGS[@]}"

python -u eval/insta_pipeline.py \
    ${INSTA_ARGS[@]} \
    > logs/agents-${RANK}.log 2>&1

screen -XS llm-server quit



# for AGENT_MODEL_NAME in /data/matrix/projects/rsalakhugroup/btrabucc/qwen3-1.7b-*x-0.9s-judgments/; do sbatch --export=AGENT_MODEL_NAME=$AGENT_MODEL_NAME eval/start_insta_pipeline.sbatch; done
