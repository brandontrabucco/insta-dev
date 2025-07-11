#!/bin/bash

NUM_REFINE_STEPS=5

START_INSTA_PIPELINE_SCRIPT="datagen/gemini-2.5-flash/start_insta_pipeline.sbatch"
UPLOAD_TASKS_SCRIPT="datagen/gemini-2.5-flash/upload_tasks.sbatch"

START_DATASET="data-for-agents/insta-150k-v1"
DATASET_SPLIT="test"

BASE_INPUT_DATA_DIR="${NFS_DIR}/btrabucc/refiner_test/gemini-2.5-flash-refiner"
BASE_DATASET="btrabucco/refiner"

INPUT_DATA_DIR=${BASE_INPUT_DATA_DIR}-step0
DATASET=${START_DATASET}

EXPORT_LIST=DATASET=${DATASET},DATASET_SPLIT=${DATASET_SPLIT},INPUT_DATA_DIR=${INPUT_DATA_DIR},SET_EXPLORATION_MODE="--set_exploration_mode"
JOB_ID=$(sbatch --parsable --export=${EXPORT_LIST} ${START_INSTA_PIPELINE_SCRIPT})

for STEP in $(seq 1 ${NUM_REFINE_STEPS}); do

if (( STEP < 3 )); then

# upload all tasks to huggingface
SUCCESS_THRESHOLD=2.0

else

# upload only the failed tasks
SUCCESS_THRESHOLD=1.0

fi

DATASET=${BASE_DATASET}-step${STEP}

EXPORT_LIST=DATASET=${DATASET},DATASET_SPLIT=${DATASET_SPLIT},INPUT_DATA_DIR=${INPUT_DATA_DIR},SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD}
JOB_ID=$(sbatch --dependency=afterok:${JOB_ID} --parsable --export=${EXPORT_LIST} ${UPLOAD_TASKS_SCRIPT})

INPUT_DATA_DIR=${BASE_INPUT_DATA_DIR}-step${STEP}

EXPORT_LIST=DATASET=${DATASET},DATASET_SPLIT=${DATASET_SPLIT},INPUT_DATA_DIR=${INPUT_DATA_DIR}
JOB_ID=$(sbatch --dependency=afterok:${JOB_ID} --parsable --export=${EXPORT_LIST} ${START_INSTA_PIPELINE_SCRIPT})

done