#!/bin/bash

MODEL_PATH=${MODEL_PATH:-"./qwen-1.5b-sft"}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-"./qwen-1.5b-grpo-n0"}

PROJECT_NAME=${PROJECT_NAME:-"verl_qwen_grpo"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen2.5_1.5b_grpo_n0_lr1e-5-large-batch"}

TRAIN_FILES=${TRAIN_FILES:-"./verl/insta-150k-v2-grpo-n0.parquet"}
VAL_FILES=${VAL_FILES:-"./verl/insta-150k-v2-grpo-n0.parquet"}

VERL_LOG=${VERL_LOG:-"./verl/trainer.log"}

set -x  # Enable debugging output

TRAINER_ARGS=(
    algorithm.adv_estimator=grpo 
    custom_reward_function.path=./verl/reward_func.py 
    custom_reward_function.name='compute_score' 
    trainer.default_local_dir=${DEFAULT_LOCAL_DIR}
    trainer.critic_warmup=0 
    trainer.logger=['console','wandb'] 
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=8 
    trainer.nnodes=1 
    trainer.save_freq=100000
    trainer.test_freq=40 
    trainer.total_epochs=5
)

DATASET_ARGS=(
    data.train_files=${TRAIN_FILES}
    data.val_files=${VAL_FILES}
    data.train_batch_size=1024 
    data.max_prompt_length=7680 
    data.max_response_length=512 
    data.filter_overlong_prompts=True 
    data.truncation='error' 
)

ACTOR_ARGS=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.actor.ppo_mini_batch_size=256 
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.actor.use_kl_loss=True 
    actor_rollout_ref.actor.kl_loss_coef=0.001 
    actor_rollout_ref.actor.kl_loss_type=low_var_kl 
    actor_rollout_ref.actor.entropy_coeff=0 
    actor_rollout_ref.model.use_remove_padding=True 
    actor_rollout_ref.model.enable_gradient_checkpointing=True 
    actor_rollout_ref.actor.fsdp_config.param_offload=True 
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True 
    actor_rollout_ref.ref.fsdp_config.param_offload=True 
)

ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 
    actor_rollout_ref.rollout.name=vllm 
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 
    actor_rollout_ref.rollout.n=5 
)

python3 -m verl.trainer.main_ppo \
    ${ROLLOUT_ARGS[@]} \
    ${ACTOR_ARGS[@]} \
    ${DATASET_ARGS[@]} \
    ${TRAINER_ARGS[@]} \
    > ${VERL_LOG} 2>&1

chmod -R 777 ${DEFAULT_LOCAL_DIR}
chmod -R 777 ${VERL_LOG}

chmod -R 777 wandb
chmod -R 777 outputs
