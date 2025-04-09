#!/bin/bash

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

TRAINER_ARGS=(
    algorithm.adv_estimator=grpo 
    custom_reward_function.path=./verl/reward_func.py 
    custom_reward_function.name='compute_score' 
    trainer.critic_warmup=0 
    trainer.logger=['console','wandb'] 
    trainer.project_name='verl_qwen_grpo' 
    trainer.experiment_name='qwen2.5_1.5b_grpo_n0' 
    trainer.n_gpus_per_node=8 
    trainer.nnodes=1 
    trainer.save_freq=-1 
    trainer.test_freq=50 
    trainer.total_epochs=10
)

DATA_ARGS=(
    data.train_files=./verl/insta-150k-v2-grpo-n0.parquet 
    data.val_files=./verl/insta-150k-v2-grpo-n0.parquet 
    data.train_batch_size=256 
    data.max_prompt_length=7680 
    data.max_response_length=512 
    data.filter_overlong_prompts=True 
    data.truncation='error' 
)

ACTOR_ARGS=(
    actor_rollout_ref.model.path=./qwen-1.5b-sft 
    actor_rollout_ref.actor.optim.lr=1e-6 
    actor_rollout_ref.model.use_remove_padding=True 
    actor_rollout_ref.actor.ppo_mini_batch_size=32 
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 
    actor_rollout_ref.actor.use_kl_loss=True 
    actor_rollout_ref.actor.kl_loss_coef=0.001 
    actor_rollout_ref.actor.kl_loss_type=low_var_kl 
    actor_rollout_ref.actor.entropy_coeff=0 
    actor_rollout_ref.model.enable_gradient_checkpointing=True 
    actor_rollout_ref.actor.fsdp_config.param_offload=True 
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True 
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 
    actor_rollout_ref.rollout.name=vllm 
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 
    actor_rollout_ref.rollout.n=5 
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 
    actor_rollout_ref.ref.fsdp_config.param_offload=True 
)

python3 -m verl.trainer.main_ppo \
    ${ACTOR_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINER_ARGS[@]} $@