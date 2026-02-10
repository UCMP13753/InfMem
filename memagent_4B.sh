#!/bin/bash
set -x

# at least 1 nodes, 4nodes=3~4days to converge
NNODES=1
NGPUS_PER_NODE=8
TORCHDYNAMO_VERBOSE=1
YOUR_PROJECT_NAME=
ROLLOUT_TYPE=vllm
YOUR_RUN_NAME=
PROJ_DIR=RL_models/$YOUR_RUN_NAME
DATASET_ROOT=/

MODEL_PATH=Qwen/Qwen3-4B
VAL_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_dev.parquet"
TRAIN_PATH="${DATASET_ROOT}/hotpotqa/hotpotqa_train_32k.parquet"

test_files="['$VAL_PATH']"
# Please note that recurrent framewrok will use max_length defined in task config.
# These two values are just for vLLM to decide max_model_length.
MAXLEN=10240 
MAX_NEW_TOKEN=1536

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory_prethink \
    recurrent.memory_prethink.config.chunk_size=5000 \
    recurrent.memory_prethink.config.max_final_response_length=1536 \
    algorithm.adv_estimator=grpo \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=50 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    trainer.logger=['console','wandb'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$test_files \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=128 \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='thread' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.rollout.name=${ROLLOUT_TYPE} \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=$YOUR_PROJECT_NAME \
    trainer.experiment_name=$YOUR_RUN_NAME \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=15 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR \
    trainer.total_epochs=30 2>&1 | tee logs/${YOUR_RUN_NAME}.log