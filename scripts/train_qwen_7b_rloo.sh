# require 8 GPUs
# pre-download model by executing
# huggingface-cli download Qwen/Qwen2.5-7B --cache-dir ckpts

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=ckpts/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796

python3 -m verl.trainer.main_ppo \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    algorithm.adv_estimator=rloo \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    data.train_files=orz.parquet \
    data.val_files=olympiadbench.parquet \
    data.max_prompt_length=2048 \
    data.max_response_length=6144 \
    actor_rollout_ref.model.use_remove_padding=True \
    trainer.total_epochs=1 \
    data.train_batch_size=128 \
    actor_rollout_ref.rollout.n=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    trainer.project_name=OpenReasoner \
    trainer.experiment_name=qwen2.5-7b-rloo \
    trainer.test_freq=8