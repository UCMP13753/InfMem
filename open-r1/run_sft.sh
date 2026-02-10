

export NCCL_DEBUG_SUBSYS=COLL
export MODELNAME=Qwen/Qwen3-4B                #Qwen/Qwen3-4B
export EXP_NAME=qwen3_4B_thinking_shorter_mixed_traj_distilled
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/prethink_memagent.yaml \
    --dataset_name \
    --model_name_or_path $MODELNAME \
    --output_dir path_to_${EXP_NAME} \
    --wandb_entity  --wandb_project  --run_name $EXP_NAME \
    > logs/${EXP_NAME}.log 2>&1 &