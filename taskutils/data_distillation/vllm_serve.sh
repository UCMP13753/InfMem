export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup vllm serve Qwen/Qwen3-32B \
    --tensor_parallel_size 8 \
    --gpu-memory-utilization 0.9 \
    --served-model-name qwen32b \
    --port 8000 \
    > qwen32b.log 2>&1 &

