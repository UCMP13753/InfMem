export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nohup env vllm serve ucmp137538/infmem-4B \
    --tensor_parallel_size 8 \
    --gpu-memory-utilization 0.9 \
    --served-model-name qwen \
    --port 8000 \
    > logs/vllm_serve.log 2>&1 &
    