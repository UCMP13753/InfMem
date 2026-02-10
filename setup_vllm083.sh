#!/usr/bin/env bash
set -e  # stop on first error
set -x  # print each command for debugging

# ===============================
# 1. Create and activate conda env (optional)
# ===============================
# Uncomment the following lines if using conda
# ENV_NAME=xagent_env
# conda create -y -n $ENV_NAME python=3.10
# conda activate $ENV_NAME

# ===============================
# 2. Install core dependencies
# ===============================

pip install --upgrade pip setuptools wheel
pip install --no-cache-dir "vllm==0.8.3" 
pip install --no-cache-dir  "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata \
    "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff

# ===============================
# 3. Install FlashAttention
# ===============================
# wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# ===============================
# 4. Install FlashInfer
# ===============================
# wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl
pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl

# pip uninstall -y pynvml nvidia-ml-py && \
#     pip install --no-cache-dir --upgrade "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

# pip install --no-cache-dir verl[vllm] -U
pip install nltk
pip install rank-bm25
pip install -U "qwen-agent[gui,rag,code_interpreter,mcp]"

pip install rouge
pip install scikit-learn
pip install tenacity
pip install transformers==4.53.0