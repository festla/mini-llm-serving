#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export HF_HUB_OFFLINE=1

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype half \
  --gpu-memory-utilization 0.75 \
  --generation-config vllm \
  --disable-cascade-attn \
  --no-enable-prefix-caching \
  --host 127.0.0.1 \
  --port 8000
