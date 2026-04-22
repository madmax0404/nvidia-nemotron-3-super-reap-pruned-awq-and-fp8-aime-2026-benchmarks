#!/usr/bin/env bash
# Start vLLM server for reap pruned Nemotron 3 Super 120B
#
# Usage:
#   bash src/start_vllm_server.sh

set -euo pipefail

# export HF_HOME=/workspace/.hf_home
# export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1
# export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
# export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
# export VLLM_USE_FLASHINFER_MOE_FP16=1
# --performance-mode throughput
# --mamba-ssm-cache-dtype float16

MODEL="${MODEL:-madmax0404/nvidia-nemotron-3-super-reap-pruned-checkpoint-47-fp8}"
MODEL_NAME="${MODEL_NAME:-nemotron-3-super}"
# TP="${TP:-2}"
# DP="${DP:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.96}"
PORT="${PORT:-8000}"
# MAX_LORA_RANK="${MAX_LORA_RANK:-32}"

vllm serve "$MODEL" \
  --served-model-name "$MODEL_NAME" \
  --port "$PORT" \
  --async-scheduling \
  --dtype auto \
  --kv-cache-dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser nemotron_v3 \
  --max-num-seqs 128