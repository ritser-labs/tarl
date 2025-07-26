#!/bin/bash
# NCCL debugging and optimization
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_TIMEOUT=3600
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

# Execute the training command
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=4 \
  src/open_r1/grpo.py \
  --config recipes/grpo.yaml "$@"
