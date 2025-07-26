#!/bin/bash

# Kill any existing processes
pkill -f grpo 2>/dev/null || true

# NCCL Environment Variables for containerized environments
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Increase NCCL timeout to 2 hours (7200 seconds)
export NCCL_TIMEOUT=7200

# Disable InfiniBand (common issue in containers)
export NCCL_IB_DISABLE=1

# Use ethernet for communication
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_READ=0

# Specify network interface (loopback for local training)
export NCCL_SOCKET_IFNAME=lo

# Disable peer-to-peer communication (can cause hangs)
export NCCL_P2P_DISABLE=1

# Force NCCL to use sockets
export NCCL_TREE_THRESHOLD=0

# Additional stability settings
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

# PyTorch distributed settings
export TORCH_DISTRIBUTED_DEBUG=INFO

echo "Starting multi-GPU training with 3 processes (4 GPUs, 1 reserved for vLLM)..."
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT seconds"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "NCCL_P2P_DISABLE: $NCCL_P2P_DISABLE"

# Activate virtual environment and run training
source openr1/bin/activate

ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file recipes/accelerate_configs/zero2.yaml \
  --num_processes=3 \
  --main_process_port=29504 \
  src/open_r1/grpo.py \
  --config recipes/grpo.yaml "$@" 