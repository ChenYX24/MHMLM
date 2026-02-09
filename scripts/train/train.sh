#!/bin/bash

# 使用4卡训练 qwen3_sft_epoch2_1.yaml 和 qwen3_sft_epoch2_2.yaml

set -e

export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1

CODE_DIR="/data1/chenyuxuan/MHMLM"
CONFIG1="${CODE_DIR}/configs/qwen3_sft_epoch2_1.yaml"
CONFIG2="${CODE_DIR}/configs/qwen3_sft_epoch2_2.yaml"
MASTER_PORT=29506
NPROC_PER_NODE=4  # 使用4张卡

echo "=========================================="
echo "Training Qwen3 Epoch2 Configs (4 GPUs)"
echo "=========================================="
echo "Config 1: $CONFIG1"
echo "Config 2: $CONFIG2"
echo "GPUs: 4"
echo "Master Port: $MASTER_PORT"
echo "=========================================="

cd "$CODE_DIR"

# ========== Step 1: Train with qwen3_sft_epoch2_1.yaml ==========
echo ""
echo "=========================================="
echo "Step 1: Training with qwen3_sft_epoch2_1.yaml"
echo "=========================================="

if [ ! -f "$CONFIG1" ]; then
    echo "❌ Config file not found: $CONFIG1"
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
    train_sft.py "$CONFIG1"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Training with config1 failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "✅ Config1 training completed successfully!"
echo "=========================================="

# ========== Step 2: Train with qwen3_sft_epoch2_2.yaml ==========
echo ""
echo "=========================================="
echo "Step 2: Training with qwen3_sft_epoch2_2.yaml"
echo "=========================================="

if [ ! -f "$CONFIG2" ]; then
    echo "❌ Config file not found: $CONFIG2"
    exit 1
fi

# 使用不同的master_port避免冲突
MASTER_PORT2=$((MASTER_PORT + 1))

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT2 \
    train_sft.py "$CONFIG2"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Training with config2 failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "✅ Config2 training completed successfully!"
echo "=========================================="

echo ""
echo "=========================================="
echo "✅ All training completed successfully!"
echo "=========================================="
