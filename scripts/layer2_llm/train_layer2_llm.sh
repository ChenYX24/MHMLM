#!/bin/bash
# 第二阶段：训练 LLM（使用第一阶段生成的数据）

# 默认配置
CODE_DIR="/data1/chenyuxuan/MHMLM"
CONFIG_FILE="${CODE_DIR}/configs/qwen3_sft_epoch2_layer2.yaml"
OUTPUT_DIR="/data1/chenyuxuan/checkpoint/qwen3_8b_layer2_llm"
TRAIN_DATA="${CODE_DIR}/scripts/layer2_llm/data/training_data.jsonl"

# 允许通过命令行参数覆盖
if [ "$1" != "" ]; then
    CONFIG_FILE="$1"
fi
if [ "$2" != "" ]; then
    OUTPUT_DIR="$2"
fi

# 激活环境
source /home/chenyuxuan/miniconda3/etc/profile.d/conda.sh
conda activate llam3.2

cd ${CODE_DIR}

# 训练命令
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --master_port=29500 \
    --nproc_per_node=4 \
    train_sft.py \
    --config ${CONFIG_FILE} \
    --output_dir ${OUTPUT_DIR}

echo "✅ 训练完成！"
