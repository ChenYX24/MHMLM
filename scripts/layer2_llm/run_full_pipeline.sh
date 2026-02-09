#!/bin/bash
# Layer2-LLM 完整流程：生成数据 -> 训练 -> 评测
# 使用方法: bash scripts/layer2_llm/run_full_pipeline.sh

set -e  # 只启用错误退出，不启用未定义变量检查（避免 conda 激活问题）

# ============================================
# 配置参数
# ============================================

# 项目根目录
MHMLM_ROOT="${MHMLM_ROOT:-/data1/chenyuxuan/MHMLM}"
cd "$MHMLM_ROOT"

# 环境配置
CONDA_ENV="${CONDA_ENV:-llam3.2}"
# CUDA_VISIBLE_DEVICES 用于训练（多 GPU）
# 数据生成阶段会单独设置（单 GPU）
TRAIN_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5,6,7}"
NUM_GPUS="${NUM_GPUS:-4}"

# ============================================
# 阶段1: 生成训练数据
# ============================================

# 输入数据（使用 ChemBench）
USE_CHEMBENCH="${USE_CHEMBENCH:-1}"  # 1: 使用 ChemBench, 0: 使用文件
CHEMBENCH_TASK="${CHEMBENCH_TASK:-product}"  # product, retro, yield
CHEMBENCH_SPLIT="${CHEMBENCH_SPLIT:-dev}"  # dev, test (ChemBench 没有 train，使用 dev 作为训练数据)
TRAIN_DATA_INPUT="${TRAIN_DATA_INPUT:-}"  # 如果指定则使用文件，否则使用 ChemBench
TRAIN_DATA_OUTPUT="${TRAIN_DATA_OUTPUT:-${MHMLM_ROOT}/scripts/layer2_llm/data/training_data_${CHEMBENCH_TASK}_${CHEMBENCH_SPLIT}.jsonl}"

# 模型配置（用于生成数据）
GEN_CONFIG="${GEN_CONFIG:-${MHMLM_ROOT}/configs/qwen3_sft_epoch2_3.yaml}"
# 默认使用 GPU 6（如果用户分配了 GPU 6 和 7，数据生成用 GPU 6）
GEN_DEVICE="${GEN_DEVICE:-cuda:6}"
GEN_TASK_TYPE="${GEN_TASK_TYPE:-reaction_prediction}"

# ============================================
# 阶段2: 训练 LLM
# ============================================

# 训练配置
TRAIN_CONFIG="${TRAIN_CONFIG:-${MHMLM_ROOT}/configs/qwen3_sft_epoch2_3.yaml}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/data1/chenyuxuan/checkpoint/qwen3_8b_layer2_llm_$(date +%Y%m%d_%H%M%S)}"
TRAIN_MASTER_PORT="${TRAIN_MASTER_PORT:-29500}"

# ============================================
# 阶段3: 评测 ChemBench
# ============================================

# 评测配置
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${MHMLM_ROOT}/eval_chembench_layer2_llm_$(date +%Y%m%d_%H%M%S)}"
# 默认使用 GPU 7（如果用户分配了 GPU 6 和 7，评测用 GPU 7）
EVAL_DEVICE="${EVAL_DEVICE:-cuda:7}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
TOKEN_CLASSIFIER_PATH="${TOKEN_CLASSIFIER_PATH:-/data1/chenyuxuan/checkpoint/gnn_classifier/qwen3_mlp_token_head.pt}"

# ============================================
# 脚本路径
# ============================================

GENERATE_SCRIPT="${MHMLM_ROOT}/scripts/layer2_llm/generate_training_data.py"
TRAIN_SCRIPT="${MHMLM_ROOT}/train_sft.py"
EVAL_SCRIPT="${MHMLM_ROOT}/scripts/eval/eval_layer2_chembench.py"

# ============================================
# 开始执行
# ============================================

echo "============================================"
echo "Layer2-LLM 完整流程"
echo "============================================"
echo "项目根目录:     $MHMLM_ROOT"
echo "训练 GPU:      $TRAIN_CUDA_VISIBLE_DEVICES"
echo "训练输出:      $TRAIN_OUTPUT_DIR"
echo "评测输出:      $EVAL_OUTPUT_DIR"
echo "============================================"

# 激活环境
source /data1/chenyuxuan/MHMLM/.venv/bin/activate

# ============================================
# 阶段1: 生成训练数据
# ============================================

echo ""
echo "============================================"
echo "阶段1: 生成训练数据"
echo "============================================"
if [ -n "$TRAIN_DATA_INPUT" ]; then
    echo "输入:  $TRAIN_DATA_INPUT"
else
    echo "输入:  ChemBench ($CHEMBENCH_TASK/$CHEMBENCH_SPLIT)"
fi
echo "输出:  $TRAIN_DATA_OUTPUT"
echo "配置:  $GEN_CONFIG"
echo "设备:  $GEN_DEVICE"
echo ""

# 创建输出目录
mkdir -p "$(dirname "$TRAIN_DATA_OUTPUT")"

# 生成训练数据
# 处理设备映射：如果 GEN_DEVICE 是 cuda:X，设置 CUDA_VISIBLE_DEVICES=X，然后使用 cuda:0
# 注意：设置 CUDA_VISIBLE_DEVICES 后，物理 GPU 会被重新映射为逻辑 GPU 0,1,2...
# 所以如果设置 CUDA_VISIBLE_DEVICES=6，那么物理 GPU 6 会变成逻辑 GPU 0
GEN_DEVICE_MAPPED="$GEN_DEVICE"
if [[ "$GEN_DEVICE" == cuda:* ]]; then
    GPU_ID=$(echo "$GEN_DEVICE" | sed 's/cuda://')
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    GEN_DEVICE_MAPPED="cuda:0"
    echo "📌 数据生成设备映射: 物理 GPU $GPU_ID -> 逻辑 GPU 0"
fi

GEN_ARGS=(
    --output "$TRAIN_DATA_OUTPUT"
    --config "$GEN_CONFIG"
    --task_type "$GEN_TASK_TYPE"
    --device "$GEN_DEVICE_MAPPED"
)

# 如果指定了输入文件，使用文件；否则使用 ChemBench
if [ -n "$TRAIN_DATA_INPUT" ] && [ -f "$TRAIN_DATA_INPUT" ]; then
    echo "📂 使用输入文件: $TRAIN_DATA_INPUT"
    GEN_ARGS+=( --input "$TRAIN_DATA_INPUT" )
else
    echo "📂 使用 ChemBench 数据: task=$CHEMBENCH_TASK, split=$CHEMBENCH_SPLIT"
    GEN_ARGS+=( --use_chembench )
    GEN_ARGS+=( --chembench_task "$CHEMBENCH_TASK" )
    GEN_ARGS+=( --chembench_split "$CHEMBENCH_SPLIT" )
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python "$GENERATE_SCRIPT" "${GEN_ARGS[@]}"

if [ ! -f "$TRAIN_DATA_OUTPUT" ]; then
    echo "❌ 训练数据生成失败！"
    exit 1
fi

DATA_COUNT=$(wc -l < "$TRAIN_DATA_OUTPUT")
echo "✅ 训练数据生成完成: $DATA_COUNT 条"

# ============================================
# 阶段2: 训练 LLM
# ============================================

echo ""
echo "============================================"
echo "阶段2: 训练 LLM"
echo "============================================"
echo "训练数据:  $TRAIN_DATA_OUTPUT"
echo "配置文件:  $TRAIN_CONFIG"
echo "输出目录:  $TRAIN_OUTPUT_DIR"
echo "GPU数量:   $NUM_GPUS"
echo ""

# 检查训练数据
if [ ! -f "$TRAIN_DATA_OUTPUT" ]; then
    echo "❌ 训练数据不存在: $TRAIN_DATA_OUTPUT"
    exit 1
fi

# 检查配置文件
if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "❌ 训练配置文件不存在: $TRAIN_CONFIG"
    exit 1
fi

# 创建临时配置文件（更新数据路径）
TEMP_CONFIG="${TRAIN_CONFIG%.yaml}_layer2_temp.yaml"
cp "$TRAIN_CONFIG" "$TEMP_CONFIG"

# 更新数据路径（使用 Python 或 sed）
python3 << EOF
import yaml
import sys

with open("$TEMP_CONFIG", 'r') as f:
    config = yaml.safe_load(f)

# 更新数据路径
if 'data' not in config:
    config['data'] = {}
config['data']['dataset_path'] = "$TRAIN_DATA_OUTPUT"

with open("$TEMP_CONFIG", 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"✅ 已更新配置文件: $TEMP_CONFIG")
print(f"   数据路径: $TRAIN_DATA_OUTPUT")
EOF

# 使用临时配置文件
ACTUAL_TRAIN_CONFIG="$TEMP_CONFIG"

# 创建输出目录
mkdir -p "$TRAIN_OUTPUT_DIR"

# 训练 LLM
echo "开始训练..."
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" torchrun \
    --master_port="$TRAIN_MASTER_PORT" \
    --nproc_per_node="$NUM_GPUS" \
    "$TRAIN_SCRIPT" \
    --config "$ACTUAL_TRAIN_CONFIG" \
    --output_dir "$TRAIN_OUTPUT_DIR"

# 清理临时配置文件
if [ -f "$TEMP_CONFIG" ]; then
    rm "$TEMP_CONFIG"
    echo "✅ 已清理临时配置文件"
fi

# 检查训练是否成功（查找最新的 checkpoint）
# 等待一下，确保文件系统同步
sleep 2

LATEST_CKPT=$(find "$TRAIN_OUTPUT_DIR" -name "checkpoint-*" -type d 2>/dev/null | sort -V | tail -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "⚠️  警告: 未找到训练 checkpoint，可能训练失败或还在训练中"
    echo "   输出目录: $TRAIN_OUTPUT_DIR"
    echo ""
    read -p "是否使用输出目录作为 checkpoint 继续评测？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 已取消评测"
        exit 1
    fi
    # 使用输出目录作为 checkpoint
    MOLAWARE_CKPT="$TRAIN_OUTPUT_DIR"
    echo "   使用输出目录: $MOLAWARE_CKPT"
else
    # 使用最新的 checkpoint
    MOLAWARE_CKPT="$LATEST_CKPT"
    echo "✅ 训练完成，找到 checkpoint: $MOLAWARE_CKPT"
    
    # 检查是否有 llm 子目录（某些配置可能使用）
    if [ -d "$MOLAWARE_CKPT/llm" ]; then
        MOLAWARE_CKPT="$MOLAWARE_CKPT/llm"
        echo "   使用 llm 子目录: $MOLAWARE_CKPT"
    fi
fi

# ============================================
# 阶段3: 评测 ChemBench
# ============================================

echo ""
echo "============================================"
echo "阶段3: 评测 ChemBench"
echo "============================================"
echo "Checkpoint:  $MOLAWARE_CKPT"
echo "输出目录:    $EVAL_OUTPUT_DIR"
echo "评测划分:    $EVAL_SPLIT"
echo ""

# 创建输出目录
mkdir -p "$EVAL_OUTPUT_DIR"

# 评测三个任务
TASKS=("product" "retro" "yield")

for task in "${TASKS[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo "评测任务: $task"
    echo "--------------------------------------------"
    
    # 处理设备映射：如果 EVAL_DEVICE 是 cuda:X，设置 CUDA_VISIBLE_DEVICES=X，然后使用 cuda:0
    EVAL_CUDA_VISIBLE_DEVICES=""
    EVAL_DEVICE_MAPPED="$EVAL_DEVICE"
    if [[ "$EVAL_DEVICE" == cuda:* ]]; then
        EVAL_GPU_ID=$(echo "$EVAL_DEVICE" | sed 's/cuda://')
        EVAL_CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID"
        EVAL_DEVICE_MAPPED="cuda:0"
        echo "📌 评测设备映射: 物理 GPU $EVAL_GPU_ID -> 逻辑 GPU 0"
    fi
    
    CUDA_VISIBLE_DEVICES="$EVAL_CUDA_VISIBLE_DEVICES" python "$EVAL_SCRIPT" \
        --task "$task" \
        --split "$EVAL_SPLIT" \
        --molaware_ckpt "$MOLAWARE_CKPT" \
        --token_classifier_path "$TOKEN_CLASSIFIER_PATH" \
        --device "$EVAL_DEVICE_MAPPED" \
        --dtype bf16 \
        --out_dir "$EVAL_OUTPUT_DIR" \
        --use_layer2_pipeline 1 \
        --max_new_tokens 256 \
        --temperature 0.2 \
        --top_p 0.9
    
    echo "✅ 任务 $task 评测完成"
done

# ============================================
# 总结
# ============================================

echo ""
echo "============================================"
echo "✅ 完整流程执行完成！"
echo "============================================"
echo ""
echo "训练数据:     $TRAIN_DATA_OUTPUT"
echo "训练输出:     $TRAIN_OUTPUT_DIR"
echo "评测输出:     $EVAL_OUTPUT_DIR"
echo ""
echo "评测结果文件:"
echo "  - $EVAL_OUTPUT_DIR/pred_product.jsonl"
echo "  - $EVAL_OUTPUT_DIR/pred_retro.jsonl"
echo "  - $EVAL_OUTPUT_DIR/pred_yield.jsonl"
echo ""
echo "详细结果:"
for task in "${TASKS[@]}"; do
    echo "  - $EVAL_OUTPUT_DIR/chembench4k_${task}_${EVAL_SPLIT}_predictions.jsonl"
    echo "  - $EVAL_OUTPUT_DIR/chembench4k_${task}_${EVAL_SPLIT}_summary.json"
done
echo ""
