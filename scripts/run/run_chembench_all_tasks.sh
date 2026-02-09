#!/bin/bash
# ChemBench 三任务评测脚本
# 评测 product, retro, yield 三个任务

set -euo pipefail

# ============================================
# 路径配置
# ============================================

# Qwen cpt+sft checkpoint
MOLAWARE_CKPT="${MOLAWARE_CKPT:-/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200}"

# Mol classifier
TOKEN_CLASSIFIER_PATH="${TOKEN_CLASSIFIER_PATH:-/data1/chenyuxuan/checkpoint/gnn_classifier/qwen3_mlp_token_head.pt}"

# Base LLM (可选)
BASE_LLM_PATH="${BASE_LLM_PATH:-}"

# ============================================
# 运行配置
# ============================================
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-}"
DTYPE="${DTYPE:-bf16}"

# ============================================
# 生成参数
# ============================================
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
REALTIME_MOL="${REALTIME_MOL:-1}"

# ============================================
# 评测参数
# ============================================
SPLIT="${SPLIT:-test}"     # test 或 dev
MAX_SAMPLES="${MAX_SAMPLES:-}"  # 空表示全量

# ============================================
# Layer2 参数
# ============================================
USE_LAYER2_PIPELINE="${USE_LAYER2_PIPELINE:-1}"  # 1: 启用 Layer2 pipeline, 0: 禁用
LAYER2_TASK_TYPE="${LAYER2_TASK_TYPE:-}"  # 可选: reaction_prediction, yield_prediction, product_prediction, reactant_prediction

# ============================================
# 输出目录
# ============================================
OUTPUT_DIR="${OUTPUT_DIR:-/data1/chenyuxuan/MHMLM/eval_chembench_$(date +%Y%m%d_%H%M%S)}"

# ============================================
# 脚本路径
# ============================================
MHMLM_ROOT="${MHMLM_ROOT:-/data1/chenyuxuan/MHMLM}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${MHMLM_ROOT}/scripts/eval/eval_layer2_chembench.py}"

cd "$MHMLM_ROOT"

# 创建输出目录并确保有写入权限
mkdir -p "$OUTPUT_DIR"
# 如果目录已存在但不可写，创建新的带时间戳的目录
if [ ! -w "$OUTPUT_DIR" ]; then
    echo "⚠️  警告: 输出目录不可写，创建新目录..."
    OUTPUT_DIR="${MHMLM_ROOT}/eval_chembench_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$OUTPUT_DIR"
    echo "   使用新目录: $OUTPUT_DIR"
fi

echo "============================================"
echo "ChemBench 三任务评测"
echo "============================================"
echo "Split:                   $SPLIT"
echo "CUDA_VISIBLE_DEVICES:    $CUDA_VISIBLE_DEVICES"
echo "Device:                  $DEVICE"
echo "Dtype:                   $DTYPE"
echo "MolAware Checkpoint:     $MOLAWARE_CKPT"
echo "Token Classifier:        $TOKEN_CLASSIFIER_PATH"
echo "Output Dir:              $OUTPUT_DIR"
echo "Max Samples:             ${MAX_SAMPLES:-<all>}"
echo "Gen Params:              max_tokens=$MAX_NEW_TOKENS temp=$TEMPERATURE top_p=$TOP_P realtime_mol=$REALTIME_MOL"
echo "Layer2 Pipeline:          ${USE_LAYER2_PIPELINE:-0} (${LAYER2_TASK_TYPE:-auto})"
echo "============================================"

# 构建基础参数
BASE_ARGS=(
    --molaware_ckpt "$MOLAWARE_CKPT"
    --token_classifier_path "$TOKEN_CLASSIFIER_PATH"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --split "$SPLIT"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --realtime_mol "$REALTIME_MOL"
)

# 可选参数
[[ -n "$BASE_LLM_PATH" ]] && BASE_ARGS+=( --base_llm_path "$BASE_LLM_PATH" )
[[ -n "$DEVICE_MAP" ]] && BASE_ARGS+=( --device_map "$DEVICE_MAP" )
[[ -n "$MAX_SAMPLES" ]] && BASE_ARGS+=( --max_samples "$MAX_SAMPLES" )

# Layer2 参数
BASE_ARGS+=( --use_layer2_pipeline "$USE_LAYER2_PIPELINE" )
[[ -n "$LAYER2_TASK_TYPE" ]] && BASE_ARGS+=( --layer2_task_type "$LAYER2_TASK_TYPE" )

# 任务列表
TASKS=("product" "retro" "yield")

# 评测每个任务
for task in "${TASKS[@]}"; do
    echo ""
    echo "============================================"
    echo "评测任务: $task ($SPLIT split)"
    echo "时间: $(date)"
    echo "============================================"
    
    ARGS=("${BASE_ARGS[@]}")
    ARGS+=( --task "$task" )
    ARGS+=( --out_dir "$OUTPUT_DIR" )
    
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python "$EVAL_SCRIPT" "${ARGS[@]}"
    
    echo "✅ 任务 $task 完成"
done

echo ""
echo "============================================"
echo "所有任务评测完成！"
echo "============================================"
echo ""
echo "输出文件:"
echo "  - $OUTPUT_DIR/pred_product.jsonl"
echo "  - $OUTPUT_DIR/pred_retro.jsonl"
echo "  - $OUTPUT_DIR/pred_yield.jsonl"
echo ""
echo "详细结果:"
echo "  - $OUTPUT_DIR/chembench4k_product_${SPLIT}_predictions.jsonl"
echo "  - $OUTPUT_DIR/chembench4k_retro_${SPLIT}_predictions.jsonl"
echo "  - $OUTPUT_DIR/chembench4k_yield_${SPLIT}_predictions.jsonl"
echo ""
echo "摘要文件:"
echo "  - $OUTPUT_DIR/chembench4k_product_${SPLIT}_summary.json"
echo "  - $OUTPUT_DIR/chembench4k_retro_${SPLIT}_summary.json"
echo "  - $OUTPUT_DIR/chembench4k_yield_${SPLIT}_summary.json"
echo ""
