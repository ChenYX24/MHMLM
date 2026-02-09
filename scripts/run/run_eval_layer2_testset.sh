#!/bin/bash
set -euo pipefail

# ============================================
# Evaluate Layer2 testset (eval_layer2_testset.py + score_and_visualize_layer2.py)
# ============================================

# -------- Paths --------
PROJECT_ROOT="${PROJECT_ROOT:-/data1/chenyuxuan/Layer2}"
MHMLM_ROOT="${MHMLM_ROOT:-/data1/chenyuxuan/MHMLM}"

LAYER2_TESTSET="${LAYER2_TESTSET:-${PROJECT_ROOT}/data/ord_layer2/layer2_test.jsonl}"

# few-shot dev pool (jsonl / jsonl.gz)
DEVSET_PATH="${DEVSET_PATH:-/data1/chenyuxuan/Layer2/data/ord_layer2_v2/layer2_val.jsonl.gz}"   # e.g. /data1/lvchangwei/LLM/SMolInstruct/data/constructed_dev/dev.jsonl

MOLAWARE_CKPT="${MOLAWARE_CKPT:-/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200/llm}"
TOKEN_CLASSIFIER_PATH="${TOKEN_CLASSIFIER_PATH:-/data1/lvchangwei/LLM/Lora/qwen_mlp_token_classifier.pt}"
BASE_LLM_PATH="${BASE_LLM_PATH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-${MHMLM_ROOT}/eval_results/eval_results/layer2_testset_$(date +%Y%m%d_%H%M%S)}"

# -------- Runtime / GPU --------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
DEVICE="${DEVICE:-cuda:0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
DTYPE="${DTYPE:-bf16}"

# -------- Generation --------
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
REALTIME_MOL="${REALTIME_MOL:-0}"

# -------- Few-shot --------
FEW_SHOT_K="${FEW_SHOT_K:-1}"                 # 0=zero-shot
FEW_SHOT_SEED="${FEW_SHOT_SEED:-42}"
FEW_SHOT_SAME_DATASET="${FEW_SHOT_SAME_DATASET:-0}"  # 1=prefer same dataset_id

# -------- Optional limits --------
MAX_SAMPLES="${MAX_SAMPLES}"   # 空表示全量：MAX_SAMPLES=""
TASKS="${TASKS:-}"                # e.g. "mask_product predict_yield_full"

# -------- Scoring sampling options (if your score script supports them, otherwise忽略) --------
SCORE_MAX_SAMPLES="${SCORE_MAX_SAMPLES:-}"     # 空=全量
SAMPLE_MODE="${SAMPLE_MODE:-head}"             # head|random|stride（你的score脚本不一定支持）
SAMPLE_SEED="${SAMPLE_SEED:-42}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-2}"

# -------- Script paths --------
EVAL_SCRIPT="${MHMLM_ROOT}/scripts/eval/eval_layer2_testset.py"
SCORE_SCRIPT="${MHMLM_ROOT}/scripts/postprocess/score_and_visualize_layer2.py"

mkdir -p "$OUTPUT_DIR"
cd "$MHMLM_ROOT"

echo "============================================"
echo "Running Layer2 evaluation (eval_layer2_testset.py)"
echo "CUDA_VISIBLE_DEVICES        = $CUDA_VISIBLE_DEVICES"
echo "DEVICE / DEVICE_MAP         = $DEVICE / ${DEVICE_MAP:-<single-gpu>}"
echo "Testset                     = $LAYER2_TESTSET"
echo "Out dir                     = $OUTPUT_DIR"
echo "Few-shot                    = $FEW_SHOT_K (seed=$FEW_SHOT_SEED same_dataset=$FEW_SHOT_SAME_DATASET dev=${DEVSET_PATH:-<none>})"
echo "Gen                         = max_new_tokens=$MAX_NEW_TOKENS temp=$TEMPERATURE top_p=$TOP_P realtime_mol=$REALTIME_MOL"
echo "Max samples                 = ${MAX_SAMPLES:-<all>}"
echo "Tasks                       = ${TASKS:-<all>}"
echo "PYTORCH_CUDA_ALLOC_CONF     = ${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
echo "============================================"

ARGS=(
  --testset_path "$LAYER2_TESTSET"
  --output_dir "$OUTPUT_DIR"
  --molaware_ckpt "$MOLAWARE_CKPT"
  --token_classifier_path "$TOKEN_CLASSIFIER_PATH"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --realtime_mol "$REALTIME_MOL"
  --few_shot_k "$FEW_SHOT_K"
  --few_shot_seed "$FEW_SHOT_SEED"
  --few_shot_same_dataset "$FEW_SHOT_SAME_DATASET"
)

[[ -n "$BASE_LLM_PATH" ]] && ARGS+=( --base_llm_path "$BASE_LLM_PATH" )
[[ -n "$DEVICE_MAP" ]] && ARGS+=( --device_map "$DEVICE_MAP" )
[[ -n "$MAX_SAMPLES" ]] && ARGS+=( --max_samples "$MAX_SAMPLES" )

# only pass devset_path when few-shot enabled
if [[ "$FEW_SHOT_K" -gt 0 ]]; then
  if [[ -z "${DEVSET_PATH}" ]]; then
    echo "[ERROR] FEW_SHOT_K>0 but DEVSET_PATH is empty. Please set DEVSET_PATH to a dev jsonl/jsonl.gz."
    exit 1
  fi
  ARGS+=( --devset_path "$DEVSET_PATH" )
fi

if [[ -n "$TASKS" ]]; then
  # shellcheck disable=SC2206
  TASK_ARR=($TASKS)
  ARGS+=( --tasks "${TASK_ARR[@]}" )
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python "$EVAL_SCRIPT" "${ARGS[@]}"

echo ""
echo "============================================"
echo "Evaluation completed! Results saved to:"
echo "  $OUTPUT_DIR"
echo "============================================"

echo ""
echo "============================================"
echo "Scoring and visualizing results..."
echo "============================================"

# score脚本默认只需要 --results_dir
SCORE_ARGS=( --results_dir "$OUTPUT_DIR" )

# 如果你自己的 v2 score 脚本支持采样参数，再打开下面这些
# [[ -n "$SCORE_MAX_SAMPLES" ]] && SCORE_ARGS+=( --max_samples "$SCORE_MAX_SAMPLES" )
# SCORE_ARGS+=( --sample_mode "$SAMPLE_MODE" --seed "$SAMPLE_SEED" )
# [[ "$SAMPLE_MODE" == "stride" ]] && SCORE_ARGS+=( --stride "$SAMPLE_STRIDE" )

python "$SCORE_SCRIPT" "${SCORE_ARGS[@]}"

echo ""
echo "============================================"
echo "All done! Check:"
echo "  - Predictions: $OUTPUT_DIR/*_predictions.jsonl"
echo "  - Metrics:     $OUTPUT_DIR/task_metrics.json"
echo "  - Plots:       $OUTPUT_DIR/plots/"
echo "============================================"
