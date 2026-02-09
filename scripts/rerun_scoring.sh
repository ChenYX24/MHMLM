#!/bin/bash
# 重新运行评分脚本，使用修复后的代码

# 四个预测目录
DIRS=(
    "/data1/chenyuxuan/MHMLM/test_output_eval_qwen_v24-20251216-225348_checkpoint-800"
    "/data1/chenyuxuan/MHMLM/test_output_eval_qwen_v6-20251217-220917_checkpoint-5938"
    "/data1/chenyuxuan/MHMLM/test_output_eval_qwen_GNN_nofreeze_checkpoint-39"
    "/data1/chenyuxuan/MHMLM/test_output_eval_qwen_LLM_nofreeze_checkpoint-400"
)

SCRIPT_DIR="/data1/lvchangwei/LLM/SMolInstruct"

for DIR in "${DIRS[@]}"; do
    echo "========================================="
    echo "Scoring: $DIR"
    echo "========================================="
    
    python "${SCRIPT_DIR}/score_smolinstruct.py" \
        --prediction_dir "$DIR" \
        --save_json "${DIR}/scored_results.json"
    
    echo ""
done

echo "All scoring completed!"

