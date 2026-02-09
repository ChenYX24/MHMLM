#!/bin/bash
# Drug Optimization 评估启动脚本
#
# 用法:
#   ./run_drug_optim_eval.sh config/llm_cpt_sft.yaml
#   ./run_drug_optim_eval.sh config/diffusion_base.yaml
#   ./run_drug_optim_eval.sh config/llm_base.yaml --skip-score

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
VENV_PATH="/data1/chenyuxuan/MHMLM/.venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Activated venv: $VENV_PATH"
fi

if [ $# -lt 1 ]; then
    echo "用法: $0 <config.yaml> [其他参数...]"
    echo ""
    echo "示例:"
    echo "  $0 config/llm_cpt_sft.yaml"
    echo "  $0 config/diffusion_base.yaml"
    echo "  $0 config/llm_base.yaml --skip-score"
    exit 1
fi

CONFIG="$1"
shift

echo "=========================================="
echo "Drug Optimization Evaluation"
echo "Config: $CONFIG"
echo "=========================================="

python run_drug_optim_eval.py --config "$CONFIG" "$@"
