#!/usr/bin/env bash
set -euo pipefail

# 切到：当前脚本所在目录的「父目录」
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# 现在 PWD 就是你想要的目录
export PYTHONPATH="${PWD}"

# 参数默认值
: "${DEVICE:=cuda:0}"
: "${SEED:=0}"

python3 -m ldmol_component \
  --device "${DEVICE}" \
  --seed "${SEED}"
