#!/usr/bin/env bash
set -euo pipefail

source /data1/chenyuxuan/MHMLM/.venv/bin/activate

for mode in 1 2 3 4; do
    echo "Running debug_sft_tester with mode=$mode..."
    python scripts/dev/debug_sft_tester.py --mode="$mode"
    echo "Finished mode=$mode"
    echo "------------------------"
done
