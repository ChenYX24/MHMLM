#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只算分的脚本 - 对已有的预测结果进行评分
用法：
    python scripts/score_only.py \
        --prediction_dir /data1/chenyuxuan/MHMLM/test_output_eval_qwen_GNN_nofreeze_checkpoint-39_fewshot \
        --save_json /data1/chenyuxuan/MHMLM/test_output_eval_qwen_GNN_nofreeze_checkpoint-39_fewshot/scored_results.json
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_project_root))

from eval.eval_smolinstruct import run_scoring


def main():
    parser = argparse.ArgumentParser(description="只对已有预测结果进行评分")
    parser.add_argument(
        "--prediction_dir",
        type=str,
        default="",
        help="包含预测结果 jsonl 文件的目录（默认: $MHMLM_ROOT/eval_results/results）",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="保存评分结果的 JSON 文件路径（可选）",
    )
    parser.add_argument(
        "--score_workers",
        type=int,
        default=1,
        help="评分时使用的进程数（1=单进程，>1=多进程）",
    )
    parser.add_argument(
        "--skip_tasks",
        type=str,
        default="",
        help="跳过这些任务（逗号分隔）",
    )
    
    args = parser.parse_args()
    
    # Default prediction root: MHMLM_ROOT/eval_results/results
    import os
    mhmlm_root = Path(os.environ.get("MHMLM_ROOT", "/data1/chenyuxuan/MHMLM"))
    default_pred_root = mhmlm_root / "eval_results" / "results"
    pred_dir = Path(args.prediction_dir).expanduser().resolve() if args.prediction_dir else default_pred_root
    if not pred_dir.exists():
        print(f"错误：目录不存在: {pred_dir}")
        sys.exit(1)
    
    if not pred_dir.is_dir():
        print(f"错误：不是目录: {pred_dir}")
        sys.exit(1)
    
    # 调用评分函数
    save_json = args.save_json if args.save_json else str(pred_dir / "scored_results.json")
    run_scoring(
        pred_dir,
        save_json=save_json,
        score_workers=args.score_workers,
        skip_tasks=args.skip_tasks,
    )
    
    print(f"\n[INFO] 评分完成，结果已保存到: {save_json}")


if __name__ == "__main__":
    main()

