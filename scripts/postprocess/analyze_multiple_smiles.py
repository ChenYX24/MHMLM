#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计评测结果中哪些样本包含多个SMILES
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def extract_all_smiles_from_text(text: str) -> List[str]:
    """从文本中提取所有SMILES字符串"""
    if not text:
        return []
    
    all_smiles = []
    
    # 1. 匹配所有 <SMILES> ... </SMILES> 格式
    smiles_pattern = r'<SMILES>\s*([^<]+)\s*</SMILES>'
    matches = re.findall(smiles_pattern, text, re.IGNORECASE)
    for match in matches:
        cand = match.strip()
        if cand and not re.fullmatch(r"\d+(?:\.\d+)?%?", cand):
            all_smiles.append(cand)
    
    # 2. 尝试解析JSON格式，提取products数组
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "products" in data:
            products = data["products"]
            if isinstance(products, list):
                for prod in products:
                    if isinstance(prod, str) and prod.strip():
                        all_smiles.append(prod.strip())
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # 3. 检查是否包含用 '.' 分隔的多个SMILES
    if not all_smiles:
        # 检查文本中是否包含 '.' 分隔的长字符串（可能是多个SMILES）
        tokens = text.split()
        for token in tokens:
            token = token.strip()
            if '.' in token and len(token) > 20:
                parts = token.split('.')
                for part in parts:
                    part = part.strip()
                    if part and len(part) >= 3 and re.search(r'[A-Za-z]', part):
                        all_smiles.append(part)
    
    return all_smiles


def analyze_predictions_file(pred_file: Path) -> Dict[str, Any]:
    """分析单个预测文件"""
    task_name = pred_file.stem.replace("_predictions", "")
    stats = {
        "task": task_name,
        "total_samples": 0,
        "multi_smiles_pred": 0,  # 预测包含多个SMILES
        "multi_smiles_true": 0,  # 真实值包含多个SMILES
        "multi_smiles_both": 0,  # 两者都包含多个SMILES
        "single_smiles": 0,  # 都是单个SMILES
        "examples": {
            "multi_pred": [],
            "multi_true": [],
            "multi_both": []
        }
    }
    
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                stats["total_samples"] += 1
                
                pred = result.get("prediction", "")
                true = result.get("ground_truth", "")
                reaction_id = result.get("reaction_id", "")
                
                pred_smiles_list = extract_all_smiles_from_text(pred)
                true_smiles_list = extract_all_smiles_from_text(true)
                
                pred_is_multi = len(pred_smiles_list) > 1
                true_is_multi = len(true_smiles_list) > 1
                
                if pred_is_multi and true_is_multi:
                    stats["multi_smiles_both"] += 1
                    if len(stats["examples"]["multi_both"]) < 5:
                        stats["examples"]["multi_both"].append({
                            "reaction_id": reaction_id,
                            "pred_count": len(pred_smiles_list),
                            "true_count": len(true_smiles_list),
                            "pred": pred[:200] + "..." if len(pred) > 200 else pred,
                            "true": true[:200] + "..." if len(true) > 200 else true
                        })
                elif pred_is_multi:
                    stats["multi_smiles_pred"] += 1
                    if len(stats["examples"]["multi_pred"]) < 5:
                        stats["examples"]["multi_pred"].append({
                            "reaction_id": reaction_id,
                            "pred_count": len(pred_smiles_list),
                            "pred": pred[:200] + "..." if len(pred) > 200 else pred,
                            "true": true[:200] + "..." if len(true) > 200 else true
                        })
                elif true_is_multi:
                    stats["multi_smiles_true"] += 1
                    if len(stats["examples"]["multi_true"]) < 5:
                        stats["examples"]["multi_true"].append({
                            "reaction_id": reaction_id,
                            "true_count": len(true_smiles_list),
                            "pred": pred[:200] + "..." if len(pred) > 200 else pred,
                            "true": true[:200] + "..." if len(true) > 200 else true
                        })
                else:
                    stats["single_smiles"] += 1
                    
            except Exception as e:
                continue
    
    return stats


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Analyze which samples contain multiple SMILES in prediction/ground truth.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="Directory containing *_predictions.jsonl (default: $MHMLM_ROOT/eval_results/eval_results)",
    )
    args = parser.parse_args()

    mhmlm_root = Path(os.environ.get("MHMLM_ROOT", "/data1/chenyuxuan/MHMLM"))
    default_root = mhmlm_root / "eval_results" / "eval_results"
    results_dir = Path(args.results_dir).expanduser().resolve() if args.results_dir else default_root
    
    if not results_dir.exists():
        print(f"[ERROR] Results directory does not exist: {results_dir}")
        return
    
    prediction_files = list(results_dir.glob("*_predictions.jsonl"))
    
    if not prediction_files:
        print(f"[ERROR] No prediction files found in: {results_dir}")
        return
    
    print("="*80)
    print("Multiple SMILES Analysis")
    print("="*80)
    print(f"Results directory: {results_dir}\n")
    
    all_stats = []
    for pred_file in sorted(prediction_files):
        stats = analyze_predictions_file(pred_file)
        all_stats.append(stats)
    
    # 打印统计结果
    for stats in all_stats:
        print(f"\nTask: {stats['task']}")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Single SMILES (both): {stats['single_smiles']} ({stats['single_smiles']/stats['total_samples']*100:.1f}%)")
        print(f"  Multiple SMILES in prediction: {stats['multi_smiles_pred']} ({stats['multi_smiles_pred']/stats['total_samples']*100:.1f}%)")
        print(f"  Multiple SMILES in ground truth: {stats['multi_smiles_true']} ({stats['multi_smiles_true']/stats['total_samples']*100:.1f}%)")
        print(f"  Multiple SMILES in both: {stats['multi_smiles_both']} ({stats['multi_smiles_both']/stats['total_samples']*100:.1f}%)")
        
        # 打印示例
        if stats['examples']['multi_pred']:
            print(f"\n  Examples - Multiple SMILES in prediction:")
            for ex in stats['examples']['multi_pred'][:3]:
                print(f"    Reaction ID: {ex['reaction_id']}")
                print(f"      Pred ({ex['pred_count']} SMILES): {ex['pred']}")
                print(f"      True: {ex['true']}")
        
        if stats['examples']['multi_true']:
            print(f"\n  Examples - Multiple SMILES in ground truth:")
            for ex in stats['examples']['multi_true'][:3]:
                print(f"    Reaction ID: {ex['reaction_id']}")
                print(f"      Pred: {ex['pred']}")
                print(f"      True ({ex['true_count']} SMILES): {ex['true']}")
        
        if stats['examples']['multi_both']:
            print(f"\n  Examples - Multiple SMILES in both:")
            for ex in stats['examples']['multi_both'][:3]:
                print(f"    Reaction ID: {ex['reaction_id']}")
                print(f"      Pred ({ex['pred_count']} SMILES): {ex['pred']}")
                print(f"      True ({ex['true_count']} SMILES): {ex['true']}")
    
    # 汇总统计
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    total_samples = sum(s['total_samples'] for s in all_stats)
    total_multi_pred = sum(s['multi_smiles_pred'] for s in all_stats)
    total_multi_true = sum(s['multi_smiles_true'] for s in all_stats)
    total_multi_both = sum(s['multi_smiles_both'] for s in all_stats)
    total_single = sum(s['single_smiles'] for s in all_stats)
    
    print(f"Total samples across all tasks: {total_samples}")
    print(f"  Single SMILES: {total_single} ({total_single/total_samples*100:.1f}%)")
    print(f"  Multiple SMILES in prediction: {total_multi_pred} ({total_multi_pred/total_samples*100:.1f}%)")
    print(f"  Multiple SMILES in ground truth: {total_multi_true} ({total_multi_true/total_samples*100:.1f}%)")
    print(f"  Multiple SMILES in both: {total_multi_both} ({total_multi_both/total_samples*100:.1f}%)")
    
    # 保存详细统计到JSON
    output_file = results_dir / "multiple_smiles_stats.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Detailed statistics saved to: {output_file}")


if __name__ == "__main__":
    main()
