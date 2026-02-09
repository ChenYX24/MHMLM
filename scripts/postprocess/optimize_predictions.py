#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化预测结果的脚本 - 对 JSONL 文件中的 pred 字段进行优化
用法：
    python scripts/optimize_predictions.py \
        --prediction_dir /data1/chenyuxuan/MHMLM/1228results_baseline/LlaSMol-Mistral-7B-merged
"""

import argparse
import json
import re
import sys
from pathlib import Path

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_project_root))

# 任务类型定义（与 eval_smolinstruct.py 保持一致）
SMILES_TASKS = {
    "forward_synthesis",
    "retrosynthesis",
    "molecule_generation",
    "name_conversion-i2s",
}
TEXT_TASKS = {"molecule_captioning"}
FORMULA_ELEMENT_TASKS = {"name_conversion-i2f", "name_conversion-s2f"}
FORMULA_SPLIT_TASKS = {"name_conversion-s2i"}
NUMBER_TASKS = {"property_prediction-esol", "property_prediction-lipo"}
BOOLEAN_TASKS = {
    "property_prediction-bbbp",
    "property_prediction-clintox",
    "property_prediction-hiv",
    "property_prediction-sider",
}

# 正则表达式
SMILES_TOKEN_RE = re.compile(r"([A-Za-z0-9@+\-\[\]\(\)=#\\/%.]+)")
NUMBER_TOKEN_RE = re.compile(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?")
# 匹配包含小数点或负号的数字（更可能是实际数值，而不是SMILES中的数字）
# 匹配格式：-4.07, -4.883, 4.5, -20.86 等
NUMBER_WITH_DECIMAL_OR_NEGATIVE_RE = re.compile(r"[-+]?\d+\.\d+|[-+]\d+\.\d*[eE][-+]?\d+")


def _is_task_file(p: Path) -> bool:
    """判断是否为任务文件"""
    return p.is_file() and p.suffix == ".jsonl" and p.name != "eval_summary.json"


def _extract_based_on_gold_format(text: str, gold: str, task_name: str) -> str:
    """
    根据gold的格式从text中提取最相似的内容
    
    Args:
        text: 要提取的文本（通常是answer_only或raw_output）
        gold: gold标准答案，用于判断格式
        task_name: 任务名称
    
    Returns:
        提取出的预测结果
    """
    if not text or not gold:
        return ""
    
    text = str(text).strip()
    gold = str(gold).strip()
    
    # 1. 数值任务：gold是数字，提取数字（优先提取包含小数点或负号的数字）
    if task_name in NUMBER_TASKS:
        # 先移除SMILES标签，避免提取SMILES中的数字
        text_without_smiles = re.sub(r"<SMILES[^>]*>.*?</SMILES>", "", text, flags=re.IGNORECASE | re.DOTALL)
        
        # 优先在非SMILES部分查找包含小数点或负号的数字
        matches = NUMBER_WITH_DECIMAL_OR_NEGATIVE_RE.findall(text_without_smiles)
        if matches:
            return matches[-1]  # 返回最后一个（通常是答案）
        
        # 如果在非SMILES部分没找到，在整个文本中查找
        matches = NUMBER_WITH_DECIMAL_OR_NEGATIVE_RE.findall(text)
        if matches:
            return matches[-1]
        
        # 如果还没找到，尝试提取所有数字，选择看起来像数值的（包含小数点或负号）
        all_numbers = NUMBER_TOKEN_RE.findall(text_without_smiles)
        likely_numbers = [n for n in all_numbers if '.' in n or n.startswith('-') or n.startswith('+')]
        if likely_numbers:
            return likely_numbers[-1]
        
        # 最后尝试在整个文本中查找
        all_numbers = NUMBER_TOKEN_RE.findall(text)
        likely_numbers = [n for n in all_numbers if '.' in n or n.startswith('-') or n.startswith('+')]
        if likely_numbers:
            return likely_numbers[-1]
        
        return ""
    
    # 2. SMILES任务：gold是SMILES字符串，提取SMILES内容
    if task_name in SMILES_TASKS:
        # 先尝试提取<SMILES>标签内的内容（支持各种标签变体）
        smiles_pattern = re.compile(r"<SMILES[^>]*>(.*?)</SMILE[S]?>", re.IGNORECASE | re.DOTALL)
        matches = smiles_pattern.findall(text)
        if matches:
            # 返回最后一个匹配（通常是答案）
            content = matches[-1].strip()
            if content:
                return content
        
        # 如果没有标签，尝试提取看起来像SMILES的字符串
        m = SMILES_TOKEN_RE.search(text)
        if m:
            return m.group(1)
        return ""
    
    # 3. 布尔任务：gold是Yes/No，提取Yes/No
    if task_name in BOOLEAN_TASKS:
        # 先尝试提取<BOOLEAN>标签内的内容
        boolean_pattern = re.compile(r"<BOOLEAN[^>]*>(.*?)</BOOLEAN[T]?>", re.IGNORECASE | re.DOTALL)
        matches = boolean_pattern.findall(text)
        if matches:
            content = matches[-1].strip()
            if content:
                # 标准化为Yes/No
                content_lower = content.lower()
                if "yes" in content_lower or "true" in content_lower or "toxic" in content_lower:
                    return "Yes"
                elif "no" in content_lower or "false" in content_lower or "non-toxic" in content_lower:
                    return "No"
        
        # 提取Yes/No（不区分大小写）
        yes_no_pattern = re.compile(r"\b(Yes|No)\b", re.IGNORECASE)
        matches = yes_no_pattern.findall(text)
        if matches:
            # 返回最后一个匹配（通常是答案）
            result = matches[-1]
            # 标准化为Yes/No
            return "Yes" if result.lower() == "yes" else "No"
        return ""
    
    # 4. 文本任务：gold是文本，提取文本（排除SMILES标签）
    if task_name in TEXT_TASKS:
        # 移除SMILES标签及其内容
        text_cleaned = re.sub(r"<SMILES[^>]*>.*?</SMILES>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text_cleaned = text_cleaned.strip()
        if text_cleaned:
            return text_cleaned
        # 如果没有文本，返回原文本（去除标签）
        return re.sub(r"<[^>]+>", "", text).strip()
    
    # 5. 公式任务
    if task_name in FORMULA_ELEMENT_TASKS:
        # name_conversion-i2f, name_conversion-s2f: 从<MOLFORMULA>标签提取分子式
        # 先尝试提取<MOLFORMULA>标签内的内容（支持各种标签变体，包括有空格的情况）
        molformula_pattern = re.compile(r"<MOLFORMULA[^>]*>(.*?)</MOLFORMATULA[T]?\s*>", re.IGNORECASE | re.DOTALL)
        matches = molformula_pattern.findall(text)
        if matches:
            # 返回最后一个匹配（通常是答案）
            content = matches[-1].strip()
            if content:
                return content
        
        # 如果没有MOLFORMULA标签，返回空字符串（保留原始pred）
        return ""
    
    if task_name in FORMULA_SPLIT_TASKS:
        # name_conversion-s2i: 从<IUPAC>标签提取IUPAC名称
        # 先尝试提取<IUPAC>标签内的内容（支持各种标签变体）
        iupac_pattern = re.compile(r"<IUPAC[^>]*>(.*?)</IUPAC[C]?\s*>", re.IGNORECASE | re.DOTALL)
        matches = iupac_pattern.findall(text)
        if matches:
            # 返回最后一个匹配（通常是答案）
            content = matches[-1].strip()
            if content:
                return content
        
        # 如果没有IUPAC标签，返回空字符串（保留原始pred）
        return ""
    
    # 默认：返回清理后的文本
    return text.strip()


def optimize_predictions_in_file(file_path: Path, backup: bool = True) -> dict:
    """
    优化文件中的所有预测
    
    Args:
        file_path: JSONL 文件路径
        backup: 是否创建备份文件
    
    Returns:
        统计信息字典
    """
    stats = {
        "total": 0,
        "optimized": 0,
        "failed": 0,
        "unchanged": 0,
    }
    
    # 创建备份
    if backup:
        backup_path = file_path.with_suffix(file_path.suffix + ".backup")
        if backup_path.exists():
            print(f"[WARN] 备份文件已存在，跳过备份: {backup_path}")
        else:
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"[INFO] 已创建备份: {backup_path}")
    
    # 读取所有记录
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
                stats["total"] += 1
            except json.JSONDecodeError as e:
                print(f"[ERROR] 解析 JSON 失败 (行 {stats['total']+1}): {e}")
                stats["failed"] += 1
    
    # 默认任务名称（从文件名）
    default_task_name = file_path.stem
    
    print(f"[INFO] 处理文件: {file_path.name}, 默认任务: {default_task_name}, 记录数: {stats['total']}")
    
    # 优化每条记录的 pred 字段
    for i, record in enumerate(records):
        original_pred = record.get("pred", None)
        raw_output = record.get("raw_output", "")
        answer_only = record.get("answer_only", None)
        
        # 确定任务名称（优先使用记录中的 task 字段）
        task_name = record.get("task", default_task_name)
        gold = record.get("gold", "")
        
        # 尝试重新提取预测（根据gold的格式）
        try:
            # 优先使用 answer_only，其次使用 raw_output
            source_text = None
            if answer_only and isinstance(answer_only, str) and answer_only.strip():
                source_text = answer_only
            elif raw_output and isinstance(raw_output, str) and raw_output.strip():
                source_text = raw_output
            
            if source_text:
                optimized_pred = _extract_based_on_gold_format(source_text, gold, task_name)
            else:
                optimized_pred = original_pred
            
            # 如果提取成功，更新 pred 字段
            if optimized_pred and optimized_pred.strip():
                if optimized_pred != original_pred:
                    record["pred"] = optimized_pred
                    stats["optimized"] += 1
                else:
                    stats["unchanged"] += 1
            else:
                # 如果提取失败，保留原始 pred（如果存在）
                if not original_pred or not str(original_pred).strip():
                    stats["failed"] += 1
                else:
                    stats["unchanged"] += 1
        except Exception as e:
            print(f"[ERROR] 优化失败 (记录 {i+1}): {e}")
            stats["failed"] += 1
    
    # 写回文件
    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="优化预测结果")
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="包含预测结果 jsonl 文件的目录",
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="不创建备份文件",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="",
        help="只处理匹配该模式的文件（可选，如 '*.jsonl'）",
    )
    
    args = parser.parse_args()
    
    pred_dir = Path(args.prediction_dir).expanduser().resolve()
    if not pred_dir.exists():
        print(f"[ERROR] 目录不存在: {pred_dir}")
        sys.exit(1)
    
    if not pred_dir.is_dir():
        print(f"[ERROR] 不是目录: {pred_dir}")
        sys.exit(1)
    
    # 查找所有任务文件
    files = [p for p in pred_dir.iterdir() if _is_task_file(p)]
    
    if args.file_pattern:
        from fnmatch import fnmatch
        files = [f for f in files if fnmatch(f.name, args.file_pattern)]
    
    if not files:
        print(f"[WARN] 未在目录中发现任务文件：{pred_dir}")
        sys.exit(0)
    
    print(f"[INFO] 找到 {len(files)} 个文件需要处理")
    
    # 处理每个文件
    total_stats = {
        "total": 0,
        "optimized": 0,
        "failed": 0,
        "unchanged": 0,
    }
    
    for file_path in sorted(files):
        print(f"\n{'='*60}")
        stats = optimize_predictions_in_file(file_path, backup=not args.no_backup)
        
        # 汇总统计
        for key in total_stats:
            total_stats[key] += stats[key]
        
        print(f"[INFO] 完成: {file_path.name}")
        print(f"  - 总计: {stats['total']}")
        print(f"  - 已优化: {stats['optimized']}")
        print(f"  - 未变化: {stats['unchanged']}")
        print(f"  - 失败: {stats['failed']}")
    
    # 打印总统计
    print(f"\n{'='*60}")
    print(f"[INFO] 全部完成！")
    print(f"  - 处理文件数: {len(files)}")
    print(f"  - 总记录数: {total_stats['total']}")
    print(f"  - 已优化: {total_stats['optimized']}")
    print(f"  - 未变化: {total_stats['unchanged']}")
    print(f"  - 失败: {total_stats['failed']}")


if __name__ == "__main__":
    main()

