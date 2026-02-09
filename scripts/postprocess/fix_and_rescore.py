#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复预测提取并重新评分
用法：
    python scripts/fix_and_rescore.py \
        --prediction_dir /data1/chenyuxuan/MHMLM/1125results_baseline/LlaSMol-Mistral-7B-merged_fewshot \
        --backup
"""

import argparse
import json
import sys
import re
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_project_root))

from eval.extract_prediction import extract_prediction_from_raw

# 任务类型定义（与eval_smolinstruct.py保持一致）
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
FORMULA_TOKEN_RE = re.compile(r"([A-Za-z0-9\(\)\.\+\-]+)")
NUMBER_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
BOOL_TOKEN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
SMILES_TAG_RE = re.compile(r"<SMILES>\s*([A-Za-z0-9@+\-\[\]\(\)=#\\/%.]+)\s*</SMILES>", re.IGNORECASE)


def _canonical_bool(text: str) -> str:
    """将文本规范化为 Yes/No"""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    text = text.rstrip('.,;:!?')
    
    yes_values = {"yes", "y", "true", "t", "1", "positive", "toxic", "unsafe", "harmful"}
    no_values = {"no", "n", "false", "f", "0", "negative", "non-toxic", "non toxic", "nontoxic", "safe", "non-harmful"}
    
    if text in yes_values:
        return "Yes"
    elif text in no_values:
        return "No"
    
    m = BOOL_TOKEN_RE.search(text)
    if m:
        v = m.group(1).lower()
        return "Yes" if v == "yes" else "No"
    
    if "toxic" in text and "non" not in text and "not" not in text:
        return "Yes"
    elif "non-toxic" in text or "nontoxic" in text or ("non" in text and "toxic" in text):
        return "No"
    
    return ""


def _extract_core_answer(text: str, task_name: str) -> str:
    """从文本中提取核心答案"""
    if not text or not isinstance(text, str):
        return ""
    text = str(text).strip()
    
    if task_name in TEXT_TASKS:
        return text
    
    if task_name in SMILES_TASKS:
        # 先尝试从<SMILES>标签中提取
        m = SMILES_TAG_RE.search(text)
        if m:
            return m.group(1).strip()
        
        # 如果没有完整的标签，尝试从</SMILES>之前提取
        if "</SMILES>" in text:
            before_close = text.split("</SMILES>")[0]
            # 从后往前找最长的SMILES字符串
            matches = list(SMILES_TOKEN_RE.finditer(before_close))
            if matches:
                # 取最后一个匹配（通常是最完整的SMILES）
                return matches[-1].group(1).strip()
        
        # 然后尝试从文本中提取SMILES（找最长的匹配）
        all_matches = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            matches = list(SMILES_TOKEN_RE.finditer(line))
            if matches:
                all_matches.extend(matches)
        
        if all_matches:
            # 找到最长的SMILES字符串（通常是最完整的）
            longest_match = max(all_matches, key=lambda m: len(m.group(1)))
            return longest_match.group(1).strip()
        
        return text
    
    if task_name in FORMULA_ELEMENT_TASKS or task_name in FORMULA_SPLIT_TASKS:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = FORMULA_TOKEN_RE.search(line)
            if m:
                return m.group(1)
        return text
    
    if task_name in NUMBER_TASKS:
        m = NUMBER_TOKEN_RE.search(text)
        if m:
            return m.group(0)
        return text
    
    if task_name in BOOLEAN_TASKS:
        return _canonical_bool(text)
    
    # 默认：返回第一行非空内容
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def extract_prediction_improved(obj: dict, task_name: str) -> Optional[str]:
    """
    改进的预测提取函数
    优先级：pred > answer_only > raw_output
    """
    # 1. 如果pred字段存在且有效，检查是否需要重新提取
    pred = None
    need_re_extract = False
    
    if pred is not None and isinstance(pred, str) and pred.strip():
        pred_stripped = pred.strip()
        # 对于布尔任务，如果pred不完整（如只有"The"），需要重新提取
        if task_name in BOOLEAN_TASKS:
            bool_result = _canonical_bool(pred_stripped)
            if not bool_result or len(pred_stripped) < 2:
                need_re_extract = True
            else:
                return bool_result
        # 对于SMILES任务，如果pred很短且不像是SMILES，需要重新提取
        elif task_name in SMILES_TASKS:
            if len(pred_stripped) < 5 or not SMILES_TOKEN_RE.search(pred_stripped):
                need_re_extract = True
            else:
                return pred_stripped
        else:
            # 其他任务，如果pred看起来不完整（比如只有单个词且很短），可能需要重新提取
            if len(pred_stripped) < 2:
                need_re_extract = True
            else:
                return pred_stripped
    else:
        # pred为空或None，需要提取
        need_re_extract = True
    
    if not need_re_extract:
        return None
    
    # 2. 优先从answer_only提取
    answer_only = obj.get("answer_only", None)
    if answer_only and isinstance(answer_only, str) and answer_only.strip():
        # 对于布尔任务，如果answer_only包含SMILES标签，这是错误的输出，应该忽略
        if task_name in BOOLEAN_TASKS and "<SMILES>" in answer_only:
            # 跳过，尝试从raw_output提取
            pass
        else:
            extracted = _extract_core_answer(answer_only.strip(), task_name)
            if extracted and extracted.strip():
                return extracted
    
    # 3. 从raw_output提取
    raw_output = obj.get("raw_output", None)
    if raw_output and isinstance(raw_output, str) and raw_output.strip():
        try:
            extracted = extract_prediction_from_raw(
                raw_output, task_name,
                text_tasks=TEXT_TASKS,
                smiles_tasks=SMILES_TASKS,
                formula_element_tasks=FORMULA_ELEMENT_TASKS,
                formula_split_tasks=FORMULA_SPLIT_TASKS,
                number_tasks=NUMBER_TASKS,
                boolean_tasks=BOOLEAN_TASKS,
                answer_only=answer_only,  # 传递answer_only作为备选
            )
            if extracted and extracted.strip():
                # 对于布尔任务，再次验证提取结果
                if task_name in BOOLEAN_TASKS:
                    bool_result = _canonical_bool(extracted.strip())
                    if bool_result:
                        return bool_result
                    # 如果提取的结果不是有效的布尔值，返回None
                    return None
                return extracted.strip()
        except Exception as e:
            # 如果提取失败，尝试直接从answer_only提取（如果之前没试过）
            if answer_only and task_name not in BOOLEAN_TASKS:
                extracted = _extract_core_answer(answer_only.strip(), task_name)
                if extracted and extracted.strip():
                    return extracted
    
    return None


def fix_jsonl_file(jsonl_path: Path, backup: bool = False) -> tuple[int, int]:
    """
    修复单个jsonl文件
    返回: (修复数量, 总数量)
    """
    if not jsonl_path.exists():
        print(f"[WARN] 文件不存在: {jsonl_path}")
        return 0, 0
    
    # 备份
    if backup:
        backup_path = jsonl_path.with_suffix(jsonl_path.suffix + ".backup")
        if not backup_path.exists():
            import shutil
            shutil.copy2(jsonl_path, backup_path)
            print(f"[INFO] 已备份到: {backup_path}")
    
    task_name = jsonl_path.stem
    fixed_count = 0
    total_count = 0
    fixed_entries = []
    
    # 读取并修复
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
                total_count += 1
                
                # 获取任务名
                task_from_obj = obj.get("task", task_name)
                
                # 提取预测
                new_pred = extract_prediction_improved(obj, task_from_obj)
                
                # 检查是否需要更新
                old_pred = obj.get("pred", "")
                old_pred_str = old_pred if old_pred is not None else ""
                
                if new_pred is not None and new_pred.strip():
                    # 有新提取的结果
                    new_pred_str = new_pred.strip()
                    if old_pred_str != new_pred_str:
                        obj["pred"] = new_pred_str
                        fixed_count += 1
                elif old_pred_str and old_pred_str.strip():
                    # 如果新提取失败但旧pred存在，保留旧pred
                    # 但需要检查旧pred是否有效
                    if task_from_obj in BOOLEAN_TASKS:
                        # 对于布尔任务，验证旧pred是否有效
                        bool_result = _canonical_bool(old_pred_str.strip())
                        if not bool_result:
                            # 旧pred无效，清空
                            obj["pred"] = ""
                            fixed_count += 1
                else:
                    # 如果新旧都为空，确保pred是空字符串
                    if old_pred_str != "":
                        obj["pred"] = ""
                        fixed_count += 1
                
                fixed_entries.append(obj)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON解析失败 (line {total_count+1}): {e}")
                continue
            except Exception as e:
                print(f"[ERROR] 处理失败 (line {total_count+1}): {e}")
                continue
    
    # 写回文件
    if fixed_entries:
        with jsonl_path.open("w", encoding="utf-8") as f:
            for obj in fixed_entries:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    return fixed_count, total_count


def main():
    parser = argparse.ArgumentParser(description="修复预测提取并重新评分")
    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="包含预测结果 jsonl 文件的目录",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="修复前备份原始文件",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="修复后重新评分",
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
        help="评分时使用的进程数",
    )
    
    args = parser.parse_args()
    
    pred_dir = Path(args.prediction_dir).expanduser().resolve()
    if not pred_dir.exists():
        print(f"错误：目录不存在: {pred_dir}")
        sys.exit(1)
    
    if not pred_dir.is_dir():
        print(f"错误：不是目录: {pred_dir}")
        sys.exit(1)
    
    # 查找所有jsonl文件
    jsonl_files = list(pred_dir.glob("*.jsonl"))
    jsonl_files = [f for f in jsonl_files if not f.name.startswith("_")]
    
    if not jsonl_files:
        print(f"[WARN] 未找到jsonl文件: {pred_dir}")
        sys.exit(1)
    
    print(f"[INFO] 找到 {len(jsonl_files)} 个jsonl文件")
    
    # 修复每个文件
    total_fixed = 0
    total_samples = 0
    for jsonl_file in sorted(jsonl_files):
        print(f"\n[INFO] 处理: {jsonl_file.name}")
        fixed, total = fix_jsonl_file(jsonl_file, backup=args.backup)
        total_fixed += fixed
        total_samples += total
        print(f"  - 修复: {fixed}/{total} 条记录")
    
    print(f"\n[INFO] 修复完成: 共修复 {total_fixed}/{total_samples} 条记录")
    
    # 重新评分
    if args.rescore:
        print("\n[INFO] 开始重新评分...")
        from eval.eval_smolinstruct import run_scoring
        
        save_json = args.save_json if args.save_json else str(pred_dir / "metrics.json")
        run_scoring(
            pred_dir,
            save_json=save_json,
            score_workers=args.score_workers,
        )
        print(f"\n[INFO] 评分完成，结果已保存到: {save_json}")


if __name__ == "__main__":
    main()

