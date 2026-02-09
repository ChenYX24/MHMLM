"""ADMET 评分模块 - 封装 admet_reasoning_richness.py 的评分逻辑"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def score_output(
    output_file: Path,
    output_dir: Path,
    w_main: float = 1.0,
    w_bonus: float = 1.0,
) -> Dict[str, Any]:
    """
    对 output.txt 进行 ADMET 评分
    
    Args:
        output_file: output.txt 路径（包含 pred_smiles）
        output_dir: 输出目录
        w_main: 主奖励权重
        w_bonus: reasoning 奖励权重
        
    Returns:
        评分结果字典
    """
    # 添加当前目录到路径（admet_reasoning_richness.py 在同目录）
    scoring_dir = Path(__file__).parent
    sys.path.insert(0, str(scoring_dir))
    
    from admet_reasoning_richness import main_from_extracted
    
    # 解析 output.txt，生成评分所需的中间文件
    orig_jsonl_path = output_dir / "tmp_orig.jsonl"
    pred_smiles_path = output_dir / "tmp_pred_smiles.txt"
    summary_path = output_dir / "scoring_summary.json"
    
    logger.info(f"Parsing output file: {output_file}")
    
    with (
        output_file.open("r", encoding="utf-8") as fin,
        orig_jsonl_path.open("w", encoding="utf-8") as f_orig,
        pred_smiles_path.open("w", encoding="utf-8") as f_pred,
    ):
        # 跳过表头
        header = fin.readline()
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            
            row_id, orig_smiles, source_cap, target_cap, gt_smiles, pred_smiles = parts[:6]
            
            # 写入 orig_jsonl（评分脚本需要的格式）
            orig_record = {
                "input": f"Original SMILES: {orig_smiles}\n\nADMET Profile:\n{source_cap}"
            }
            f_orig.write(json.dumps(orig_record, ensure_ascii=False) + "\n")
            
            # 写入 pred_smiles
            f_pred.write(pred_smiles + "\n")
    
    logger.info("Running ADMET scoring...")
    
    # 调用评分函数
    main_from_extracted(
        orig_jsonl=str(orig_jsonl_path),
        after_smi_path=str(pred_smiles_path),
        out_path=str(summary_path),
        w_main=w_main,
        w_bonus=w_bonus,
    )
    
    # 清理临时文件
    orig_jsonl_path.unlink(missing_ok=True)
    pred_smiles_path.unlink(missing_ok=True)
    
    # 读取并返回结果
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            # 读取最后一行（averages）
            lines = f.readlines()
            if lines:
                result = json.loads(lines[-1])
                logger.info(f"Scoring completed: {summary_path}")
                return result
    
    logger.warning("No scoring result generated")
    return {}
