#!/usr/bin/env python
"""
Drug Optimization 评估主入口

用法:
    python run_drug_optim_eval.py --config config/llm_cpt_sft.yaml
    python run_drug_optim_eval.py --config config/diffusion_base.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def setup_logging(log_file: Path) -> None:
    """配置日志"""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置格式
    fmt = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    
    # 同时输出到文件和控制台
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_path: Path) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_tester(config: Dict[str, Any]):
    """根据配置创建测试器"""
    model_type = config.get("model_type", "llm")
    
    if model_type == "llm":
        from testers import LLMTester
        return LLMTester(config)
    elif model_type == "diffusion":
        from testers import DiffusionTester
        return DiffusionTester(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Drug Optimization 评估")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="配置文件路径（YAML）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认: eval_output/<model_name>）",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="跳过测试阶段，直接对已有的 output.txt 评分",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="跳过评分阶段",
    )
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    model_name = config.get("model_name", "unknown")
    
    # 确定输出目录
    base_dir = Path(__file__).parent
    output_dir = args.output_dir or base_dir / "eval_output" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    log_file = output_dir / "test_log.log"
    setup_logging(log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Drug Optimization Evaluation")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Model: {model_name} ({config.get('model_type')})")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    
    # 保存运行配置
    run_info = {
        "config_path": str(args.config),
        "model_name": model_name,
        "model_type": config.get("model_type"),
        "output_dir": str(output_dir),
        "timestamp": datetime.now().isoformat(),
    }
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    
    output_file = output_dir / "output.txt"
    
    # ===== 测试阶段 =====
    if not args.skip_test:
        logger.info("[Phase 1] Testing...")
        tester = create_tester(config)
        summary = tester.run(output_dir)
        
        # 保存测试摘要
        test_summary = {
            "total": summary.total,
            "success": summary.success,
            "failed": summary.failed,
        }
        with (output_dir / "test_summary.json").open("w", encoding="utf-8") as f:
            json.dump(test_summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Test completed: {summary.success}/{summary.total} success")
        logger.info("")
    else:
        logger.info("[Phase 1] Testing skipped")
        if not output_file.exists():
            logger.error(f"output.txt not found: {output_file}")
            return 1
    
    # ===== 评分阶段 =====
    if not args.skip_score:
        logger.info("[Phase 2] Scoring...")
        
        from scoring import score_output
        
        # 重定向评分日志
        scoring_log = output_dir / "scoring_log.log"
        
        result = score_output(
            output_file=output_file,
            output_dir=output_dir,
        )
        
        if result:
            logger.info("Scoring result:")
            for key, value in result.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
        
        logger.info("")
    else:
        logger.info("[Phase 2] Scoring skipped")
    
    logger.info("=" * 60)
    logger.info("Evaluation completed!")
    logger.info(f"Results: {output_dir}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
