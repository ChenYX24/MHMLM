"""测试器基类"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """单条测试结果"""
    row_id: int
    original_smiles: str
    source_caption: str
    target_caption: str
    gt_smiles: str
    pred_smiles: str
    thinking: str = ""  # 思考过程（可选）
    
    def to_tsv_row(self, include_thinking: bool = False) -> str:
        """转换为 TSV 行"""
        # 清理字段中的换行符和制表符
        def clean(s: str) -> str:
            return s.replace("\n", " ").replace("\t", " ").replace("\r", "")
        
        cols = [
            str(self.row_id),
            self.original_smiles,
            clean(self.source_caption),
            clean(self.target_caption),
            self.gt_smiles,
            self.pred_smiles,
        ]
        if include_thinking:
            cols.append(clean(self.thinking))
        return "\t".join(cols)
    
    @staticmethod
    def tsv_header(include_thinking: bool = False) -> str:
        """TSV 表头"""
        cols = ["row_id", "original_smiles", "source_caption", "target_caption", "gt_smiles", "pred_smiles"]
        if include_thinking:
            cols.append("thinking")
        return "\t".join(cols)


@dataclass
class TestSummary:
    """测试汇总"""
    total: int = 0
    success: int = 0
    failed: int = 0
    results: List[TestResult] = field(default_factory=list)


class BaseTester(ABC):
    """测试器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "unknown")
        self.ckpt = config.get("ckpt")
        self.input_data = config.get("input_data")
        self.batch_size = config.get("batch_size", 1)
        self.include_thinking = config.get("include_thinking")
        
    @abstractmethod
    def load_model(self) -> None:
        """加载模型"""
        pass
    
    @abstractmethod
    def predict(self, sample: Dict[str, Any]) -> str | Dict[str, str]:
        """单条预测，返回 SMILES 或 {smiles, thinking} 字典"""
        pass
    
    def predict_batch(self, samples: List[Dict[str, Any]]) -> List[str | Dict[str, str]]:
        """批量预测（默认逐条调用 predict，子类可覆盖）"""
        return [self.predict(s) for s in samples]
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """加载测试数据"""
        pass
    
    def _parse_prediction(self, pred) -> tuple[str, str]:
        """解析预测结果，返回 (smiles, thinking)"""
        if isinstance(pred, dict):
            return pred.get("smiles", ""), pred.get("thinking", "")
        return pred, ""
    
    def run(self, output_dir: Path) -> TestSummary:
        """运行测试（支持批量推理和思考过程输出）"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "output.txt"
        
        logger.info(f"Loading model: {self.model_name}")
        self.load_model()
        
        logger.info(f"Loading data: {self.input_data}")
        samples = self.load_data()
        
        summary = TestSummary(total=len(samples))
        batch_size = self.batch_size
        include_thinking = self.include_thinking
        
        logger.info(f"Running inference on {len(samples)} samples (batch_size={batch_size}, include_thinking={include_thinking})...")
        
        with output_file.open("w", encoding="utf-8") as f:
            # 写入表头
            f.write(TestResult.tsv_header(include_thinking=include_thinking) + "\n")
            
            # 分批处理
            for batch_start in range(0, len(samples), batch_size):
                batch_end = min(batch_start + batch_size, len(samples))
                batch_samples = samples[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))
                
                try:
                    # 批量预测
                    predictions = self.predict_batch(batch_samples)
                    
                    # 处理结果
                    for idx, sample, pred in zip(batch_indices, batch_samples, predictions):
                        pred_smiles, thinking = self._parse_prediction(pred)
                        result = TestResult(
                            row_id=idx,
                            original_smiles=sample["orig_smiles"],
                            source_caption=sample["orig_cap"],
                            target_caption=sample["tgt_cap"],
                            gt_smiles=sample["gt_smiles"],
                            pred_smiles=pred_smiles,
                            thinking=thinking,
                        )
                        summary.results.append(result)
                        summary.success += 1
                        f.write(result.to_tsv_row(include_thinking=include_thinking) + "\n")
                        
                except Exception as e:
                    logger.warning(f"Batch {batch_start}-{batch_end} failed: {e}")
                    summary.failed += len(batch_samples)
                
                # 进度日志
                if batch_end % max(10, batch_size) == 0 or batch_end == len(samples):
                    logger.info(f"Progress: {batch_end}/{len(samples)}")
                    f.flush()  # 及时写入磁盘
        
        logger.info(f"Test completed: {summary.success}/{summary.total} success")
        return summary
