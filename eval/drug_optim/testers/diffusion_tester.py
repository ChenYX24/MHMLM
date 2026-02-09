"""Diffusion 测试器 - 使用 DDS 算法优化分子"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseTester

logger = logging.getLogger(__name__)


class DiffusionTester(BaseTester):
    """Diffusion 测试器，使用 LDMolInferer DDS 接口"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.inferer = None
        self.ldmol_config = config.get("ldmol_config")
        
        # 设备配置: "cuda:0" / "0" / "auto"（默认 cuda:0）
        device_cfg = config.get("device", "cuda:0")
        if device_cfg.isdigit():
            self.device = f"cuda:{device_cfg}"
        elif device_cfg == "auto":
            self.device = "cuda"
        else:
            self.device = device_cfg
        
    def load_model(self) -> None:
        """加载 Diffusion 模型"""
        # 添加项目路径以导入 LDMolInferer
        project_root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(project_root))
        
        from modules.ldmol_component.LDMolInferer import LDMolInferer
        
        logger.info(f"Loading Diffusion model from: {self.ckpt}")
        logger.info(f"Device: {self.device}")
        
        config_path = self.ldmol_config
        
        self.inferer = LDMolInferer(
            config_path=config_path,
            device=self.device,
        )
        
        # 如果有 checkpoint 路径，加载权重
        if self.ckpt:
            import torch
            state_dict = torch.load(self.ckpt, map_location=self.device)
            # 根据 checkpoint 结构加载
            if "model" in state_dict:
                self.inferer.model.load_state_dict(state_dict["model"], strict=False)
            else:
                self.inferer.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded checkpoint: {self.ckpt}")
        
        logger.info("Diffusion model loaded successfully")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """加载 test_dds.txt 数据（4 列格式）"""
        samples = []
        with Path(self.input_data).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 4:
                    continue
                orig_smiles, orig_cap, gt_smiles, tgt_cap = parts
                samples.append({
                    "orig_smiles": orig_smiles,
                    "orig_cap": orig_cap,
                    "gt_smiles": gt_smiles,
                    "tgt_cap": tgt_cap,
                })
        return samples
    
    def predict(self, sample: Dict[str, Any]) -> str:
        """使用 DDS 算法优化分子"""
        pred_smiles = self.inferer.generate_smi_dds(
            input_smiles=sample["orig_smiles"],
            source_text=sample["orig_cap"],
            target_text=sample["tgt_cap"],
        )
        return pred_smiles
