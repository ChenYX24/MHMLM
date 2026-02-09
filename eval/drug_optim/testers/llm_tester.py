"""LLM 测试器 - 使用 chat 接口生成分子（支持批量推理）"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseTester

logger = logging.getLogger(__name__)


class LLMTester(BaseTester):
    """LLM 测试器，使用 Qwen chat 接口，支持批量推理"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        
        # 设备配置: "auto" / "cuda:0" / "0,1,2" 等
        self.device = config.get("device", "auto")
        
        # 批量推理配置
        self.batch_size = config.get("batch_size", 1)
        
        # 生成参数
        self.max_new_tokens = config.get("max_new_tokens", 256)
        self.temperature = config.get("temperature", 0.7)
        self.do_sample = config.get("do_sample", True)
        
        # Qwen3 思考模式：False 禁用思考链，直接输出答案
        self.enable_thinking = config.get("enable_thinking", False)
    
    def _parse_device_map(self) -> str | dict:
        """解析设备配置，返回 device_map 参数"""
        device = self.device
        
        if device in ("auto", "balanced", "sequential"):
            return device
        
        # 单卡: "cuda:0" 或 "0"
        if device.startswith("cuda:"):
            gpu_id = device.split(":")[1]
            if "," not in gpu_id:
                return {"": f"cuda:{gpu_id}"}
        
        if device.isdigit():
            return {"": f"cuda:{device}"}
        
        # 多卡: "0,1,2" -> 设置可见设备后用 auto
        if "," in device:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            logger.info(f"Set CUDA_VISIBLE_DEVICES={device}")
            return "auto"
        
        return "auto"
        
    def load_model(self) -> None:
        """加载 LLM 模型"""
        logger.info(f"Loading LLM from: {self.ckpt}")
        logger.info(f"Device config: {self.device}")
        
        device_map = self._parse_device_map()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.ckpt,
            trust_remote_code=True,
        )
        # 批量推理需要 padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # 生成任务使用左 padding
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.ckpt,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        
        logger.info(f"LLM loaded successfully (batch_size={self.batch_size}, enable_thinking={self.enable_thinking})")
    
    def load_data(self) -> List[Dict[str, Any]]:
        """加载 test_text2smi.jsonl 数据"""
        samples = []
        with Path(self.input_data).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                samples.append({
                    "prompt": obj["prompt"],
                    "gt_smiles": obj["ground_truth"],
                    "orig_smiles": obj["orig_smiles"],
                    "orig_cap": obj["orig_cap"],
                    "tgt_cap": obj["tgt_cap"],
                })
        return samples
    
    def predict(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """使用 LLM 生成 SMILES（单条），返回 {smiles, thinking}"""
        results = self.predict_batch([sample])
        return results[0]
    
    def predict_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """批量生成 SMILES，返回 [{smiles, thinking}, ...]"""
        if not samples:
            return []
        
        # 构建所有 prompt
        texts = []
        for sample in samples:
            messages = [{"role": "user", "content": sample["prompt"]}]
            # Qwen3: enable_thinking 控制是否启用思考链
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            texts.append(text)
        
        # 批量编码（带 padding）
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        
        # 批量生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 批量解码
        results = []
        for i, output in enumerate(outputs):
            # 跳过输入部分，只保留生成的 token
            generated_tokens = output[inputs["input_ids"].shape[1]:]
            generated = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            smiles, thinking = self._extract_smiles_and_thinking(generated)
            results.append({"smiles": smiles, "thinking": thinking})
        
        return results
    
    def _extract_smiles_and_thinking(self, text: str) -> tuple[str, str]:
        """
        从生成文本中提取 SMILES 和思考过程
        
        支持格式：
        1. <think>思考过程</think> + ```smiles\nSMILES\n```
        2. 纯 ```smiles\nSMILES\n```
        3. 纯文本（第一行）
        
        Returns:
            (smiles, thinking) 元组
        """
        thinking = ""
        content = text
        
        # 1. 提取 <think>...</think> 思考过程
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            # 移除思考部分，处理剩余内容
            content = text[think_match.end():].strip()
        elif text.startswith("<think>"):
            # 思考未完成（被截断），整个内容都是思考
            thinking = text[7:].strip()  # 跳过 "<think>"
            content = ""
        
        # 2. 从 content 提取 SMILES
        smiles = self._extract_smiles_from_content(content)
        
        return smiles, thinking
    
    def _extract_smiles_from_content(self, text: str) -> str:
        """从内容中提取 SMILES"""
        if not text:
            return ""
        
        # 尝试匹配 ```smiles ... ``` 代码块
        pattern = r"```smiles\s*\n?([^\n`]+)\n?```"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 尝试匹配任意代码块
        pattern = r"```\s*\n?([^\n`]+)\n?```"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        
        # 回退：返回第一行非空文本（跳过注释和空行）
        for line in text.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("```"):
                return line
        
        return ""
