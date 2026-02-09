#!/usr/bin/env python3
"""
单独推理 Layer2 模型
用法: python scripts/layer2/infer_layer2.py --input data.jsonl --output predictions.jsonl
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.layer2_component import Layer2Inferer


def main():
    # 默认路径
    DEFAULT_CONFIG = "/data1/chenyuxuan/MHMLM/modules/layer2_component/layer2_config.yaml"
    DEFAULT_GVP_CKPT = "/data1/chenyuxuan/checkpoint/gvp_weights_best.pt"
    DEFAULT_INPUT = "/data1/chenyuxuan/Layer2/data/test.jsonl"
    DEFAULT_OUTPUT = "/data1/chenyuxuan/MHMLM/scripts/layer2/data/predictions.jsonl"
    
    parser = argparse.ArgumentParser(description="Layer2 推理")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help=f"输入文件（JSONL，每行包含 reactant_smiles），默认: {DEFAULT_INPUT}")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help=f"输出文件（JSONL），默认: {DEFAULT_OUTPUT}")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help=f"Layer2 配置文件路径，默认: {DEFAULT_CONFIG}")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备，默认: cuda:0")
    parser.add_argument("--gvp_ckpt", type=str, default=DEFAULT_GVP_CKPT, help=f"GVP checkpoint 路径，默认: {DEFAULT_GVP_CKPT}")
    
    args = parser.parse_args()
    
    # 初始化 Layer2Inferer
    print("📦 初始化 Layer2Inferer...")
    inferer = Layer2Inferer(
        config_path=args.config,
        device=args.device,
        gvp_ckpt_path=args.gvp_ckpt,
    )
    
    # 加载输入数据
    print(f"📂 加载输入: {args.input}")
    inputs = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                inputs.append(data)
            except json.JSONDecodeError:
                inputs.append({"reactant_smiles": line})
    
    print(f"   找到 {len(inputs)} 条数据")
    
    # 推理
    print("🔄 开始推理...")
    results = []
    for i, item in enumerate(inputs):
        reactant_smiles = item.get("reactant_smiles", "")
        if not reactant_smiles:
            continue
        
        try:
            # 预测
            output = inferer.predict(reactant_smiles=reactant_smiles)
            
            result = {
                "reactant_smiles": reactant_smiles,
                "yield_bin": int(output['yield_bin']),
                "yield_reg": float(output['yield_reg']),
                "embedding": output['embedding'].cpu().tolist(),
            }
            results.append(result)
            
            if (i + 1) % 100 == 0:
                print(f"   已处理 {i + 1}/{len(inputs)} 条")
                
        except Exception as e:
            print(f"❌ 处理第 {i} 条数据时出错: {e}")
            continue
    
    # 保存结果
    print(f"💾 保存结果到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✅ 完成！共处理 {len(results)} 条数据")


if __name__ == "__main__":
    main()
