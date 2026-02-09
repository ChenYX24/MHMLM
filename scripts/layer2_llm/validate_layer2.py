#!/usr/bin/env python3
"""
Layer2 模型验证脚本
使用 Layer2 测试集验证模型输出是否正常
"""

import sys
import json
import math
from pathlib import Path
from typing import List, Dict, Any
import torch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.layer2_component.Layer2Inferer import Layer2Inferer
from modules.layer2_component.gvp_embedder import build_gvp_encoder

def load_test_samples(test_file: str, max_samples: int = 10) -> List[Dict[str, Any]]:
    """加载测试样本"""
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            if line.strip():
                samples.append(json.loads(line))
    return samples

def validate_layer2(
    layer2_inferer: Layer2Inferer,
    test_samples: List[Dict[str, Any]],
    device: str = "cuda:0"
):
    """验证 Layer2 模型"""
    print(f"\n{'='*80}")
    print(f"Layer2 模型验证")
    print(f"{'='*80}\n")
    print(f"测试样本数: {len(test_samples)}\n")
    
    yield_bins = []
    yield_regs = []
    bin_logits_stats = []
    
    for i, sample in enumerate(test_samples):
        print(f"\n样本 {i+1}/{len(test_samples)}:")
        print(f"  反应: {sample.get('rxn', 'N/A')[:100]}...")
        
        # 提取反应物 SMILES（从 tokens 中提取）
        tokens = sample.get("tokens", [])
        
        # 从 tokens 中提取反应物信息
        reactant_smiles_list = []
        reactant_embeddings = []
        amount_info_list = []
        
        for token in tokens:
            role = token.get("reaction_role", token.get("role", ""))
            if role == "REACTANT":
                # 尝试从 token 中获取 SMILES（如果有）
                smiles = token.get("smiles") or token.get("reactant_smiles")
                emb = token.get("emb")
                
                if emb is not None:
                    reactant_embeddings.append(torch.tensor(emb, device=device, dtype=torch.float32))
                    if smiles:
                        reactant_smiles_list.append(smiles)
                    else:
                        reactant_smiles_list.append("C")  # 占位符
                    
                    # 提取 amount 信息（从 log 值反推，但这里我们只用于验证，可以简化）
                    amt_moles_log = token.get("amt_moles_log")
                    amt_mass_log = token.get("amt_mass_log")
                    amt_volume_log = token.get("amt_volume_log")
                    
                    # 如果有 log 值，需要反 log1p（但这里我们只用于验证，可以简化）
                    # 注意：amount_info 应该使用原始值，不是 log 值
                    amount_info_list.append({
                        "moles": math.expm1(amt_moles_log) if amt_moles_log is not None else 1.0,
                        "mass": math.expm1(amt_mass_log) if amt_mass_log is not None else 0.0,
                        "volume": math.expm1(amt_volume_log) if amt_volume_log is not None else 0.0,
                    })
        
        if not reactant_embeddings:
            print("  ⚠️  未找到反应物，跳过")
            continue
        
        # 使用 Layer2 预测
        try:
            # 如果有多个反应物，使用列表；否则使用单个
            if len(reactant_embeddings) == 1:
                gvp_embedding = reactant_embeddings[0]
                reactant_smiles = reactant_smiles_list[0] if reactant_smiles_list else "C"
                amount_info = amount_info_list[0] if amount_info_list else None
            else:
                gvp_embedding = reactant_embeddings
                reactant_smiles = reactant_smiles_list if reactant_smiles_list else ["C"] * len(reactant_embeddings)
                amount_info = amount_info_list if amount_info_list else None
            
            result = layer2_inferer.predict(
                reactant_smiles=reactant_smiles,
                gvp_embedding=gvp_embedding,
                amount_info=amount_info,
            )
            
            yield_bin = result['yield_bin']
            yield_reg = result['yield_reg']
            yield_bins.append(yield_bin)
            yield_regs.append(yield_reg)
            
            print(f"  yield_bin: {yield_bin} (区间: {yield_bin*10}%-{(yield_bin+1)*10}%)")
            print(f"  yield_reg: {yield_reg:.3f} ({yield_reg*100:.1f}%)")
            
            # 如果有 logits 信息，显示
            if 'logits' in result:
                logits = result['logits']
                probs = result.get('probs', torch.softmax(torch.tensor(logits), dim=0).tolist())
                logits_std = result.get('logits_std', 0.0)
                logits_range = result.get('logits_range', 0.0)
                print(f"  bin_logits: {[f'{x:.2f}' for x in logits]}")
                print(f"  bin_probs: {[f'{x:.3f}' for x in probs]}")
                print(f"  logits_std: {logits_std:.4f}, logits_range: {logits_range:.4f}")
                if logits_std < 0.1 or logits_range < 0.5:
                    print(f"  ⚠️  警告: logits 变化很小，可能模型有问题")
            
            # 如果有真实值，比较
            if "yield_bin" in sample:
                true_bin = sample["yield_bin"]
                true_reg = sample.get("yield_reg", 0.0)
                bin_correct = "✅" if yield_bin == true_bin else "❌"
                reg_diff = abs(yield_reg - true_reg)
                print(f"  真实值: bin={true_bin}, reg={true_reg:.3f}")
                print(f"  预测: bin={yield_bin} {bin_correct}, reg={yield_reg:.3f} (误差: {reg_diff:.3f})")
        
        except Exception as e:
            print(f"  ❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 统计信息
    if yield_bins:
        print(f"\n{'='*80}")
        print(f"统计信息:")
        print(f"{'='*80}")
        print(f"yield_bin 分布: {dict(zip(*torch.unique(torch.tensor(yield_bins), return_counts=True)))}")
        print(f"yield_bin 范围: {min(yield_bins)} - {max(yield_bins)}")
        print(f"yield_reg 范围: {min(yield_regs):.3f} - {max(yield_regs):.3f}")
        print(f"yield_reg 均值: {sum(yield_regs)/len(yield_regs):.3f}")
        print(f"yield_reg 标准差: {torch.std(torch.tensor(yield_regs)).item():.3f}")
        
        # 检查 yield_bin 是否总是相同
        if len(set(yield_bins)) == 1:
            print(f"\n⚠️  警告: 所有样本的 yield_bin 都相同 ({yield_bins[0]})，可能模型有问题")
        else:
            print(f"\n✅ yield_bin 有变化，范围: {min(yield_bins)} - {max(yield_bins)}")
        
        # 检查 yield_reg 的变化范围
        reg_range = max(yield_regs) - min(yield_regs)
        if reg_range < 0.1:
            print(f"\n⚠️  警告: yield_reg 变化范围很小 ({reg_range:.3f})，可能模型输出不够敏感")
        else:
            print(f"\n✅ yield_reg 有合理变化，范围: {reg_range:.3f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="验证 Layer2 模型")
    parser.add_argument("--test_file", type=str, 
                        default="/data1/chenyuxuan/Layer2/data/pretrain/dev.jsonl",
                        help="测试文件路径")
    parser.add_argument("--config", type=str,
                        default=str(project_root / "modules" / "layer2_component" / "layer2_config.yaml"),
                        help="Layer2 配置文件路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--max_samples", type=int, default=20, help="最大测试样本数")
    
    args = parser.parse_args()
    
    # 加载测试数据
    test_file = Path(args.test_file)
    if not test_file.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        print(f"   请提供有效的测试文件路径")
        return
    
    print(f"📂 加载测试数据: {test_file}")
    test_samples = load_test_samples(str(test_file), args.max_samples)
    print(f"   加载了 {len(test_samples)} 个样本\n")
    
    # 初始化 Layer2
    print(f"📦 初始化 Layer2...")
    layer2_inferer = Layer2Inferer(
        config_path=args.config,
        device=args.device,
    )
    print(f"✅ Layer2 初始化完成\n")
    
    # 验证
    validate_layer2(layer2_inferer, test_samples, args.device)

if __name__ == "__main__":
    main()
