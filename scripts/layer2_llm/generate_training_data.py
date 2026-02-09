#!/usr/bin/env python3
"""
第一阶段：生成 LLM 训练数据
Pipeline: query -> LLM -> Layer2 -> 生成训练数据

生成的数据格式：
{
    "input": "原始 query",
    "intermediate": "第一轮 LLM 输出（包含反应物 SMILES）",
    "layer2_info": {
        "yield_bin": 5,
        "yield_reg": 0.75,
        "embedding": [...]
    },
    "output": "最终 LLM 输出（使用 layer2 信息后）"
}
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sft_tester import MolAwareGenerator2


def load_queries(input_path: str) -> List[Dict[str, Any]]:
    """加载查询数据"""
    queries = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                queries.append(data)
            except json.JSONDecodeError:
                # 如果不是 JSON，当作纯文本
                queries.append({"input": line})
    return queries


def load_chembench_data(task: str = "product", split: str = "train") -> List[Dict[str, Any]]:
    """从 ChemBench 加载数据"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Please install datasets: pip install datasets")
    
    REPO_ID = "AI4Chem/ChemBench4K"
    BENCH_FILES = {
        "product": {
            "dev": "dev/Product_Prediction_benchmark.json",
            "test": "test/Product_Prediction_benchmark.json",
        },
        "retro": {
            "dev": "dev/Retrosynthesis_benchmark.json",
            "test": "test/Retrosynthesis_benchmark.json",
        },
        "yield": {
            "dev": "dev/Yield_Prediction_benchmark.json",
            "test": "test/Yield_Prediction_benchmark.json",
        },
    }
    
    if task not in BENCH_FILES:
        raise ValueError(f"Unsupported task: {task}")
    
    # ChemBench 没有 train split，使用 dev 作为训练数据
    if split == "train":
        print("[INFO] ChemBench 没有 train split，使用 dev 作为训练数据")
        split = "dev"
    
    if split not in BENCH_FILES[task]:
        raise ValueError(f"Unsupported split: {split}")
    
    relpath = BENCH_FILES[task][split]
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{relpath}"
    
    print(f"[INFO] Loading ChemBench data: {task}/{split}")
    print(f"[INFO] URL: {url}")
    
    ds = load_dataset("json", data_files={split: url}, split=split)
    
    # 转换为查询格式
    queries = []
    for sample in ds:
        question = sample.get("question", "")
        queries.append({"input": question})
    
    print(f"[INFO] Loaded {len(queries)} samples from ChemBench")
    return queries


def save_results(results: List[Dict[str, Any]], output_path: str):
    """保存结果到 JSONL"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def generate_training_data(
    generator: MolAwareGenerator2,
    queries: List[Dict[str, Any]],
    output_path: str,
    task_type: Optional[str] = None,
):
    """
    生成训练数据
    
    Args:
        generator: MolAwareGenerator2 实例
        queries: 查询列表
        output_path: 输出文件路径
        task_type: 任务类型（如 "reaction_prediction"）
    """
    results = []
    
    for i, query_data in enumerate(tqdm(queries, desc="生成训练数据")):
        query = query_data.get("input", query_data.get("query", ""))
        if not query:
            continue
        
        try:
            # 使用 generate_with_layer2 生成完整 pipeline 的结果，并获取中间结果
            result = generator.generate_with_layer2(
                prompt=query,
                add_dialog_wrapper=True,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                task_type=task_type,
                return_intermediate=True,  # 返回中间结果
            )
            
            # result 现在是一个字典，包含 first_response, layer2_info, final_response
            if isinstance(result, dict):
                first_response = result.get("first_response", "")
                layer2_info = result.get("layer2_info", {})
                final_response = result.get("final_response", "")
                
                # 解析第一轮输出中的 JSON（包含分子信息和角色）
                molecules_info = None
                try:
                    import json
                    import re
                    # 尝试从 first_response 中提取 JSON
                    json_match = re.search(r'\{.*\}', first_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        parsed = json.loads(json_str)
                        if "molecules" in parsed:
                            molecules_info = parsed["molecules"]
                except:
                    pass
                
                # 构建训练数据
                training_item = {
                    "input": query,
                    "intermediate": first_response,  # 第一轮 JSON 输出
                    "molecules_info": molecules_info,  # 解析后的分子信息（包含角色）
                    "layer2_info": {
                        "yield_bin": layer2_info.get("yield_bin") if layer2_info else None,
                        "yield_reg": layer2_info.get("yield_reg") if layer2_info else None,
                        # embedding 是 tensor，只保存形状信息（实际 embedding 在训练时动态生成）
                        "embedding_shape": list(layer2_info.get("embedding", torch.tensor([])).shape) if layer2_info and layer2_info.get("embedding") is not None else None,
                    },
                    "output": final_response,  # 最终 LLM 输出
                }
            else:
                # 兼容旧接口：如果返回的是字符串
                training_item = {
                    "input": query,
                    "intermediate": "",
                    "layer2_info": {},
                    "output": result,
                }
            
            results.append(training_item)
            
        except Exception as e:
            print(f"❌ 处理查询 {i} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    save_results(results, output_path)
    print(f"✅ 已保存 {len(results)} 条训练数据到 {output_path}")


def main():
    # 默认路径
    DEFAULT_INPUT = None  # 默认使用 ChemBench
    DEFAULT_OUTPUT = "/data1/chenyuxuan/MHMLM/scripts/layer2_llm/data/training_data.jsonl"
    DEFAULT_CONFIG = "/data1/chenyuxuan/MHMLM/configs/qwen3_sft_epoch2_2.yaml"
    
    parser = argparse.ArgumentParser(description="生成 Layer2-LLM 联合训练数据")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, 
                       help="输入查询文件（JSONL），如果不指定则使用 ChemBench 数据")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, 
                       help=f"输出训练数据文件（JSONL），默认: {DEFAULT_OUTPUT}")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, 
                       help=f"模型配置文件路径，默认: {DEFAULT_CONFIG}")
    parser.add_argument("--task_type", type=str, default="reaction_prediction", help="任务类型，默认: reaction_prediction")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备，默认: cuda:0")
    
    # ChemBench 相关参数
    parser.add_argument("--use_chembench", action="store_true", help="使用 ChemBench 数据（如果不指定 --input 则默认启用）")
    parser.add_argument("--chembench_task", type=str, default="product", choices=["product", "retro", "yield"], 
                       help="ChemBench 任务类型，默认: product")
    parser.add_argument("--chembench_split", type=str, default="train", choices=["train", "dev", "test"],
                       help="ChemBench 数据划分，默认: train")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config.endswith('.yaml') or args.config.endswith('.yml'):
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            train_cfg = yaml.safe_load(f)
    else:
        # 假设是 JSON
        with open(args.config, 'r', encoding='utf-8') as f:
            train_cfg = json.load(f)
    
    # 将训练配置转换为生成器配置格式
    # 检查设备是否可用
    device = args.device
    if device.startswith("cuda:"):
        import torch
        if not torch.cuda.is_available():
            print(f"⚠️  警告: CUDA 不可用，但设备设置为 {device}")
            print("   尝试使用 CPU 或检查 CUDA_VISIBLE_DEVICES 环境变量")
            # 如果 CUDA 不可用，回退到 CPU
            device = "cpu"
        else:
            # 检查指定的 GPU 是否在可见范围内
            gpu_id = int(device.split(":")[-1])
            visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible_gpus:
                visible_list = [int(x) for x in visible_gpus.split(",") if x.strip().isdigit()]
                if visible_list:
                    # CUDA_VISIBLE_DEVICES 重新映射了 GPU ID
                    # 如果设置了 CUDA_VISIBLE_DEVICES=0,1,2,3，那么 cuda:0 实际指向第一个可见 GPU
                    # 所以如果指定了 cuda:0，应该使用 cuda:0（因为已经重新映射）
                    # 但如果指定的 ID 超出了可见列表范围，使用第一个可见的 GPU
                    if gpu_id >= len(visible_list):
                        device = "cuda:0"  # 使用第一个可见 GPU（重新映射后是 cuda:0）
                        print(f"⚠️  警告: GPU {gpu_id} 超出可见范围，使用 cuda:0（映射到物理 GPU {visible_list[0]}）")
            else:
                # 如果没有设置 CUDA_VISIBLE_DEVICES，直接使用指定的设备
                pass
    
    cfg = {
        "ckpt_dir": train_cfg.get("paths", {}).get("checkpoint_dir") or train_cfg.get("paths", {}).get("llm_name_or_path"),
        "device": device,
        "dtype": "bf16",  # 默认使用 bf16
        "debug": False,
    }
    
    # 添加 token_classifier_path（如果存在）
    token_classifier_path = train_cfg.get("paths", {}).get("mlp_token_classifier_path")
    if token_classifier_path:
        cfg["token_classifier_path"] = token_classifier_path
    
    # 默认启用 Layer2（这是 Layer2 训练数据生成脚本）
    import yaml
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent
    layer2_config_path = project_root / "modules" / "layer2_component" / "layer2_config.yaml"
    if layer2_config_path.exists():
        with open(layer2_config_path, 'r', encoding='utf-8') as f:
            layer2_config = yaml.safe_load(f)
        # 传递配置文件路径，让 Layer2Inferer 自己加载
        cfg["layer2"] = {
            "config_path": str(layer2_config_path),
            **layer2_config,  # 也包含配置内容，供其他部分使用
        }
    else:
        print(f"[WARNING] Layer2 config not found at {layer2_config_path}, using defaults")
        cfg["layer2"] = {
            "config_path": None,
            "checkpoint_path": "/data1/chenyuxuan/Layer2/ckpt/0115/layer2_pretrain.pt",
            "gvp_root": "/data1/chenyuxuan/MSMLM",
            "gvp_ckpt_path": "/data1/chenyuxuan/checkpoint/gvp_weights_best.pt",
        }
    
    # 确保 train 配置存在并设置 use_layer2
    if "train" not in cfg:
        cfg["train"] = {}
    cfg["train"]["use_layer2"] = True
    print(f"[INFO] Layer2 enabled in config (train.use_layer2=True)")
    
    # 检查必要的配置
    if not cfg.get("ckpt_dir"):
        raise ValueError("配置文件中缺少 checkpoint_dir 或 llm_name_or_path")
    
    print(f"📦 使用 checkpoint: {cfg['ckpt_dir']}")
    
    # 初始化生成器
    print("📦 初始化模型...")
    generator = MolAwareGenerator2()
    generator.load(cfg)
    
    # 加载查询
    if args.input:
        print(f"📂 从文件加载查询: {args.input}")
        queries = load_queries(args.input)
        print(f"   找到 {len(queries)} 条查询")
    elif args.use_chembench or not args.input:
        # 如果没有指定输入文件，默认使用 ChemBench
        print(f"📂 从 ChemBench 加载数据")
        queries = load_chembench_data(task=args.chembench_task, split=args.chembench_split)
        print(f"   找到 {len(queries)} 条查询")
    else:
        print("❌ 错误: 必须指定 --input 或使用 --use_chembench")
        return
    
    # 生成训练数据
    print("🔄 开始生成训练数据...")
    generate_training_data(
        generator=generator,
        queries=queries,
        output_path=args.output,
        task_type=args.task_type,
    )
    
    print("✅ 完成！")


if __name__ == "__main__":
    main()
