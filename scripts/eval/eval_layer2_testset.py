#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 sft_tester.py 对 Layer2 测试集进行 LLM 评测。

从 Layer2 测试集 JSONL 文件中读取数据，提取 llm_templates 字段，
对每个任务（task1_mask_product, task2a_predict_yield_full, task2b_predict_product_and_yield,
task3_mask_role, task4_mask_reactant）进行评测。
"""

import sys
import os
import json
import argparse
import re
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sft_tester import MolAwareGenerator2


# ================================
# Prompt template + few-shot
# ================================

TASK_INSTRUCTIONS: Dict[str, str] = {
    # SMILES span completion
    "mask_product": (
        "You are a chemistry reaction assistant. "
        "Return ONLY the missing product as a SMILES string. "
        "Do not output JSON. Do not add any extra text."
    ),
    "mask_reactant": (
        "You are a chemistry reaction assistant. "
        "Return ONLY the missing reactant as a SMILES string. "
        "Do not output JSON. Do not add any extra text."
    ),
    # Yield regression
    "predict_yield_full": (
        "You are a chemistry reaction assistant. "
        "Predict the isolated reaction yield as a percentage in [0, 100]. "
        "Return ONLY a single-line JSON object with exactly this key: "
        "{\"yield_percent\": float}. "
        "yield_percent must be in [0, 100]. "
        "Do not include any other keys, text, code fences, or line breaks."
    ),
    # Product + yield
    "predict_product_and_yield": (
        "You are a chemistry reaction assistant. "
        "Jointly predict the major organic product(s) and the isolated yield. "
        "Return ONLY a single-line JSON object with exactly these keys: "
        "{\"products\": [\"SMILES\", ...], \"yield_percent\": float}. "
        "products must be a non-empty list of SMILES strings for the major organic products "
        "(do NOT include reactants, solvents, catalysts, metals, or counter-ions). "
        "yield_percent must be in [0, 100]. "
        "Do not include any other keys, text, code fences, or line breaks."
    ),
    # Role classification
    "mask_role": (
        "You are a chemistry reaction assistant. "
        "Predict the missing role/category label. "
        "Output ONLY the label text and nothing else."
    ),
}

def _strip_conflicting_instructions(text: str) -> str:
    """
    Dev few-shot inputs sometimes contain instructions like 'Return ONLY a single-line JSON object...'.
    These conflict with TASK_INSTRUCTIONS. We strip those lines so few-shot doesn't teach JSON output.
    """
    if not text:
        return ""
    lines = str(text).splitlines()
    drop_patterns = [
        r"return\s+only.*json",
        r"schema\s*:",
        r"confidence",
        r"yield_range",
        r"is_multiple_products",
        r"do not include markdown",
        r"code fences",
        r"single-line\s+json",
    ]
    kept: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            kept.append(ln)
            continue
        low = s.lower()
        if any(re.search(p, low) for p in drop_patterns):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _extract_first_smiles(text: str) -> Optional[str]:
    if not text:
        return None
    text = str(text)
    m = re.search(r"<SMILES>\s*([^<]+)\s*</SMILES>", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            products = obj.get("products")
            if isinstance(products, list) and products:
                for p in products:
                    if isinstance(p, str) and p.strip():
                        return p.strip()
    except Exception:
        pass
    tokens = re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#\\/%.]+", text)
    tokens = [t for t in tokens if re.search(r"[A-Za-z]", t)]
    if not tokens:
        return None
    return max(tokens, key=len)


def _extract_yield_percent(text: str) -> Optional[float]:
    if not text:
        return None
    text = str(text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k in ("yield_percent", "yield percent", "yieldPercent", "yield"):
                if k in obj:
                    v = obj[k]
                    try:
                        fv = float(v)
                        if 0 <= fv <= 100:
                            return fv
                        if 100 < fv <= 10000:
                            return fv / 100.0
                    except Exception:
                        pass
    except Exception:
        pass
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        try:
            fv = float(m.group(1))
            if 0 <= fv <= 100:
                return fv
        except Exception:
            pass
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    if m:
        try:
            fv = float(m.group(1))
            if 0 <= fv <= 100:
                return fv
        except Exception:
            pass
    return None


def _yield_to_bin(y: float) -> int:
    """Map 0-100 yield_percent to 0-9 decile bin."""
    try:
        y = float(y)
    except Exception:
        return 0
    if y < 0:
        y = 0.0
    if y > 100:
        y = 100.0
    b = int(y // 10)
    return max(0, min(9, b))


def _format_fewshot_example(task_name: str, ex_inp: str, ex_out: str) -> Optional[Tuple[str, str]]:
    """Rewrite dev few-shot examples so outputs match our strict TASK_INSTRUCTIONS."""
    ex_inp = _strip_conflicting_instructions(ex_inp)
    ex_out = str(ex_out).strip() if ex_out is not None else ""
    if not ex_inp or not ex_out:
        return None

    if task_name in {"mask_product", "mask_reactant"}:
        smi = _extract_first_smiles(ex_out)
        if not smi:
            return None
        # 直接输出裸 SMILES，不再包裹标签
        return (ex_inp, smi)

    if task_name == "predict_yield_full":
        y = _extract_yield_percent(ex_out)
        if y is None:
            return None
        return (ex_inp, json.dumps({"yield_percent": float(f"{y:.2f}")}, ensure_ascii=False))

    if task_name == "predict_product_and_yield":
        smi = _extract_first_smiles(ex_out)
        y = _extract_yield_percent(ex_out)
        if not smi or y is None:
            return None
        return (
            ex_inp,
            json.dumps(
                {"products": [smi], "yield_percent": float(f"{y:.2f}")},
                ensure_ascii=False,
            ),
        )

    if task_name == "mask_role":
        return (ex_inp, ex_out.strip())

    return (ex_inp, ex_out.strip())


def _compact_text(x: str, max_chars: int = 4000) -> str:
    """Keep prompts stable & avoid huge examples."""
    if x is None:
        return ""
    x = str(x).strip()
    # collapse excessive whitespace
    x = re.sub(r"\n{3,}", "\n\n", x)
    x = re.sub(r"[ \t]{2,}", " ", x)
    if len(x) > max_chars:
        x = x[: max_chars - 12].rstrip() + "\n...<truncated>"
    return x


def build_prompt(
    task_name: str,
    raw_input: str,
    few_shot_examples: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Wrap dataset-provided input with a stricter instruction layer + optional few-shot."""
    instruction = TASK_INSTRUCTIONS.get(task_name, "Answer the question.")
    raw_input = _compact_text(raw_input)

    parts: List[str] = []
    parts.append(instruction)

    if few_shot_examples:
        parts.append("\n\n### Examples")
        for i, (ex_inp, ex_out) in enumerate(few_shot_examples, 1):
            parts.append(f"\nExample {i}:")
            parts.append("Input:")
            parts.append(_compact_text(ex_inp))
            parts.append("Output:")
            parts.append(_compact_text(ex_out, max_chars=1200))

    parts.append("\n\n### Now solve")
    parts.append("Input:")
    parts.append(raw_input)
    parts.append("Output:")
    return "\n".join(parts).strip() + "\n"


def load_few_shot_pool(devset_path: Path) -> List[Dict[str, Any]]:
    """Load dev set jsonl(.gz) to build a few-shot pool."""
    if not devset_path:
        return []
    if not devset_path.exists():
        raise FileNotFoundError(f"Devset path not found: {devset_path}")
    return load_layer2_testset(devset_path)


def build_few_shot_examples_by_task(
    dev_samples: List[Dict[str, Any]],
    k: int,
    seed: int = 42,
    prefer_same_dataset_id: bool = False,
) -> Dict[str, List[Tuple[str, str]]]:
    """Pre-sample few-shot examples for each task.

    If prefer_same_dataset_id=True, we will later re-sample per test sample (slower). For now we keep
    it simple: global per-task sampling.
    """
    if k <= 0 or not dev_samples:
        return {}

    rng = np.random.RandomState(seed)

    pool_by_task: Dict[str, List[Tuple[str, str]]] = {}
    for s in dev_samples:
        tasks = extract_tasks_from_sample(s)
        for tname, t in tasks.items():
            if "input" in t and "output" in t and t["input"] and t["output"]:
                pool_by_task.setdefault(tname, []).append((t["input"], t["output"]))

    few_shot_by_task: Dict[str, List[Tuple[str, str]]] = {}
    for tname, pool in pool_by_task.items():
        if not pool:
            continue
        if len(pool) <= k:
            few_shot_by_task[tname] = pool
        else:
            idx = rng.choice(len(pool), size=k, replace=False)
            few_shot_by_task[tname] = [pool[i] for i in idx]

    return few_shot_by_task


def load_layer2_testset(testset_path: Path) -> List[Dict[str, Any]]:
    """加载 Layer2 测试集 JSONL 文件"""
    data = []
    print(f"[INFO] Loading testset from: {testset_path}")
    
    if testset_path.suffix == ".gz":
        import gzip
        with gzip.open(testset_path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(testset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    
    print(f"[INFO] Loaded {len(data)} samples")
    return data


def extract_tasks_from_sample(sample: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    从样本中提取所有任务的 llm_templates。
    返回: {task_name: {input, output, metadata, ...}}
    """
    llm_templates = sample.get("llm_templates", {})
    if not llm_templates:
        return {}
    
    # 提取所有任务
    tasks = {}
    task_mapping = {
        "task1_mask_product": "mask_product",
        "task2a_predict_yield_full": "predict_yield_full",
        "task2b_predict_product_and_yield": "predict_product_and_yield",
        "task3_mask_role": "mask_role",
        "task4_mask_reactant": "mask_reactant",
    }
    
    for task_key, task_short_name in task_mapping.items():
        if task_key in llm_templates:
            tasks[task_short_name] = {
                **llm_templates[task_key],
                "reaction_id": sample.get("reaction_id"),
                "dataset_id": sample.get("dataset_id"),
                "task_key": task_key,
            }
    
    return tasks


def run_llm_evaluation(
    generator: MolAwareGenerator2,
    testset_path: Path,
    output_dir: Path,
    task_names: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    few_shot_k: int = 0,
    devset_path: Optional[Path] = None,
    few_shot_seed: int = 42,
    few_shot_same_dataset: int = 0,
    **gen_kwargs
):
    """
    对测试集进行 LLM 评测
    
    Args:
        generator: MolAwareGenerator2 实例
        testset_path: 测试集 JSONL 路径
        output_dir: 输出目录
        task_names: 要评测的任务列表，None 表示所有任务
        max_samples: 最大样本数（用于测试）
        batch_size: 批处理大小
        **gen_kwargs: 传递给 generator.generate 的参数
    """
    # 加载测试集
    samples = load_layer2_testset(testset_path)

    # Build few-shot pool (optional) - rewrite dev examples to match TASK_INSTRUCTIONS
    few_shot_pool: Dict[str, List[Tuple[str, str, Optional[str]]]] = {}
    if few_shot_k and devset_path is not None:
        dev_samples = load_few_shot_pool(devset_path)
        for s in dev_samples:
            tasks = extract_tasks_from_sample(s)
            for tname, tdata in tasks.items():
                if task_names and tname not in task_names:
                    continue
                inp = tdata.get("input", "")
                out = tdata.get("output", "")
                dsid = tdata.get("dataset_id")
                formatted = _format_fewshot_example(tname, inp, out)
                if not formatted:
                    continue
                few_shot_pool.setdefault(tname, []).append((formatted[0], formatted[1], dsid))
        for tname in list(few_shot_pool.keys()):
            if not few_shot_pool[tname]:
                few_shot_pool.pop(tname, None)
        if not few_shot_pool:
            print("[WARNING] few-shot enabled but no dev examples found; falling back to zero-shot.")
    
    # 收集所有任务的数据
    all_tasks_data = {}
    for sample in samples:
        tasks = extract_tasks_from_sample(sample)
        for task_name, task_data in tasks.items():
            if task_names and task_name not in task_names:
                continue
            if task_name not in all_tasks_data:
                all_tasks_data[task_name] = []
            all_tasks_data[task_name].append(task_data)
    
    # If max_samples is set, sample evenly across tasks (stratified by task type).
    if max_samples is not None:
        task_list = list(all_tasks_data.keys())
        if task_list:
            base = max_samples // len(task_list)
            rem = max_samples % len(task_list)
            for i, t in enumerate(task_list):
                quota = base + (1 if i < rem else 0)
                all_tasks_data[t] = all_tasks_data[t][:quota]

    print(f"[INFO] Found tasks: {list(all_tasks_data.keys())}")
    for task_name, task_samples in all_tasks_data.items():
        print(f"  {task_name}: {len(task_samples)} samples")
    
    # 为每个任务创建输出目录和文件
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}
    for task_name in all_tasks_data.keys():
        output_file = output_dir / f"{task_name}_predictions.jsonl"
        output_files[task_name] = open(output_file, "w", encoding="utf-8")
    
    # 对每个任务进行评测
    for task_name, task_samples in all_tasks_data.items():
        print(f"\n[INFO] Evaluating task: {task_name} ({len(task_samples)} samples)")
        output_file = output_files[task_name]
        
        for task_data in tqdm(task_samples, desc=f"Task {task_name}"):
            raw_input_text = task_data["input"]
            ground_truth = task_data["output"]
            metadata = task_data.get("metadata", {})

            # Few-shot select examples (optional)
            examples: Optional[List[Tuple[str, str]]] = None
            if few_shot_k and task_name in few_shot_pool and few_shot_pool[task_name]:
                pool = few_shot_pool[task_name]
                if few_shot_same_dataset:
                    cur_dsid = task_data.get("dataset_id")
                    filtered = [x for x in pool if x[2] == cur_dsid]
                    if len(filtered) >= 1:
                        pool = filtered

                k = min(int(few_shot_k), len(pool))

                # sample without replacement; deterministic per-sample (order-independent)
                rid = str(task_data.get("reaction_id") or "")
                h = hashlib.md5(rid.encode("utf-8")).hexdigest()[:8]
                local_seed = int(h, 16) + int(few_shot_seed)
                rng_local = np.random.RandomState(local_seed)
                idx = rng_local.choice(len(pool), size=k, replace=False)
                examples = [(pool[i][0], pool[i][1]) for i in idx]

            # Build stricter prompt wrapper
            input_text = build_prompt(task_name, raw_input_text, examples)
            
            # 调用 LLM 生成
            try:
                prediction = generator.generate(
                    input_text,
                    add_dialog_wrapper=True,
                    skip_special_tokens=True,
                    **gen_kwargs  # 包含 realtime_mol, max_new_tokens, temperature, top_p 等参数
                )
                
                # 保存结果
                result = {
                    "reaction_id": task_data.get("reaction_id"),
                    "dataset_id": task_data.get("dataset_id"),
                    "task": task_name,
                    "input": input_text,
                    "raw_input": raw_input_text,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "metadata": metadata,
                    "few_shot_k": int(len(examples)) if examples else 0,
                }
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()
            except Exception as e:
                print(f"[ERROR] Error processing sample {task_data.get('reaction_id')}: {e}")
                result = {
                    "reaction_id": task_data.get("reaction_id"),
                    "dataset_id": task_data.get("dataset_id"),
                    "task": task_name,
                    "input": input_text,
                    "raw_input": raw_input_text,
                    "ground_truth": ground_truth,
                    "prediction": None,
                    "error": str(e),
                    "metadata": metadata,
                    "few_shot_k": int(len(examples)) if examples else 0,
                }
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()
    
    # 关闭所有文件
    for f in output_files.values():
        f.close()
    
    print(f"\n[INFO] Evaluation completed. Results saved to: {output_dir}")
    print(f"[INFO] To calculate metrics and generate plots, run:")
    print(f"     python scripts/postprocess/score_and_visualize_layer2.py --results_dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Layer2 testset using LLM")
    parser.add_argument(
        "--testset_path",
        type=str,
        required=True,
        help="Path to Layer2 testset JSONL file (e.g., layer2_test.jsonl or layer2_test.jsonl.gz)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--molaware_ckpt",
        type=str,
        required=True,
        help="Path to MolAwareGenerator2 checkpoint directory"
    )
    parser.add_argument(
        "--token_classifier_path",
        type=str,
        help="Path to token classifier model (optional)"
    )
    parser.add_argument(
        "--base_llm_path",
        type=str,
        default=None,
        help="Base LLM path for loading tokenizer (optional, will try to infer from checkpoint if not provided)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device map for multi-GPU (default: None for single GPU, 'auto' for multi-GPU)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["float32", "float16", "bf16"],
        help="Data type (default: bf16)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Tasks to evaluate (default: all tasks). Options: mask_product, predict_yield_full, predict_product_and_yield, mask_role, mask_reactant"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing). Will be split evenly across tasks."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--realtime_mol",
        type=int,
        default=0,
        help="Whether to use realtime molecule generation (0 or 1)"
    )

    # Few-shot controls
    parser.add_argument(
        "--few_shot_k",
        type=int,
        default=0,
        help="Number of few-shot examples to prepend for each sample (default: 0 = zero-shot)"
    )
    parser.add_argument(
        "--devset_path",
        type=str,
        default=None,
        help="Dev set JSONL(.gz) used to sample few-shot examples (required when --few_shot_k>0)"
    )
    parser.add_argument(
        "--few_shot_seed",
        type=int,
        default=42,
        help="Random seed for few-shot sampling (deterministic per reaction_id)"
    )
    parser.add_argument(
        "--few_shot_same_dataset",
        type=int,
        default=0,
        help="If 1, sample few-shot examples from the same dataset_id when possible"
    )
    
    args = parser.parse_args()
    
    # 初始化生成器
    config = {
        "ckpt_dir": args.molaware_ckpt,
        "device": args.device,
        "device_map": args.device_map,
        "dtype": args.dtype,
        "debug": False,
    }
    config["enable_ldmol"] = False   # Layer2 评测不需要 LDMol
    config["ldmol_device"] = "cpu"
    if args.token_classifier_path:
        config["token_classifier_path"] = args.token_classifier_path
    if args.base_llm_path:
        config["base_llm_path"] = args.base_llm_path
    
    print("[INFO] Loading MolAwareGenerator2...")
    generator = MolAwareGenerator2()
    generator.load(config)
    print("[INFO] Generator loaded successfully")
    
    # 运行评测
    run_llm_evaluation(
        generator=generator,
        testset_path=Path(args.testset_path),
        output_dir=Path(args.output_dir),
        task_names=args.tasks,
        max_samples=args.max_samples,
        few_shot_k=args.few_shot_k,
        devset_path=Path(args.devset_path) if args.devset_path else None,
        few_shot_seed=args.few_shot_seed,
        few_shot_same_dataset=args.few_shot_same_dataset,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        realtime_mol=bool(args.realtime_mol),
    )


if __name__ == "__main__":
    main()
