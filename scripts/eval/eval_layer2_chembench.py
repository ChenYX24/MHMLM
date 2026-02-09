#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval Layer2 (MolAwareGenerator2) on ChemBench4K benchmarks loaded from HuggingFace repo files.

Supported tasks:
  - reactant : Retrosynthesis_benchmark.json (reactant prediction / retrosynthesis)
  - product  : Product_Prediction_benchmark.json (forward product prediction)
  - yield    : Yield_Prediction_benchmark.json (yield prediction)

Core idea:
  1) Load benchmark JSON from HF (AI4Chem/ChemBench4K)
  2) Convert question -> Layer2-friendly prompt
  3) Use MolAwareGenerator2.generate() to get structured output
  4) Map structured output -> A/B/C/D
     - reactant: set fingerprint similarity (symmetric best-match)
     - product : max fingerprint similarity
     - yield   : closest number (0-100)
  5) Add per-sample field: choice_gold_similarity (fingerprint similarity between predicted option and gold option)
     - reactant/product only; yield -> None
  6) Summary outputs: accuracy + avg_choice_gold_similarity (over samples where defined)

Notes:
  - Strongly recommend RDKit for reactant/product matching.
  - To avoid OOM in pure text tasks, we pass config["enable_ldmol"]=False by default.
    (Requires you to patch sft_tester.py to honor enable_ldmol flag; otherwise it's ignored.)

Usage:
  python eval_layer2_chembench.py --task reactant --split test --molaware_ckpt /path/to/ckpt --device cuda:0 --dtype bf16 --out_dir ./out
  python eval_layer2_chembench.py --task product  --split test --molaware_ckpt /path/to/ckpt --device cuda:0 --dtype bf16 --out_dir ./out
  python eval_layer2_chembench.py --task yield    --split test --molaware_ckpt /path/to/ckpt --device cuda:0 --dtype bf16 --out_dir ./out
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

try:
    from datasets import load_dataset
except Exception as e:
    raise RuntimeError("Please install datasets: pip install datasets") from e

# ----------------------------
# Import your generator
# Ensure this script is placed where sft_tester.py is importable (repo root / scripts dir ok if sys.path adjusted).
# ----------------------------
_project_root = Path(__file__).parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_project_root.parent) not in sys.path:
    sys.path.insert(0, str(_project_root.parent))

from sft_tester import MolAwareGenerator2  # noqa: E402

# ----------------------------
# Optional RDKit
# ----------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

CHOICES = ["A", "B", "C", "D"]

# reaction smiles patterns: 2-arrow and 3-part (> >)
_RXN_SMILES_2ARROW = re.compile(r"([A-Za-z0-9@\+\-\[\]\(\)\\\/%=#$\.]+>>[A-Za-z0-9@\+\-\[\]\(\)\\\/%=#$\.]+)")
_RXN_SMILES_3ARROW = re.compile(
    r"([A-Za-z0-9@\+\-\[\]\(\)\\\/%=#$\.]+>[A-Za-z0-9@\+\-\[\]\(\)\\\/%=#$\.]*>[A-Za-z0-9@\+\-\[\]\(\)\\\/%=#$\.]+)"
)
# loose smiles-ish token candidates
_SMILES_TOKEN_RE = re.compile(r"[A-Za-z0-9@\+\-\[\]\(\)\\\/%=#$\.]{3,}")


# ============================================================
# HF benchmark mapping (repo-relative)
# ============================================================
REPO_ID = "AI4Chem/ChemBench4K"
BENCH_FILES = {
    "product": {
        "test": "test/Product_Prediction_benchmark.json",
        "dev":  "dev/Product_Prediction_benchmark.json",
    },
    "retro": {
        "test": "test/Retrosynthesis_benchmark.json",
        "dev":  "dev/Retrosynthesis_benchmark.json",
    },
    "reactant": {  # 兼容旧名称
        "test": "test/Retrosynthesis_benchmark.json",
        "dev":  "dev/Retrosynthesis_benchmark.json",
    },
    "yield": {
        "test": "test/Yield_Prediction_benchmark.json",
        "dev":  "dev/Yield_Prediction_benchmark.json",
    },
}


def hf_resolve_url(repo_id: str, relpath: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{relpath}"


def load_benchmark(task: str, split: str, revision: str = "main"):
    if task not in BENCH_FILES:
        raise ValueError(f"Unsupported task: {task}")
    if split not in ("test", "dev"):
        raise ValueError("split must be 'test' or 'dev' (ChemBench uses dev/test folders)")
    relpath = BENCH_FILES[task][split]
    url = hf_resolve_url(REPO_ID, relpath, revision=revision)
    ds = load_dataset("json", data_files={split: url}, split=split)
    return ds, url


# ============================================================
# Parsing utils
# ============================================================
def extract_choice_letter(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b([ABCD])\b", text.strip())
    return m.group(1) if m else None


def extract_rxn_smiles(text: str) -> Optional[str]:
    if not text:
        return None
    m = _RXN_SMILES_2ARROW.search(text)
    if m:
        return m.group(1)
    m = _RXN_SMILES_3ARROW.search(text)
    if m:
        return m.group(1)
    return None


def _canon_smiles(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    if not _HAS_RDKIT:
        return s
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def extract_valid_smiles_list(text: str, max_mols: int = 32) -> List[str]:
    """
    Extract SMILES candidates from text and validate with RDKit if available.
    """
    if not text:
        return []
    out: List[str] = []
    for tok in _SMILES_TOKEN_RE.findall(text):
        if re.fullmatch(r"\d+(\.\d+)?", tok):
            continue
        cs = _canon_smiles(tok)
        if cs:
            out.append(cs)
        if len(out) >= max_mols:
            break
    return sorted(set(out))


def extract_yield_number(text: str) -> Optional[float]:
    """
    Parse a yield number from text (0-100). Accept '37.3', '37.3%', etc.
    """
    if not text:
        return None
    nums = re.findall(r"(\d+(?:\.\d+)?)\s*%?", text)
    for n in nums:
        try:
            v = float(n)
        except Exception:
            continue
        if 0.0 <= v <= 100.0:
            return v
    return None


# ============================================================
# Fingerprint similarity
# ============================================================
def _fp(smiles: str, radius: int = 2, nbits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def reactant_set_similarity(pred: List[str], opt: List[str]) -> float:
    """
    Symmetric set similarity:
      s = (avg_i max_j sim(pred_i, opt_j) + avg_j max_i sim(opt_j, pred_i)) / 2
    If RDKit unavailable -> Jaccard overlap on strings.
    """
    if not pred or not opt:
        return 0.0
    if not _HAS_RDKIT:
        ps, os = set(pred), set(opt)
        return len(ps & os) / max(1, len(ps | os))

    pred_fps = [x for x in (_fp(s) for s in pred) if x is not None]
    opt_fps = [x for x in (_fp(s) for s in opt) if x is not None]
    if not pred_fps or not opt_fps:
        return 0.0

    def avg_best(src, tgt):
        tot = 0.0
        for s in src:
            tot += max(float(DataStructs.TanimotoSimilarity(s, t)) for t in tgt)
        return tot / len(src)

    return 0.5 * (avg_best(pred_fps, opt_fps) + avg_best(opt_fps, pred_fps))


def max_mol_similarity(mols1: List[str], mols2: List[str]) -> float:
    """
    max_{i in mols1, j in mols2} Tanimoto(fp(i), fp(j)).
    If RDKit unavailable -> string overlap (0/1).
    """
    if not mols1 or not mols2:
        return 0.0
    if not _HAS_RDKIT:
        return 1.0 if (set(mols1) & set(mols2)) else 0.0

    fps1 = [x for x in (_fp(s) for s in mols1) if x is not None]
    fps2 = [x for x in (_fp(s) for s in mols2) if x is not None]
    if not fps1 or not fps2:
        return 0.0

    best = 0.0
    for a in fps1:
        for b in fps2:
            best = max(best, float(DataStructs.TanimotoSimilarity(a, b)))
    return best


# ============================================================
# Prompt builders
# ============================================================
def build_prompt_reactant(sample: Dict[str, Any]) -> str:
    q = str(sample.get("question", "")).strip()
    rxn = extract_rxn_smiles(q)

    product = None
    if rxn:
        if ">>" in rxn:
            _, right = rxn.split(">>", 1)
            product = _canon_smiles(right)
        else:
            parts = rxn.split(">")
            if len(parts) >= 3:
                product = _canon_smiles(parts[-1])

    if not product:
        mols = extract_valid_smiles_list(q, max_mols=16)
        if mols:
            product = mols[-1]

    if product:
        return (
            f"Question: {q}\n\n"
            "You are given a target product. Predict the reactants (SMILES) that can produce it under ideal conditions.\n"
            f"Target product SMILES: {product}\n"
            "Only output the reactant SMILES separated by '.' (no extra words).\n"
        )
    return (
        f"Question: {q}\n\n"
        "Predict the reactants (SMILES) for the following retrosynthesis question.\n"
        "Only output the reactant SMILES separated by '.' (no extra words).\n"
    )


def build_prompt_product(sample: Dict[str, Any]) -> str:
    q = str(sample.get("question", "")).strip()
    rxn = extract_rxn_smiles(q)

    reactants = None
    if rxn:
        if ">>" in rxn:
            left, _ = rxn.split(">>", 1)
            reactants = left
        else:
            parts = rxn.split(">")
            if len(parts) >= 3:
                reactants = parts[0]

    if reactants:
        return (
            f"Question: {q}\n\n"
            "You are given reactants. Predict the major product SMILES under ideal conditions.\n"
            f"Reactants SMILES: {reactants}\n"
            "Only output the product SMILES (no extra words).\n"
        )
    return (
        f"Question: {q}\n\n"
        "Predict the major product SMILES for the following chemistry question.\n"
        "Only output the product SMILES (no extra words).\n"
    )


def build_prompt_yield(sample: Dict[str, Any]) -> str:
    q = str(sample.get("question", "")).strip()
    rxn = extract_rxn_smiles(q)
    if rxn:
        return (
            f"Question: {q}\n\n"
            "Predict the reaction yield (0-100) under ideal conditions.\n"
            f"Reaction SMILES: {rxn}\n"
            "Only output a number.\n"
        )
    return (
        f"Question: {q}\n\n"
        "Predict the reaction yield (0-100) under ideal conditions.\n"
        "Only output a number.\n"
    )


# ============================================================
# Map model output -> A/B/C/D
# ============================================================
def pick_choice_reactant(sample: Dict[str, Any], pred_text: str) -> Tuple[str, Dict[str, Any]]:
    direct = extract_choice_letter(pred_text)
    if direct in CHOICES:
        return direct, {"mode": "direct_letter"}

    pred_smiles = extract_valid_smiles_list(pred_text, max_mols=32)

    scores: Dict[str, float] = {}
    option_smiles: Dict[str, List[str]] = {}
    for c in CHOICES:
        opt_txt = str(sample.get(c, ""))
        opt_smiles = extract_valid_smiles_list(opt_txt, max_mols=32)
        option_smiles[c] = opt_smiles
        scores[c] = reactant_set_similarity(pred_smiles, opt_smiles)

    best = max(CHOICES, key=lambda k: scores.get(k, -1.0))
    meta = {
        "mode": "fingerprint_set" if _HAS_RDKIT else "string_jaccard",
        "pred_smiles": pred_smiles,
        "option_smiles": option_smiles,
        "scores": scores,
    }
    return best, meta


def pick_choice_product(sample: Dict[str, Any], pred_text: str) -> Tuple[str, Dict[str, Any]]:
    direct = extract_choice_letter(pred_text)
    if direct in CHOICES:
        return direct, {"mode": "direct_letter"}

    pred_mols = extract_valid_smiles_list(pred_text, max_mols=8)

    scores: Dict[str, float] = {}
    option_mols: Dict[str, List[str]] = {}
    for c in CHOICES:
        opt_txt = str(sample.get(c, ""))
        opt_mols = extract_valid_smiles_list(opt_txt, max_mols=8)
        option_mols[c] = opt_mols
        scores[c] = max_mol_similarity(pred_mols, opt_mols)

    best = max(CHOICES, key=lambda k: scores.get(k, -1.0))
    meta = {
        "mode": "max_tanimoto" if _HAS_RDKIT else "string_match",
        "pred_mols": pred_mols,
        "option_mols": option_mols,
        "scores": scores,
    }
    return best, meta


def pick_choice_yield(sample: Dict[str, Any], pred_text: str, layer2_yield_reg: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
    """
    选择最接近预测产率的选项。
    
    Args:
        sample: 样本数据
        pred_text: LLM 输出的文本（LLM 应该已经根据 Layer2 信息生成了答案）
        layer2_yield_reg: Layer2 预测的产率回归值（0-1），仅用于记录，不用于选择
    """
    direct = extract_choice_letter(pred_text)
    if direct in CHOICES:
        return direct, {"mode": "direct_letter"}

    # 只从 LLM 的文本输出中提取产率（不直接使用 Layer2 的预测值）
    # Layer2 的信息已经通过 embedding 传递给 LLM，LLM 应该基于这些信息生成答案
    pred_y = extract_yield_number(pred_text)
    mode = "text_extracted"

    option_y: Dict[str, Optional[float]] = {}
    diffs: Dict[str, float] = {}
    for c in CHOICES:
        y = extract_yield_number(str(sample.get(c, "")))
        option_y[c] = y
        if pred_y is None or y is None:
            diffs[c] = float("inf")
        else:
            diffs[c] = abs(pred_y - y)

    if pred_y is None:
        return "A", {"mode": "no_yield_parsed", "pred_yield": None, "option_yields": option_y, "diffs": diffs, "layer2_yield_reg": layer2_yield_reg}

    best = min(CHOICES, key=lambda k: diffs.get(k, float("inf")))
    return best, {"mode": mode, "pred_yield": pred_y, "option_yields": option_y, "diffs": diffs, "layer2_yield_reg": layer2_yield_reg}


# ============================================================
# choice_gold_similarity (pred option vs gold option)
# ============================================================
def compute_choice_gold_similarity(task: str, sample: Dict[str, Any], pred_choice: str, gold_choice: str) -> Optional[float]:
    if task in ["reactant", "retro"]:
        chosen_set = extract_valid_smiles_list(str(sample.get(pred_choice, "")), max_mols=32)
        gold_set = extract_valid_smiles_list(str(sample.get(gold_choice, "")), max_mols=32)
        return reactant_set_similarity(chosen_set, gold_set)
    if task == "product":
        chosen_mols = extract_valid_smiles_list(str(sample.get(pred_choice, "")), max_mols=8)
        gold_mols = extract_valid_smiles_list(str(sample.get(gold_choice, "")), max_mols=8)
        return max_mol_similarity(chosen_mols, gold_mols)
    # yield has no fingerprint similarity
    return None


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", type=str, required=True, choices=["product", "retro", "reactant", "yield"])
    ap.add_argument("--split", type=str, default="test", choices=["test", "dev"])
    ap.add_argument("--revision", type=str, default="main", help="HF repo revision/tag/commit for AI4Chem/ChemBench4K")
    ap.add_argument("--out_dir", type=str, required=True)

    # MolAware / model
    ap.add_argument("--molaware_ckpt", type=str, required=True, help="Path to MolAwareGenerator2 checkpoint dir")
    ap.add_argument("--token_classifier_path", type=str, default=None)
    ap.add_argument("--base_llm_path", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--device_map", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["float32", "float16", "bf16"])

    # Generation
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--realtime_mol", type=int, default=0)

    # LDMol toggle (requires sft_tester.py patch to honor enable_ldmol)
    ap.add_argument("--enable_ldmol", type=int, default=0, help="0: disable LDMol (recommended for chembench); 1: enable")
    ap.add_argument("--ldmol_device", type=str, default="cpu", help="Device for LDMol when enabled, e.g. cpu or cuda:0")
    
    # Layer2 pipeline
    ap.add_argument("--use_layer2_pipeline", type=int, default=0, help="0: disable Layer2 pipeline; 1: enable Layer2 pipeline")
    ap.add_argument("--layer2_task_type", type=str, default=None, help="Task type for Layer2: 'reaction_prediction', 'yield_prediction', 'product_prediction', 'reactant_prediction'")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark
    ds, src_url = load_benchmark(args.task, args.split, revision=args.revision)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Init generator
    cfg = {
        "ckpt_dir": args.molaware_ckpt,
        "device": args.device,
        "device_map": args.device_map,
        "dtype": args.dtype,
        "debug": False,

        # prevent OOM for chembench
        "enable_ldmol": bool(args.enable_ldmol),
        "ldmol_device": args.ldmol_device,
    }
    if args.token_classifier_path:
        cfg["token_classifier_path"] = args.token_classifier_path
    if args.base_llm_path:
        cfg["base_llm_path"] = args.base_llm_path
    
    # 如果使用 Layer2 pipeline，需要启用 Layer2
    if args.use_layer2_pipeline:
        # 加载 Layer2 配置
        import yaml
        # 从脚本位置找到项目根目录
        # scripts/eval/eval_layer2_chembench.py -> scripts/eval -> scripts -> project_root
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent.parent  # scripts/eval -> scripts -> project_root
        layer2_config_path = project_root / "modules" / "layer2_component" / "layer2_config.yaml"
        
        if layer2_config_path.exists():
            with open(layer2_config_path, 'r', encoding='utf-8') as f:
                layer2_config = yaml.safe_load(f)
            # 传递配置文件路径，让 Layer2Inferer 自己加载
            cfg["layer2"] = {
                "config_path": str(layer2_config_path),  # Layer2Inferer 需要这个路径
                **layer2_config,  # 也包含配置内容，供其他部分使用
            }
            print(f"[INFO] Layer2 config loaded from: {layer2_config_path}")
        else:
            print(f"[WARNING] Layer2 config not found at {layer2_config_path}, using defaults")
            # 使用默认配置
            cfg["layer2"] = {
                "config_path": None,  # 使用默认路径
                "checkpoint_path": "/data1/chenyuxuan/Layer2/ckpt/0115/layer2_pretrain.pt",
                "gvp_root": "/data1/chenyuxuan/MSMLM",
                "gvp_ckpt_path": "/data1/chenyuxuan/checkpoint/gvp_weights_best.pt",
            }
        
        # 确保 train 配置存在并设置 use_layer2
        if "train" not in cfg:
            cfg["train"] = {}
        cfg["train"]["use_layer2"] = True
        print(f"[INFO] Layer2 enabled in config (train.use_layer2=True)")

    print("[INFO] Loading MolAwareGenerator2...")
    gen = MolAwareGenerator2()
    gen.load(cfg)
    print("[INFO] Generator loaded.")
    print(f"[INFO] Benchmark source: {src_url}")
    print(f"[INFO] RDKit available: {_HAS_RDKIT}")
    print(f"[INFO] Layer2 pipeline: {'enabled' if args.use_layer2_pipeline else 'disabled'}")
    if args.use_layer2_pipeline:
        layer2_task_type = args.layer2_task_type
        if layer2_task_type is None:
            if args.task == "yield":
                layer2_task_type = "yield_prediction"
            elif args.task in ["retro", "reactant"]:
                layer2_task_type = "reactant_prediction"
            elif args.task == "product":
                layer2_task_type = "product_prediction"
            else:
                layer2_task_type = "reaction_prediction"
        print(f"[INFO] Layer2 task type: {layer2_task_type}")

    # 详细预测结果（保留中间过程）
    pred_path = out_dir / f"chembench4k_{args.task}_{args.split}_predictions.jsonl"
    # 简化答案（task, idx, gold, pred, correct）
    simple_pred_path = out_dir / f"pred_{args.task}.jsonl"
    summary_path = out_dir / f"chembench4k_{args.task}_{args.split}_summary.json"

    correct = 0
    total = 0
    sims: List[float] = []  # choice_gold_similarity list (reactant/product)

    # 打开两个文件：详细结果和简化答案
    with pred_path.open("w", encoding="utf-8") as wf, \
         simple_pred_path.open("w", encoding="utf-8") as sf:
        for i, sample in enumerate(tqdm(ds, desc=f"Eval {args.task}/{args.split}")):
            gold = str(sample.get("answer", "")).strip().upper()
            if gold not in CHOICES:
                gold = "A"

            # build prompt based on task
            if args.task == "product":
                prompt = build_prompt_product(sample)
            elif args.task in ["retro", "reactant"]:
                prompt = build_prompt_reactant(sample)
            elif args.task == "yield":
                prompt = build_prompt_yield(sample)
            else:
                raise ValueError(f"Unknown task: {args.task}")

            # generate
            try:
                # 确定 Layer2 task_type（如果未指定，根据任务自动推断）
                layer2_task_type = args.layer2_task_type
                if layer2_task_type is None and args.use_layer2_pipeline:
                    if args.task == "yield":
                        layer2_task_type = "yield_prediction"
                    elif args.task in ["retro", "reactant"]:
                        layer2_task_type = "reactant_prediction"
                    elif args.task == "product":
                        layer2_task_type = "product_prediction"
                    else:
                        layer2_task_type = "reaction_prediction"
                
                # 使用 Layer2 pipeline 或普通生成
                if args.use_layer2_pipeline:
                    result = gen.generate(
                        prompt,
                        add_dialog_wrapper=True,
                        skip_special_tokens=True,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        realtime_mol=bool(args.realtime_mol),
                        use_layer2_pipeline=True,
                        task_type=layer2_task_type,
                        return_intermediate=True,  # 返回中间结果以便记录
                    )
                    # 如果返回的是字典（中间结果），提取最终输出
                    if isinstance(result, dict):
                        pred_text = result.get("final_response", result.get("first_response", ""))
                        layer2_info = result.get("layer2_info", {})
                    else:
                        pred_text = result
                        layer2_info = {}
                else:
                    pred_text = gen.generate(
                        prompt,
                        add_dialog_wrapper=True,
                        skip_special_tokens=True,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        realtime_mol=bool(args.realtime_mol),
                    )
                    layer2_info = {}
                err = None
            except Exception as e:
                pred_text = None
                layer2_info = {}
                err = str(e)

            # map to choice based on task
            if pred_text is None:
                chosen = "A"
                meta = {"mode": "generation_error", "error": err}
            else:
                if args.task == "product":
                    chosen, meta = pick_choice_product(sample, pred_text)
                elif args.task in ["retro", "reactant"]:
                    chosen, meta = pick_choice_reactant(sample, pred_text)
                elif args.task == "yield":
                    # 对于 yield 任务，如果 Layer2 已经预测了产率，优先使用
                    layer2_yield_reg = None
                    if args.use_layer2_pipeline and layer2_info:
                        layer2_yield_reg = layer2_info.get("yield_reg")
                    chosen, meta = pick_choice_yield(sample, pred_text, layer2_yield_reg=layer2_yield_reg)
                else:
                    chosen = "A"
                    meta = {"mode": "unknown_task"}

            is_correct = (chosen == gold)
            total += 1
            correct += int(is_correct)

            choice_gold_sim = compute_choice_gold_similarity(args.task, sample, chosen, gold)
            if choice_gold_sim is not None:
                sims.append(choice_gold_sim)
            
            # 提取正确选项的产率信息（如果有）
            gold_yield = None
            if args.use_layer2_pipeline and layer2_info:
                # 尝试从正确选项中提取产率
                gold_choice_text = str(sample.get(gold, ""))
                gold_yield = extract_yield_number(gold_choice_text)
                if gold_yield is not None:
                    # 输出正确选项的产率信息
                    print(f"✅ 正确选项 {gold} 的产率: {gold_yield:.1f}%")

            # 详细记录（保留中间过程）
            rec = {
                "idx": i,
                "task": args.task,
                "split": args.split,
                "source_url": src_url,
                "question": sample.get("question", ""),
                "choices": {c: sample.get(c, "") for c in CHOICES},
                "gold": gold,
                "prompt": prompt,
                "model_output": pred_text,
                "pred_choice": chosen,
                "correct": is_correct,
                "choice_gold_similarity": choice_gold_sim,  # ✅ 新字段
                "meta": meta,
            }
            # 如果使用了 Layer2，添加 Layer2 信息
            if args.use_layer2_pipeline and layer2_info:
                embedding = layer2_info.get("embedding")
                embedding_shape = None
                if embedding is not None:
                    try:
                        if hasattr(embedding, "shape"):
                            embedding_shape = list(embedding.shape)
                        elif isinstance(embedding, (list, tuple)) and len(embedding) > 0:
                            if hasattr(embedding[0], "shape"):
                                embedding_shape = list(embedding[0].shape)
                    except:
                        pass
                rec["layer2_info"] = {
                    "yield_bin": layer2_info.get("yield_bin"),
                    "yield_reg": layer2_info.get("yield_reg"),
                    "embedding_shape": embedding_shape,
                }
                # 添加正确选项的产率信息（如果有）
                if gold_yield is not None:
                    rec["layer2_info"]["gold_yield"] = gold_yield
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            # 简化答案（只保留关键信息）
            simple_rec = {
                "task": args.task,
                "idx": i,
                "gold": gold,
                "pred": chosen,
                "correct": is_correct,
            }
            sf.write(json.dumps(simple_rec, ensure_ascii=False) + "\n")

    acc = correct / max(1, total)
    avg_sim = (sum(sims) / len(sims)) if sims else None

    summary = {
        "repo": REPO_ID,
        "revision": args.revision,
        "source_url": src_url,
        "task": args.task,
        "split": args.split,
        "n": total,
        "acc": acc,
        "has_rdkit": _HAS_RDKIT,

        # similarity summary
        "avg_choice_gold_similarity": avg_sim,
        "n_with_similarity": len(sims),

        "molaware_ckpt": args.molaware_ckpt,
        "gen": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "realtime_mol": bool(args.realtime_mol),
        },
        "ldmol": {
            "enable_ldmol": bool(args.enable_ldmol),
            "ldmol_device": args.ldmol_device,
        },
        "layer2": {
            "use_layer2_pipeline": bool(args.use_layer2_pipeline),
            "layer2_task_type": args.layer2_task_type or "auto",
        },
        "outputs": {
            "detailed_predictions_jsonl": str(pred_path),
            "simple_answers_jsonl": str(simple_pred_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[RESULT] task={args.task} split={args.split} N={total} Acc={acc:.4f} avg_choice_gold_sim={avg_sim}")
    print(f"[INFO] Saved detailed predictions: {pred_path}")
    print(f"[INFO] Saved simple answers: {simple_pred_path}")
    print(f"[INFO] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
