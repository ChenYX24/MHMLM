#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 MolAwareGenerator2 在 SMolInstruct 上批量推理并自动打分的完整脚本（集成 MLP 标 <mol> + GNN pipeline + Few-shot）。

功能：
1. 调用 MolAwareGenerator2 对 SMolInstruct 各子任务做批量推理，生成 <task>.jsonl。
2. 在推理前，用 MLP token classifier 对 prompt 自动打 <mol> 标签：
   - prompt -> tagger(prompts) -> 带 <mol> 的 prompt -> MolAwareGenerator2.generate(...)
3. 支持评测时 few-shot（从 dev 目录抽样，按 task 生成 prefix）：
   - 使用 --few_shot, --few_shot_dir 等参数启用
   - few-shot 示例信息会写入 _fewshot_meta/ 目录并回填到输出 jsonl
4. 调用 utils.metrics 对每个 <task>.jsonl 打分，支持：
   - 评测阶段每个子任务带 tqdm 进度条（逐样本读取）
   - 可选多进程按 task 粒度并行加速评测 (--score_workers)

用法示例（with GNN + Few-shot）：
    CUDA_VISIBLE_DEVICES=3 python eval_smolinstruct.py \
        --raw_data_dir ./constructed_test \
        --template_dir ./data/template/instruction_tuning \
        --output_dir ./predictions_llama32_sft_withgnn_fewshot \
        --molaware_ckpt /data1/chenyuxuan/MSMLM/model/llama3.2-chem-sft-gnn/1123_llm_gnn_loss/epoch2/checkpoint-13000 \
        --token_classifier_path /data1/lvchangwei/LLM/Lora/llama_mlp_token_classifier.pt \
        --realtime_mol 1 \
        --max_new_tokens 1024 \
        --temperature 0.2 \
        --top_p 0.9 \
        --few_shot 3 \
        --few_shot_dir ./constructed_dev \
        --few_shot_seed 42 \
        --prompt_style strict \
        --save_json ./predictions_llama32_sft_withgnn_fewshot/metrics.json

baseline（不走 GNN，只用同样的 prompt + <mol>）：
    ... 把 --realtime_mol 0，换个 output_dir 即可。
"""

# 确保stdout和stderr使用UTF-8编码，避免日志文件中的中文乱码
import sys
import io
import os

# 设置环境变量确保UTF-8编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 如果stdout/stderr不是UTF-8，则重新包装
if hasattr(sys.stdout, 'buffer') and (not hasattr(sys.stdout, 'encoding') or sys.stdout.encoding != 'utf-8'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except (AttributeError, ValueError):
        pass
if hasattr(sys.stderr, 'buffer') and (not hasattr(sys.stderr, 'encoding') or sys.stderr.encoding != 'utf-8'):
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except (AttributeError, ValueError):
        pass

import argparse
import json
import os
import re
import sys
import hashlib
import random
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========= 部分一：导入 SMolInstruct 的评测逻辑（构造 prompt & 写预测文件） =========

# 同目录的 eval_smolinstruct_batch.py
# 尝试相对导入，如果失败则使用绝对导入
try:
    from . import eval_smolinstruct_batch as evaluator
except ImportError:
    # 如果相对导入失败（如直接运行脚本），使用绝对导入
    import sys
    import os
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import eval_smolinstruct_batch as evaluator

# ========= 部分二：导入 MHMLM 代码里的 MolAwareGenerator2 + MLP 标注工具 =========

# 使用当前项目（MHMLM）的 sft_tester
# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from sft_tester import MolAwareGenerator2
from modules.data_loader import create_batch_tag_text_function

# ========= 部分三：导入 metrics（打分）和预测提取函数 =========

# 尝试导入 metrics 模块（优先使用参考脚本的 metrics）
_metrics_imported = False
_metrics_paths = [
    "/data1/lvchangwei/LLM/SMolInstruct/utils",  # 参考脚本的 metrics
    None,  # 当前项目的 utils.metrics（会在下面设置）
]

# 先尝试从参考脚本导入
try:
    ref_metrics_path = "/data1/lvchangwei/LLM/SMolInstruct/utils"
    if os.path.exists(ref_metrics_path):
        if ref_metrics_path not in sys.path:
            sys.path.insert(0, ref_metrics_path)
        from metrics import (
            calculate_smiles_metrics,
            calculate_formula_metrics,
            calculate_text_metrics,
            calculate_number_metrics,
            calculate_boolean_metrics,
        )
        _metrics_imported = True
        print(f"[INFO] 使用参考脚本的 metrics 模块: {ref_metrics_path}")
except ImportError as e:
    pass

# 如果参考脚本的导入失败，尝试从当前项目导入
if not _metrics_imported:
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from utils.metrics import (
            calculate_smiles_metrics,
            calculate_formula_metrics,
            calculate_text_metrics,
            calculate_number_metrics,
            calculate_boolean_metrics,
        )
        _metrics_imported = True
        print(f"[INFO] 使用当前项目的 metrics 模块: {project_root}/utils/metrics")
    except ImportError as e:
        import sys as _sys
        if len(_sys.argv) > 1 and _sys.argv[1] not in ('--help', '-h'):
            print(f"[ERROR] utils.metrics 模块导入失败: {e}")
            print(f"[ERROR] 请确保 rdkit 等依赖已安装，或参考脚本路径正确")
        _metrics_imported = False

# 如果都导入失败，创建占位符函数（会返回0，不推荐）
if not _metrics_imported:
    def calculate_smiles_metrics(preds, golds, metrics=None):
        return {"num_all": len(preds), "t1_rdk_fps": 0.0}
    
    def calculate_formula_metrics(preds, golds, metrics=None):
        return {"num_all": len(preds), "num_t1_ele_match": 0}
    
    def calculate_text_metrics(preds, golds):
        return {"num_all": len(preds), "bleu4": 0.0, "rouge_l": 0.0}
    
    def calculate_number_metrics(preds, golds):
        return {"RMSE": 0.0}
    
    def calculate_boolean_metrics(preds, golds):
        return {"num_all": len(preds), "f1_score": 0.0, "mcc": 0.0}

# 导入预测提取函数（优化过的版本）
try:
    from eval.extract_prediction import extract_prediction_from_raw
    EXTRACT_PREDICTION_AVAILABLE = True
except ImportError:
    # 如果导入失败，定义占位符函数
    def extract_prediction_from_raw(raw_output, task_name, **kwargs):
        return str(raw_output).strip()
    EXTRACT_PREDICTION_AVAILABLE = False

# ---------- 任务分组（与 SMolInstruct 对齐） ----------
SMILES_TASKS = {
    "forward_synthesis",
    "retrosynthesis",
    "molecule_generation",
    "name_conversion-i2s",
}
SMILES_TASKS_MULTIMETRIC = {"retrosynthesis"}  # multiple_match
TEXT_TASKS = {"molecule_captioning"}
FORMULA_ELEMENT_TASKS = {"name_conversion-i2f", "name_conversion-s2f"}
FORMULA_SPLIT_TASKS = {"name_conversion-s2i"}
NUMBER_TASKS = {"property_prediction-esol", "property_prediction-lipo"}
BOOLEAN_TASKS = {
    "property_prediction-bbbp",
    "property_prediction-clintox",
    "property_prediction-hiv",
    "property_prediction-sider",
}

DEFAULT_REPLACE_SEMICOLON_TASKS = {
    "forward_synthesis",
    "retrosynthesis",
    "molecule_generation",
    "name_conversion-i2s",
    "name_conversion-s2i",
}

# ---------- 从原始文本中抽"核心答案"的工具（用于 fewshot） ----------
SMILES_TOKEN_RE = re.compile(r"([A-Za-z0-9@+\-\[\]\(\)=#\\/%.]+)")
FORMULA_TOKEN_RE = re.compile(r"([A-Za-z0-9\(\)\.\+\-]+)")
NUMBER_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
BOOL_TOKEN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)

def _canonical_bool(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    m = BOOL_TOKEN_RE.search(text)
    if not m:
        return ""
    v = m.group(1).lower()
    return "Yes" if v == "yes" else "No"

def _extract_core_answer(text: str, task_name: str) -> str:
    """从文本中提取核心答案（用于 fewshot 示例）。"""
    if text is None:
        return ""
    text = str(text)

    if task_name in TEXT_TASKS:
        return text.strip()

    if task_name in SMILES_TASKS:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = SMILES_TOKEN_RE.search(line)
            if m:
                return m.group(1)
        return text.strip()

    if task_name in FORMULA_ELEMENT_TASKS or task_name in FORMULA_SPLIT_TASKS:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = FORMULA_TOKEN_RE.search(line)
            if m:
                return m.group(1)
        return text.strip()

    if task_name in NUMBER_TASKS:
        m = NUMBER_TOKEN_RE.search(text)
        if m:
            return m.group(0)
        return text.strip()

    if task_name in BOOLEAN_TASKS:
        return _canonical_bool(text)

    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


# ------------------- Few-shot (evaluation-time) helpers -------------------

def _load_jsonl_safe(path: Path):
    """安全地加载 jsonl 文件，跳过无效行。"""
    if not path or not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _get_obj_id(obj: dict) -> str:
    """尽量从样本里取稳定 id；没有则用内容 hash 兜底。"""
    if not isinstance(obj, dict):
        return ""
    for k in ("id", "uid", "uuid", "sample_id"):
        if k in obj and obj[k] is not None:
            return str(obj[k])
    # fallback: hash messages/prompt
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

def _extract_user_assistant_from_obj(obj: dict):
    """从对象中提取 user 和 assistant 文本。"""
    if not isinstance(obj, dict):
        return None, None

    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs:
        user_text, asst_text = None, None
        for m in msgs:
            if not isinstance(m, dict):
                continue
            r = m.get("role")
            c = m.get("content")
            if r == "user" and user_text is None:
                user_text = c
            elif r == "assistant" and asst_text is None:
                asst_text = c
        if user_text is not None and asst_text is not None:
            return str(user_text), str(asst_text)

    user_text = (
        obj.get("prompt")
        or obj.get("input")
        or obj.get("instruction")
        or obj.get("question")
        or obj.get("query")
        or obj.get("text")
    )
    asst_text = (
        obj.get("answer")
        or obj.get("output")
        or obj.get("gold")
        or obj.get("target")
        or obj.get("completion")
        or obj.get("label")
    )
    if user_text is None or asst_text is None:
        return None, None

    return str(user_text), str(asst_text)

def _find_task_jsonl(dev_dir: Path, task: str) -> Optional[Path]:
    """在 dev_dir 中查找任务对应的 jsonl 文件。"""
    if dev_dir is None:
        return None
    if not dev_dir.exists():
        return None

    exact = dev_dir / f"{task}.jsonl"
    if exact.exists():
        return exact

    cand = sorted(dev_dir.glob("*.jsonl"))
    for p in cand:
        if task in p.stem:
            return p
    return None

def _build_prompt_suffix_for_task(task: str, prompt_style: str) -> str:
    """根据任务类型和 prompt_style 构建输出格式后缀。"""
    if prompt_style not in ("compact", "strict"):
        return ""

    if task in SMILES_TASKS:
        return "\n\nIMPORTANT: Output format: output ONLY a single SMILES string. No explanation, no additional text, just the SMILES.\n"
    if task in BOOLEAN_TASKS:
        return "\n\nIMPORTANT: Output format: output ONLY \"Yes\" or \"No\" (capitalized, exactly one word). No explanation, no additional text.\n"
    if task in NUMBER_TASKS:
        return "\n\nIMPORTANT: Output format: output ONLY a number (e.g., 4.5 or -2.3). No units, no text, just the number.\n"
    if task in FORMULA_ELEMENT_TASKS or task in FORMULA_SPLIT_TASKS:
        if task in FORMULA_SPLIT_TASKS:
            return "\n\nIMPORTANT: Output format: output ONLY the IUPAC name in standard format. No explanation, no additional text.\n"
        else:
            return "\n\nIMPORTANT: Output format: output ONLY the molecular formula. No explanation, no additional text.\n"
    return "\n\nIMPORTANT: Output format: output ONLY the answer. No additional text or explanation.\n"

def _build_fewshot_prefix_and_meta(
    task: str,
    few_shot_k: int,
    few_shot_dir: str,
    seed: int = 42,
    max_chars: int = 6000,
    store_full: bool = True,
) -> Tuple[str, Optional[dict]]:
    """
    返回 (fewshot_prefix, fewshot_meta)
    fewshot_meta 会记录 used ids / dev_path / examples 等，用于写 meta 文件 + 回填输出 jsonl。
    """
    if not few_shot_k or few_shot_k <= 0:
        return "", None
    if not few_shot_dir:
        return "", None

    dev_dir = Path(few_shot_dir)
    fp = _find_task_jsonl(dev_dir, task)
    if fp is None:
        return "", None

    pool: List[dict] = []
    for obj in _load_jsonl_safe(fp):
        ex_id = _get_obj_id(obj)
        u, a = _extract_user_assistant_from_obj(obj)
        if not u or not a:
            continue
        a_core = _extract_core_answer(a, task)
        if not a_core:
            continue
        pool.append({
            "id": ex_id,
            "user": u.strip(),
            "answer": a_core.strip(),
            "user_full": u.strip() if store_full else "",
            "answer_full": str(a).strip() if store_full else "",
        })
        if len(pool) >= max(few_shot_k * 12, 4000):
            break

    if not pool:
        return "", None

    h = int(hashlib.md5(task.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF
    rng = random.Random(seed ^ h)
    rng.shuffle(pool)
    shots = pool[:few_shot_k]

    lines = []
    lines.append("Here are some examples:\n")
    for i, ex in enumerate(shots, start=1):
        lines.append(f"### Example {i} (id={ex['id']})")
        lines.append(ex["user"])
        lines.append(ex["answer"])
        lines.append("")
    lines.append("### Now solve the next problem.\n")

    prefix = "\n".join(lines)
    if max_chars and len(prefix) > max_chars:
        prefix = prefix[:max_chars]
        if not prefix.endswith("\n"):
            prefix += "\n"

    meta = {
        "task": task,
        "few_shot_k": few_shot_k,
        "few_shot_seed": seed,
        "dev_path": str(fp),
        "fewshot_ids": [ex["id"] for ex in shots],
        "examples": shots,
    }
    return prefix, meta

def _write_fewshot_meta_file(output_dir: Path, task: str, meta: dict) -> Path:
    """将 fewshot meta 信息写入文件。"""
    meta_dir = output_dir / "_fewshot_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    outp = meta_dir / f"{task}.json"
    outp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return outp

def _attach_fewshot_meta_to_task_output(task_outfile: Path, meta_file: Path, meta: dict):
    """把 few-shot 信息写回该 task 的输出 jsonl 每一行。"""
    if not task_outfile.exists():
        return
    tmp = task_outfile.with_suffix(".jsonl.tmp")
    fewshot_ids = meta.get("fewshot_ids", [])
    few_shot_k = meta.get("few_shot_k", 0)
    few_shot_seed = meta.get("few_shot_seed", None)

    with task_outfile.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["fewshot_meta_file"] = str(meta_file)
            obj["fewshot_ids"] = fewshot_ids
            obj["fewshot_k"] = few_shot_k
            obj["fewshot_seed"] = few_shot_seed
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    tmp.replace(task_outfile)


# ========= 一：构建"MLP 分子标注器"（批量给 prompt 打 <mol>） =========

def build_mlp_mol_tagger(args, device: str, generator: "MolAwareGenerator2" = None):
    """
    构建一个 batch 版的分子标注函数：List[str] -> List[str]
    
    优先使用 MolAwareGenerator2 内部的 token_classifier_head（如果已加载），
    否则回退到独立加载的方式。

    - 如果没有提供 --token_classifier_path，则返回 None（不做标注）
    - generator: 可选的 MolAwareGenerator2 实例，如果已加载 token_classifier_head 则直接使用
    """
    if not args.token_classifier_path:
        print("[Tagger] No --token_classifier_path provided, skip MLP tagging.")
        return None

    # 优先使用 generator 内部的 token_classifier_head
    if generator is not None and hasattr(generator.model, 'token_classifier_head') and generator.model.token_classifier_head is not None:
        print("[Tagger] Using token_classifier_head from MolAwareGenerator2 (already loaded)")
        offline_head = generator.model.token_classifier_head
        llm = generator.model.llm
        tokenizer = generator.tokenizer
    else:
        # 回退到独立加载方式
        print("[Tagger] Token classifier not found in generator, loading independently...")
        tag_llm_path = args.tag_llm_path or os.path.join(args.molaware_ckpt, "llm")
        print(f"[Tagger] Using LLM for tagging: {tag_llm_path}")
        print(f"[Tagger] Loading token classifier from: {args.token_classifier_path}")

        # 加载临时 LLM 以获取 hidden_size
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        temp_llm = AutoModelForCausalLM.from_pretrained(
            tag_llm_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=None,
        )
        
        # 使用 model_init 中的函数加载 token classifier
        from modules.model_init import init_offline_token_classifier
        offline_head = init_offline_token_classifier(
            llm=temp_llm,
            mlp_token_classifier_path=args.token_classifier_path,
            device=device,
        )
        
        # 清理临时 LLM
        del temp_llm
        torch.cuda.empty_cache()
        
        if offline_head is None:
            raise RuntimeError(f"Failed to load token classifier from {args.token_classifier_path}")

        # 加载用于标注的 LLM + tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(tag_llm_path, use_fast=False)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(tag_llm_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        llm = AutoModelForCausalLM.from_pretrained(
            tag_llm_path,
            torch_dtype=dtype,
            device_map={"": device},
        )
        llm.eval()

    # 用 data_loader 里的函数构建 batch tagger
    local_rank = int(device.split(":")[-1]) if device.startswith("cuda") and ":" in device else 0
    tag_func = create_batch_tag_text_function(
        tokenizer=tokenizer,
        llm=llm,
        offline_token_head=offline_head,
        local_rank=local_rank,
        max_length=args.offline_tagging_max_length,
        batch_size=args.offline_tagging_batch_size,
    )
    print("[Tagger] MLP-based <mol> tagger is ready.")
    return tag_func


# ========= 二：封装给 evaluator 用的 call_model_func =========

def build_call_model(gen: "MolAwareGenerator2",
                     max_new_tokens: int,
                     temperature: float,
                     top_p: float,
                     batch_size: int,
                     tag_func=None,
                     realtime_mol: bool = True,
                     enable_gnn_trigger: bool = False,
                     verbose_gnn: bool = False,
                     repetition_penalty: float = 1.06,
                     no_repeat_ngram_size: int = 3):
    """
    返回一个可被 evaluator.evaluate_file 调用的函数：

        outputs, _ = call_model_func([prompt1, prompt2, ...])

    - tag_func: 已废弃，generate 函数会自动使用 token_classifier_head 进行 tagging（如果已加载）
    - realtime_mol: True = 走 GNN pipeline，False = 只当普通 token（baseline）。
    - enable_gnn_trigger: 是否在 prompt 中注入复读 SMILES 的指令以触发 GNN
    - verbose_gnn: 是否启用详细的 GNN 日志
    """
    supports_batch = {"ok": None}  # None=未知, True=支持 batch, False=不支持

    def _maybe_inject_gnn_trigger(p: str) -> str:
        """如果需要，在 prompt 中注入 GNN 触发指令。"""
        if not enable_gnn_trigger:
            return p
        extra = (
            "\n\n[Important] Before answering, you MUST first repeat all molecule SMILES strings "
            "exactly as they appear in the question on a separate line, then answer the question."
        )
        return p + extra

    def _call_model(prompts: List[str], task: Optional[str] = None):
        """
        调用模型生成函数。
        
        Args:
            prompts: 输入prompt列表
            task: 任务名称（可选），用于传递给 generate 函数的 task_type 参数
        """
        outputs: List[str] = []
        raw_outputs_list = []

        for i in tqdm(range(0, len(prompts), max(1, batch_size)), desc="Generating", unit="prompt"):
            raw_batch = prompts[i:i + batch_size]

            # generate 函数会自动使用 token_classifier_head 进行 tagging（如果已加载）
            # 所以这里直接使用原始 prompt，但可能需要注入 GNN trigger
            batch_prompts = [_maybe_inject_gnn_trigger(p) for p in raw_batch]

            # 2) 如果已经确认不支持 batch，或者当前就是单条，就逐条 generate
            if realtime_mol:
                for p in tqdm(batch_prompts, desc="Generating", unit="prompt"):
                    generate_kwargs = {
                        "add_dialog_wrapper": True,
                        "realtime_mol": realtime_mol,
                        "max_new_tokens": max_new_tokens,
                        "do_sample": temperature > 0,
                        "temperature": temperature if temperature > 0 else 1.0,
                        "top_k": 0,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                        "no_repeat_ngram_size": no_repeat_ngram_size,
                        "skip_special_tokens": True,
                        "verbose_logging": verbose_gnn,  # 根据参数控制详细日志
                    }
                    # 如果提供了 task，传递给 generate 函数
                    if task is not None:
                        generate_kwargs["task_type"] = task
                    
                    text = gen.generate(p, **generate_kwargs)
                    outputs.append(text)
                    raw_outputs_list.append(text)
            else:
                generate_kwargs = {
                    "add_dialog_wrapper": True,
                    "realtime_mol": realtime_mol,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": temperature > 0,
                    "temperature": temperature if temperature > 0 else 1.0,
                    "top_k": 0,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "no_repeat_ngram_size": no_repeat_ngram_size,
                    "skip_special_tokens": True,
                    "verbose_logging": verbose_gnn,  # 启用详细日志
                }
                # 如果提供了 task，传递给 generate 函数
                if task is not None:
                    generate_kwargs["task_type"] = task
                
                batch_out = gen.generate(batch_prompts, **generate_kwargs)

                # 如果返回的是一个 str，说明模型没把 list 当 batch 处理
                if isinstance(batch_out, str):
                    raise TypeError("MolAwareGenerator2.generate returned str for batched input")

                batch_out_list = list(batch_out)
                
                if len(batch_out_list) != len(batch_prompts):
                    raise ValueError(
                        f"batched generate returned {len(batch_out_list)} outputs "
                        f"for {len(batch_prompts)} prompts"
                    )

                outputs.extend(batch_out_list)
                raw_outputs_list.extend(batch_out_list) 

        return outputs, raw_outputs_list

    return _call_model


# ========= 三：推理阶段（按 task 批量跑 MolAwareGenerator2） =========

def run_inference(args) -> Path:
    """跑 MolAware 推理，生成 <task>.jsonl 预测文件，返回预测目录 Path。"""
    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    template_dir = Path(args.template_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备与多卡设置：如果可见 GPU 数量 >1，则启用 device_map="auto" 多卡推理
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            device = "cuda:0"
            device_map = "auto"
            print(f"[INFO] Detected {n_gpus} GPUs, using multi-GPU inference with device_map='auto'")
        else:
            device = "cuda:0"
            device_map = None
    else:
        device = "cpu"
        device_map = None

    dtype_flag = "bf16" if device.startswith("cuda") else "fp32"

    print(f"[INFO] Loading MolAware checkpoint from {args.molaware_ckpt} on {device} (dtype={dtype_flag}, device_map={device_map})")
    gen_cfg = {
        "ckpt_dir": args.molaware_ckpt,
        "device": device,
        "dtype": dtype_flag,
        "debug": False,
    }
    if device_map is not None:
        gen_cfg["device_map"] = device_map
    # 如果提供了 base_llm_path，用于加载 tokenizer
    if args.base_llm_path:
        gen_cfg["base_llm_path"] = args.base_llm_path
    # 如果 MolAwareGenerator2.load 里用得到 classifier，可以顺带传进去；实际 tagging 我们在外面做
    if args.token_classifier_path:
        gen_cfg["token_classifier_path"] = args.token_classifier_path
    # 是否启用 thinking 模式（默认关闭）
    gen_cfg["enable_thinking"] = getattr(args, "enable_thinking", False)
    
    # 尝试使用 Flash Attention 2（如果支持且用户要求）
    if getattr(args, "use_flash_attention", False) and device.startswith("cuda"):
        try:
            # 检查是否有 flash-attn 库
            import flash_attn
            gen_cfg["use_flash_attention_2"] = True
            print("[INFO] 已启用 Flash Attention 2 加速")
        except ImportError:
            print("[WARN] Flash Attention 2 未安装，忽略 --use_flash_attention 选项")

    generator = MolAwareGenerator2()
    generator.load(gen_cfg)
    
    # 根据参数控制 verbose_logging（默认启用，但可通过 --disable_verbose_logging 禁用以加速）
    if hasattr(generator.model, '_verbose_logging'):
        generator.model._verbose_logging = not getattr(args, "disable_verbose_logging", False)
        if not generator.model._verbose_logging:
            print("[INFO] 已禁用详细日志输出以加速推理")
    # 构建 evaluator 用的模型调用函数
    call_model_func = build_call_model(
        generator,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        tag_func=None,  # generate 函数应该自动处理 tagging
        realtime_mol=bool(args.realtime_mol),
        enable_gnn_trigger=getattr(args, "enable_gnn_trigger", False),
        repetition_penalty=getattr(args, "repetition_penalty", 1.06),
        no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 3),
        verbose_gnn=getattr(args, "verbose_gnn", False),
    )
    
    # 保存tokenizer引用，供后续提取函数使用
    tokenizer_for_extraction = generator.tokenizer if hasattr(generator, 'tokenizer') and generator.tokenizer is not None else None

    # 先收集所有 jsonl 任务文件
    files = sorted([p for p in raw_data_dir.iterdir() if evaluator.is_target_jsonl(p)])

    # 如果命令行指定了 --include_tasks，就按任务名过滤一遍
    if getattr(args, "include_tasks", ""):
        allow = {t.strip() for t in args.include_tasks.split(",") if t.strip()}
        before_n = len(files)
        files = [
            p for p in files
            if evaluator.task_name_from_filename(p) in allow
        ]
        print(
            f"[INFO] 启用 include_tasks 过滤：原有 {before_n} 个任务文件，"
            f"保留 {len(files)} 个：{', '.join(sorted(allow))}"
        )

    if not files:
        print(f"[WARN] 未在 {raw_data_dir} 找到要评测的 *.jsonl 文件 "
              f"(include_tasks={getattr(args, 'include_tasks', '')})")
        return output_dir

    print(f"[INFO] 将评测 {len(files)} 个任务文件：")
    for p in files:
        task_name = evaluator.task_name_from_filename(p)
        print(f"  - {p.name}  (task={task_name})")

    total = 0
    summary = []

    # 这里按 task 粒度给一个总的 tqdm（推理阶段）
    for jsonl_path in tqdm(files, desc="Inference over tasks", unit="task"):
        task = evaluator.task_name_from_filename(jsonl_path)
        template_path = template_dir / f"{task}.json"
        if not template_path.exists():
            print(f"[WARN] 模板不存在：{template_path}，跳过该任务")
            continue

        # --- per-task prompt injection (few-shot + strict suffix) ---
        fewshot_prefix, fewshot_meta = _build_fewshot_prefix_and_meta(
            task=task,
            few_shot_k=getattr(args, "few_shot", 0),
            few_shot_dir=getattr(args, "few_shot_dir", ""),
            seed=getattr(args, "few_shot_seed", 42),
            max_chars=getattr(args, "few_shot_max_chars", 6000),
            store_full=True,
        )
        suffix = _build_prompt_suffix_for_task(task, getattr(args, "prompt_style", "default"))

        # 存储完整prompt列表，用于后续写入jsonl
        full_prompts_list = []
        
        def _call_model_task(prompts, _base=call_model_func, _pre=fewshot_prefix, _suf=suffix, _task=task):
            """包装 call_model_func，添加 fewshot prefix 和 suffix，并传递 task 参数。"""
            full_prompts = []
            
            for p in prompts:
                full_prompt = p
                if _pre:
                    full_prompt = _pre + "\n\n" + full_prompt
                if _suf:
                    full_prompt = full_prompt + _suf
                full_prompts.append(full_prompt)
            
            # 保存完整prompt列表供evaluate_file使用
            full_prompts_list.clear()
            full_prompts_list.extend(full_prompts)
            
            # 调用 call_model_func 并传递 task 参数
            return _base(full_prompts, task=_task)

        task_name, n = evaluator.evaluate_file(
            call_model_func=_call_model_task,
            jsonl_path=jsonl_path,
            template_path=template_path,
            output_dir=output_dir,
            is_chat_model=True,
            cot=args.cot,
            verbose_every=args.verbose_every,
            data_limit=args.data_limit,
            full_prompts_getter=lambda: full_prompts_list.copy() if full_prompts_list else None,
            tokenizer=tokenizer_for_extraction,  # 传递tokenizer用于动态提取
        )

        # --- write few-shot meta + attach to output jsonl ---
        task_outfile = output_dir / f"{task_name}.jsonl"
        if fewshot_meta is not None:
            meta_file = _write_fewshot_meta_file(output_dir, task_name, fewshot_meta)
            _attach_fewshot_meta_to_task_output(task_outfile, meta_file, fewshot_meta)
            print(f"[INFO] Task={task_name} few-shot meta saved: {meta_file}")

        summary.append(
            {"task": task_name, "num_samples": n, "outfile": str(task_outfile)}
        )
        total += n

    summary_path = output_dir / "eval_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": args.molaware_ckpt,
                "raw_data_dir": str(raw_data_dir),
                "template_dir": str(template_dir),
                "output_dir": str(output_dir),
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "cot": args.cot,
                "realtime_mol": bool(args.realtime_mol),
                "few_shot": getattr(args, "few_shot", 0),
                "few_shot_dir": getattr(args, "few_shot_dir", ""),
                "few_shot_seed": getattr(args, "few_shot_seed", 42),
                "prompt_style": getattr(args, "prompt_style", "default"),
                "total_samples": total,
                "per_task": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] 推理汇总文件已写入：{summary_path}")
    print(f"[INFO] 推理阶段完成 {len(summary)} 个任务，共 {total} 条样本。")

    return output_dir


# ========= 四：打分逻辑（基本沿用原 score_smolinstruct.py，增加 tqdm & 多进程） =========

def _is_task_file(p: Path) -> bool:
    return p.is_file() and p.suffix == ".jsonl" and p.name != "eval_summary.json"


def _read_task_file(
    path: Path,
    replace_semicolon: bool = False,
    task_name: str = "",
    use_tqdm: bool = False,  # 与参考脚本保持一致，默认不使用 tqdm
) -> Tuple[List[Optional[List[str]]], List[List[str]]]:
    """
    返回：
      pred_list: List[Optional[List[str]]]
      gold_list: List[List[str]]

    metrics 期望的是"列表的列表"，即每样本可包含 k 个备选预测/答案（此处 k=1）
    与参考脚本保持一致，不使用 tqdm 进度条
    """
    pred_list: List[Optional[List[str]]] = []
    gold_list: List[List[str]] = []

    print(f"replace_semicolon={replace_semicolon}，读取文件：{path}")
    
    # 与参考脚本保持一致，不使用 tqdm
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # gold 统一收成 list[str]（即使只有 1 个）
            gold = obj["gold"]
            if isinstance(gold, str):
                golds = [gold]
            elif isinstance(gold, list):
                # 兼容极少数情况：gold 已经是 list
                golds = [str(x) for x in gold]
            else:
                golds = [str(gold)]

            # 可选替换分号
            if replace_semicolon:
                golds = [g.replace(";", ".") for g in golds]

            # pred：None 或 str
            # 如果 pred 为空或无效，尝试从 answer_only 或 raw_output 提取（使用优化过的提取函数）
            pred = obj.get("pred", None)
            task_name_from_obj = obj.get("task", Path(path).stem)
            
            if pred is None or (isinstance(pred, str) and not pred.strip()):
                # 优先从 answer_only 提取
                answer_only = obj.get("answer_only", None)
                raw_output = obj.get("raw_output", "")
                
                if EXTRACT_PREDICTION_AVAILABLE:
                    try:
                        # 如果提供了answer_only，优先使用它
                        if answer_only and isinstance(answer_only, str) and answer_only.strip():
                            pred = extract_prediction_from_raw(
                                raw_output, task_name_from_obj,
                                text_tasks=TEXT_TASKS,
                                smiles_tasks=SMILES_TASKS,
                                formula_element_tasks=FORMULA_ELEMENT_TASKS,
                                formula_split_tasks=FORMULA_SPLIT_TASKS,
                                number_tasks=NUMBER_TASKS,
                                boolean_tasks=BOOLEAN_TASKS,
                                answer_only=answer_only,  # 传递answer_only作为优先源
                            )
                        elif raw_output:
                            pred = extract_prediction_from_raw(
                                raw_output, task_name_from_obj,
                                text_tasks=TEXT_TASKS,
                                smiles_tasks=SMILES_TASKS,
                                formula_element_tasks=FORMULA_ELEMENT_TASKS,
                                formula_split_tasks=FORMULA_SPLIT_TASKS,
                                number_tasks=NUMBER_TASKS,
                                boolean_tasks=BOOLEAN_TASKS,
                            )
                    except Exception:
                        pred = None
            
            if pred is None or (isinstance(pred, str) and not pred.strip()):
                pred_list.append(None)
            else:
                if isinstance(pred, str):
                    preds = [pred]
                elif isinstance(pred, list):
                    preds = [str(x) for x in pred]
                else:
                    preds = [str(pred)]
                if replace_semicolon:
                    preds = [p.replace(";", ".") for p in preds]
                pred_list.append(preds)

            gold_list.append(golds)

    return pred_list, gold_list


def _score_one_task(task: str, preds: List, golds: List) -> Dict:
    """按任务类型选择 metrics 函数并返回指标 dict。"""
    if task in SMILES_TASKS:
        if task in SMILES_TASKS_MULTIMETRIC:
            return calculate_smiles_metrics(preds, golds, metrics=("exact_match", "fingerprint", "multiple_match"))
        return calculate_smiles_metrics(preds, golds)

    if task in TEXT_TASKS:
        return calculate_text_metrics(preds, golds)

    if task in FORMULA_ELEMENT_TASKS:
        return calculate_formula_metrics(preds, golds, metrics=("element_match",))

    if task in FORMULA_SPLIT_TASKS:
        return calculate_formula_metrics(preds, golds, metrics=("split_match",))

    if task in NUMBER_TASKS:
        return calculate_number_metrics(preds, golds)

    if task in BOOLEAN_TASKS:
        return calculate_boolean_metrics(preds, golds)

    # 未知任务：直接抛错，提醒补逻辑
    raise Exception(f"未知任务 {task}，请在 _score_one_task 中添加对应逻辑")


def _score_single_file(args) -> Tuple[str, Dict]:
    """
    供多进程使用的单文件评测函数。
    args: (fp_str,)
    """
    fp_str = args[0] if isinstance(args, tuple) else args
    fp = Path(fp_str)
    task = fp.stem
    preds, golds = _read_task_file(
        fp,
        replace_semicolon=(task in DEFAULT_REPLACE_SEMICOLON_TASKS),
    )
    metrics = _score_one_task(task, preds, golds)
    return task, metrics


def run_scoring(pred_dir: Path, save_json: str = "", score_workers: int = 1, skip_tasks: str = "") -> Dict[str, Dict]:
    """对 prediction_dir 下所有 <task>.jsonl 文件打分并打印 summary。"""
    pred_dir = pred_dir.expanduser().resolve()
    assert pred_dir.is_dir(), f"Not a directory: {pred_dir}"

    files = [p for p in pred_dir.iterdir() if _is_task_file(p)]
    tasks = [p.stem for p in files]

    # hack：跳过 property_prediction-sider（如果你不想评它）
    if "property_prediction-sider" in tasks:
        print("[INFO] 跳过 property_prediction-sider")
        files = [p for p in files if p.stem != "property_prediction-sider"]
        tasks = [t for t in tasks if t != "property_prediction-sider"]

    # 支持跳过指定任务
    skip_set = set()
    if skip_tasks:
        skip_set = {t.strip() for t in skip_tasks.split(",") if t.strip()}
        if skip_set:
            print(f"[INFO] 将跳过以下任务: {', '.join(sorted(skip_set))}")
            files = [p for p in files if p.stem not in skip_set]
            tasks = [t for t in tasks if t not in skip_set]

    all_results: Dict[str, Dict] = {}
    if not files:
        print(f"[WARN] 未在目录中发现任务文件：{pred_dir}")
        return all_results

    print(f"[INFO] 将评测 {len(files)} 个任务：")
    for p in files:
        print("  -", p.name)

    # 单进程：与参考脚本保持一致，不使用 tqdm
    if score_workers <= 1:
        for fp in files:
            task = fp.stem
            print(f"\n===== {task} =====")
            preds, golds = _read_task_file(
                fp,
                replace_semicolon=(task in DEFAULT_REPLACE_SEMICOLON_TASKS),
            )
            metrics = _score_one_task(task, preds, golds)
            all_results[task] = metrics

            for k, v in metrics.items():
                print(f"{k}:\t{v}")
            print()
    else:
        # 多进程：按 task 粒度并行，显示整体 task 进度条
        n_workers = cpu_count() if score_workers <= 0 else score_workers
        print(f"[INFO] 使用多进程并行评测，workers={n_workers}")
        args_list = [str(fp) for fp in files]
        with Pool(processes=n_workers) as pool:
            for task, metrics in tqdm(
                pool.imap_unordered(_score_single_file, args_list),
                total=len(args_list),
                desc="Scoring tasks",
            ):
                all_results[task] = metrics
                print(f"\n===== {task} =====")
                for k, v in metrics.items():
                    print(f"{k}:\t{v}")
                print()

    # 可选保存 JSON
    if save_json:
        outp = Path(save_json).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 指标已写入：{outp}")

    # 打印总览
    print("\n===== All Results Summary =====")

    ordered_tasks = [
        "molecule_generation",
        "molecule_captioning",
        "name_conversion-i2f",
        "name_conversion-i2s",
        "name_conversion-s2f",
        "name_conversion-s2i",
        "forward_synthesis",
        "retrosynthesis",
        "property_prediction-bbbp",
        "property_prediction-clintox",
        "property_prediction-esol",
        "property_prediction-hiv",
        "property_prediction-lipo",
    ]

    for task in ordered_tasks:
        if task not in all_results:
            continue
        metrics = all_results[task]

        if task == "molecule_generation":
            num_all = metrics.get("num_all", 0)
            rdk = metrics.get("t1_rdk_fps", 0) * 100
            print(f"molecule_generation\t{num_all}\tfps={rdk:.1f}")

        elif task == "molecule_captioning":
            num_all = metrics.get("num_all", 0)
            bleu4 = metrics.get("bleu4", 0) * 100
            rouge_l = metrics.get("rouge_l", 0) * 100
            print(f"molecule_captioning\t{num_all}\tbleu-4={bleu4:.1f}\trouge-l={rouge_l:.1f}")

        elif task == "name_conversion-i2f":
            num_all = metrics.get("num_all", 1)
            ele = metrics.get("num_t1_ele_match", 0) / num_all * 100
            print(f"i2f\t{num_all}\tEM={ele:.1f}")

        elif task == "name_conversion-i2s":
            num_all = metrics.get("num_all", 0)
            rdk = metrics.get("t1_rdk_fps", 0) * 100
            print(f"i2s\t{num_all}\tfps={rdk:.1f}")

        elif task == "name_conversion-s2f":
            num_all = metrics.get("num_all", 1)
            ele = metrics.get("num_t1_ele_match", 0) / num_all * 100
            print(f"s2f\t{num_all}\tEM={ele:.1f}")

        elif task == "name_conversion-s2i":
            num_all = metrics.get("num_all", 1)
            split = metrics.get("num_t1_split_match", 0) / num_all * 100
            print(f"s2i\t{num_all}\tEM={split:.1f}")

        elif task == "forward_synthesis":
            num_all = metrics.get("num_all", 0)
            rdk = metrics.get("t1_rdk_fps", 0) * 100
            print(f"forward_synthesis\t{num_all}\tfps={rdk:.1f}")

        elif task == "retrosynthesis":
            num_all = metrics.get("num_all", 0)
            rdk = metrics.get("t1_rdk_fps", 0) * 100
            print(f"Retrosynthesis\t{num_all}\tfps={rdk:.1f}")

        elif task == "property_prediction-bbbp":
            num_all = metrics.get("num_all", 0)
            f1 = metrics.get("f1_score", 0) * 100
            mcc = metrics.get("mcc", 0)
            print(f"bbbp\t{num_all}\tf1={f1:.1f}\tmcc={mcc:.2f}")

        elif task == "property_prediction-clintox":
            num_all = metrics.get("num_all", 0)
            f1 = metrics.get("f1_score", 0) * 100
            mcc = metrics.get("mcc", 0)
            print(f"clintox\t{num_all}\tf1={f1:.1f}\tmcc={mcc:.2f}")

        elif task == "property_prediction-esol":
            rmse = metrics.get("RMSE", 0)
            print(f"esol\tRMSE={rmse:.2f}")

        elif task == "property_prediction-hiv":
            num_all = metrics.get("num_all", 0)
            f1 = metrics.get("f1_score", 0) * 100
            mcc = metrics.get("mcc", 0)
            print(f"hiv\t{num_all}\tf1={f1:.1f}\tmcc={mcc:.2f}")

        elif task == "property_prediction-lipo":
            rmse = metrics.get("RMSE", 0)
            print(f"lipo\tRMSE={rmse:.2f}")

    print("\n[INFO] 所有任务指标按指定顺序打印完成。")

    return all_results


def run_external_scoring(pred_dir: Path, script_path: str):
    """调用外部评分脚本。"""
    cmd = [script_path, str(pred_dir)]
    print(f"[INFO] 调用外部评分脚本：{' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ========= 五：参数解析 & 主入口 =========

def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 MolAwareGenerator2 在 SMolInstruct 上批量推理并自动打分（支持 MLP 标 <mol> + GNN pipeline）。"
    )
    # 推理相关
    parser.add_argument("--raw_data_dir", type=str, required=True, help="原始数据目录（包含 *.jsonl）")
    parser.add_argument("--template_dir", type=str, required=True, help="模板目录（包含 *.json）")
    parser.add_argument("--output_dir", type=str, default="predictions_molaware",
                        help="推理结果与指标输出目录")
    parser.add_argument("--molaware_ckpt", type=str, required=True,
                        help="MolAware checkpoint 根目录（包含 llm/ extras/）")
    parser.add_argument("--base_llm_path", type=str, default="",
                        help="基础 LLM 路径（用于加载 tokenizer，如果 checkpoint 中无法加载）")
    parser.add_argument("--token_classifier_path", type=str, default="",
                        help="MLP token classifier 权重路径（如果为空则不做 MLP 标注）")
    parser.add_argument("--tag_llm_path", type=str, default="",
                        help="用于 MLP 标注的 LLM 路径（默认用 molaware_ckpt/llm）")
    parser.add_argument("--offline_tagging_max_length", type=int, default=512,
                        help="MLP 标注时的最大序列长度")
    parser.add_argument("--offline_tagging_batch_size", type=int, default=16,
                        help="MLP 标注时的 batch size（默认16以加速推理）")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.06, help="重复惩罚系数（默认1.06）")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="防止重复n-gram的大小（默认3）")
    parser.add_argument("--data_limit", type=int, default=0, help="每个任务限制样本数（0 表示全部）")
    parser.add_argument("--cot", action="store_true",
                        help="是否启用 chain-of-thought，不添加 'Please only output the answer.'")
    parser.add_argument("--verbose_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="推理时一次送入模型的样本数（仅在模型支持批量 generate 时生效，默认8以加速推理）")
    parser.add_argument(
        "--include_tasks", type=str, default="",
        help="只跑这些任务，逗号分隔，比如: molecule_generation,forward_synthesis",
    )
    parser.add_argument(
        "--realtime_mol", type=int, default=1,
        help="1=开启 MolAware realtime_mol(GNN pipeline)，0=关闭（baseline）",
    )
    parser.add_argument("--few_shot", type=int, default=0,
                        help="Number of few-shot examples to prepend per task (0=disable).")
    parser.add_argument("--few_shot_dir", type=str, default="",
                        help="Dev set directory (jsonl) used to sample few-shot examples.")
    parser.add_argument("--few_shot_seed", type=int, default=42,
                        help="Random seed for few-shot sampling.")
    parser.add_argument("--few_shot_max_chars", type=int, default=6000,
                        help="Max characters for few-shot prefix (to control context length).")
    parser.add_argument("--prompt_style", type=str, default="default",
                        choices=["default", "compact", "strict"],
                        help="Evaluation prompt style; 'compact/strict' append stricter output-format suffix.")
    parser.add_argument("--enable_gnn_trigger", action="store_true",
                        help="prompt 增加复读 SMILES 触发 GNN")
    parser.add_argument("--verbose_gnn", action="store_true",
                        help="realtime_mol=1 时更详细日志（会降低推理速度）")
    parser.add_argument("--disable_verbose_logging", action="store_true",
                        help="禁用模型内部的详细日志输出以加速推理")
    parser.add_argument("--detection_interval", type=int, default=5,
                        help="实体检测间隔（每N个边界token检测一次，默认5，增大可加速但可能降低精度）")
    parser.add_argument("--use_flash_attention", action="store_true",
                        help="使用 Flash Attention 2 加速（如果模型支持）")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="启用 thinking 模式（仅适用于支持该功能的模型，如 Intern-S1，默认关闭）")
    parser.add_argument("--device_ids", type=str, default="",
                        help="逗号分隔GPU编号，例如 '5,6' 并行推理（暂未实现多GPU并行）")
    parser.add_argument("--seed", type=int, default=42,
                        help="全局随机种子")

    # 打分相关
    parser.add_argument("--save_json", type=str, default="",
                        help="将所有任务指标写入该 JSON 文件（可选）")
    parser.add_argument("--prediction_dir", type=str, default="",
                        help="若仅想对已有结果打分，可单独指定预测目录；为空则使用 --output_dir")
    parser.add_argument("--skip_inference", action="store_true",
                        help="只打分，不重新跑推理（需要 --prediction_dir 或 --output_dir 已存在结果）")
    parser.add_argument("--skip_scoring", action="store_true",
                        help="只跑推理，不打分")
    parser.add_argument("--score_workers", type=int, default=1,
                        help="评测阶段使用的 CPU 进程数：1=单进程（每个 task 带样本级 tqdm），>1=多进程按 task 并行")
    parser.add_argument("--skip_tasks", type=str, default="",
                        help="跳过这些任务（逗号分隔）")
    parser.add_argument("--use_external_scoring", action="store_true",
                        help="使用外部脚本打分")
    parser.add_argument("--external_score_script", type=str,
                        default="/data1/lvchangwei/LLM/SMolInstruct/score_smolinstruct.sh",
                        help="外部评分脚本路径")

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置全局随机种子
    seed = int(getattr(args, "seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Step 1: 推理
    if args.skip_inference:
        pred_dir = Path(args.prediction_dir or args.output_dir).expanduser().resolve()
        print(f"[INFO] 跳过推理，直接从 {pred_dir} 读取预测进行打分。")
    else:
        pred_dir = run_inference(args)

    # Step 2: 打分
    if args.skip_scoring:
        print("[INFO] 已完成推理，但根据参数设置跳过打分。")
        return

    if getattr(args, "use_external_scoring", False):
        run_external_scoring(pred_dir, args.external_score_script)
    else:
        run_scoring(
            pred_dir,
            save_json=args.save_json,
            score_workers=args.score_workers,
            skip_tasks=getattr(args, "skip_tasks", "")
        )


if __name__ == "__main__":
    main()
