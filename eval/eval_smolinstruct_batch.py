#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测 Instruct/基座模型 在 SMolInstruct 多任务数据上的零样本表现，并生成 compute_metrics.py 可直接读取的预测文件。
修改版：
- 不再使用 data_dir 下现成 prompt
- 基于 raw_data_dir 和 template_dir 生成 prompt：
  * 每条数据随机选一个模板文件中的一条模板，将 <INPUT> 替换为数据中的 input
  * 若 --cot=False，则在生成的 prompt 末尾加上 " Please only output the answer."
- 增加参数：
  * --raw_data_dir
  * --template_dir
  * --template_seed
  * --cot
"""

import argparse
import json
import os
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from functools import partial

from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ========== 任务集合（与 compute_metrics.py 对齐） ==========
SMILES_TASKS = {
    "forward_synthesis",
    "retrosynthesis",
    "molecule_generation",
    "name_conversion-i2s",
}
NUMERIC_TASKS = {
    "property_prediction-esol",
    "property_prediction-lipo",
}
BOOLEAN_TASKS = {
    "property_prediction-bbbp",
    "property_prediction-clintox",
    "property_prediction-hiv",
    "property_prediction-sider",
}
SMILES_TASKS_MULTIMETRIC = {"retrosynthesis"}
TEXT_TASKS = {
    "molecule_captioning",
}
FORMULA_TASKS = {
    "name_conversion-i2f",
    "name_conversion-s2f",
    "name_conversion-s2i",
}


# ========== 基础工具 ==========
def is_target_jsonl(p: Path) -> bool:
    """识别目标 jsonl：*.jsonl"""
    return p.is_file() and p.suffix.lower() == ".jsonl"


def task_name_from_filename(p: Path) -> str:
    """例：property_prediction-esol.jsonl -> property_prediction-esol"""
    return p.stem


def load_jsonl_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```") and s.endswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
        s = s.strip()
    s = re.sub(r'^\s*(?:Answer\s*:|The answer is\s*: )\s*', "", s, flags=re.I)
    s = s.strip().strip('"').strip("'").strip()
    return s


def extract_number(text: str) -> str:
    if not text:
        return ""
    up = strip_code_fences(text)
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', up)
    return m.group(0) if m else ""

def extract_molecular_formula(text: str) -> str:
    """
    从给定文本中提取第一个看起来像“分子式”的字符串（如 C16H17N5 或 C24H18N2O3S2 …）。
    如果找不到则返回空字符串。
    """
    # 匹配模式：
    # - 首字母大写 A-Z 开始，后跟可能的小写字母 a-z（如 “Cl”, “Br” 等元素符号）
    # - 紧接一个或多个数字（或无数字，表示 1 个原子）
    # - 重复上述结构若干次
    # - 整体至少两个元素，或长度较长
    pattern = re.compile(r'\b(?:[A-Z][a-z]?[\d]*){2,}\b')
    match = pattern.search(text)
    if match:
        return match.group(0)
    return ""


def extract_yes_no(text: str) -> str:
    if not text:
        return ""
    t = strip_code_fences(text).lower()
    m = re.search(r'\b(yes|no)\b', t)
    if m:
        return m.group(1)
    m = re.search(r'\b(true|false)\b', t)
    if m:
        return "yes" if m.group(1) == "true" else "no"
    m = re.search(r'\b(y|n)\b', t)
    if m:
        return "yes" if m.group(1) == "y" else "no"
    return ""


_SMILES_CHARS = r"A-Za-z0-9@+\-\[\]\(\)=#\$\/\\\.\:%"


def extract_smiles(text: str) -> str:
    """从生成文本中提取更可靠的 SMILES 字符串。"""
    import re
    if not text:
        return ""
    s = strip_code_fences(text or "")
    lines = s.splitlines() if s else []

    # 候选：完全由 SMILES 字符组成
    pattern = re.compile(rf'([{_SMILES_CHARS}]+)')
    candidates = []
    for ln in lines:
        candidates.extend(pattern.findall(ln))
    if not candidates:
        return ""

    # 过滤：至少长度 2，且包含数字或典型 SMILES 特征字符
    def is_potential_smiles(tok: str) -> bool:
        if len(tok) < 2:
            return False
        has_digit = any(c.isdigit() for c in tok)
        has_special = any(c in "=#()[]@+/-\\" for c in tok)
        return has_digit or has_special

    filtered = [c for c in candidates if is_potential_smiles(c)]
    if not filtered:
        return ""

    # 选最长那个
    best = max(filtered, key=len)
    if best.endswith('.'):
        best = best[:-1]
    return best


def clean_text(text: str) -> str:
    s = strip_code_fences(text)
    return s.splitlines()[0].strip() if s else ""


def postprocess_by_task(task: str, raw_output: str) -> List[Optional[str]]:
    # if task in NUMERIC_TASKS:
    #     num = extract_number(raw_output)
    #     return [num if num is not None else ""]
    # if task in BOOLEAN_TASKS:
    #     yn = extract_yes_no(raw_output)
    #     return [yn if yn is not None else ""]
    # if task in SMILES_TASKS:
    #     smi = extract_smiles(raw_output)
    #     return [smi if smi is not None else ""]
    # if task in TEXT_TASKS:
    #     return [clean_text(raw_output)]
    # if task in FORMULA_TASKS:
    #     return [clean_text(raw_output)]
    # raise Exception(f"未知任务 {task}，请在 postprocess_by_task 中添加对应逻辑")

    # 先不用list
    if task in NUMERIC_TASKS:
        num = extract_number(raw_output)
        return num
    if task in BOOLEAN_TASKS:
        yn = extract_yes_no(raw_output)
        return yn
    if task in SMILES_TASKS:
        smi = extract_smiles(raw_output)
        return smi
    if task in TEXT_TASKS:
        return clean_text(raw_output)
    if task in FORMULA_TASKS:
        if task == "name_conversion-s2f" or task == "name_conversion-i2f":
            return extract_molecular_formula(raw_output)
        else:
            return clean_text(raw_output)
    raise Exception(f"未知任务 {task}，请在 postprocess_by_task 中添加对应逻辑")


# ========== 推理 ==========
def call_local_transformers_batch(
    tokenizer,
    model,
    prompts: List[Union[str, List[Dict[str, str]]]],
    is_chat: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if is_chat:
        rendered: List[str] = []
        for p in prompts:
            if isinstance(p, str):
                messages = [{"role": "user", "content": p}]
            else:
                messages = p
            rendered.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        enc = tokenizer(rendered, return_tensors="pt", padding=True)
    else:
        raise Exception("当前仅支持 is_chat=True 的模型")

    inputs = {k: v.to(model.device) for k, v in enc.items()}

    do_sample = temperature > 0
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "repetition_penalty": 1.06,
        "no_repeat_ngram_size": 3,
    }
    if do_sample:
        generate_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
        })

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    outs = tokenizer.batch_decode(output_ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs], output_ids


# ========== 新逻辑：基于模板构建 prompt ==========
def build_prompts_from_template(raw_rows: List[Dict], template_list: List[Dict], cot: bool) -> List[str]:
    prompts = []
    for row in raw_rows:
        template = random.choice(template_list)
        # 支持多种数据格式：
        # 1. SMolInstruct 格式：有 "input" 字段
        # 2. messages 格式：从 messages 中提取 user content
        if "input" in row:
            input_str = row.get("input", "")
        elif "messages" in row:
            # 从 messages 格式中提取 user 消息
            messages = row.get("messages", [])
            user_msg = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
            input_str = user_msg
        else:
            input_str = ""
        
        prompt = template["input"].replace("<INPUT>", input_str)
        # 不填充 OUTPUT，占位即可
        prompt = prompt.replace("<OUTPUT>", "")
        if not cot:
            # 使用更明确的格式说明
            prompt = prompt.strip() + "\n\nPlease only output the answer without any explanation or additional text."
        prompts.append(prompt)
    return prompts


# ========== 主流程 ==========
def evaluate_file(
    call_model_func,
    jsonl_path: Path,
    template_path: Path,
    output_dir: Path,
    is_chat_model: bool,
    cot: bool,
    verbose_every: int = 50,
    data_limit: int = 0,
    full_prompts_getter=None,  # 可选的函数，用于获取完整prompt列表（包含few-shot prefix）
    tokenizer=None,  # 可选的tokenizer，用于动态获取special tokens和assistant标记
) -> Tuple[str, int]:
    task = task_name_from_filename(jsonl_path)
    rows = load_jsonl_rows(jsonl_path)
    if data_limit > 0:
        rows = rows[:data_limit]
    print(f"[INFO] Evaluating {task}, {len(rows)} samples from {jsonl_path}")

    template_list = json.loads(template_path.read_text(encoding="utf-8"))
    out_path = output_dir / f"{task}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    batch_size = 32

    with out_path.open("w", encoding="utf-8") as fw:
        for start in tqdm(range(0, len(rows), batch_size), desc=f"Evaluating {task}"):
            end = min(start + batch_size, len(rows))
            batch = rows[start:end]
            prompts = build_prompts_from_template(batch, template_list, cot=cot)
            outputs, raw_outputs_full = call_model_func(prompts=prompts)
            
            # 如果返回了完整的原始输出，使用它；否则使用处理后的输出
            if raw_outputs_full is not None and len(raw_outputs_full) == len(outputs):
                raw_outputs = raw_outputs_full
            else:
                # 向后兼容：如果没有返回完整原始输出，使用处理后的输出
                raw_outputs = outputs
            
            # 获取完整prompt列表（如果提供了getter函数，如few-shot场景）
            full_prompts = None
            if full_prompts_getter is not None:
                full_prompts = full_prompts_getter()
                # 确保长度匹配
                if full_prompts and len(full_prompts) == len(raw_outputs):
                    prompts_to_use = full_prompts
                else:
                    prompts_to_use = prompts
            else:
                prompts_to_use = prompts

            for local_idx, (row, raw_output, prompt) in enumerate(zip(batch, raw_outputs, prompts_to_use), start=1):
                # ✅ 不再二次处理：raw_output 就是 generate 返回的文本
                # ✅ answer_only 与 raw_output 完全相同（即 raw_answer）
                answer_only = raw_output

                # 从 answer_only 提取 pred（用于打分），不改动 raw_output/answer_only 本身
                try:
                    # 导入任务类型定义
                    try:
                        from eval.eval_smolinstruct import TEXT_TASKS, SMILES_TASKS, FORMULA_ELEMENT_TASKS, FORMULA_SPLIT_TASKS, NUMBER_TASKS, BOOLEAN_TASKS
                    except ImportError:
                        # 如果导入失败，使用默认任务集合
                        TEXT_TASKS = {"molecule_captioning"}
                        SMILES_TASKS = {"forward_synthesis", "retrosynthesis", "molecule_generation", "name_conversion-i2s"}
                        FORMULA_ELEMENT_TASKS = {"name_conversion-i2f", "name_conversion-s2f"}
                        FORMULA_SPLIT_TASKS = {"name_conversion-s2i"}
                        NUMBER_TASKS = {"property_prediction-esol", "property_prediction-lipo"}
                        BOOLEAN_TASKS = {"property_prediction-bbbp", "property_prediction-clintox", "property_prediction-hiv", "property_prediction-sider"}
                    
                    from eval.extract_prediction import extract_prediction_from_raw
                    # 使用answer_only提取pred
                    pred = extract_prediction_from_raw(
                        raw_output=None,  # 不使用raw_output
                        task_name=task,
                        answer_only=answer_only,  # 使用answer_only
                        text_tasks=TEXT_TASKS,
                        smiles_tasks=SMILES_TASKS,
                        formula_element_tasks=FORMULA_ELEMENT_TASKS,
                        formula_split_tasks=FORMULA_SPLIT_TASKS,
                        number_tasks=NUMBER_TASKS,
                        boolean_tasks=BOOLEAN_TASKS,
                    )
                except (ImportError, Exception) as e:
                    # 如果提取失败，使用原来的postprocess逻辑
                    pred = postprocess_by_task(task, raw_output)
                
                # 支持多种数据格式：
                # 1. SMolInstruct 格式：有 "gold" 和 "input" 字段
                # 2. messages 格式：从 messages 中提取 assistant content 作为 gold
                gold = row.get("gold") or row.get("output")
                if not gold and "messages" in row:
                    # 从 messages 格式中提取 assistant 消息作为 gold
                    messages = row.get("messages", [])
                    gold = next((msg.get("content", "") for msg in messages if msg.get("role") == "assistant"), "")
                
                input_str = row.get("input", "")
                if not input_str and "messages" in row:
                    # 从 messages 格式中提取 user 消息作为 input
                    messages = row.get("messages", [])
                    input_str = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "")
                
                item_out = {
                    "prompt": prompt,  # prompt现在已经是完整prompt（包含few-shot prefix和suffix）
                    "gold": gold,
                    "pred": pred,
                    "input": input_str,
                    "raw_output": raw_output,  # 原始完整输出
                    "answer_only": answer_only,  # 只有answer的版本（从assistant提取，移除think）
                    "sample_id": row.get("sample_id"),
                    "task": task,
                }
                # print(f"------------")
                # print(f"prompt: {prompt}")
                # print(f"------------")
                # print(f"raw_output: {raw_output}")
                # print(f"------------")
                # print(f"pred: {pred}")
                if "target" in row:
                    item_out["target"] = row.get("target")
                fw.write(json.dumps(item_out, ensure_ascii=False) + "\n")
                n += 1

                global_idx = start + local_idx
                # if verbose_every and (global_idx) % verbose_every == 0:
                #     print(f"[{task}] {global_idx} samples processed. Example:")
                #     print("PROMPT:", (prompt[:300] + "...") if len(prompt) > 300 else prompt)
                #     print("RAW  :", (raw_output[:300] + "...") if len(raw_output) > 300 else raw_output)
                #     print("PRED :", pred)

    print(f"[DONE] {task}: wrote {n} lines -> {out_path}")
    return task, n


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on SMolInstruct raw_data + template to generate prompts and predictions for compute_metrics.py.")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="原始数据目录（包含 *.jsonl）")
    parser.add_argument("--template_dir", type=str, required=True, help="模板目录（包含 *.json）")
    parser.add_argument("--data_limit", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="predictions_smol")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--sampling_seed", type=int, default=42)
    parser.add_argument("--template_seed", type=int, default=42)
    parser.add_argument("--cot", action="store_true", help="是否启用 chain-of-thought，不添加 'Please only output the answer.'")
    parser.add_argument("--verbose_every", type=int, default=50)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if args.dtype == "auto":
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    print(f"[INFO] Loading model: {args.model} on {device} (dtype={torch_dtype})")

    if args.data_limit < 0:
        raise ValueError("--data_limit 需为非负整数")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        assert tokenizer.eos_token is not None
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(model, "config"):
        model.config.use_cache = True
    if hasattr(model, "generation_config"):
        try:
            model.generation_config.use_cache = True
        except Exception:
            pass
    model.eval()

    is_chat = "instruct" in args.model.lower() or "chat" in args.model.lower()
    print(f"[INFO] Is Chat Model: {is_chat}")

    if args.temperature > 0:
        torch.manual_seed(int(args.sampling_seed))
    random.seed(args.template_seed)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    call_model = partial(
        call_local_transformers_batch,
        tokenizer=tokenizer,
        model=model,
        is_chat=is_chat,
        **sampling_params,
    )

    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    template_dir = Path(args.template_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    files = sorted([p for p in raw_data_dir.iterdir() if is_target_jsonl(p)])

    if not files:
        print(f"[WARN] 未在目录中找到 *.jsonl ：{raw_data_dir}")
        return

    print(f"[INFO] 将评测 {len(files)} 个任务文件：")
    for p in files:
        print(f"  - {p.name}")

    total = 0
    summary = []
    for jsonl_path in files:
        task = task_name_from_filename(jsonl_path)
        template_path = template_dir / f"{task}.json"
        if not template_path.exists():
            print(f"[WARN] 模板不存在：{template_path}，跳过该任务")
            continue

        task_name, n = evaluate_file(
            call_model_func=call_model,
            jsonl_path=jsonl_path,
            template_path=template_path,
            output_dir=output_dir,
            is_chat_model=is_chat,
            cot=args.cot,
            verbose_every=args.verbose_every,
            data_limit=args.data_limit,
        )
        summary.append({"task": task_name, "num_samples": n, "outfile": str(output_dir / f"{task_name}.jsonl")})
        total += n

    summary_path = output_dir / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "gen_config": sampling_params,
            "raw_data_dir": str(raw_data_dir),
            "template_dir": str(template_dir),
            "output_dir": str(output_dir),
            "total_samples": total,
            "per_task": summary,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 汇总文件已写入：{summary_path}")
    print("[INFO] 接下来可运行：python compute_metrics.py --prediction_dir", str(output_dir))


if __name__ == "__main__":
    main()