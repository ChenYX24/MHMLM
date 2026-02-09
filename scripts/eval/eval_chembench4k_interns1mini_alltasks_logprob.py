#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ChemBench4K (AI4Chem/ChemBench4K) ALL benchmark tasks evaluation for Intern-S1-mini
using logprob scoring (NO free-form generation), with tqdm progress bars.

Why logprob scoring:
- Some chat models reply "Okay"/"Sure" even with strict "A/B/C/D only" instruction.
- Logprob scoring guarantees a valid A/B/C/D prediction.

How:
1) List all files under split dir (default: test/) that end with "_benchmark.json"
   from HuggingFace repo.
2) For each benchmark json file:
   - Load dataset via datasets.load_dataset(data_files=...)
   - For each example, build chat-formatted prompt via AutoProcessor.apply_chat_template
   - Compute next-token logprobs for "A/B/C/D" and pick max.
   - If a letter tokenizes into multiple tokens (rare), fallback to multi-token scoring.

Output:
- out_dir/pred_{benchmark_name}.jsonl
- out_dir/summary.json

Usage:
  conda activate interns1_chembench
  CUDA_VISIBLE_DEVICES=0 python eval_chembench4k_interns1mini_alltasks_logprob.py \
    --model_path /data1/chenyuxuan/base_model/Intern-S1-mini \
    --out_dir /data1/chenyuxuan/MHMLM/eval_chembench_interns1mini_alltasks_logprob \
    --device cuda:0 --dtype bf16 \
    --split test --use_hf 1 \
    --max_items -1 \
    --debug_scores 0
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from datasets import load_dataset  # type: ignore
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

try:
    from huggingface_hub import HfApi  # type: ignore
    HAS_HF_HUB = True
except Exception:
    HAS_HF_HUB = False

DATASET_REPO = "AI4Chem/ChemBench4K"
CHOICES = ["A", "B", "C", "D"]


def list_benchmark_files(split: str) -> List[str]:
    """
    List all benchmark json files under split/ (e.g., 'test/') in the HF dataset repo.
    """
    if not HAS_HF_HUB:
        raise RuntimeError("huggingface_hub not installed. Please: pip install -U huggingface_hub")
    api = HfApi()
    files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
    prefix = f"{split}/"
    bench = []
    for f in files:
        if f.startswith(prefix) and f.endswith("_benchmark.json"):
            bench.append(f)
    bench.sort()
    if not bench:
        raise RuntimeError(f"No *_benchmark.json found under {split}/ in {DATASET_REPO}")
    return bench


def load_task_hf(file_path_in_repo: str, split: str) -> List[Dict[str, Any]]:
    """
    file_path_in_repo looks like: 'test/Name_Conversion_benchmark.json'
    split should be 'test' to match.
    """
    if not HAS_DATASETS:
        raise RuntimeError("datasets not installed. Please: pip install -U datasets")
    # datasets expects data_files as mapping split->file
    data_files = {split: file_path_in_repo}
    ds = load_dataset(DATASET_REPO, data_files=data_files, split=split)
    return [dict(x) for x in ds]


def format_prompt_mcq(ex: Dict[str, Any], task_name: str) -> str:
    """
    ChemBench4K benchmark json format: has 'question', 'A','B','C','D','answer'
    We build a strict MCQ prompt; prediction will be done by logprob scoring.
    """
    q = (ex.get("question") or "").strip()
    a = (ex.get("A") or "").strip()
    b = (ex.get("B") or "").strip()
    c = (ex.get("C") or "").strip()
    d = (ex.get("D") or "").strip()

    return (
        "You are taking a multiple-choice chemistry benchmark.\n"
        "Choose the single best option.\n"
        "You MUST output ONLY one letter among A, B, C, D.\n"
        "No explanation. No punctuation. No extra text.\n\n"
        f"Benchmark: {task_name}\n\n"
        f"Question:\n{q}\n\n"
        "Options:\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n"
        f"D. {d}\n\n"
        "Answer (A/B/C/D):"
    )


def build_chat_text(processor, prompt: str) -> str:
    """
    Intern-S1-mini recommended: AutoProcessor.apply_chat_template.
    """
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


@torch.no_grad()
def score_multi_token_append(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    append_ids: List[int],
) -> float:
    """
    Sum log P(append_ids | input_ids) for multi-token appended sequence.
    (Used only if "A"/"B"/"C"/"D" tokenizes into >1 token, which is rare.)
    """
    device = input_ids.device
    base_len = input_ids.shape[1]

    append = torch.tensor([append_ids], dtype=torch.long, device=device)
    full_ids = torch.cat([input_ids, append], dim=1)

    full_attn = torch.cat(
        [attention_mask, torch.ones((1, len(append_ids)), dtype=attention_mask.dtype, device=device)],
        dim=1
    )

    out = model(input_ids=full_ids, attention_mask=full_attn)
    logits = out.logits  # (1, L, V)

    score = 0.0
    for j, tok_id in enumerate(append_ids):
        pos = base_len + j
        prev_logits = logits[0, pos - 1]  # (V,)
        log_probs = F.log_softmax(prev_logits, dim=-1)
        score += float(log_probs[tok_id].item())
    return score


@torch.no_grad()
def predict_choice_logprob_fast(
    model,
    processor,
    chat_text: str,
    device: str
) -> Tuple[str, Dict[str, float]]:
    """
    Most questions cost ~1 forward pass:
      - Run model once on prompt to get next-token logits (last position).
      - Score single-token candidates 'A','B','C','D' from that distribution.
    Fallback for multi-token candidates (rare): use extra forward(s).
    """
    tok = processor.tokenizer
    enc = tok(chat_text, return_tensors="pt", padding=False, truncation=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # forward once
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    last_logits = out.logits[0, -1]  # (V,)
    last_logprobs = F.log_softmax(last_logits, dim=-1)

    scores: Dict[str, float] = {}
    for c in CHOICES:
        c_ids = tok(c, add_special_tokens=False).input_ids
        if len(c_ids) == 1:
            scores[c] = float(last_logprobs[c_ids[0]].item())
        else:
            # rare fallback
            scores[c] = score_multi_token_append(model, input_ids, attention_mask, c_ids)

    pred = max(scores.items(), key=lambda kv: kv[1])[0]
    return pred, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    ap.add_argument("--split", type=str, default="test", choices=["test", "train", "validation"])
    ap.add_argument("--use_hf", type=int, default=1, help="must be 1 (HF datasets)")
    ap.add_argument("--max_items", type=int, default=-1, help="-1 means all items per benchmark file")
    ap.add_argument("--only_files", type=str, default="", help="comma-separated exact file names (e.g., Name_Conversion_benchmark.json)")
    ap.add_argument("--debug_scores", type=int, default=0, help="write per-choice scores into jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_hf != 1:
        raise ValueError("This script is designed for HF loading only. Please set --use_hf 1")

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Load processor/model and force to single GPU device (NO device_map=auto)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # Ensure tokenizer has pad token id (not required for forward, but keep consistent)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(args.device)

    # List benchmark files
    bench_files_full = list_benchmark_files(args.split)  # like ['test/XXX_benchmark.json', ...]
    if args.only_files.strip():
        only = {x.strip() for x in args.only_files.split(",") if x.strip()}
        bench_files_full = [f for f in bench_files_full if f.split("/")[-1] in only]
        bench_files_full.sort()
        if not bench_files_full:
            raise RuntimeError(f"--only_files provided but matched nothing. only={sorted(list(only))}")

    summary: Dict[str, Any] = {}
    overall_total = 0
    overall_correct = 0

    pbar_tasks = tqdm(bench_files_full, desc="Benchmarks", unit="file")
    for file_path in pbar_tasks:
        bench_name = file_path.split("/")[-1].replace(".json", "")  # e.g. Name_Conversion_benchmark
        exs = load_task_hf(file_path_in_repo=file_path, split=args.split)
        if args.max_items is not None and args.max_items > 0:
            exs = exs[:args.max_items]

        out_jsonl = out_dir / f"pred_{bench_name}.jsonl"
        f = out_jsonl.open("w", encoding="utf-8")

        total = 0
        correct = 0

        pbar = tqdm(exs, desc=f"{bench_name}", unit="ex", leave=False)
        for i, ex in enumerate(pbar):
            gold = (ex.get("answer") or "").strip().upper()
            # Skip malformed
            if gold not in CHOICES:
                continue

            prompt = format_prompt_mcq(ex, bench_name)
            chat_text = build_chat_text(processor, prompt)

            pred, scores = predict_choice_logprob_fast(model, processor, chat_text, args.device)

            ok = (pred == gold)
            total += 1
            correct += int(ok)

            rec: Dict[str, Any] = {
                "benchmark": bench_name,
                "file": file_path,
                "idx": i,
                "gold": gold,
                "pred": pred,
                "correct": ok,
            }
            if args.debug_scores == 1:
                rec["scores"] = scores
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if total > 0:
                pbar.set_postfix(acc=f"{correct/total:.3f}")

        f.close()

        acc = correct / total if total > 0 else 0.0
        summary[bench_name] = {
            "file": file_path,
            "acc": acc,
            "correct": correct,
            "total": total,
        }

        overall_total += total
        overall_correct += correct
        pbar_tasks.set_postfix(overall_acc=f"{(overall_correct/overall_total) if overall_total else 0.0:.3f}")

    summary["overall"] = {
        "acc": (overall_correct / overall_total) if overall_total else 0.0,
        "correct": overall_correct,
        "total": overall_total,
        "num_benchmarks": len(bench_files_full),
        "split": args.split,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
