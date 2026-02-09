#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intern-S1-mini evaluation on MMLU (cais/mmlu) for 5 columns:
HS Chem / College Chem / Organic Chem / Physical Chem / General Sci

Important:
- cais/mmlu does NOT provide config names 'organic_chemistry', 'physical_chemistry', 'general_science'
  So we approximate these columns by aggregating existing subjects.
- Use logprob scoring (no free-form generation) => no invalid outputs.

Outputs:
- out_dir/predictions.jsonl
- out_dir/summary.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


CHOICES = ["A", "B", "C", "D"]


# ======== Column -> cais/mmlu subjects mapping ========
# You can adjust this mapping to better match your paper's definition.
COLUMN_TO_SUBJECTS: Dict[str, List[str]] = {
    "HS Chem": ["high_school_chemistry"],
    "College Chem": ["college_chemistry"],

    # cais/mmlu has no organic_chemistry; approximate with biology/chem related reasoning
    # If you have a better mapping definition from your paper, replace here.
    "Organic Chem": ["college_chemistry", "high_school_chemistry"],

    # cais/mmlu has no physical_chemistry; approximate with physics-related science
    "Physical Chem": ["college_physics", "high_school_physics", "conceptual_physics"],

    # cais/mmlu has no general_science; approximate with broader science subjects
    "General Sci": ["astronomy", "college_physics", "high_school_physics", "conceptual_physics"],
}


def format_prompt(question: str, choices: List[str]) -> str:
    lines = [
        "Choose the correct option (A, B, C, or D). Return only one letter.",
        "",
        f"Question: {question}",
        "Choices:",
        f"A. {choices[0]}",
        f"B. {choices[1]}",
        f"C. {choices[2]}",
        f"D. {choices[3]}",
        "",
        "Answer:",
    ]
    return "\n".join(lines)


@torch.no_grad()
def predict_logprob_choice(model, tokenizer, prompt: str, device: str) -> Tuple[str, Dict[str, float]]:
    """
    Score A/B/C/D by next-token logprob at prompt end (fast path).
    If letter tokenizes to multiple tokens (rare), fall back to multi-token scoring.
    """
    enc = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    out = model(input_ids=input_ids, attention_mask=attn)
    last_logits = out.logits[0, -1]  # vocab
    last_logp = F.log_softmax(last_logits, dim=-1)

    def score_multi(ids: List[int]) -> float:
        base_len = input_ids.size(1)
        append = torch.tensor([ids], dtype=torch.long, device=device)
        full = torch.cat([input_ids, append], dim=1)
        full_attn = torch.cat([attn, torch.ones((1, len(ids)), dtype=attn.dtype, device=device)], dim=1)
        o = model(input_ids=full, attention_mask=full_attn)
        logits = o.logits[0]  # (L, V)
        s = 0.0
        for j, tok_id in enumerate(ids):
            pos = base_len + j
            s += float(F.log_softmax(logits[pos - 1], dim=-1)[tok_id].item())
        return s

    scores: Dict[str, float] = {}
    for c in CHOICES:
        ids = tokenizer(c, add_special_tokens=False).input_ids
        if len(ids) == 1:
            scores[c] = float(last_logp[ids[0]].item())
        else:
            scores[c] = score_multi(ids)

    pred = max(scores.items(), key=lambda kv: kv[1])[0]
    return pred, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_items_per_subject", type=int, default=-1, help="-1 means all")
    ap.add_argument("--debug_scores", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(args.device)
    model.eval()

    pred_path = out_dir / "predictions.jsonl"
    sum_path = out_dir / "summary.json"

    summary: Dict[str, Any] = {"by_column": {}, "overall": {}}
    overall_total = 0
    overall_correct = 0

    with pred_path.open("w", encoding="utf-8") as f_out:
        for col_name, subjects in COLUMN_TO_SUBJECTS.items():
            col_total = 0
            col_correct = 0

            # iterate subjects
            for subject in subjects:
                ds = load_dataset("cais/mmlu", subject, split="test")
                if args.max_items_per_subject > 0:
                    ds = ds.select(range(min(len(ds), args.max_items_per_subject)))

                pbar = tqdm(range(len(ds)), desc=f"{col_name}::{subject}", ncols=110)
                for i in pbar:
                    ex = ds[i]
                    q = ex["question"]
                    choices = ex["choices"]
                    gold = CHOICES[int(ex["answer"])]

                    prompt = format_prompt(q, choices)
                    pred, scores = predict_logprob_choice(model, tokenizer, prompt, args.device)
                    ok = (pred == gold)

                    rec = {
                        "column": col_name,
                        "subject": subject,
                        "idx": int(i),
                        "gold": gold,
                        "pred": pred,
                        "correct": ok,
                    }
                    if args.debug_scores == 1:
                        rec["scores"] = scores

                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                    col_total += 1
                    overall_total += 1
                    if ok:
                        col_correct += 1
                        overall_correct += 1

                    if col_total > 0:
                        pbar.set_postfix(acc=f"{col_correct/col_total:.3f}")

            summary["by_column"][col_name] = {
                "subjects": subjects,
                "acc": col_correct / max(col_total, 1),
                "correct": col_correct,
                "total": col_total,
            }

    summary["overall"] = {
        "acc": overall_correct / max(overall_total, 1),
        "correct": overall_correct,
        "total": overall_total,
    }

    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
