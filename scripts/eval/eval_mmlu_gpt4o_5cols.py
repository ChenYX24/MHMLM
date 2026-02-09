#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MMLU (cais/mmlu) evaluation using GPT-4o API, aligned with Intern-S1-mini 5-column header:
HS Chem / College Chem / Organic Chem / Physical Chem / General Sci

- Uses requests to call /chat/completions (same style as eval_chembench4k_gpt4o.py)
- Multi-threading with ThreadPoolExecutor
- Retry with exponential backoff + jitter
- Saves per-example jsonl and summary.json
- Supports aggregated columns: one column = multiple mmlu subjects

Usage:
  python eval_mmlu_gpt4o_5cols.py \
    --out_dir /path/to/out \
    --max_workers 16 \
    --model gpt-4o \
    --max_items_per_subject -1

Env vars:
  OPENAI_API_KEY
  OPENAI_BASE_URL
"""

from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

try:
    from datasets import load_dataset  # type: ignore
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False


CHOICES = ["A", "B", "C", "D"]

# ===== Column -> subjects mapping (EDIT THIS to match your table definition) =====
COLUMN_TO_SUBJECTS: Dict[str, List[str]] = {
    "HS Chem": ["high_school_chemistry"],
    "College Chem": ["college_chemistry"],

    # cais/mmlu has NO organic_chemistry config -> approximate (replace with your official mapping if you have)
    "Organic Chem": ["high_school_chemistry", "college_chemistry"],

    # cais/mmlu has NO physical_chemistry config -> approximate with physics-related subjects
    "Physical Chem": ["conceptual_physics", "high_school_physics", "college_physics"],

    # cais/mmlu has NO general_science config -> approximate as broader science bucket
    "General Sci": ["astronomy", "conceptual_physics", "high_school_physics", "college_physics"],
}


def extract_choice_from_text(text: str) -> Optional[str]:
    if text is None:
        return None
    t = text.strip().upper()
    # common patterns: "C", "Answer: C", "The answer is C"
    for c in CHOICES:
        if t == c:
            return c
    # find first standalone A/B/C/D
    for c in CHOICES:
        if t.startswith(c) or f" {c}" in t or f"\n{c}" in t:
            return c
    if t and t[0] in CHOICES:
        return t[0]
    return None


def format_mmlu_prompt(question: str, choices: List[str], subject: str, column: str) -> str:
    # Strict MCQ prompt: only output one letter
    return (
        "You are taking a multiple-choice test.\n"
        "Choose the single best option.\n"
        "You MUST output ONLY one letter among A, B, C, D.\n"
        "No explanation. No punctuation. No extra text.\n\n"
        f"Column: {column}\n"
        f"Subject: {subject}\n\n"
        f"Question:\n{question}\n\n"
        "Options:\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n\n"
        "Answer (A/B/C/D):"
    )


def call_chat_completions(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    max_retries: int = 6,
    timeout: int = 60,
) -> str:
    """
    requests -> POST {base_url}/chat/completions
    retry with backoff + jitter
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    system_prompt = (
        "You are a strict grader. "
        "Output ONLY one letter among A, B, C, D. "
        "No other words."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 5,
    }

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            out = (data["choices"][0]["message"]["content"] or "").strip()
            return out
        except Exception as e:
            if attempt < max_retries:
                sleep_s = (2 ** (attempt - 1)) + random.random()
                time.sleep(sleep_s)
            else:
                return ""  # let parser mark invalid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model", type=str, default="gpt-4o")
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--max_retries", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=60)

    ap.add_argument("--max_items_per_subject", type=int, default=-1, help="-1 means all")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--api_key", type=str, default="")
    ap.add_argument("--base_url", type=str, default="")
    args = ap.parse_args()

    if not HAS_DATASETS:
        raise RuntimeError("datasets not installed. pip install datasets")

    random.seed(args.seed)

    api_key = args.api_key.strip() or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.base_url.strip() or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is empty. Set env var or pass --api_key")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    # Build jobs
    jobs: List[Dict[str, Any]] = []
    for col, subjects in COLUMN_TO_SUBJECTS.items():
        for subject in subjects:
            ds = load_dataset("cais/mmlu", subject, split="test")
            if args.max_items_per_subject and args.max_items_per_subject > 0:
                ds = ds.select(range(min(len(ds), args.max_items_per_subject)))

            for i in range(len(ds)):
                ex = ds[i]
                q = ex["question"]
                choices = ex["choices"]
                gold = CHOICES[int(ex["answer"])]

                prompt = format_mmlu_prompt(q, choices, subject=subject, column=col)
                jobs.append({
                    "column": col,
                    "subject": subject,
                    "idx": int(i),
                    "gold": gold,
                    "prompt": prompt,
                })

    # Run jobs with concurrency
    results_buffer: List[Dict[str, Any]] = []
    col_stats: Dict[str, Dict[str, int]] = {c: {"total": 0, "correct": 0, "invalid": 0} for c in COLUMN_TO_SUBJECTS}
    overall = {"total": 0, "correct": 0, "invalid": 0}

    def process_job(job: Dict[str, Any]) -> Dict[str, Any]:
        raw = call_chat_completions(
            api_key=api_key,
            base_url=base_url,
            model=args.model,
            prompt=job["prompt"],
            max_retries=args.max_retries,
            timeout=args.timeout,
        )
        pred = extract_choice_from_text(raw)
        ok = (pred == job["gold"]) if pred in CHOICES else False
        return {
            "column": job["column"],
            "subject": job["subject"],
            "idx": job["idx"],
            "gold": job["gold"],
            "pred": pred,
            "correct": ok,
            "raw_gen": raw,
        }

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(process_job, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="MMLU jobs", unit="ex"):
            rec = fut.result()
            results_buffer.append(rec)

            col = rec["column"]
            col_stats[col]["total"] += 1
            overall["total"] += 1

            if rec["pred"] not in CHOICES:
                col_stats[col]["invalid"] += 1
                overall["invalid"] += 1
            if rec["correct"]:
                col_stats[col]["correct"] += 1
                overall["correct"] += 1

    # Write jsonl (sorted for reproducibility)
    results_buffer.sort(key=lambda r: (r["column"], r["subject"], r["idx"]))
    with pred_path.open("w", encoding="utf-8") as f:
        for r in results_buffer:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Build summary
    summary: Dict[str, Any] = {"by_column": {}, "overall": {}}
    for col, st in col_stats.items():
        acc = st["correct"] / st["total"] if st["total"] else 0.0
        summary["by_column"][col] = {
            "subjects": COLUMN_TO_SUBJECTS[col],
            "acc": acc,
            "correct": st["correct"],
            "total": st["total"],
            "invalid": st["invalid"],
        }
    summary["overall"] = {
        "acc": overall["correct"] / overall["total"] if overall["total"] else 0.0,
        "correct": overall["correct"],
        "total": overall["total"],
        "invalid": overall["invalid"],
        "model": args.model,
        "base_url": base_url,
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
