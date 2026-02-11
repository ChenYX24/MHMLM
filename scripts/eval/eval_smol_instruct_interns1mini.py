#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run SMolInstruct evaluation using Intern-S1-mini (local causal LM), reusing your existing
eval_smolinstruct_batch.py pipeline (data reading, prompt building, prediction extraction, scoring).

Key features:
- Dynamic import evaluator from an absolute path (no PYTHONPATH dependency)
- Patch transformers tokenizer base for Intern-S1-mini remote tokenizer compatibility:
  fixes missing _added_tokens_encoder/_decoder/_update_trie for newer transformers versions
- Per-task limit (e.g., 200 samples each task)
- Batch inference with progress bar (provided by evaluator)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Tuple, Callable, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Dynamic import evaluator by path
# -----------------------------
from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_loader, module_from_spec


def import_module_from_path(name: str, path: str):
    loader = SourceFileLoader(name, path)
    spec = spec_from_loader(loader.name, loader)
    mod = module_from_spec(spec)
    loader.exec_module(mod)
    return mod


# -----------------------------
# Patch tokenizer base for Intern-S1-mini tokenizer compatibility
# -----------------------------
def patch_tokenizer_base_for_interns1():
    """
    Intern-S1-mini remote tokenizer code expects some private attributes/methods that
    may not exist in newer transformers (e.g., 4.46+):
      - self._added_tokens_encoder
      - self._added_tokens_decoder
      - self._update_trie()

    We patch PreTrainedTokenizerBase.__init__ to ensure these exist *before* add_tokens is called.
    """
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if getattr(PreTrainedTokenizerBase, "_interns1_patch_applied", False):
        return

    orig_init = PreTrainedTokenizerBase.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)

        # Ensure legacy private dicts exist and point to the public ones
        if not hasattr(self, "_added_tokens_encoder"):
            try:
                self._added_tokens_encoder = self.added_tokens_encoder
            except Exception:
                self._added_tokens_encoder = {}

        if not hasattr(self, "_added_tokens_decoder"):
            try:
                self._added_tokens_decoder = self.added_tokens_decoder
            except Exception:
                self._added_tokens_decoder = {}

        # Some legacy tokenizers call _update_trie after adding tokens
        if not hasattr(self, "_update_trie"):
            setattr(self, "_update_trie", lambda *a, **k: None)

    PreTrainedTokenizerBase.__init__ = patched_init  # type: ignore
    PreTrainedTokenizerBase._interns1_patch_applied = True  # type: ignore


# -----------------------------
# Model inference wrapper
# -----------------------------
@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    # tokenize
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=getattr(tokenizer, "model_max_length", 4096),
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    do_sample = temperature is not None and float(temperature) > 1e-6
    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        temperature=float(temperature) if do_sample else None,
        top_p=float(top_p) if do_sample else None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    # remove None keys
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    out_ids = model.generate(**enc, **gen_kwargs)

    # decode only the newly generated part (optional)
    # Here we decode full and let extractor handle it; more robust.
    texts = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True, help="Intern-S1-mini local path or HF id")
    ap.add_argument("--raw_data_dir", type=str, required=True, help="constructed_test directory containing *.jsonl tasks")
    ap.add_argument("--template_dir", type=str, required=True, help="instruction_tuning template dir containing <task>.json")
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--per_task_limit", type=int, default=200, help="cap each task to N samples (0=no cap)")
    ap.add_argument("--include_tasks", type=str, default="", help="comma-separated task names to run (optional)")

    ap.add_argument(
        "--evaluator_path",
        type=str,
        default="/data1/chenyuxuan/MHMLM/eval/eval_smolinstruct_batch.py",
        help="absolute path to eval_smolinstruct_batch.py",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose_every", type=int, default=0)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- import evaluator
    evaluator_path = Path(args.evaluator_path).expanduser().resolve()
    if not evaluator_path.exists():
        raise FileNotFoundError(f"evaluator_path not found: {evaluator_path}")
    evaluator = import_module_from_path("eval_smolinstruct_batch", str(evaluator_path))

    # --- patch tokenizer base BEFORE loading tokenizer
    patch_tokenizer_base_for_interns1()

    # --- dtype
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but device is cuda:*")

    # --- load tokenizer / model (trust_remote_code is needed for Intern-S1-mini)
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )

    # safety: ensure eos/pad
    if tokenizer.eos_token_id is None and tokenizer.eos_token:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None,  # single device for stability
    )
    model.to(device)
    model.eval()

    # --- build call_model_func compatible with evaluator.evaluate_file
    def call_model_func(prompts: List[str], task: str = "") -> Tuple[List[str], List[str]]:
        # batch in chunks
        all_out = []
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i : i + args.batch_size]
            out = generate_batch(
                model=model,
                tokenizer=tokenizer,
                prompts=chunk,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            all_out.extend(out)
        # evaluator expects (outputs, raw_outputs_full)
        return all_out, all_out

    # --- discover task jsonl
    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    template_dir = Path(args.template_dir).expanduser().resolve()
    if not raw_data_dir.exists():
        raise FileNotFoundError(raw_data_dir)
    if not template_dir.exists():
        raise FileNotFoundError(template_dir)

    files = sorted([p for p in raw_data_dir.iterdir() if evaluator.is_target_jsonl(p)])
    if args.include_tasks.strip():
        allow = {x.strip() for x in args.include_tasks.split(",") if x.strip()}
        files = [p for p in files if evaluator.task_name_from_filename(p) in allow]

    if not files:
        raise RuntimeError(f"No task jsonl found under {raw_data_dir}")

    # --- run per task
    total = 0
    per_task = []
    print(f"[INFO] tasks={len(files)}  raw_data_dir={raw_data_dir}")

    for jsonl_path in files:
        task = evaluator.task_name_from_filename(jsonl_path)
        template_path = template_dir / f"{task}.json"
        if not template_path.exists():
            print(f"[WARN] template missing: {template_path} (skip)")
            continue

        print(f"[INFO] Evaluating {task} from {jsonl_path}")
        task_name, n = evaluator.evaluate_file(
            call_model_func=lambda prompts, _task=task: call_model_func(prompts, task=_task),
            jsonl_path=jsonl_path,
            template_path=template_path,
            output_dir=out_dir,
            is_chat_model=False,          # local causal LM
            cot=False,
            verbose_every=args.verbose_every if args.verbose_every > 0 else 10**18,
            data_limit=(args.per_task_limit if args.per_task_limit and args.per_task_limit > 0 else 0),
        )
        per_task.append({"task": task_name, "num_samples": n, "outfile": str(out_dir / f"{task_name}.jsonl")})
        total += n

    summary = {
        "model": args.model_name_or_path,
        "raw_data_dir": str(raw_data_dir),
        "template_dir": str(template_dir),
        "out_dir": str(out_dir),
        "total_samples": total,
        "per_task": per_task,
        "gen": {
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "per_task_limit": args.per_task_limit,
        },
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[DONE] total={total}  summary={out_dir / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
