#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, os, random, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

try:
    import pyarrow as pa  # noqa
    import pyarrow.parquet as pq
    HAS_ARROW = True
except Exception:
    HAS_ARROW = False

try:
    from rdkit import Chem  # type: ignore
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False


TAG_RE = re.compile(r"<([A-Z_]+)>\s*(.*?)\s*</\1>", flags=re.DOTALL)

def extract_tags(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not text:
        return out
    for m in TAG_RE.finditer(text):
        out[m.group(1)] = m.group(2).strip()
    return out

def extract_tag(text: str, tag: str) -> str:
    if not text:
        return ""
    m = re.search(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, flags=re.DOTALL)
    return (m.group(1).strip() if m else "")

def canonicalize_smiles(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if not HAS_RDKIT:
        return s
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return ""
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""

def parse_number(text: str) -> Optional[float]:
    if text is None:
        return None
    t = text.strip()
    if not t:
        return None
    x = extract_tag(t, "NUMBER")
    if x:
        t = x
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", t)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


@dataclass
class APIConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int


def call_chat(cfg: APIConfig, prompt: str) -> str:
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    system = (
        "You are a chemistry assistant.\n"
        "Follow the instruction strictly.\n"
        "If tagged output is required, output ONLY the tagged answer.\n"
        "Do NOT add explanations.\n"
    )
    payload = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    for a in range(1, cfg.max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout)
            r.raise_for_status()
            j = r.json()
            return (j["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            if a == cfg.max_retries:
                return ""
            time.sleep((2 ** (a - 1)) + random.random())


def infer_expected_tag(sample: Dict[str, Any]) -> Optional[str]:
    left = sample.get("output_core_tag_left")
    if isinstance(left, str) and left.startswith("<") and left.endswith(">"):
        return left[1:-1]
    task = str(sample.get("task", "")).lower()
    if "caption" in task:
        return None
    if "yield" in task or "property" in task:
        return "NUMBER"
    return "SMILES"


def score_one(sample: Dict[str, Any], raw: str) -> Dict[str, Any]:
    task = sample.get("task", "")
    gold = sample.get("output", "") or ""
    gold_tags = extract_tags(gold)
    pred_tags = extract_tags(raw)
    exp = infer_expected_tag(sample)

    rec = {
        "sample_id": sample.get("sample_id", ""),
        "task": task,
        "gold": gold,
        "raw_gen": raw,
        "pred": None,
        "valid": False,
        "correct": False,
        "metric": {},
    }

    if exp is None:
        pred = raw.strip()
        rec["pred"] = pred
        rec["valid"] = bool(pred)
        rec["correct"] = (pred == gold.strip()) if pred else False
        return rec

    pred_core = pred_tags.get(exp, "")
    if not pred_core:
        pred_core = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    rec["pred"] = pred_core

    if exp == "NUMBER":
        g = parse_number(gold_tags.get("NUMBER", gold))
        p = parse_number(pred_core)
        rec["valid"] = p is not None
        if g is not None and p is not None:
            rec["metric"]["abs_err"] = abs(p - g)
        return rec

    if exp == "SMILES":
        g = canonicalize_smiles(gold_tags.get("SMILES", ""))
        p = canonicalize_smiles(pred_core)
        rec["metric"]["gold_can"] = g
        rec["metric"]["pred_can"] = p
        rec["valid"] = bool(p)
        rec["correct"] = (p == g) if (p and g) else False
        return rec

    g = (gold_tags.get(exp, "") or "").strip()
    p = (pred_core or "").strip()
    rec["valid"] = bool(p)
    rec["correct"] = (p == g) if (p and g) else False
    return rec


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, Dict[str, Any]] = {}
    ot = ov = oc = 0

    for r in records:
        t = r.get("task", "UNKNOWN")
        st = by_task.setdefault(t, {"total": 0, "valid": 0, "em_correct": 0, "n_num": 0, "mae_sum": 0.0, "mse_sum": 0.0})
        st["total"] += 1
        if r.get("valid"):
            st["valid"] += 1
        if r.get("correct"):
            st["em_correct"] += 1
        if "abs_err" in r.get("metric", {}):
            ae = float(r["metric"]["abs_err"])
            st["n_num"] += 1
            st["mae_sum"] += ae
            st["mse_sum"] += ae * ae

    out: Dict[str, Any] = {"by_task": {}, "overall": {}}
    for t, st in by_task.items():
        total = st["total"]
        valid = st["valid"]
        emc = st["em_correct"]
        ot += total
        ov += valid
        oc += emc
        row: Dict[str, Any] = {
            "total": total,
            "valid_pct": valid / total if total else 0.0,
            "em": emc / total if total else 0.0,
            "em_correct": emc,
        }
        if st["n_num"] > 0:
            n = st["n_num"]
            row["mae"] = st["mae_sum"] / n
            row["rmse"] = (st["mse_sum"] / n) ** 0.5
            row["n_numeric"] = n
        out["by_task"][t] = row

    out["overall"] = {
        "total": ot,
        "valid_pct": ov / ot if ot else 0.0,
        "em": oc / ot if ot else 0.0,
        "em_correct": oc,
    }
    return out


def normalize_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Try to map various field names into:
      - input
      - output
      - task
      - sample_id (optional)
      - output_core_tag_left/right (optional)
    """
    out = dict(ex)

    # common aliases
    if "prompt" in out and "input" not in out:
        out["input"] = out["prompt"]
    if "instruction" in out and "input" not in out:
        out["input"] = out["instruction"]
    if "response" in out and "output" not in out:
        out["output"] = out["response"]
    if "answer" in out and "output" not in out:
        out["output"] = out["answer"]
    if "id" in out and "sample_id" not in out:
        out["sample_id"] = out["id"]

    # ensure types
    out["input"] = str(out.get("input", "") or "")
    out["output"] = str(out.get("output", "") or "")
    out["task"] = str(out.get("task", "unknown") or "unknown")
    out["sample_id"] = str(out.get("sample_id", "") or "")

    return out


def load_records_from_dir(data_dir: Path, max_files: int = 0) -> List[Dict[str, Any]]:
    files: List[Path] = []
    for ext in ["*.jsonl", "*.json", "*.parquet", "*.arrow", "*.csv"]:
        files.extend(data_dir.rglob(ext))
    files = sorted(set(files))

    if max_files and max_files > 0:
        files = files[:max_files]

    if not files:
        raise RuntimeError(f"No data files found under: {data_dir}")

    records: List[Dict[str, Any]] = []

    for fp in files:
        suf = fp.suffix.lower()
        if suf == ".jsonl":
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(normalize_example(json.loads(line)))
        elif suf == ".json":
            obj = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                for ex in obj:
                    if isinstance(ex, dict):
                        records.append(normalize_example(ex))
            elif isinstance(obj, dict):
                # maybe dict with "data"
                if "data" in obj and isinstance(obj["data"], list):
                    for ex in obj["data"]:
                        if isinstance(ex, dict):
                            records.append(normalize_example(ex))
        elif suf in [".parquet", ".arrow"]:
            if not HAS_ARROW:
                raise RuntimeError("Need pyarrow to read parquet/arrow. Install: pip install pyarrow")
            if suf == ".parquet":
                table = pq.read_table(str(fp))
            else:
                # arrow file: try read as IPC
                import pyarrow.ipc as ipc
                with fp.open("rb") as f:
                    reader = ipc.open_file(f)
                    table = reader.read_all()
            df = table.to_pandas()
            for _, row in df.iterrows():
                records.append(normalize_example(row.to_dict()))
        elif suf == ".csv":
            if not HAS_PANDAS:
                raise RuntimeError("Need pandas to read csv. Install: pip install pandas")
            df = pd.read_csv(fp)
            for _, row in df.iterrows():
                records.append(normalize_example(row.to_dict()))

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str, help="Directory containing extracted data files.")
    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--use_test_subset", default=1, type=int, help="simulate <=200 per task")
    ap.add_argument("--tasks", default="", type=str, help="comma separated task names; empty=all")
    ap.add_argument("--max_files", default=0, type=int, help="debug: only read first N files")

    ap.add_argument("--model", default="gpt-4o", type=str)
    ap.add_argument("--max_workers", default=16, type=int)
    ap.add_argument("--temperature", default=0.0, type=float)
    ap.add_argument("--max_tokens", default=128, type=int)
    ap.add_argument("--timeout", default=60, type=int)
    ap.add_argument("--max_retries", default=6, type=int)
    ap.add_argument("--api_key", default="", type=str)
    ap.add_argument("--base_url", default="", type=str)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    random.seed(args.seed)

    api_key = args.api_key.strip() or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.base_url.strip() or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is empty.")

    cfg = APIConfig(api_key, base_url, args.model, args.temperature, args.max_tokens, args.timeout, args.max_retries)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    sum_path = out_dir / "summary.json"

    data_dir = Path(args.data_dir)
    records_raw = load_records_from_dir(data_dir, max_files=args.max_files)
    print(f"[INFO] loaded {len(records_raw)} raw records from {data_dir}")

    task_filter = set([t.strip() for t in args.tasks.split(",") if t.strip()]) if args.tasks.strip() else None
    cap = 200 if args.use_test_subset == 1 else None
    per_task: Dict[str, int] = {}

    samples: List[Dict[str, Any]] = []
    for ex in records_raw:
        t = ex.get("task", "unknown")
        if task_filter and t not in task_filter:
            continue
        if cap is not None:
            c = per_task.get(t, 0)
            if c >= cap:
                continue
            per_task[t] = c + 1
        samples.append(ex)

    print(f"[INFO] filtered to {len(samples)} samples (cap per task={cap})")

    jobs: List[Tuple[int, Dict[str, Any]]] = list(enumerate(samples))

    def run_one(i: int, s: Dict[str, Any]) -> Dict[str, Any]:
        raw = call_chat(cfg, s.get("input", ""))
        rec = score_one(s, raw)
        rec["idx"] = i
        return rec

    outs: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [pool.submit(run_one, i, s) for i, s in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="SMolInstruct eval", ncols=110):
            outs.append(fut.result())

    outs.sort(key=lambda r: (r.get("task", ""), r.get("sample_id", ""), r.get("idx", 0)))
    with pred_path.open("w", encoding="utf-8") as f:
        for r in outs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = aggregate(outs)
    summary["meta"] = {
        "model": args.model,
        "base_url": base_url,
        "data_dir": str(data_dir),
        "use_test_subset_simulated": bool(args.use_test_subset == 1),
        "per_task_cap": cap,
        "has_rdkit": HAS_RDKIT,
        "has_pyarrow": HAS_ARROW,
        "has_pandas": HAS_PANDAS,
    }
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
