#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMolInstruct postprocess (robust extraction) + score
- Robustly extracts answers from answer_only/raw_output (prefers content after the last instruction line)
- Supports special wrapper tokens like <SMILES>...</SMILES>, [SMILES]...[/SMILES], <NUMBER>...</NUMBER>, etc.
- In-place updates `pred` back to the original task jsonl files (optional, default ON)
- Writes a .bak backup per task file by default (safe rollback)
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

# ---------------- Task groups ----------------
SMILES_TASKS = {
    "forward_synthesis",
    "retrosynthesis",
    "molecule_generation",
    "name_conversion-i2s",
}
SMILES_TASKS_MULTIMETRIC = {"retrosynthesis"}
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

KNOWN_TASKS = (
    SMILES_TASKS
    | TEXT_TASKS
    | FORMULA_ELEMENT_TASKS
    | FORMULA_SPLIT_TASKS
    | NUMBER_TASKS
    | BOOLEAN_TASKS
)

TYPE_NAMES = {"SMILES", "FORMULA", "NUMBER", "BOOLEAN"}

# ---------------- Metrics import ----------------
_metrics_imported = False
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
        print(f"[INFO] Using reference metrics: {ref_metrics_path}")
except Exception:
    pass

if not _metrics_imported:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from utils.metrics import (
            calculate_smiles_metrics,
            calculate_formula_metrics,
            calculate_text_metrics,
            calculate_number_metrics,
            calculate_boolean_metrics,
        )
        _metrics_imported = True
        print(f"[INFO] Using project metrics: {repo_root}/utils/metrics")
    except Exception as e:
        print(f"[ERROR] Failed to import metrics: {e}")
        _metrics_imported = False

if not _metrics_imported:
    raise RuntimeError("Cannot import metrics. Please ensure metrics module is available.")

# ---------------- Robust extraction helpers ----------------

INSTRUCTION_PAT = re.compile(
    r"(?is)(please\s+only\s+output\s+the\s+answer\s+without\s+any\s+explanation\s+or\s+additional\s+text\.?)"
)

def _strip(s: str) -> str:
    return s.strip().strip("\u200b").strip()

def _first_nonempty(*xs: Optional[str]) -> Optional[str]:
    for x in xs:
        if isinstance(x, str) and x.strip():
            return x
    return None

def _after_last_instruction(text: str) -> str:
    """Take substring after the LAST occurrence of the common instruction line."""
    if not text:
        return text
    matches = list(INSTRUCTION_PAT.finditer(text))
    if not matches:
        return text
    m = matches[-1]
    return text[m.end():].lstrip()

def _extract_between_tokens(text: str, start_tokens: List[str], end_tokens: List[str]) -> Optional[str]:
    if not text:
        return None
    t_low = text.lower()
    for st in start_tokens:
        si = t_low.find(st.lower())
        if si < 0:
            continue
        after = si + len(st)
        for et in end_tokens:
            ei = t_low.find(et.lower(), after)
            if ei < 0:
                continue
            val = text[after:ei]
            return _strip(val)
    return None

def _cleanup_tag_junk(s: str) -> str:
    # Handles cases like "<SMILES> 1.23</</SMILES>" -> "1.23"
    s = s.strip()
    s = re.sub(r"</\s*</", "", s)
    s = re.sub(r"</\s*/?\s*$", "", s)
    return s.strip()

def _extract_wrapped_field(text: str, field: str) -> Optional[str]:
    f = field.upper()
    starts = [f"<{f}>", f"<{f.lower()}>", f"[{f}]", f"[{f.lower()}]"]
    ends   = [f"</{f}>", f"</{f.lower()}>", f"[/{f}]", f"[/{f.lower()}]"]
    val = _extract_between_tokens(text, starts, ends)
    if val:
        return _cleanup_tag_junk(val)

    # Label style: "FIELD: xxx"
    m = re.search(rf"(?im)^\s*{re.escape(f)}\s*[:：]\s*(.+?)\s*$", text)
    if m:
        return _strip(m.group(1))
    m = re.search(rf"(?im)^\s*{re.escape(f.lower())}\s*[:：]\s*(.+?)\s*$", text)
    if m:
        return _strip(m.group(1))
    return None

def _try_parse_json_answer(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip()
    lb, rb = t.find("{"), t.rfind("}")
    if lb < 0 or rb <= lb:
        return None
    cand = t[lb:rb+1]
    try:
        obj = json.loads(cand)
    except Exception:
        return None
    if isinstance(obj, dict):
        for k in ["answer", "smiles", "SMILES", "formula", "FORMULA", "output", "pred", "value", "boolean"]:
            if k in obj and isinstance(obj[k], (str, int, float, bool)):
                return str(obj[k])
    return None

def _extract_last_number(text: str) -> Optional[str]:
    if not text:
        return None
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    return nums[-1] if nums else None

def _bool_to_yesno(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().lower()

    # strip common prefixes like "Answer:", "Final:", etc.
    t = re.sub(r"(?i)^\s*(answer|output|boolean|final)\s*[:：]\s*", "", t).strip()

    # keep only the first token to avoid "No." / "no (because ...)" / trailing explanations
    t = re.split(r"[\s,;:：\.\!\?\)\]\}]+", t, maxsplit=1)[0].strip()

    if t in {"true", "yes", "y", "1", "positive", "pos"}:
        return "Yes"
    if t in {"false", "no", "n", "0", "negative", "neg"}:
        return "No"
    return None

def _drop_wrappers_keep_content(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s*(.*?)\s*<\s*/\s*\1\s*>", r"\2", text, flags=re.S)
    text = re.sub(r"\[\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]\s*(.*?)\s*\[\s*/\s*\1\s*\]", r"\2", text, flags=re.S)
    return text

def _extract_plausible_smiles(text: str) -> Optional[str]:
    if not text:
        return None
    plain = _drop_wrappers_keep_content(text)
    lines = [ln.strip() for ln in plain.splitlines() if ln.strip()]
    if not lines:
        return None
    tail = lines[-1]
    tokens = re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/\.\:%]+", tail)
    atom_pat = re.compile(r"(Br|Cl|Si|Na|Ca|Li|Al|Mg|Fe|Zn|Cu|Ag|Au|B|C|N|O|P|S|F|I)", re.I)
    for tok in tokens:
        if len(tok) < 3:
            continue
        if tok.lower() in {"text", "smiles", "answer", "output"}:
            continue
        if atom_pat.search(tok):
            return tok
    return tail

def extract_pred_for_task(task: str, answer_only: Optional[str], raw_output: Optional[str], input_field: Optional[str]) -> Optional[Any]:
    text = _first_nonempty(answer_only, raw_output)
    if not text:
        return None
    text = _strip(text)

    # Prefer JSON field if present
    j = _try_parse_json_answer(text)
    if j is not None and str(j).strip():
        text = str(j).strip()

    # Cut to content AFTER the last instruction, to avoid picking numbers like "pH 7.4"
    tail = _after_last_instruction(text)
    tail = tail if tail.strip() else text  # fallback

    if task in SMILES_TASKS:
        s = _extract_wrapped_field(text, "SMILES") or _extract_wrapped_field(tail, "SMILES")
        if s is None:
            s = _extract_plausible_smiles(tail)
        if not s:
            return None
        if input_field and _strip(input_field) and _strip(s) == _strip(input_field):
            return None
        return s

    if task in FORMULA_ELEMENT_TASKS:
        f = _extract_wrapped_field(text, "FORMULA") or _extract_wrapped_field(tail, "FORMULA")
        if f is None:
            plain = _drop_wrappers_keep_content(tail)
            m = re.search(r"\b([A-Z][a-z]?\d*)+(?:[-+]\d+)?\b", plain)
            f = m.group(0) if m else None
        return f if f and f.strip() else None

    if task in FORMULA_SPLIT_TASKS:
        name = (_extract_wrapped_field(text, "IUPAC") or _extract_wrapped_field(tail, "IUPAC")
                or _extract_wrapped_field(text, "NAME") or _extract_wrapped_field(tail, "NAME")
                or _extract_wrapped_field(text, "FORMULA") or _extract_wrapped_field(tail, "FORMULA"))
        if name is None:
            name = _drop_wrappers_keep_content(tail).strip()
        lines = [ln.strip() for ln in name.splitlines() if ln.strip()] if name else []
        return lines[-1] if lines else None

    if task in BOOLEAN_TASKS:
        b = _extract_wrapped_field(text, "BOOLEAN") or _extract_wrapped_field(tail, "BOOLEAN")
        if b is None:
            b = _drop_wrappers_keep_content(tail).strip()
        # take last non-empty line, map to Yes/No
        lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
        cand = lines[-1] if lines else b
        return _bool_to_yesno(cand)

    if task in NUMBER_TASKS:
        n = (_extract_wrapped_field(text, "NUMBER") or _extract_wrapped_field(tail, "NUMBER")
             or _extract_wrapped_field(text, "VALUE") or _extract_wrapped_field(tail, "VALUE"))
        if n is None:
            n = _extract_last_number(tail)
        else:
            n = _extract_last_number(n)
        return n

    if task in TEXT_TASKS:
        plain = _drop_wrappers_keep_content(tail).strip()
        lines = [ln.strip() for ln in plain.splitlines() if ln.strip()]
        return (" ".join(lines[-3:]) if lines else None)

    return None

# ---------------- IO, in-place fix, scoring ----------------

def _is_task_file(p: Path) -> bool:
    return (
        p.is_file()
        and p.suffix == ".jsonl"
        and p.name != "eval_summary.json"
        and p.stem in KNOWN_TASKS
    )

def _read_jsonl_lines(fp: Path) -> List[dict]:
    objs = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                objs.append(json.loads(line))
            except Exception:
                # skip malformed lines to avoid breaking scoring
                continue
    return objs

def _write_jsonl_atomic(fp: Path, objs: List[dict]) -> None:
    tmp = fp.with_suffix(fp.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp, fp)

def _needs_reextract(task: str, obj: dict, force_overwrite_pred: bool) -> bool:
    if force_overwrite_pred:
        return True
    pred = obj.get("pred", None)
    if pred is None:
        return True
    if isinstance(pred, str):
        t = pred.strip()
        if not t:
            return True
        if t.upper() in TYPE_NAMES:
            return True
        # For number/bool tasks, many wrong preds are from prompt leakage; we re-extract if answer_only exists
        if task in (NUMBER_TASKS | BOOLEAN_TASKS) and obj.get("answer_only"):
            return True
    return False

def _normalize_pred_for_storage(task: str, pred: Any) -> Any:
    if pred is None:
        return None
    if task in BOOLEAN_TASKS and isinstance(pred, str):
        yn = _bool_to_yesno(pred)
        return yn if yn is not None else pred.strip()
    if task in NUMBER_TASKS and isinstance(pred, str):
        n = _extract_last_number(pred)
        return n if n is not None else pred.strip()
    if isinstance(pred, str):
        return pred.strip()
    return pred

def read_fix_and_build_lists(
    fp: Path,
    inplace_fix: bool,
    backup: bool,
    force_overwrite_pred: bool,
    replace_semicolon: bool,
) -> Tuple[List[Optional[List[str]]], List[List[str]], int]:
    """
    Returns preds(list-of-list or None), golds(list-of-list), and num_updated for this file.
    Optionally writes updated preds back into fp.
    """
    task = fp.stem
    objs = _read_jsonl_lines(fp)

    updated = 0
    preds: List[Optional[List[str]]] = []
    golds: List[List[str]] = []

    for obj in objs:
        task_name = obj.get("task", task)

        # gold normalize
        gold = obj.get("gold", "")
        if isinstance(gold, str):
            g_list = [gold]
        elif isinstance(gold, list):
            g_list = [str(x) for x in gold]
        else:
            g_list = [json.dumps(gold, ensure_ascii=False)] if isinstance(gold, dict) else [str(gold)]

        if replace_semicolon:
            g_list = [g.replace(";", ".") for g in g_list]

        # pred fix (re-extract)
        if _needs_reextract(task_name, obj, force_overwrite_pred):
            new_pred = extract_pred_for_task(
                task=task_name,
                answer_only=obj.get("answer_only"),
                raw_output=obj.get("raw_output", obj.get("raw_gen", "")),
                input_field=obj.get("input"),
            )
            new_pred = _normalize_pred_for_storage(task_name, new_pred)

            # overwrite if we got something (or pred was None)
            if new_pred is not None or obj.get("pred") is None:
                if obj.get("pred") != new_pred:
                    obj["pred"] = new_pred
                    updated += 1

        # Always normalize existing pred for NUMBER/BOOLEAN tasks (case/punctuation/format)
        # This fixes cases like pred="no" vs gold="No", or pred containing stray punctuation.
        cur_pred = obj.get("pred", None)
        norm_pred = _normalize_pred_for_storage(task_name, cur_pred)
        if norm_pred != cur_pred:
            obj["pred"] = norm_pred
            updated += 1

        pred = obj.get("pred", None)

        if pred is None or (isinstance(pred, str) and not pred.strip()):
            preds.append(None)
        else:
            if isinstance(pred, str):
                p_list = [pred]
            elif isinstance(pred, list):
                p_list = [str(x) for x in pred]
            else:
                p_list = [str(pred)]
            if replace_semicolon:
                p_list = [p.replace(";", ".") for p in p_list]
            preds.append(p_list)

        golds.append(g_list)

    if inplace_fix and updated > 0:
        try:
            if backup:
                bak = fp.with_suffix(fp.suffix + ".bak")
                if not bak.exists():
                    bak.write_text(fp.read_text(encoding="utf-8"), encoding="utf-8")
            _write_jsonl_atomic(fp, objs)
        except Exception as e:
            print(f"[WARN] In-place fix failed for {fp.name}: {e}")

    return preds, golds, updated

def _score_one_task(task: str, preds: List, golds: List) -> Dict:
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

    raise Exception(f"Unknown task: {task}")

def _score_single_file(args):
    fp_str, inplace_fix, backup, force_overwrite_pred, _show_sample_progress = args
    fp = Path(fp_str)
    task = fp.stem

    preds, golds, updated = read_fix_and_build_lists(
        fp=fp,
        inplace_fix=inplace_fix,
        backup=backup,
        force_overwrite_pred=force_overwrite_pred,
        replace_semicolon=(task in DEFAULT_REPLACE_SEMICOLON_TASKS),
    )
    metrics = _score_one_task(task, preds, golds)
    metrics["_num_pred_updated"] = int(updated)
    return task, metrics

def print_summary(all_results: Dict[str, Dict]):
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
        "property_prediction-sider",
    ]

    for task in ordered_tasks:
        if task not in all_results:
            continue
        m = all_results[task]
        updated = m.get("_num_pred_updated", 0)

        if task in {"molecule_generation", "name_conversion-i2s", "forward_synthesis", "retrosynthesis"}:
            num_all = m.get("num_all", 0)
            fps = m.get("t1_rdk_fps", 0) * 100
            print(f"{task}\t{num_all}\tfps={fps:.1f}\tupdated={updated}")

        elif task == "molecule_captioning":
            num_all = m.get("num_all", 0)
            bleu4 = m.get("bleu4", 0) * 100
            rouge_l = m.get("rouge_l", 0) * 100
            print(f"{task}\t{num_all}\tbleu-4={bleu4:.1f}\trouge-l={rouge_l:.1f}\tupdated={updated}")

        elif task in {"name_conversion-i2f", "name_conversion-s2f"}:
            num_all = m.get("num_all", 1)
            ele = m.get("num_t1_ele_match", 0) / max(1, num_all) * 100
            print(f"{task}\t{num_all}\tEM={ele:.1f}\tupdated={updated}")

        elif task == "name_conversion-s2i":
            num_all = m.get("num_all", 1)
            split = m.get("num_t1_split_match", 0) / max(1, num_all) * 100
            print(f"{task}\t{num_all}\tEM={split:.1f}\tupdated={updated}")

        elif task in {"property_prediction-bbbp", "property_prediction-clintox", "property_prediction-hiv", "property_prediction-sider"}:
            num_all = m.get("num_all", 0)
            f1 = m.get("f1_score", 0) * 100
            mcc = m.get("mcc", 0)
            print(f"{task}\t{num_all}\tf1={f1:.1f}\tmcc={mcc:.2f}\tupdated={updated}")

        elif task in {"property_prediction-esol", "property_prediction-lipo"}:
            rmse = m.get("RMSE", 0)
            print(f"{task}\tRMSE={rmse:.2f}\tupdated={updated}")

def main():
    import argparse
    ap = argparse.ArgumentParser("SMolInstruct postprocess+score (in-place pred fix)")
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--save_json", type=str, default="")
    ap.add_argument("--score_workers", type=int, default=1)
    ap.add_argument("--skip_tasks", type=str, default="")
    ap.add_argument("--inplace_fix", type=int, default=1, help="1: write fixed pred back to original jsonl")
    ap.add_argument("--backup", type=int, default=1, help="1: create .bak backup per task file (first time)")
    ap.add_argument("--force_overwrite_pred", type=int, default=0, help="1: always re-extract and overwrite pred")
    ap.add_argument("--show_sample_progress", type=int, default=0, help="reserved (kept for compatibility)")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir).expanduser().resolve()
    assert pred_dir.is_dir(), f"Not a directory: {pred_dir}"

    files = [p for p in pred_dir.iterdir() if _is_task_file(p)]
    if not files:
        print(f"[WARN] No known task jsonl found in: {pred_dir}")
        return

    skip_set = set()
    if args.skip_tasks.strip():
        skip_set = {t.strip() for t in args.skip_tasks.split(",") if t.strip()}
        files = [p for p in files if p.stem not in skip_set]
        if skip_set:
            print("[INFO] Skip tasks:", ", ".join(sorted(skip_set)))

    print(f"[INFO] Found {len(files)} task files in {pred_dir}")
    for p in sorted(files):
        print("  -", p.name)

    all_results: Dict[str, Dict] = {}
    inplace_fix = bool(args.inplace_fix)
    backup = bool(args.backup)
    force_overwrite_pred = bool(args.force_overwrite_pred)

    if args.score_workers <= 1:
        for fp in tqdm(sorted(files), desc="Scoring tasks", unit="task"):
            task, metrics = _score_single_file((str(fp), inplace_fix, backup, force_overwrite_pred, False))
            all_results[task] = metrics
    else:
        n_workers = cpu_count() if args.score_workers <= 0 else args.score_workers
        args_list = [(str(fp), inplace_fix, backup, force_overwrite_pred, False) for fp in files]
        print(f"[INFO] Parallel scoring with workers={n_workers}")
        with Pool(processes=n_workers) as pool:
            for task, metrics in tqdm(
                pool.imap_unordered(_score_single_file, args_list),
                total=len(args_list),
                desc="Scoring tasks",
                unit="task",
            ):
                all_results[task] = metrics

    if args.save_json:
        outp = Path(args.save_json).expanduser().resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] Saved metrics to: {outp}")

    print_summary(all_results)

if __name__ == "__main__":
    main()
