#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a combined MolAwareCausalLM checkpoint into:
- <dst>/llm/          : pure HF LLM weights/config
- <dst>/extras/       : gvp_encoder.pt, mol_adapter.pt, diffusion_adapter.pt (optional diffusion.pt)
- <dst>/molaware_metadata.json
- <dst>/tokenizer*    : tokenizer files (copied/saved if available)

Design principles:
- NEVER add/expand tokens inside this tool. Match src checkpoint exactly.
- Align cfg.vocab_size and model's embedding rows to src tokenizer length BEFORE load_state_dict.
"""

import os
import re
import json
import argparse
from typing import Dict, Any, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

try:
    from safetensors.torch import load_file as safe_load
    HAVE_SAFE = True
except Exception:
    HAVE_SAFE = False


# ---------------- IO helpers ----------------
def find_weight_file(src_dir: str) -> Tuple[str, str]:
    """
    Return (path, fmt) where fmt in {"safetensors","bin","bin-index"}.
    """
    st = os.path.join(src_dir, "model.safetensors")
    if os.path.isfile(st):
        return st, "safetensors"
    bi = os.path.join(src_dir, "pytorch_model.bin")
    if os.path.isfile(bi):
        return bi, "bin"
    idx = os.path.join(src_dir, "pytorch_model.bin.index.json")
    if os.path.isfile(idx):
        return idx, "bin-index"
    raise FileNotFoundError("No model.safetensors / pytorch_model.bin / *.bin.index.json found in src.")


def load_any_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        if not HAVE_SAFE:
            raise RuntimeError("safetensors not available; pip install safetensors or provide .bin")
        return safe_load(path, device="cpu")
    if path.endswith(".bin"):
        return torch.load(path, map_location="cpu")
    if path.endswith(".json") and path.endswith("bin.index.json"):
        with open(path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        sd = {}
        base = os.path.dirname(path)
        # 使用集合避免重复加载 shard
        for shard in sorted(set(weight_map.values())):
            shard_path = os.path.join(base, shard)
            part = torch.load(shard_path, map_location="cpu")
            sd.update(part)
        return sd
    raise ValueError(f"Unsupported weight file: {path}")


# ---------------- split heuristics ----------------
def split_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Separate into blocks: llm, gvp_encoder, mol_adapter, diffusion_adapter, diffusion (optional).
    - If keys have 'llm.' prefix -> strip it and put to llm
    - Else classify by known prefixes or common transformer names (LLaMA).
    """
    parts: Dict[str, Dict[str, torch.Tensor]] = {
        "llm": {},
        "gvp_encoder": {},
        "mol_adapter": {},
        "diffusion_adapter": {},
        "diffusion": {},
    }

    def put(part: str, k: str, v: torch.Tensor):
        parts[part][k] = v

    for k, v in sd.items():
        if k.startswith("llm."):
            put("llm", k[len("llm."):], v)
            continue

        for prefix, part_name in [
            ("gvp_encoder.", "gvp_encoder"),
            ("mol_adapter.", "mol_adapter"),
            ("diffusion_adapter.", "diffusion_adapter"),
            ("diffusion.", "diffusion"),
        ]:
            if k.startswith(prefix):
                put(part_name, k[len(prefix):], v)
                break
        else:
            # Heuristic for LLM block (LLaMA-style)
            if (
                k.startswith("model.") or
                k.startswith("transformer.") or
                k.startswith("layers.") or
                k.startswith("lm_head.") or
                k.startswith("norm.") or
                k.startswith("embed_tokens.")
            ):
                put("llm", k, v)
            else:
                # Nested module names like foo.bar.gvp_encoder.x
                if ".gvp_encoder." in k:
                    put("gvp_encoder", k.split(".gvp_encoder.", 1)[1], v)
                elif ".mol_adapter." in k:
                    put("mol_adapter", k.split(".mol_adapter.", 1)[1], v)
                elif ".diffusion_adapter." in k:
                    put("diffusion_adapter", k.split(".diffusion_adapter.", 1)[1], v)
                elif ".diffusion." in k:
                    put("diffusion", k.split(".diffusion.", 1)[1], v)
                else:
                    # Unrecognized — ignore silently (safer for forward compatibility)
                    pass
    return parts


# ---------------- sanity checks ----------------
def infer_src_vocab_from_weights(llm_subdict: Dict[str, torch.Tensor]) -> int:
    """
    Try to infer vocab size from embed/lm_head shapes if tokenizer is absent.
    """
    for name in ["model.embed_tokens.weight", "embed_tokens.weight"]:
        w = llm_subdict.get(name)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            return w.shape[0]
    # Fallback: check lm_head
    w = llm_subdict.get("lm_head.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return w.shape[0]
    return -1


def ensure_llm_vocab_size_matches_config(cfg: AutoConfig, src_vocab_size: int):
    """
    Mutate cfg.vocab_size if needed to match src vocab size before model instantiation.
    """
    if getattr(cfg, "vocab_size", None) != src_vocab_size and src_vocab_size > 0:
        cfg.vocab_size = src_vocab_size


def align_model_embeddings_to_vocab(model: torch.nn.Module, expected_vocab: int):
    """
    Resize input/output embeddings to expected_vocab if mismatch (before load_state_dict).
    """
    cur = model.get_input_embeddings().weight.shape[0]
    if cur != expected_vocab:
        model.resize_token_embeddings(expected_vocab)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source checkpoint dir (contains model.safetensors or pytorch_model.bin)")
    ap.add_argument("--dst", required=True, help="Destination dir for split output")
    ap.add_argument("--base-llm", default=None, help="(Optional) Base LLM name/path to load config/tokenizer if src missing")
    ap.add_argument("--save-diffusion", action="store_true", help="Also save diffusion.pt (can be very large)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    llm_dir = os.path.join(args.dst, "llm")
    extras_dir = os.path.join(args.dst, "extras")
    os.makedirs(llm_dir, exist_ok=True)
    os.makedirs(extras_dir, exist_ok=True)

    # 1) Load raw state_dict
    weight_path, fmt = find_weight_file(args.src)
    print(f"[info] loading weights: {weight_path} ({fmt})")
    sd_all = load_any_state_dict(weight_path)
    print(f"[info] total tensors: {len(sd_all)}")

    # 2) Split
    parts = split_state_dict(sd_all)
    print(
        "[info] llm params: {llm}, gvp: {gvp}, mol_adapter: {mol}, diff_adapter: {da}, diffusion: {diff}".format(
            llm=len(parts["llm"]),
            gvp=len(parts["gvp_encoder"]),
            mol=len(parts["mol_adapter"]),
            da=len(parts["diffusion_adapter"]),
            diff=len(parts["diffusion"]),
        )
    )
    if not parts["llm"]:
        raise RuntimeError("No LLM weights detected. Are you sure the source is a combined MolAware checkpoint?")

    # 3) Tokenizer — always prefer src tokenizer; do NOT add tokens here
    tok = None
    if any(os.path.isfile(os.path.join(args.src, f)) for f in ["tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt"]):
        tok = AutoTokenizer.from_pretrained(args.src, use_fast=True)
        tok.save_pretrained(args.dst)
        print(f"[info] tokenizer saved to: {args.dst}")
    elif args.base_llm:
        tok = AutoTokenizer.from_pretrained(args.base_llm, use_fast=True)
        tok.save_pretrained(args.dst)
        print(f"[info] tokenizer copied from base-llm to: {args.dst}")
    else:
        print("[warn] No tokenizer found; you may need to save it manually later.")

    # Infer src_vocab_size
    src_vocab_from_tok = len(tok) if tok is not None else -1
    src_vocab_from_w = infer_src_vocab_from_weights(parts["llm"])
    src_vocab_size = src_vocab_from_tok if src_vocab_from_tok > 0 else src_vocab_from_w
    if src_vocab_size <= 0:
        raise RuntimeError("Failed to infer src vocab size from tokenizer and weights.")

    print(f"[info] src_vocab_size = {src_vocab_size} (tokenizer: {src_vocab_from_tok}, weights: {src_vocab_from_w})")

    # 4) Config — prefer src/config.json; else base-llm
    if os.path.isfile(os.path.join(args.src, "config.json")):
        cfg = AutoConfig.from_pretrained(args.src)
    elif args.base_llm:
        cfg = AutoConfig.from_pretrained(args.base_llm)
    else:
        raise RuntimeError("No config.json in src and no --base-llm provided; cannot build LLM.")

    # Force cfg.vocab_size to match src vocab BEFORE constructing model
    ensure_llm_vocab_size_matches_config(cfg, src_vocab_size)

    # 5) Build empty LLM and pre-align its embedding rows to src vocab size
    llm = AutoModelForCausalLM.from_config(cfg)
    align_model_embeddings_to_vocab(llm, src_vocab_size)

    # 6) Now safe to load weights (no size mismatch)
    missing, unexpected = llm.load_state_dict(parts["llm"], strict=False)
    # 常见无害 missing：position_ids / rotary 缓存；无害 unexpected：有的实现会多保存一些 buffer
    print(f"[info] load llm: missing={len(missing)}, unexpected={len(unexpected)} (strict=False)")
    if missing:
        # 打印前几个帮助排查
        print("        missing (head):", missing[:8])
    if unexpected:
        print("        unexpected (head):", unexpected[:8])

    # Sanity check: confirm embedding rows == src_vocab_size
    emb_rows = llm.get_input_embeddings().weight.shape[0]
    lm_head_rows = getattr(llm.get_output_embeddings(), "weight", None)
    lm_head_rows = lm_head_rows.shape[0] if isinstance(lm_head_rows, torch.Tensor) else "N/A"
    print(f"[info] embeddings rows: input={emb_rows}, lm_head={lm_head_rows}")

    # 7) Save LLM block in HF format
    llm.save_pretrained(llm_dir)
    print(f"[ok] saved HF LLM to: {llm_dir}")

    # 8) Save extras (state_dicts)
    meta_extras: Dict[str, str] = {}

    def _save_sub(name: str, sub: Dict[str, torch.Tensor], filename: str):
        if sub:
            path = os.path.join(extras_dir, filename)
            torch.save(sub, path)
            meta_extras[name] = os.path.relpath(path, args.dst)
            print(f"[ok] saved {name} -> {path}")

    _save_sub("gvp_encoder", parts["gvp_encoder"], "gvp_encoder.pt")
    _save_sub("mol_adapter", parts["mol_adapter"], "mol_adapter.pt")
    _save_sub("diffusion_adapter", parts["diffusion_adapter"], "diffusion_adapter.pt")
    if args.save_diffusion:
        _save_sub("diffusion", parts["diffusion"], "diffusion.pt")

    # 9) Metadata for MolAwareCausalLM.from_pretrained
    meta = {
        "class": "MolAwareCausalLM",
        "version": 1,
        "extras": meta_extras,
        # 这里不强行写死 mol_token，避免和不同项目冲突；若需要可在下游覆盖
        "mol_token": "<mol>"
    }
    with open(os.path.join(args.dst, "molaware_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote molaware_metadata.json")

    print("\n[done] Split complete.\n"
          f" - LLM:     {llm_dir}\n"
          f" - EXTRAS:  {extras_dir}\n"
          f" - META:    {os.path.join(args.dst, 'molaware_metadata.json')}\n"
          f" - TOKENIZER in {args.dst} (if available)\n")


if __name__ == "__main__":
    main()
