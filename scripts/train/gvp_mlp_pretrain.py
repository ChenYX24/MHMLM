#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-GPU MLP pretrain on top of a frozen GNN (SMILES -> target_embedding).

- 纯 PyTorch、单卡、尽量精简、无多线程 DataLoader（num_workers=0）。
- Step 1: 用冻结的 GNN 批量编码 SMILES -> 缓存 enc_feats.npy / enc_targets.npy / encode_meta.json。
- Step 2: 训练 MLP 适配器（MSE + cosine），支持 TensorBoard 与 CSV 日志。
- 输出：最佳权重 mlp_adapter.pt 与训练元信息 training_meta.json。

数据格式：.jsonl / .json / .csv / .pkl
  每条样本需包含：
    - "smiles": string
    - "target"（或 .pkl 中的 "gnn_embedding"）: 1-D list/array，目标向量
"""

import os
import sys
from pathlib import Path

# 自动添加项目根目录到 Python 路径，以便导入 modules
_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent  # scripts/ -> MHMLM/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import csv
import json
import time
import argparse
import importlib
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader


# ----------------------------- Utils -----------------------------

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def atomic_write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def import_by_path(path: str):
    """'pkg.mod:ClassName' -> Class object"""
    if ":" not in path:
        raise ValueError(f"--gnn-class must be 'pkg.mod:Class', got: {path}")
    mod_name, cls_name = path.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)

def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = F.normalize(pred, dim=-1)
    t = F.normalize(target, dim=-1)
    return 1.0 - (p * t).sum(dim=-1)


# ----------------------------- MLP Adapter -----------------------------

class MLPAdapter(nn.Module):
    """简洁 MLP: [in] -> (Linear+GELU)*(num_layers-1) -> Linear[out]"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 1536, num_layers: int = 2):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------- Dataset -----------------------------

@dataclass
class Sample:
    smiles: str
    target: np.ndarray  # (D,)

class SmilesTargetDataset(Dataset):
    """
    接受 .jsonl / .json / .csv / .pkl
      - 每项必须包含 "smiles" 和 "target"（或 .pkl 中的 "gnn_embedding"）
    """
    def __init__(self, data_path: str):
        self.samples: List[Sample] = self._load_samples(data_path)
        if not self.samples:
            raise RuntimeError(f"No valid samples in {data_path}")
        dims = {s.target.shape[0] for s in self.samples}
        if len(dims) != 1:
            raise ValueError(f"Inconsistent target dims: {dims}")
        self.target_dim = next(iter(dims))

    @staticmethod
    def _to_array(t) -> np.ndarray:
        if isinstance(t, str) and t.endswith(".npy") and os.path.isfile(t):
            arr = np.load(t)
        else:
            arr = np.asarray(t, dtype=np.float32)
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise ValueError(f"Target must be 1-D, got {arr.shape}")
        return arr.astype(np.float32)

    def _load_samples(self, path: str) -> List[Sample]:
        items: List[Sample] = []
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    smi = obj.get("smiles"); tgt = obj.get("target")
                    if smi is None or tgt is None:
                        continue
                    items.append(Sample(smi, self._to_array(tgt)))
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                arr = json.load(f)
            for obj in arr:
                smi = obj.get("smiles"); tgt = obj.get("target")
                if smi is None or tgt is None:
                    continue
                items.append(Sample(smi, self._to_array(tgt)))
        elif path.endswith(".csv"):
            with open(path, newline='', encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    smi = row.get("smiles"); tgt = row.get("target")
                    if smi is None or tgt is None:
                        continue
                    try:
                        tgt_v = json.loads(tgt)
                    except Exception:
                        tgt_v = [float(x) for x in tgt.split(",")]
                    items.append(Sample(smi, self._to_array(tgt_v)))
        elif path.endswith(".pkl"):
            import pickle
            with open(path, "rb") as f:
                data = pickle.load(f)
            for obj in data:
                if not obj: continue
                smi = obj.get("smiles")
                tgt = obj.get("target", obj.get("gnn_embedding"))
                if smi is None or tgt is None:
                    continue
                items.append(Sample(smi, self._to_array(tgt)))
        else:
            raise ValueError("Unsupported data; use .jsonl/.json/.csv/.pkl")
        return items

    def __len__(self): return len(self.samples)
    def __getitem__(self, i: int) -> Sample: return self.samples[i]


# ----------------------------- Frozen GNN Wrapper -----------------------------

class FrozenGNNWrapper(nn.Module):
    """
    包装用户自定义 GNN 编码器。需要实现之一：
      - forward(batch_smiles: List[str]) -> Tensor[N, D]
      - forward_from_smiles(smiles: str) -> Tensor[1, D] or [D]
    """
    def __init__(self, gnn_cls_path: str, gnn_config: Optional[Dict[str, Any]] = None,
                 gnn_ckpt: Optional[str] = None, device: str = "cpu"):
        super().__init__()
        GNNCls = import_by_path(gnn_cls_path)
        default_cfg = {
            "node_dims": (10, 1),
            "edge_dims": (1, 1),
            "hidden_scalar_dim": 256,
            "hidden_vector_dim": 16,
            "output_dim": 256,
            "num_layers": 4,
        }
        self.model = GNNCls(**(gnn_config or default_cfg))
        if gnn_ckpt and os.path.isfile(gnn_ckpt):
            sd = torch.load(gnn_ckpt, map_location="cpu")
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=False)

        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = torch.device(device)

    @torch.no_grad()
    def encode_batch_smiles(self, smiles_list: List[str]) -> Tuple[List[np.ndarray], List[int]]:
        feats: List[np.ndarray] = []
        bad: List[int] = []

        # 优先使用 GVPEncoder 提供的 batch 接口，避免静默退回到逐条编码
        if hasattr(self.model, "encode_smiles_batch"):
            out = self.model.encode_smiles_batch(smiles_list)
            if not isinstance(out, torch.Tensor):
                raise TypeError("encode_smiles_batch must return a torch.Tensor")
            arr = out.detach().float().cpu().numpy()
            if arr.ndim != 2 or arr.shape[0] != len(smiles_list):
                raise ValueError(f"encode_smiles_batch output shape mismatch: {arr.shape}, expected (N,D)")
            for i in range(arr.shape[0]):
                v = arr[i]
                if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                    bad.append(i)
                else:
                    feats.append(v.astype(np.float32))
            return feats, bad

        # 次优先：部分 GNN 可能支持 forward(list[str]) 作为 batch 接口
        if hasattr(self.model, "forward"):
            out = self.model.forward(smiles_list)
            if not isinstance(out, torch.Tensor):
                raise TypeError("GNN forward must return a torch.Tensor")
            arr = out.detach().float().cpu().numpy()
            if arr.ndim != 2 or arr.shape[0] != len(smiles_list):
                raise ValueError(f"GNN forward output shape mismatch: {arr.shape}, expected (N,D)")
            for i in range(arr.shape[0]):
                v = arr[i]
                if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                    bad.append(i)
                else:
                    feats.append(v.astype(np.float32))
            return feats, bad

        # Slow path: one-by-one
        for i, smi in enumerate(smiles_list):
            try:
                if hasattr(self.model, "forward_from_smiles"):
                    e = self.model.forward_from_smiles(smi)
                else:
                    raise AttributeError("GNN must implement forward() or forward_from_smiles()")
                if isinstance(e, torch.Tensor):
                    e = e.squeeze(0).detach().float().cpu().numpy()
                else:
                    raise TypeError("GNN encoder returned non-tensor")
                if e.ndim != 1:
                    raise ValueError(f"Embedding must be 1-D, got {e.shape}")
                if np.any(np.isnan(e)) or np.any(np.isinf(e)):
                    bad.append(i)
                else:
                    feats.append(e.astype(np.float32))
            except Exception:
                bad.append(i)
        return feats, bad


# ----------------------------- Encode or Load Cache -----------------------------

def encode_or_load(args, device, ds: SmilesTargetDataset):
    """Return (feats_all: np.ndarray [N, Din], tgts_all: np.ndarray [N, Dout])"""
    ensure_dir(args.outdir)
    feats_path = os.path.join(args.outdir, "enc_feats.npy")
    tgts_path  = os.path.join(args.outdir, "enc_targets.npy")
    meta_path  = os.path.join(args.outdir, "encode_meta.json")
    bad_path   = os.path.join(args.outdir, "bad_smiles.txt")

    if args.use_cache and os.path.isfile(feats_path) and os.path.isfile(tgts_path) and os.path.isfile(meta_path):
        print("[cache] Found encoded cache. Loading...")
        feats_all = np.load(feats_path)
        tgts_all  = np.load(tgts_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            _ = json.load(f)
        print(f"[cache] N={feats_all.shape[0]} Din={feats_all.shape[1]} Dout={tgts_all.shape[1]}")
        return feats_all, tgts_all

    # fresh encode
    print("[encode] Building frozen GNN...")
    gnn_conf = None
    if args.gnn_config and os.path.isfile(args.gnn_config):
        with open(args.gnn_config, "r", encoding="utf-8") as f:
            gnn_conf = json.load(f)
    gnn = FrozenGNNWrapper(args.gnn_class, gnn_conf, args.gnn_ckpt, device=str(device))

    bs = max(1, int(args.gnn_batch_size))
    smiles_all  = [s.smiles for s in ds.samples]
    targets_all = [s.target for s in ds.samples]

    feat_chunks, tgt_chunks = [], []
    total_bad = 0
    if os.path.exists(bad_path):
        try: os.remove(bad_path)
        except Exception: pass

    print(f"[encode] Total {len(smiles_all)} SMILES, batch={bs}")
    for st in tqdm(range(0, len(smiles_all), bs), desc="GNN encode"):
        ed   = min(len(smiles_all), st + bs)
        smiB = smiles_all[st:ed]
        tgtB = targets_all[st:ed]
        feats, bad_idx = gnn.encode_batch_smiles(smiB)

        keep_mask = np.ones(len(smiB), dtype=bool)
        for j in bad_idx:
            keep_mask[j] = False

        if bad_idx:
            with open(bad_path, "a", encoding="utf-8") as f:
                for j in bad_idx:
                    f.write(smiB[j] + "\n")
            total_bad += len(bad_idx)

        if feats:
            feat_chunks.append(np.stack(feats, axis=0).astype(np.float32))
            kept_targets = [tgtB[j] for j in range(len(tgtB)) if keep_mask[j]]
            tgt_chunks.append(np.stack(kept_targets, axis=0).astype(np.float32))

    if len(feat_chunks) == 0:
        raise RuntimeError("No valid embeddings after GNN encode (all failed).")

    feats_all = np.concatenate(feat_chunks, axis=0)
    tgts_all  = np.concatenate(tgt_chunks,  axis=0)
    print(f"[encode] Done. kept={feats_all.shape[0]} bad={total_bad} Din={feats_all.shape[1]} Dout={tgts_all.shape[1]}")

    # save cache
    np.save(feats_path, feats_all)
    np.save(tgts_path,  tgts_all)
    atomic_write_json(meta_path, {
        "kept": int(feats_all.shape[0]),
        "feat_dim": int(feats_all.shape[1]),
        "tgt_dim": int(tgts_all.shape[1]),
        "total_bad": int(total_bad),
        "bad_smiles_path": bad_path if total_bad > 0 else "",
    })
    print(f"[encode] Saved cache to: {feats_path}, {tgts_path}, {meta_path}")
    return feats_all, tgts_all


# ----------------------------- Train -----------------------------

def train_mlp(args, device, feats_all: np.ndarray, tgts_all: np.ndarray):
    # optional normalize targets
    norm_meta = {"type": "none"}
    if args.target_normalize == "zscore":
        mean = tgts_all.mean(axis=0, keepdims=True)
        std  = tgts_all.std(axis=0, keepdims=True) + 1e-6
        tgts_all = (tgts_all - mean) / std
        norm_meta = {"type": "zscore", "mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()}
    elif args.target_normalize == "l2":
        nrm = np.linalg.norm(tgts_all, axis=1, keepdims=True) + 1e-6
        tgts_all = tgts_all / nrm
        norm_meta = {"type": "l2"}

    N = feats_all.shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)

    val_n = max(1, int(N * args.val_ratio))
    val_idx, tr_idx = idx[:val_n], idx[val_n:]

    # 保持 CPU 张量，后续每个 batch 再搬到 GPU（避免 pin_memory 报错）
    x_tr = torch.from_numpy(feats_all[tr_idx])  # CPU
    y_tr = torch.from_numpy(tgts_all[tr_idx])  # CPU
    x_va = torch.from_numpy(feats_all[val_idx])  # CPU
    y_va = torch.from_numpy(tgts_all[val_idx])  # CPU

    train_ds = TensorDataset(x_tr, y_tr)
    val_ds   = TensorDataset(x_va, y_va)

    use_pin = (device.type == "cuda")
    tr_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=use_pin)
    va_loader = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=use_pin)

    in_dim  = feats_all.shape[1]
    out_dim = tgts_all.shape[1]
    model = MLPAdapter(in_dim, out_dim, args.hidden_dim, args.num_layers).to(device)

    if args.resume and os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"[resume] Loaded MLP weights from {args.resume}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    else:
        sched = None

    # 新式 GradScaler（仅在 CUDA+AMP 时启用）
    use_cuda_amp = (device.type == "cuda") and args.amp and (not args.bf16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    # 日志
    tb_writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = args.tb_logdir if args.tb_logdir else os.path.join(args.outdir, "tb_logs")
        ensure_dir(tb_dir)
        tb_writer = SummaryWriter(log_dir=tb_dir)
        print(f"[log] TensorBoard: {tb_dir}")

    csv_path = os.path.join(args.outdir, "train_log.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_file)
    csv_w.writerow(["epoch","train_mse","train_cos","val_mse","val_cos","lr","seconds"])
    csv_file.flush()
    print(f"[log] CSV: {csv_path}")

    best_val = float("inf")
    best_path = os.path.join(args.outdir, "mlp_adapter.pt")

    def run_one_epoch(split: str, epoch: int):
        is_train = (split == "train")
        loader = tr_loader if is_train else va_loader
        model.train(is_train)

        total_mse = 0.0
        total_cos = 0.0
        total_n   = 0

        for x, y in loader:
            # CPU -> GPU（或保持 CPU），避免 pin_memory 针对 GPU Tensor 报错
            if device.type == "cuda":
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            else:
                x = x.to(device)
                y = y.to(device)

            # autocast 仅在 CUDA 上启用；CPU 上用空上下文
            if device.type == "cuda" and (args.amp or args.bf16):
                ac = torch.amp.autocast(device_type="cuda", dtype=autocast_dtype)
            else:
                ac = nullcontext()

            with ac:
                pred = model(x)
                mse  = F.mse_loss(pred, y)
                cos  = cosine_loss(pred, y).mean()
                loss = args.alpha * mse + (1.0 - args.alpha) * cos

            if is_train:
                opt.zero_grad(set_to_none=True)
                if use_cuda_amp:
                    scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()

            bs = x.size(0)
            total_mse += mse.item() * bs
            total_cos += cos.item() * bs
            total_n   += bs

        return total_mse / max(1,total_n), total_cos / max(1,total_n)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_mse, tr_cos = run_one_epoch("train", ep)
        va_mse, va_cos = run_one_epoch("eval",  ep)
        dt = time.time() - t0

        if sched is not None:
            if args.scheduler == "plateau":
                val_score = args.alpha * va_mse + (1.0 - args.alpha) * va_cos
                sched.step(val_score)
            else:
                sched.step()

        lr_now = opt.param_groups[0]["lr"]
        print(f"[ep {ep:03d}] train MSE={tr_mse:.6f} COS={tr_cos:.6f} | val MSE={va_mse:.6f} COS={va_cos:.6f} | lr={lr_now:.3e} | {dt:.1f}s")
        csv_w.writerow([ep, f"{tr_mse:.6f}", f"{tr_cos:.6f}", f"{va_mse:.6f}", f"{va_cos:.6f}", f"{lr_now:.6e}", f"{dt:.3f}"])
        csv_file.flush()
        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch_mse", tr_mse, ep)
            tb_writer.add_scalar("train/epoch_cos", tr_cos, ep)
            tb_writer.add_scalar("val/epoch_mse",   va_mse, ep)
            tb_writer.add_scalar("val/epoch_cos",   va_cos, ep)
            tb_writer.add_scalar("opt/lr", lr_now, ep)
            tb_writer.flush()

        val_score = args.alpha * va_mse + (1.0 - args.alpha) * va_cos
        if val_score < best_val:
            best_val = val_score
            torch.save(model.state_dict(), best_path)
            meta = {
                "epoch": ep,
                "best_val": float(best_val),
                "in_dim": int(in_dim),
                "out_dim": int(out_dim),
                "hidden_dim": int(args.hidden_dim),
                "num_layers": int(args.num_layers),
                "alpha": float(args.alpha),
                "target_normalize": norm_meta,
                "scheduler": args.scheduler,
                "bf16": bool(args.bf16),
                "amp": bool(args.amp),
                "kept_samples": int(N),
            }
            atomic_write_json(os.path.join(args.outdir, "training_meta.json"), meta)
            print(f"  ↳ ✅ saved best to {best_path}")

    csv_file.close()
    if tb_writer is not None:
        tb_writer.close()
    print("[train] Done.")


# ----------------------------- CLI -----------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description="Single-GPU: Pretrain MLP on frozen GNN (SMILES -> target embedding)")
    ap.add_argument('--data', type=str, required=True, help='Path to .jsonl/.json/.csv/.pkl (fields: smiles, target)')
    ap.add_argument('--outdir', type=str, required=True, help='Output directory')

    # GNN
    ap.add_argument('--gnn-class', type=str, required=True, help="Import path to GNN class, e.g. 'modules.gnn:GVPEncoder'")
    ap.add_argument('--gnn-config', type=str, default=None, help='Optional JSON file with GNN init kwargs')
    ap.add_argument('--gnn-ckpt', type=str, default=None, help='Optional checkpoint for GNN state_dict')
    ap.add_argument('--gnn-batch-size', type=int, default=128, help='Batch size for GNN feature extraction')

    # MLP
    ap.add_argument('--hidden-dim', type=int, default=1536)
    ap.add_argument('--num-layers', type=int, default=2)

    # Optimization
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--scheduler', type=str, default='plateau', choices=['none', 'cosine', 'plateau'])
    ap.add_argument('--alpha', type=float, default=0.5, help='Loss mix: alpha*MSE + (1-alpha)*cosine')
    ap.add_argument('--grad-clip', type=float, default=1.0)

    # Normalization & split
    ap.add_argument('--target-normalize', type=str, default='none', choices=['none', 'zscore', 'l2'])
    ap.add_argument('--val-ratio', type=float, default=0.05)

    # Precision & device
    ap.add_argument('--bf16', action='store_true', help='Use bfloat16 autocast')
    ap.add_argument('--amp', action='store_true', help='Use mixed precision autocast & GradScaler (disabled if bf16)')
    ap.add_argument('--device', type=str, default=None, help="cuda, cuda:0, or cpu")

    # Cache & resume
    ap.add_argument('--use-cache', action='store_true', help='Use encoded cache if present; otherwise encode & save')
    ap.add_argument('--resume', type=str, default=None, help='Path to an existing mlp_adapter.pt to resume')

    # Logging
    ap.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    ap.add_argument('--tb-logdir', type=str, default=None, help='TensorBoard log dir (default: <outdir>/tb_logs)')

    # Misc
    ap.add_argument('--seed', type=int, default=42)
    return ap


# ----------------------------- Main -----------------------------

def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    if args.bf16:
        torch.set_float32_matmul_precision('medium')

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ensure_dir(args.outdir)

    print("[data] Loading dataset...")
    ds = SmilesTargetDataset(args.data)
    print(f"[data] samples={len(ds.samples)} target_dim={ds.target_dim}")

    feats_all, tgts_all = encode_or_load(args, device, ds)
    train_mlp(args, device, feats_all, tgts_all)


if __name__ == "__main__":
    main()


# python gvp_mlp_pretrain.py \
#   --data /data1/chenyuxuan/Project/MSMLM/data/traindata/chatmol/chatmol_gnn.pkl \
#   --outdir /data1/chenyuxuan/Project/MSMLM/model/gnn_mlp_single \
#   --gnn-class modules.gnn:GVPEncoder \
#   --gnn-ckpt /data1/lvchangwei/GNN/Project/GVP/checkpoints_256_wo/gvp_weights_best.pt \
#   --epochs 100 --batch-size 256 --lr 1e-5 --alpha 0.5 --num-layers 3\
#   --bf16 --amp --use-cache --tensorboard

# python gvp_mlp_pretrain.py \
#   --data /data1/chenyuxuan/Project/MSMLM/data/traindata/chatmol/chatmol_gnn.pkl \
#   --outdir /data1/chenyuxuan/Project/MSMLM/model/gnn_mlp_single \
#   --gnn-class modules.gnn:GVPEncoder \
#   --gnn-ckpt /data1/lvchangwei/GNN/Project/GVP/checkpoints_256_wo/gvp_weights_best.pt \
#   --epochs 30 \
#   --batch-size 512 \
#   --hidden-dim 2048 \
#   --num-layers 3 \
#   --lr 3e-3 \
#   --scheduler cosine \
#   --alpha 0.2 \
#   --target-normalize l2 \
#   --bf16 --amp --use-cache --tensorboard