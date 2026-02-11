#!/usr/bin/env python3
"""
Yield Calibration with K-Fold / Bootstrap Validation

Learns per-class bias correction on validation logits and validates
the improvement with K-fold cross-validation or bootstrap resampling.

Usage:
    cd /data1/chenyuxuan/MHMLM
    # K-fold validation (recommended)
    python scripts/layer2/calibrate_yield.py \
        --checkpoint /path/to/checkpoint.pt \
        --data /data1/chenyuxuan/Layer2/data/ord_layer2_v2/layer2_val.jsonl \
        --mode kfold --kfold 5

    # Bootstrap validation
    python scripts/layer2/calibrate_yield.py \
        --checkpoint /path/to/checkpoint.pt \
        --data /data1/chenyuxuan/Layer2/data/ord_layer2_v2/layer2_val.jsonl \
        --mode bootstrap --bootstrap 200

    # Both modes
    python scripts/layer2/calibrate_yield.py \
        --checkpoint /path/to/checkpoint.pt \
        --data /data1/chenyuxuan/Layer2/data/ord_layer2_v2/layer2_val.jsonl \
        --mode both
"""

import sys
import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from modules.layer2_component.model import ModelConfig, Layer2PretrainModel
from modules.layer2_component.dataset import Layer2JsonlIndexed
from modules.layer2_component.collate import collate_layer2
from modules.layer2_component.masking import EvalMaskingConfig


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model_cfg = ModelConfig(
        mol_emb_dim=config.get("mol_emb_dim", 256),
        hidden_dim=config.get("hidden_dim", 768),
        n_layers=config.get("n_layers", 8),
        n_heads=config.get("n_heads", 12),
        dropout=config.get("dropout", 0.2),
        num_roles=config.get("num_roles", 11),
        num_token_types=config.get("num_token_types", 2),
        tau=config.get("tau", 0.07),
        learnable_tau=config.get("learnable_tau", False),
        symmetric_ince=config.get("symmetric_ince", False),
        use_projection_head=config.get("use_projection_head", False),
        head_dropout=config.get("head_dropout", 0.0),
    )
    model = Layer2PretrainModel(model_cfg)
    sd = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def extract_yield_logits(model, data_path: str, batch_size: int = 64, num_workers: int = 2):
    """Extract all yield logits and labels from a dataset."""
    ds = Layer2JsonlIndexed(data_path, masking=True, masking_cfg=EvalMaskingConfig())
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_layer2,
                        num_workers=num_workers, shuffle=False)

    all_logits = []
    all_labels = []
    all_reg_pred = []
    all_reg_true = []

    for batch in loader:
        out = model(batch)
        y_mask = batch.yield_pred_mask
        idx = (y_mask > 0.5).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        logits = out["pred_yield_bin"][idx].float().cpu()
        labels = batch.yield_bin[idx].cpu()
        reg_pred = out["pred_yield_reg"][idx].float().cpu()
        reg_true = batch.yield_reg[idx].cpu()

        all_logits.append(logits)
        all_labels.append(labels)
        all_reg_pred.append(reg_pred)
        all_reg_true.append(reg_true)

    logits = torch.cat(all_logits, dim=0)  # [N, 10]
    labels = torch.cat(all_labels, dim=0)  # [N]
    reg_pred = torch.cat(all_reg_pred, dim=0)
    reg_true = torch.cat(all_reg_true, dim=0)
    return logits, labels, reg_pred, reg_true


def fit_bias(logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.1, steps: int = 500):
    """Learn per-class bias by minimizing CE loss on logits using LBFGS."""
    bias = torch.zeros(logits.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS([bias], lr=lr, max_iter=20)

    def closure():
        optimizer.zero_grad()
        calibrated = logits + bias
        loss = F.cross_entropy(calibrated, labels)
        loss.backward()
        return loss

    for _ in range(steps):
        optimizer.step(closure)

    return bias.detach().clone()


def eval_accuracy(logits: torch.Tensor, labels: torch.Tensor, bias: torch.Tensor = None):
    """Compute bin accuracy and relaxed accuracy (within +/-1 bin)."""
    if bias is not None:
        logits = logits + bias
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    relaxed = ((preds - labels).abs() <= 1).float().mean().item()
    return acc, relaxed


def eval_per_class(logits: torch.Tensor, labels: torch.Tensor, bias: torch.Tensor = None):
    """Per-class accuracy."""
    if bias is not None:
        logits = logits + bias
    preds = logits.argmax(dim=-1)
    per_class = {}
    for c in range(10):
        mask = labels == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == c).float().mean().item()
        else:
            per_class[c] = float("nan")
    return per_class


def kfold_validation(logits, labels, K=5, seed=42):
    """K-fold cross-validation of bias calibration."""
    N = logits.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(N)

    fold_size = N // K
    results = []

    for k in range(K):
        if k < K - 1:
            val_start = k * fold_size
            val_end = (k + 1) * fold_size
        else:
            val_start = k * fold_size
            val_end = N

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        train_logits = logits[train_idx]
        train_labels = labels[train_idx]
        val_logits = logits[val_idx]
        val_labels = labels[val_idx]

        # Baseline (no calibration)
        base_acc, base_relax = eval_accuracy(val_logits, val_labels)

        # Fit bias on train fold
        bias = fit_bias(train_logits, train_labels)

        # Eval on val fold
        cal_acc, cal_relax = eval_accuracy(val_logits, val_labels, bias)

        results.append({
            "fold": k,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "baseline_acc": base_acc,
            "baseline_relaxed": base_relax,
            "calibrated_acc": cal_acc,
            "calibrated_relaxed": cal_relax,
            "improvement": cal_acc - base_acc,
            "relaxed_improvement": cal_relax - base_relax,
            "bias": bias.tolist(),
        })

    return results


def bootstrap_validation(logits, labels, B=200, seed=42):
    """Bootstrap validation of bias calibration."""
    N = logits.shape[0]
    rng = np.random.RandomState(seed)

    # Overall baseline
    base_acc, base_relax = eval_accuracy(logits, labels)

    improvements = []
    relaxed_improvements = []
    biases = []

    for b in range(B):
        # Resample with replacement
        idx = rng.choice(N, size=N, replace=True)
        oob_mask = np.ones(N, dtype=bool)
        oob_mask[np.unique(idx)] = False
        oob_idx = np.where(oob_mask)[0]

        if len(oob_idx) < 5:
            continue

        train_logits = logits[idx]
        train_labels = labels[idx]
        oob_logits = logits[oob_idx]
        oob_labels = labels[oob_idx]

        # Fit on bootstrap sample
        bias = fit_bias(train_logits, train_labels)

        # Eval on OOB
        oob_base, oob_base_relax = eval_accuracy(oob_logits, oob_labels)
        oob_cal, oob_cal_relax = eval_accuracy(oob_logits, oob_labels, bias)

        improvements.append(oob_cal - oob_base)
        relaxed_improvements.append(oob_cal_relax - oob_base_relax)
        biases.append(bias.numpy())

    improvements = np.array(improvements)
    relaxed_improvements = np.array(relaxed_improvements)
    biases = np.array(biases)

    return {
        "baseline_acc": base_acc,
        "baseline_relaxed": base_relax,
        "n_bootstrap": len(improvements),
        "improvement_mean": float(improvements.mean()),
        "improvement_std": float(improvements.std()),
        "improvement_median": float(np.median(improvements)),
        "improvement_95ci": [float(np.percentile(improvements, 2.5)),
                              float(np.percentile(improvements, 97.5))],
        "pct_positive": float((improvements > 0).mean()),
        "relaxed_improvement_mean": float(relaxed_improvements.mean()),
        "relaxed_improvement_std": float(relaxed_improvements.std()),
        "bias_mean": biases.mean(axis=0).tolist(),
        "bias_std": biases.std(axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Yield calibration with K-fold / Bootstrap")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, help="Val JSONL file")
    parser.add_argument("--mode", type=str, default="both", choices=["kfold", "bootstrap", "both"])
    parser.add_argument("--kfold", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    args = parser.parse_args()

    print("=" * 60)
    print("YIELD CALIBRATION - ROBUST VALIDATION")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    print(f"Mode: {args.mode}")

    print(f"\nLoading model...")
    model = load_model(args.checkpoint)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    print(f"Extracting yield logits...")
    logits, labels, reg_pred, reg_true = extract_yield_logits(model, args.data, args.batch_size)
    N = logits.shape[0]
    print(f"Yield samples: {N}")

    # Label distribution
    dist = Counter(labels.numpy().tolist())
    print(f"Label distribution: {dict(sorted(dist.items()))}")

    # Overall baseline
    base_acc, base_relax = eval_accuracy(logits, labels)
    print(f"\n{'='*60}")
    print(f"BASELINE (no calibration)")
    print(f"{'='*60}")
    print(f"  bin_accuracy: {base_acc:.4f}")
    print(f"  relaxed_acc:  {base_relax:.4f}")

    # Regression metrics
    reg_mae = (reg_pred - reg_true).abs().mean().item()
    reg_rmse = ((reg_pred - reg_true) ** 2).mean().sqrt().item()
    print(f"  yield_mae:    {reg_mae:.4f}")
    print(f"  yield_rmse:   {reg_rmse:.4f}")

    # Per-class baseline
    pc = eval_per_class(logits, labels)
    print(f"\n  Per-class accuracy (baseline):")
    for c in range(10):
        cnt = (labels == c).sum().item()
        print(f"    bin {c}: {pc[c]:.3f} (n={cnt})")

    # Full-set calibration (for reference only - overfits!)
    print(f"\n{'='*60}")
    print(f"FULL-SET CALIBRATION (overfit reference, NOT for reporting)")
    print(f"{'='*60}")
    full_bias = fit_bias(logits, labels)
    full_acc, full_relax = eval_accuracy(logits, labels, full_bias)
    full_pc = eval_per_class(logits, labels, full_bias)
    print(f"  calibrated_acc:    {full_acc:.4f} (+{full_acc - base_acc:.4f})")
    print(f"  calibrated_relax:  {full_relax:.4f} (+{full_relax - base_relax:.4f})")
    bias_str = ", ".join(f"{b:.3f}" for b in full_bias.tolist())
    print(f"  bias: [{bias_str}]")
    print(f"\n  Per-class accuracy (full-set calibrated):")
    for c in range(10):
        cnt = (labels == c).sum().item()
        delta = full_pc[c] - pc[c] if not (np.isnan(pc[c]) or np.isnan(full_pc[c])) else 0
        print(f"    bin {c}: {full_pc[c]:.3f} ({delta:+.3f}) (n={cnt})")

    results = {
        "checkpoint": args.checkpoint,
        "n_yield_samples": N,
        "label_distribution": {str(k): v for k, v in sorted(dist.items())},
        "baseline_acc": base_acc,
        "baseline_relaxed": base_relax,
        "reg_mae": reg_mae,
        "reg_rmse": reg_rmse,
        "full_set_calibrated_acc": full_acc,
        "full_set_calibrated_relaxed": full_relax,
        "full_set_bias": full_bias.tolist(),
    }

    # K-fold
    if args.mode in ("kfold", "both"):
        print(f"\n{'='*60}")
        print(f"K-FOLD CROSS-VALIDATION (K={args.kfold})")
        print(f"{'='*60}")
        kf_results = kfold_validation(logits, labels, K=args.kfold, seed=args.seed)

        accs = [r["calibrated_acc"] for r in kf_results]
        base_accs = [r["baseline_acc"] for r in kf_results]
        imps = [r["improvement"] for r in kf_results]
        relax_imps = [r["relaxed_improvement"] for r in kf_results]

        for r in kf_results:
            sign = "+" if r["improvement"] >= 0 else ""
            print(f"  Fold {r['fold']}: baseline={r['baseline_acc']:.4f} -> "
                  f"calibrated={r['calibrated_acc']:.4f} "
                  f"({sign}{r['improvement']:.4f}) "
                  f"[relaxed: {r['baseline_relaxed']:.4f} -> {r['calibrated_relaxed']:.4f}]")

        print(f"\n  Summary:")
        print(f"    baseline acc:    {np.mean(base_accs):.4f} +/- {np.std(base_accs):.4f}")
        print(f"    calibrated acc:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        sign = "+" if np.mean(imps) >= 0 else ""
        print(f"    improvement:     {sign}{np.mean(imps):.4f} +/- {np.std(imps):.4f}")
        sign_r = "+" if np.mean(relax_imps) >= 0 else ""
        print(f"    relaxed improve: {sign_r}{np.mean(relax_imps):.4f} +/- {np.std(relax_imps):.4f}")
        print(f"    all folds positive: {all(i > 0 for i in imps)}")
        print(f"    min improvement:    {min(imps):+.4f}")
        print(f"    max improvement:    {max(imps):+.4f}")

        # Bias stability
        biases = np.array([r["bias"] for r in kf_results])
        print(f"\n  Bias stability (mean +/- std across folds):")
        for c in range(10):
            print(f"    bin {c}: {biases[:, c].mean():+.3f} +/- {biases[:, c].std():.3f}")

        results["kfold"] = {
            "K": args.kfold,
            "folds": kf_results,
            "calibrated_acc_mean": float(np.mean(accs)),
            "calibrated_acc_std": float(np.std(accs)),
            "improvement_mean": float(np.mean(imps)),
            "improvement_std": float(np.std(imps)),
            "relaxed_improvement_mean": float(np.mean(relax_imps)),
            "relaxed_improvement_std": float(np.std(relax_imps)),
            "all_positive": all(i > 0 for i in imps),
            "bias_mean": biases.mean(axis=0).tolist(),
            "bias_std": biases.std(axis=0).tolist(),
        }

    # Bootstrap
    if args.mode in ("bootstrap", "both"):
        print(f"\n{'='*60}")
        print(f"BOOTSTRAP VALIDATION (B={args.bootstrap})")
        print(f"{'='*60}")
        bs_results = bootstrap_validation(logits, labels, B=args.bootstrap, seed=args.seed)

        print(f"  baseline acc:        {bs_results['baseline_acc']:.4f}")
        sign = "+" if bs_results["improvement_mean"] >= 0 else ""
        print(f"  improvement (OOB):   {sign}{bs_results['improvement_mean']:.4f} +/- {bs_results['improvement_std']:.4f}")
        print(f"  95% CI improvement:  [{bs_results['improvement_95ci'][0]:+.4f}, {bs_results['improvement_95ci'][1]:+.4f}]")
        print(f"  % positive improve:  {bs_results['pct_positive']*100:.1f}%")
        sign_r = "+" if bs_results["relaxed_improvement_mean"] >= 0 else ""
        print(f"  relaxed improvement: {sign_r}{bs_results['relaxed_improvement_mean']:.4f} +/- {bs_results['relaxed_improvement_std']:.4f}")

        print(f"\n  Bias stability (bootstrap mean +/- std):")
        for c in range(10):
            print(f"    bin {c}: {bs_results['bias_mean'][c]:+.3f} +/- {bs_results['bias_std'][c]:.3f}")

        results["bootstrap"] = bs_results

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
