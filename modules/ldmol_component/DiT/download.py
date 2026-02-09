from __future__ import annotations

import os
import torch


def find_model(ckpt_path: str) -> dict:
    """
    Load a local checkpoint and return a state_dict.
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint
