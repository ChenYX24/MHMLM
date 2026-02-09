from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


ROLE_VOCAB: List[str] = [
    "REACTANT",
    "REAGENT",
    "SOLVENT",
    "CATALYST",
    "PRODUCT",
    "BYPRODUCT",
    "SIDE_PRODUCT",
    "WORKUP",
    "INTERNAL_STANDARD",
    "STANDARD",
    "OTHER",
]
ROLE2ID = {r: i for i, r in enumerate(ROLE_VOCAB)}


def _role_id(role: str) -> int:
    if not isinstance(role, str):
        return ROLE2ID["OTHER"]
    return ROLE2ID.get(role, ROLE2ID["OTHER"])


def _token_type_id(token_type: str) -> int:
    if token_type == "OUTCOME":
        return 1
    return 0


@dataclass(frozen=True)
class Batch:
    # input
    mol_emb: "torch.Tensor"  # [B, L, D]
    amt_feat: "torch.Tensor"  # [B, L, A]
    role_id: "torch.Tensor"  # [B, L]
    tok_type_id: "torch.Tensor"  # [B, L]
    key_padding_mask: "torch.Tensor"  # [B, L] (True means pad)

    # yield labels
    yield_reg: "torch.Tensor"  # [B]
    yield_bin: "torch.Tensor"  # [B]
    yield_pred_mask: "torch.Tensor"  # [B] (1 means compute yield loss)

    # targets for masked modeling
    emb_query_pos: "torch.Tensor"  # [M, 2] => (batch_idx, token_pos) in padded coordinates
    emb_pos: "torch.Tensor"  # [M, D] true embeddings (L2 normalized)

    amt_query_pos: "torch.Tensor"  # [K, 3] => (batch_idx, token_pos, channel_id)
    amt_true: "torch.Tensor"  # [K] true log-value
    
    # task info (optional, for task-specific evaluation)
    tasks: Optional["torch.Tensor"] = None  # [B] task indices or None


def collate_layer2(batch: List[dict[str, Any]]) -> Batch:
    """
    将 `Layer2Jsonl*` 读出的字典样本 batch 化。
    约定：
    - 输入样本已经是 dynamic mask view（包含 tokens[*].*_pred_mask 与 targets）
    - tokens 已包含 emb（被 mask 时为 None）
    """
    import torch

    if not batch:
        raise ValueError("empty batch")

    # 推断 embedding 维度（从首个非空 emb 或 targets.embedding）
    emb_dim = None
    for ex in batch:
        for t in ex.get("tokens", []) or []:
            emb = t.get("emb")
            if isinstance(emb, list) and emb:
                emb_dim = len(emb)
                break
        if emb_dim is not None:
            break
        for _, true_emb in ex.get("targets", {}).get("embedding", []) or []:
            if isinstance(true_emb, list) and true_emb:
                emb_dim = len(true_emb)
                break
        if emb_dim is not None:
            break
    if emb_dim is None:
        raise ValueError("无法推断 emb_dim：batch 内没有 emb/targets.embedding")

    max_len = max(len(ex.get("tokens") or []) for ex in batch)
    # +1 给 [CLS]
    L = max_len + 1
    B = len(batch)
    A = 10  # 3*log + 3*data_mask + 3*pred_mask + 1*volume_includes_solutes

    mol_emb = torch.zeros((B, L, emb_dim), dtype=torch.float32)
    amt_feat = torch.zeros((B, L, A), dtype=torch.float32)
    role_id = torch.full((B, L), fill_value=ROLE2ID["OTHER"], dtype=torch.long)
    tok_type_id = torch.zeros((B, L), dtype=torch.long)
    key_padding_mask = torch.ones((B, L), dtype=torch.bool)

    yield_reg = torch.zeros((B,), dtype=torch.float32)
    yield_bin = torch.zeros((B,), dtype=torch.long)
    yield_pred_mask = torch.zeros((B,), dtype=torch.float32)

    emb_query_list: List[Tuple[int, int]] = []
    emb_pos_list: List[List[float]] = []

    amt_query_list: List[Tuple[int, int, int]] = []
    amt_true_list: List[float] = []

    for bi, ex in enumerate(batch):
        tokens = ex.get("tokens") or []
        if not isinstance(tokens, list):
            raise ValueError("example.tokens 必须是 list")

        # 处理 yield_reg 和 yield_bin 可能为 None 的情况
        yield_reg_val = ex.get("yield_reg")
        yield_bin_val = ex.get("yield_bin")
        if yield_reg_val is None or yield_bin_val is None:
            # 如果 yield 为 None，不计算 yield loss
            yield_reg[bi] = 0.0
            yield_bin[bi] = 0
            yield_pred_mask[bi] = 0.0
        else:
            yield_reg[bi] = float(yield_reg_val)
            yield_bin[bi] = int(yield_bin_val)
            yield_pred_mask[bi] = float(ex.get("yield_pred_mask", 0.0))

        # CLS 永远有效
        key_padding_mask[bi, 0] = False
        role_id[bi, 0] = ROLE2ID["OTHER"]
        tok_type_id[bi, 0] = 0

        for ti, t in enumerate(tokens):
            pos = ti + 1  # shift for CLS
            key_padding_mask[bi, pos] = False

            role_id[bi, pos] = _role_id(t.get("reaction_role"))
            tok_type_id[bi, pos] = _token_type_id(t.get("token_type"))

            emb = t.get("emb")
            if isinstance(emb, list) and len(emb) == emb_dim:
                mol_emb[bi, pos] = torch.tensor(emb, dtype=torch.float32)

            moles_log = t.get("amt_moles_log")
            mass_log = t.get("amt_mass_log")
            vol_log = t.get("amt_volume_log")
            moles_data_mask = int(t.get("amt_moles_mask", 0))
            mass_data_mask = int(t.get("amt_mass_mask", 0))
            vol_data_mask = int(t.get("amt_volume_mask", 0))
            moles_pred_mask = int(t.get("amt_moles_pred_mask", 0))
            mass_pred_mask = int(t.get("amt_mass_pred_mask", 0))
            vol_pred_mask = int(t.get("amt_volume_pred_mask", 0))
            vis = t.get("volume_includes_solutes")
            vis_f = 0.0 if vis is None else (1.0 if bool(vis) else 0.0)

            # 特征维度：[moles_log, moles_data_mask, moles_pred_mask, mass_log, mass_data_mask, mass_pred_mask, vol_log, vol_data_mask, vol_pred_mask, vis]
            amt_feat[bi, pos, 0] = 0.0 if moles_log is None else float(moles_log)
            amt_feat[bi, pos, 1] = float(moles_data_mask)
            amt_feat[bi, pos, 2] = float(moles_pred_mask)
            amt_feat[bi, pos, 3] = 0.0 if mass_log is None else float(mass_log)
            amt_feat[bi, pos, 4] = float(mass_data_mask)
            amt_feat[bi, pos, 5] = float(mass_pred_mask)
            amt_feat[bi, pos, 6] = 0.0 if vol_log is None else float(vol_log)
            amt_feat[bi, pos, 7] = float(vol_data_mask)
            amt_feat[bi, pos, 8] = float(vol_pred_mask)
            amt_feat[bi, pos, 9] = float(vis_f)

            # amount targets：从 token 的 pred_mask 取（真值在 targets.amount）
            # 注：这里不直接用 token 内的值（因为被 mask 了），而使用 targets.amount 列表。

        # embedding targets（位置索引需要 +1）
        targets = ex.get("targets") or {}
        for tpos, true_emb in targets.get("embedding", []) or []:
            if not (isinstance(true_emb, list) and len(true_emb) == emb_dim):
                continue
            emb_query_list.append((bi, int(tpos) + 1))
            emb_pos_list.append([float(x) for x in true_emb])

        # amount targets：记录 (bi, tpos+1, channel_id) 与 true_val
        ch2id = {"moles": 0, "mass": 1, "volume": 2}
        for tpos, ch, true_val in targets.get("amount", []) or []:
            if ch not in ch2id:
                continue
            try:
                v = float(true_val)
            except Exception:
                continue
            amt_query_list.append((bi, int(tpos) + 1, ch2id[ch]))
            amt_true_list.append(v)

    # emb positives：确保 L2 normalize（防止上游未 normalize）
    emb_pos = torch.zeros((len(emb_pos_list), emb_dim), dtype=torch.float32)
    if emb_pos_list:
        emb_pos = torch.tensor(emb_pos_list, dtype=torch.float32)
        emb_pos = torch.nn.functional.normalize(emb_pos, p=2, dim=-1)
    emb_query_pos = torch.zeros((len(emb_query_list), 2), dtype=torch.long)
    if emb_query_list:
        emb_query_pos = torch.tensor(emb_query_list, dtype=torch.long)

    amt_query_pos = torch.zeros((len(amt_query_list), 3), dtype=torch.long)
    if amt_query_list:
        amt_query_pos = torch.tensor(amt_query_list, dtype=torch.long)
    amt_true = torch.zeros((len(amt_true_list),), dtype=torch.float32)
    if amt_true_list:
        amt_true = torch.tensor(amt_true_list, dtype=torch.float32)

    # 提取task信息（用于任务级别的评估）
    # Task映射: task1_mask_product -> 0, task2a_predict_yield_full -> 1, task2b_predict_product_and_yield -> 2, task3_mask_role -> 3, task4_mask_reactant -> 4
    task_to_id = {
        "task1_mask_product": 0,
        "task2a_predict_yield_full": 1,
        "task2b_predict_product_and_yield": 2,
        "task3_mask_role": 3,
        "task4_mask_reactant": 4,
        # 兼容旧的任务名称
        "forward": 0,
        "yield_full": 1,
        "yield_with_product": 2,
        "condition": 3,
        "retro": 4,
    }
    tasks_list = []
    for ex in batch:
        targets = ex.get("targets") or {}
        task = targets.get("task", "")
        task_id = task_to_id.get(task, -1)  # -1表示未知任务
        tasks_list.append(task_id)
    tasks_tensor = torch.tensor(tasks_list, dtype=torch.long) if tasks_list else None

    return Batch(
        mol_emb=mol_emb,
        amt_feat=amt_feat,
        role_id=role_id,
        tok_type_id=tok_type_id,
        key_padding_mask=key_padding_mask,
        yield_reg=yield_reg,
        yield_bin=yield_bin,
        yield_pred_mask=yield_pred_mask,
        emb_query_pos=emb_query_pos,
        emb_pos=emb_pos,
        amt_query_pos=amt_query_pos,
        amt_true=amt_true,
        tasks=tasks_tensor,
    )

