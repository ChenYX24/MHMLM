from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    # 允许在未安装 torch 的环境中导入本文件（例如本机仅写代码）。
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ModelConfig:
    mol_emb_dim: int = 256
    hidden_dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1

    num_roles: int = 11
    num_token_types: int = 2

    tau: float = 0.07  # InfoNCE temperature
    learnable_tau: bool = False  # 是否让 tau 可学习
    symmetric_ince: bool = False  # 是否使用对称 InfoNCE
    use_projection_head: bool = False  # 使用 MLP 投影头做 InfoNCE（训练时）
    head_dropout: float = 0.0  # Dropout applied before task heads (v4)


if nn is not None:

    class Layer2PretrainModel(nn.Module):
        def __init__(self, cfg: ModelConfig):
            super().__init__()
            assert torch is not None

            self.cfg = cfg

            self.mask_mol_emb = nn.Parameter(torch.zeros(cfg.mol_emb_dim))
            self.cls_emb = nn.Parameter(torch.zeros(cfg.hidden_dim))

            self.mol_proj = nn.Linear(cfg.mol_emb_dim, cfg.hidden_dim)
            self.amt_proj = nn.Sequential(
                nn.Linear(10, cfg.hidden_dim),  # 3*log + 3*data_mask + 3*pred_mask + 1*vis
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            self.role_emb = nn.Embedding(cfg.num_roles, cfg.hidden_dim)
            self.type_emb = nn.Embedding(cfg.num_token_types, cfg.hidden_dim)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.hidden_dim * 4,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
            self.final_ln = nn.LayerNorm(cfg.hidden_dim)

            self.emb_head = nn.Linear(cfg.hidden_dim, cfg.mol_emb_dim)

            # MLP projection head for contrastive learning (SimCLR/MoCo style)
            # Training: use emb_proj for InfoNCE (better gradient flow)
            # Inference: use emb_head (linear) for downstream embedding
            self.emb_proj = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.mol_emb_dim),
            ) if cfg.use_projection_head else None

            self.amt_head = nn.Linear(cfg.hidden_dim, 3)  # moles/mass/volume
            self.yield_bin_head = nn.Linear(cfg.hidden_dim, 10)
            self.yield_reg_head = nn.Linear(cfg.hidden_dim, 1)

            # 可学习的 temperature（如果启用）
            if cfg.learnable_tau:
                # 使用 log_tau 参数，然后 exp 得到 tau，并 clamp 到合理范围
                self.log_tau = nn.Parameter(torch.log(torch.tensor(cfg.tau)))
            else:
                self.log_tau = None

            # Head dropout (v4: regularize before task heads)
            self.head_drop = nn.Dropout(cfg.head_dropout) if cfg.head_dropout > 0 else nn.Identity()

            nn.init.normal_(self.mask_mol_emb, std=0.02)
            nn.init.normal_(self.cls_emb, std=0.02)

        def forward(self, batch: "Batch") -> dict[str, "torch.Tensor"]:
            assert torch is not None

            B, L, D = batch.mol_emb.shape
            device = next(self.parameters()).device

            mol_emb = batch.mol_emb.to(device)
            amt_feat = batch.amt_feat.to(device)
            role_id = batch.role_id.to(device)
            tok_type_id = batch.tok_type_id.to(device)
            key_padding_mask = batch.key_padding_mask.to(device)

            # CLS token：将 pos=0 的 mol_emb/amt_feat 忽略，使用可学习 cls_emb
            x = torch.zeros((B, L, self.cfg.hidden_dim), device=device, dtype=torch.float32)
            x[:, 0, :] = self.cls_emb.unsqueeze(0).expand(B, -1)

            # token embeddings（pos>=1）
            mol_in = mol_emb[:, 1:, :]
            is_zero = mol_in.abs().sum(dim=-1, keepdim=True) == 0  # mask 时为 0 向量
            mol_in = torch.where(is_zero, self.mask_mol_emb.view(1, 1, -1).expand_as(mol_in), mol_in)
            x[:, 1:, :] = (
                self.mol_proj(mol_in)
                + self.amt_proj(amt_feat[:, 1:, :])
                + self.role_emb(role_id[:, 1:])
                + self.type_emb(tok_type_id[:, 1:])
            )

            h = self.encoder(x, src_key_padding_mask=key_padding_mask)
            h = self.final_ln(h)

            cls_h = self.head_drop(h[:, 0, :])
            tok_h_drop = self.head_drop(h)

            pred_emb = self.emb_head(tok_h_drop)  # [B,L,D]
            pred_amt = self.amt_head(tok_h_drop)  # [B,L,3]
            pred_yield_bin = self.yield_bin_head(cls_h)  # [B,10]
            pred_yield_reg = self.yield_reg_head(cls_h).squeeze(-1)  # [B]

            tok_h = h  # original hidden states (for emb_proj etc.)

            result = {
                "tok_h": tok_h,
                "pred_emb": pred_emb,
                "pred_amt": pred_amt,
                "pred_yield_bin": pred_yield_bin,
                "pred_yield_reg": pred_yield_reg,
            }
            if self.emb_proj is not None:
                result["pred_emb_proj"] = self.emb_proj(tok_h_drop)  # [B,L,D]

            # Compute tau inside forward for DDP gradient tracking
            if self.log_tau is not None:
                result["tau"] = torch.exp(self.log_tau).clamp(min=0.01, max=1.0)

            return result

    def compute_losses(
        out: dict[str, "torch.Tensor"],
        batch: "Batch",
        *,
        tau: float,
        emb_lambda: float = 1.0,
        amt_lambda: float = 1.0,
        yield_weight: float = 1.0,
        yield_reg_lambda: float = 1.0,
        yield_lambda: float | None = None,  # 兼容旧接口：yield_lambda -> yield_weight
        symmetric_ince: bool = False,  # 是否使用对称 InfoNCE
        model: "Layer2PretrainModel | None" = None,  # 用于获取可学习的 tau
        yield_class_weights: "torch.Tensor | None" = None,  # Yield 分类的类别权重
        amt_channel_weights: "torch.Tensor | None" = None,  # Amount 三通道权重 [moles, mass, volume]
        yield_reg_mean: float = 0.0,  # Yield regression 的均值（用于标准化）
        yield_reg_std: float = 1.0,  # Yield regression 的标准差（用于标准化）
        yield_mode: str = "reg_only",  # Yield 预测模式: "reg_only", "bin_only", "soft_bin_only", "both"
        yield_soft_bin_temperature: float = 0.1,  # 软标签分类的温度参数（越小越尖锐）
        yield_label_smoothing: float = 0.0,  # Label smoothing for yield CE losses (v4)
    ) -> dict[str, "torch.Tensor"]:
        """
        多任务加权：
        total = emb_lambda * emb_loss + amt_lambda * amt_loss + yield_weight * yield_loss
        
        Yield 模式：
        - "reg_only": 只用回归头，yield_loss = MSE
        - "bin_only": 只用分类头，yield_loss = CrossEntropy
        - "soft_bin_only": 使用软标签 KL-Divergence，yield_loss = KL(soft_target || pred_dist)
        - "both": 同时使用分类和回归，yield_loss = CE + yield_reg_lambda * MSE
        """
        assert torch is not None and F is not None
        device = out["pred_emb"].device

        # 兼容旧接口：yield_lambda -> yield_weight
        if yield_lambda is not None:
            yield_weight = yield_lambda

        # 获取实际的 tau（可能是可学习的）
        # Prefer tau from forward output (preserves DDP gradient tracking)
        if "tau" in out:
            actual_tau = out["tau"]
        else:
            actual_tau = tau

        # -------- emb --------
        emb_loss = torch.zeros((), device=device)
        if batch.emb_query_pos.numel() > 0:
            qp = batch.emb_query_pos.to(device)  # [M,2]
            pos = batch.emb_pos.to(device)       # [M,D]

            # Use projection head output for InfoNCE if available (better training)
            emb_key = "pred_emb_proj" if "pred_emb_proj" in out else "pred_emb"
            q = out[emb_key][qp[:, 0], qp[:, 1], :]  # [M,D]
            q = F.normalize(q, p=2, dim=-1)
            pos = F.normalize(pos, p=2, dim=-1)

            logits = (q @ pos.t()) / actual_tau
            targets = torch.arange(pos.size(0), device=device, dtype=torch.long)
            
            if symmetric_ince:
                # 对称 InfoNCE：q→pos 和 pos→q 的平均
                loss_q2pos = F.cross_entropy(logits, targets)
                loss_pos2q = F.cross_entropy(logits.t(), targets)
                emb_loss = (loss_q2pos + loss_pos2q) / 2.0
            else:
                emb_loss = F.cross_entropy(logits, targets)

        # -------- amount --------
        amt_loss = torch.zeros((), device=device)
        if batch.amt_query_pos.numel() > 0:
            ap = batch.amt_query_pos.to(device)    # [K,3]
            true_v = batch.amt_true.to(device)     # [K]
            pred = out["pred_amt"][ap[:, 0], ap[:, 1], ap[:, 2]]  # [K]
            
            if amt_channel_weights is not None:
                # 按通道加权：moles > mass/volume
                channel_ids = ap[:, 2]  # [K] 每个样本的通道 ID (0=moles, 1=mass, 2=volume)
                weights = amt_channel_weights.to(device)[channel_ids]  # [K] 确保 weights 在正确的设备上
                # 加权 SmoothL1
                abs_err = (pred - true_v).abs()
                smooth_l1 = torch.where(abs_err < 1.0, 0.5 * abs_err ** 2, abs_err - 0.5)
                amt_loss = (weights * smooth_l1).mean()
            else:
                amt_loss = F.smooth_l1_loss(pred, true_v)

        # -------- yield --------
        yield_loss = torch.zeros((), device=device)
        y_mask = batch.yield_pred_mask.to(device)  # [B]
        idx = (y_mask > 0.5).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            y_bin = batch.yield_bin.to(device)[idx]
            y_reg = batch.yield_reg.to(device)[idx]
            pred_bin = out["pred_yield_bin"][idx]
            pred_reg = out["pred_yield_reg"][idx]
            
            if yield_mode == "reg_only":
                # 只用回归头
                if yield_reg_std > 0:
                    y_reg_normalized = (y_reg - yield_reg_mean) / yield_reg_std
                    yield_loss = F.mse_loss(pred_reg, y_reg_normalized)
                else:
                    yield_loss = F.mse_loss(pred_reg, y_reg)
                    
            elif yield_mode == "bin_only":
                # 只用分类头
                if yield_class_weights is not None:
                    yield_loss = F.cross_entropy(pred_bin, y_bin, weight=yield_class_weights.to(device), label_smoothing=yield_label_smoothing)
                else:
                    yield_loss = F.cross_entropy(pred_bin, y_bin, label_smoothing=yield_label_smoothing)
                    
            elif yield_mode == "soft_bin_only":
                # 软标签分类：将 y_reg 转换为软标签分布，使用 KL-Divergence
                # 将 y_reg (0-1) 映射到 bin 分布
                # 例如 y_reg=0.85 -> Bin8(0.8-0.9) 和 Bin9(0.9-1.0) 都有概率
                n_bins = 10
                bin_centers = torch.arange(n_bins, device=device, dtype=torch.float32) * 0.1 + 0.05  # [0.05, 0.15, ..., 0.95]
                
                # 计算每个样本到各 bin center 的距离
                y_reg_expanded = y_reg.unsqueeze(-1)  # [yc, 1]
                bin_centers_expanded = bin_centers.unsqueeze(0)  # [1, 10]
                distances = (y_reg_expanded - bin_centers_expanded).abs()  # [yc, 10]
                
                # 使用温度参数生成软标签分布（距离越近，概率越高）
                # 使用负距离的 softmax，温度越小分布越尖锐
                logits = -distances / max(yield_soft_bin_temperature, 1e-6)
                soft_target = F.softmax(logits, dim=-1)  # [yc, 10]
                
                # KL-Divergence: KL(soft_target || pred_dist)
                pred_dist = F.log_softmax(pred_bin, dim=-1)  # [yc, 10]
                yield_loss = F.kl_div(pred_dist, soft_target, reduction='batchmean')
                
            else:  # yield_mode == "both" (默认)
                # 同时使用分类和回归
                if yield_class_weights is not None:
                    bin_loss = F.cross_entropy(pred_bin, y_bin, weight=yield_class_weights.to(device), label_smoothing=yield_label_smoothing)
                else:
                    bin_loss = F.cross_entropy(pred_bin, y_bin, label_smoothing=yield_label_smoothing)
                
                if yield_reg_std > 0:
                    y_reg_normalized = (y_reg - yield_reg_mean) / yield_reg_std
                    reg_loss = F.mse_loss(pred_reg, y_reg_normalized)
                else:
                    reg_loss = F.mse_loss(pred_reg, y_reg)
                
                yield_loss = bin_loss + float(yield_reg_lambda) * reg_loss

        total = float(emb_lambda) * emb_loss + float(amt_lambda) * amt_loss + float(yield_weight) * yield_loss

        # 返回 raw 供日志用（不 detach，方便你未来也想自定义重组）
        # 同时提供兼容旧接口的字段名
        return {
            "loss_total": total,
            "loss_emb_raw": emb_loss,
            "loss_amt_raw": amt_loss,
            "loss_yield_raw": yield_loss,
            # 兼容旧接口
            "loss_emb": emb_loss.detach(),
            "loss_amt": amt_loss.detach(),
            "loss_yield": yield_loss.detach(),
        }

else:  # pragma: no cover

    class Layer2PretrainModel:  # type: ignore[no-redef]
        def __init__(self, cfg: ModelConfig):
            raise RuntimeError("Layer2PretrainModel 需要 torch（请在训练机安装 torch 后运行）。")


    def compute_losses(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("compute_losses 需要 torch（请在训练机安装 torch 后运行）。")
