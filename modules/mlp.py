# modules/mlp_heads.py
# -*- coding: utf-8 -*-
import math, torch
import torch.nn as nn
from typing import Optional
class SimpleMLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# ===== 版本 B：更贴近 diffusion 训练常用的 MLP（稳定性更强） =====
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

class DiffusionStyleMLP(nn.Module):
    """
    结构：x -> RMSNorm -> Linear -> SiLU -> Dropout -> Linear(zero-init) -> +residual
    - 末层权重/偏置零初始化（或很小的尺度），便于稳定收敛（常见于 EDM/DiT/ControlNet 的 head）
    - 自带 residual，有利于保留原始上下文信息
    """
    def __init__(self, d_in, d_hidden, d_out, dropout=0.0, zero_init=True, residual=True):
        super().__init__()
        self.norm = RMSNorm(d_in)
        self.fc1  = nn.Linear(d_in, d_hidden)
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(d_hidden, d_out)
        self.residual = residual and (d_in == d_out)

        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        if zero_init:
            nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
        else:
            nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        y = self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))
        if self.residual:
            y = x + y
        return y

# === 对外导出两个头（和之前名字保持一致） ===
class DiffusionMLP(nn.Module):
    """把 LLM d_model -> diffusion 条件维度"""
    def __init__(self, d_in, d_hidden, d_out, variant="diffusion", dropout=0.0):
        super().__init__()
        if variant == "simple":
            self.proj = SimpleMLP(d_in, d_hidden, d_out, dropout)
        else:
            self.proj = DiffusionStyleMLP(d_in, d_hidden, d_out, dropout, zero_init=True, residual=False)
    def forward(self, x): return self.proj(x)

class GNNMLP(nn.Module):
    """把 GNN 向量 -> LLM d_model"""
    def __init__(self, d_in, d_hidden, d_out, variant="diffusion", dropout=0.0):
        super().__init__()
        if variant == "simple":
            self.proj = SimpleMLP(d_in, d_hidden, d_out, dropout)
        else:
            self.proj = DiffusionStyleMLP(d_in, d_hidden, d_out, dropout, zero_init=False, residual=False)
    def forward(self, x): return self.proj(x)

class DiffusionAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        # 可选：按你训练里的 init 方式
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc(x)

class MLPAdapter(nn.Module):
    """
    一个用于在两个不同维度嵌入空间之间进行映射的多层感知机（MLP）适配器。
    
    例如，将 GNN 的分子嵌入空间映射到 LLM 的词嵌入空间。
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
    ):
        """
        Args:
            input_dim (int): 输入嵌入的维度。
            output_dim (int): 输出嵌入的维度。
            hidden_dim (Optional[int]): 隐藏层的维度。如果为 None，则默认为 output_dim。
            num_layers (int): MLP 隐藏层的数量（不包括输入和输出层）。
                              默认值为 2，至少为 1。
        """
        super().__init__()
        
        if num_layers < 1:
            raise ValueError("`num_layers` 必须至少为 1。")

        if hidden_dim is None:
            hidden_dim = output_dim

        layers = []
        # 第一层：从输入维度到隐藏维度
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # 额外的隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # 输出层：从隐藏维度到最终输出维度
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状为 (..., input_dim)。

        Returns:
            torch.Tensor: 映射后的输出张量，形状为 (..., output_dim)。
        """
        return self.model(x)
