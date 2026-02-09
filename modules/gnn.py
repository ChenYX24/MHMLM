# gnn.py  — GVPEncoder (node_dims=(10,1), edge_dims=(1,1))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
from gvp import GVP, GVPConvLayer, LayerNorm
# from .gvp import GVP, GVPConvLayer, LayerNorm
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import logging
from typing import Optional, Tuple, Union, List

# -------- 安静 RDKit / 警告 --------
RDLogger.DisableLog('rdApp.*')
logging.getLogger("rdkit").setLevel(logging.ERROR)

# -------- 工具：由坐标和边索引得到边的向量 & 标量(长度) --------
def _get_edge_features_from_coords(coords: torch.Tensor,
                                   edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      edge_vectors_gvp: [E, 1, 3]
      edge_length:      [E, 1]
    """
    if edge_index is None or edge_index.numel() == 0:
        device = coords.device if coords is not None else "cpu"
        return (torch.empty(0, 1, 3, dtype=torch.float, device=device),
                torch.empty(0, 1, dtype=torch.float, device=device))

    row, col = edge_index  # [2, E]
    edge_vec = coords[col] - coords[row]           # [E, 3]
    edge_vectors_gvp = edge_vec.unsqueeze(1)       # [E, 1, 3]
    edge_length = edge_vec.norm(dim=-1, keepdim=True)  # [E, 1]
    return edge_vectors_gvp, edge_length


# -------- 在线构造：10 维标量(5原子属性+5杂化onehot) --------
def _rdkit_node_scalar_10d(mol: Chem.Mol) -> torch.Tensor:
    """
    生成每原子 10 维标量:
    [atomic_num, formal_charge, is_aromatic, is_in_ring, total_hs] + hybridization_onehot(5)
    """
    from rdkit.Chem import rdchem
    hyb_map = {
        rdchem.HybridizationType.SP3:  [1, 0, 0, 0, 0],
        rdchem.HybridizationType.SP2:  [0, 1, 0, 0, 0],
        rdchem.HybridizationType.SP:   [0, 0, 1, 0, 0],
        rdchem.HybridizationType.SP3D: [0, 0, 0, 1, 0],
        rdchem.HybridizationType.SP3D2:[0, 0, 0, 0, 1],
    }
    rows = []
    for a in mol.GetAtoms():
        s = [
            float(a.GetAtomicNum()),
            float(a.GetFormalCharge()),
            float(a.GetIsAromatic()),
            float(a.IsInRing()),
            float(a.GetTotalNumHs(includeNeighbors=True)),
        ]
        hyb = hyb_map.get(a.GetHybridization(), [1, 0, 0, 0, 0])  # 默认 SP3
        rows.append(s + hyb)
    return torch.tensor(rows, dtype=torch.float)  # [N, 10]


# ============================ GVP Encoder ============================
class GVPEncoder(nn.Module):
    def __init__(self,
                 node_dims=(10, 1),      # <- 与你的离线管道对齐
                 edge_dims=(1, 1),       # <- 边长(1) + 边方向(1)
                 hidden_scalar_dim=256,
                 hidden_vector_dim=16,   # 向量通道在 GVP 内部会保持为 "多少个向量"
                 output_dim=256,
                 num_layers=4):
        super().__init__()
        self.output_dim = output_dim

        # 保留首层 GVP 的直接引用，便于检查 in_dims / 设备
        self.node_gvp_in = GVP(node_dims, (hidden_scalar_dim, hidden_vector_dim), activations=(F.relu, None))
        self.node_input = nn.Sequential(
            self.node_gvp_in,
            LayerNorm((hidden_scalar_dim, hidden_vector_dim))
        )

        self.edge_input = GVP(edge_dims, (hidden_scalar_dim, hidden_vector_dim), h_dim=hidden_scalar_dim)

        self.convs = nn.ModuleList([
            GVPConvLayer((hidden_scalar_dim, hidden_vector_dim),
                         (hidden_scalar_dim, hidden_vector_dim),
                         activations=(F.relu, None))
            for _ in range(num_layers)
        ])

        self.project = nn.Sequential(
            GVP((hidden_scalar_dim, hidden_vector_dim), (output_dim, 1), activations=(None, None))
        )

    # ---------- RDKit 3D 嵌入 ----------
    @staticmethod
    def _embed_3d_coords_rdkit(mol: Chem.Mol, max_iters=200) -> Optional[np.ndarray]:
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xF00D
            res = AllChem.EmbedMolecule(mol, params)
            if res != 0:
                for seed in (0, 1, 42, 123):
                    params.randomSeed = seed
                    if AllChem.EmbedMolecule(mol, params) == 0:
                        break
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)
            except Exception:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=max_iters)
                except Exception:
                    pass
            conf = mol.GetConformer()
            n = mol.GetNumAtoms()
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n)], dtype=np.float32)
            return coords
        except Exception:
            return None

    # ---------- 可选：从 SMILES 在线构造 Data（与离线维度一致） ----------
    def _smiles_to_data(self, smiles: str) -> Optional[Data]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        mol = Chem.RemoveHs(mol)
        if mol.GetNumAtoms() == 0:
            return None

        # 节点 10 维标量 + 1 个向量通道
        x_scalar = _rdkit_node_scalar_10d(mol)             # [N, 10]
        coords = self._embed_3d_coords_rdkit(mol)
        if coords is None:
            return None
        pos = torch.tensor(coords, dtype=torch.float)      # [N, 3]
        x_vector = pos.unsqueeze(1)                        # [N, 1, 3]

        # 边 index
        edge_index = []
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edge_index += [[i, j], [j, i]]
        if len(edge_index) == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long)
        else:
            edge_index = torch.tensor(np.array(edge_index, dtype=np.int64).T, dtype=torch.long)

        # 边向量 & 边长度
        edge_vec, edge_len = _get_edge_features_from_coords(pos, edge_index)

        data = Data(
            x=x_scalar,                  # [N, 10]
            x_vector=x_vector,           # [N, 1, 3]
            edge_index=edge_index,       # [2, E]
            edge_attr=edge_len,          # 兼容字段名: 用 edge_attr 放标量(长度)也行
            edge_scalar=edge_len,        # [E, 1]
            edge_attr_vector=edge_vec,   # [E, 1, 3]
            pos=pos,
            smiles=smiles
        )
        return data

    # ---------- 前向 ----------
    def forward(self, data: Data) -> torch.Tensor:
        """
        期望:
          data.x              : [N, 10]  或 [N, 9]（会自动补成 10）
          data.x_vector       : [N, 1, 3]
          data.edge_index     : [2, E]
          data.edge_scalar    : [E, 1]   (若无则使用 data.edge_attr)
          data.edge_attr_vector: [E, 1, 3]
          data.batch          : [N]
        """
        device = next(self.parameters()).device

        node_scalar: torch.Tensor = data.x
        node_vector: torch.Tensor = data.x_vector

        # --- 容错：如果老数据是 9 维(仅 atom type one-hot)，自动补一个默认杂化 one-hot(SP3) 变成 10 维 ---
        if node_scalar.dim() == 2 and node_scalar.size(-1) == 9:
            # 默认补 [1,0,0,0,0]
            pad = torch.zeros(node_scalar.size(0), 1, device=node_scalar.device, dtype=node_scalar.dtype)
            hyb = torch.cat([torch.ones_like(pad), torch.zeros(node_scalar.size(0), 4, device=node_scalar.device, dtype=node_scalar.dtype)], dim=1)
            node_scalar = torch.cat([node_scalar, hyb], dim=-1)  # -> [N,10]

        # 断言维度一致
        assert node_scalar.size(-1) == 10, f"node_scalar last dim must be 10, got {node_scalar.size(-1)}"
        assert node_vector.dim() == 3 and node_vector.size(-2) == 1 and node_vector.size(-1) == 3, \
            f"node_vector must be [N,1,3], got {tuple(node_vector.shape)}"

        # 边特征
        edge_scalar = getattr(data, "edge_scalar", None)
        if edge_scalar is None:
            edge_scalar = getattr(data, "edge_attr", None)  # 兼容
        edge_vector = getattr(data, "edge_attr_vector", None)

        if edge_vector is None or edge_scalar is None:
            # 若缺少，就用 pos 和 edge_index 现算
            pos = getattr(data, "pos", None)
            assert pos is not None, "pos is required to derive edge features"
            edge_vector, edge_scalar = _get_edge_features_from_coords(pos, data.edge_index)

        # 断言边维度
        assert edge_scalar.dim() == 2 and edge_scalar.size(-1) == 1, \
            f"edge_scalar must be [E,1], got {tuple(edge_scalar.shape)}"
        assert edge_vector.dim() == 3 and edge_vector.size(-2) == 1 and edge_vector.size(-1) == 3, \
            f"edge_vector must be [E,1,3], got {tuple(edge_vector.shape)}"

        # 打包成 GVP 需要的元组
        node_feats = (node_scalar, node_vector)
        edge_feats = (edge_scalar, edge_vector)

        # 编码
        h_V = self.node_input(node_feats)                 # (S_h, V_h)
        h_E = self.edge_input(edge_feats)                 # (S_h, V_h)

        for conv in self.convs:
            h_V = conv(h_V, data.edge_index, h_E)

        out_scalar, _ = self.project(h_V)                 # [N, output_dim]
        graph_embeddings = global_mean_pool(out_scalar, data.batch.to(device))
        return graph_embeddings  # [B, output_dim]

    # ---------- 便捷接口：从 SMILES 得到图嵌入 ----------
    def forward_from_smiles(self, smiles: str) -> torch.Tensor:
        data = self._smiles_to_data(smiles)
        if data is None:
            device = next(self.parameters()).device
            return torch.zeros(1, self.output_dim, device=device)

        device = next(self.parameters()).device
        N = data.x.size(0)
        data.batch = torch.zeros(N, dtype=torch.long, device=device)
        data = data.to(device)
        return self.forward(data)  # [1, output_dim]

    # ================= 新增：下游任务 heads & 路由接口 =================

    def init_task_heads(
        self,
        num_reg_tasks: int = 0,
        num_cls_tasks: int = 0,
        head_hidden_dim: Optional[int] = None,
        head_dropout: float = 0.1,
    ):
        """
        初始化 QM9 回归头 & BBBP 分类头。

        Args:
            num_reg_tasks: QM9 回归任务输出维度，例如 5 表示 [mu, alpha, homo, lumo, gap]
            num_cls_tasks: BBBP 分类任务输出维度，通常是 1（logit）
        """
        head_hidden_dim = head_hidden_dim or self.output_dim

        # QM9 回归头
        if num_reg_tasks and num_reg_tasks > 0:
            self.qm9_head = nn.Sequential(
                nn.Linear(self.output_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, num_reg_tasks),
            )
        else:
            self.qm9_head = None

        # BBBP 分类头
        if num_cls_tasks and num_cls_tasks > 0:
            self.bbbp_head = nn.Sequential(
                nn.Linear(self.output_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, num_cls_tasks),
            )
        else:
            self.bbbp_head = None

    # ---------- 编码：图对象 ----------
    def encode_graph(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        对 PyG Data/Batch 做图级编码，返回 [B, output_dim]
        """
        return self.forward(data)

    # ---------- 编码：SMILES 批 ----------
    def encode_smiles_batch(self, smiles: Union[str, List[str]]) -> torch.Tensor:
        """
        从单个或一批 SMILES 得到图级 embedding。

        Args:
            smiles: str 或 List[str]

        Returns:
            emb: [B, output_dim]
        """
        device = next(self.parameters()).device

        single = False
        if isinstance(smiles, str):
            smiles_list = [smiles]
            single = True
        else:
            smiles_list = list(smiles)

        data_list: List[Data] = []
        valid_idx: List[int] = []
        for i, smi in enumerate(smiles_list):
            d = self._smiles_to_data(smi)
            if d is None:
                continue
            data_list.append(d)
            valid_idx.append(i)

        if len(data_list) == 0:
            return torch.zeros(len(smiles_list), self.output_dim, device=device)

        batch_data = Batch.from_data_list(data_list).to(device)
        emb_valid = self.encode_graph(batch_data)  # [Nv, D]

        out = torch.zeros(len(smiles_list), self.output_dim, device=device)
        for j, i in enumerate(valid_idx):
            out[i] = emb_valid[j]

        if single:
            return out[:1]  # [1, D]
        return out         # [B, D]

    # ---------- 内部辅助：统一处理 Data/Batch 或 SMILES ----------
    def _encode_any(
        self,
        data_or_smiles: Union[Data, Batch, str, List[str]],
    ) -> torch.Tensor:
        if isinstance(data_or_smiles, (Data, Batch)):
            return self.encode_graph(data_or_smiles)
        if isinstance(data_or_smiles, (str, list, tuple)):
            return self.encode_smiles_batch(data_or_smiles)
        raise TypeError(f"Unsupported input type for GVPEncoder: {type(data_or_smiles)}")

    # ---------- QM9 回归任务前向 ----------
    def forward_qm9(
        self,
        data_or_smiles: Union[Data, Batch, str, List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        QM9 回归任务前向。

        Returns:
            pred: [B, num_reg_tasks] —— 回归值（可配合外部 mean/std 做归一化）
            emb : [B, output_dim]    —— 图级 embedding
        """
        if self.qm9_head is None:
            raise RuntimeError("qm9_head is None, 请先调用 init_task_heads(num_reg_tasks>0)")
        emb = self._encode_any(data_or_smiles)
        pred = self.qm9_head(emb)
        return pred, emb

    # ---------- BBBP 分类任务前向 ----------
    def forward_bbbp(
        self,
        data_or_smiles: Union[Data, Batch, str, List[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        BBBP 分类任务前向。

        Returns:
            logits: [B, num_cls_tasks] —— 二分类 logit（配合 BCEWithLogitsLoss）
            emb   : [B, output_dim]
        """
        if self.bbbp_head is None:
            raise RuntimeError("bbbp_head is None, 请先调用 init_task_heads(num_cls_tasks>0)")
        emb = self._encode_any(data_or_smiles)
        logits = self.bbbp_head(emb)
        return logits, emb
