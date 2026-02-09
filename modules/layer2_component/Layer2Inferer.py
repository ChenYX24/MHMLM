"""
Layer2 推理组件
用于反应产率预测和任务相关 embedding 生成
参考 LDMolInferer 的设计模式
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union
import yaml

import torch
import torch.nn as nn

from .model import ModelConfig, Layer2PretrainModel
from .gvp_embedder import smiles_to_embedding, build_gvp_encoder
from .collate import collate_layer2, ROLE_VOCAB

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).with_name("layer2_config.yaml")


class Layer2Inferer:
    """
    Layer2 推理器，支持：
    - predict: 预测反应产率和生成任务相关 embedding
    """

    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        device: str | torch.device | None = None,
        gvp_encoder: Optional[Any] = None,
        gvp_ckpt_path: Optional[str] = None,
    ) -> None:
        """
        初始化 Layer2Inferer。
        
        Args:
            config_path: 配置文件路径，默认使用同目录下的 layer2_config.yaml
            device: 推理设备，如 "cuda:0"
            gvp_encoder: 可选的 GVP encoder（如果已加载）
            gvp_ckpt_path: GVP checkpoint 路径（如果未提供 gvp_encoder）
        """
        config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        base_dir = config_path.parent
        
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            self.cfg = self._get_default_config()
        else:
            with config_path.open("r", encoding="utf-8") as f:
                self.cfg = yaml.safe_load(f)

        self.device = torch.device(device) if isinstance(device, str) else (device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        
        logger.info(f"Layer2 init: config={config_path} device={self.device}")

        # 先尝试从 checkpoint 中读取配置（如果存在）
        ckpt_path = self.cfg.get("checkpoint_path")
        ckpt_cfg = None
        if ckpt_path:
            ckpt_path = Path(ckpt_path)
            if not ckpt_path.is_absolute():
                ckpt_path = base_dir / ckpt_path
            if ckpt_path.exists():
                try:
                    ckpt_state = torch.load(str(ckpt_path), map_location="cpu")
                    if "cfg" in ckpt_state and isinstance(ckpt_state["cfg"], dict):
                        ckpt_cfg = ckpt_state["cfg"]
                        logger.info(f"✅ 从 checkpoint 读取配置: hidden_dim={ckpt_cfg.get('hidden_dim')}, n_layers={ckpt_cfg.get('n_layers')}, n_heads={ckpt_cfg.get('n_heads')}")
                except Exception as e:
                    logger.warning(f"⚠️  无法从 checkpoint 读取配置: {e}")

        # 加载模型配置（优先使用 checkpoint 中的配置）
        model_cfg = ModelConfig(
            mol_emb_dim=ckpt_cfg.get("mol_emb_dim") if ckpt_cfg else self.cfg.get("mol_emb_dim", 256),
            hidden_dim=ckpt_cfg.get("hidden_dim") if ckpt_cfg else self.cfg.get("hidden_dim", 512),
            n_layers=ckpt_cfg.get("n_layers") if ckpt_cfg else self.cfg.get("n_layers", 6),
            n_heads=ckpt_cfg.get("n_heads") if ckpt_cfg else self.cfg.get("n_heads", 8),
            dropout=ckpt_cfg.get("dropout") if ckpt_cfg else self.cfg.get("dropout", 0.1),
            num_roles=ckpt_cfg.get("num_roles") if ckpt_cfg else self.cfg.get("num_roles", 11),
            num_token_types=ckpt_cfg.get("num_token_types") if ckpt_cfg else self.cfg.get("num_token_types", 2),
            tau=ckpt_cfg.get("tau") if ckpt_cfg else self.cfg.get("tau", 0.07),
            learnable_tau=ckpt_cfg.get("learnable_tau") if ckpt_cfg else self.cfg.get("learnable_tau", False),
            symmetric_ince=ckpt_cfg.get("symmetric_ince") if ckpt_cfg else self.cfg.get("symmetric_ince", False),
        )
        
        # 加载模型
        self.model = Layer2PretrainModel(model_cfg).to(self.device)
        self.model.eval()
        
        # 加载权重
        if ckpt_path and ckpt_path.exists():
            self._load_checkpoint(str(ckpt_path))
            # 从 checkpoint 中读取 yield_reg 标准化参数
            try:
                ckpt_state = torch.load(str(ckpt_path), map_location="cpu")
                if "yield_reg_mean" in ckpt_state:
                    self.cfg["yield_reg_mean"] = float(ckpt_state["yield_reg_mean"])
                if "yield_reg_std" in ckpt_state:
                    self.cfg["yield_reg_std"] = float(ckpt_state["yield_reg_std"])
                logger.info(f"✅ 从 checkpoint 读取 yield_reg 标准化参数: mean={self.cfg.get('yield_reg_mean', 0.0):.4f}, std={self.cfg.get('yield_reg_std', 1.0):.4f}")
            except Exception as e:
                logger.warning(f"⚠️  无法从 checkpoint 读取 yield_reg 标准化参数: {e}")
        else:
            logger.warning("未指定 checkpoint_path 或文件不存在，使用随机初始化的模型")

        # 初始化 GVP encoder
        if gvp_encoder is not None:
            self.gvp_encoder = gvp_encoder
        elif gvp_ckpt_path:
            gvp_root = Path(self.cfg.get("gvp_root", "/data1/chenyuxuan/MSMLM"))
            self.gvp_encoder = build_gvp_encoder(str(self.device), gvp_root, gvp_ckpt_path)
        else:
            logger.warning("未提供 GVP encoder，predict 时需要外部提供 embedding")
            self.gvp_encoder = None
    
    def to(self, device):
        """移动到指定设备"""
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = self.model.to(self.device)
        return self

    def _get_default_config(self) -> Dict[str, Any]:
        """返回默认配置"""
        return {
            "mol_emb_dim": 256,
            "hidden_dim": 512,
            "n_layers": 6,
            "n_heads": 8,
            "dropout": 0.1,
            "num_roles": 11,
            "num_token_types": 2,
            "tau": 0.07,
            "learnable_tau": False,
            "symmetric_ince": False,
            "checkpoint_path": None,
            "gvp_root": "/data1/chenyuxuan/MSMLM",
        }

    def _load_checkpoint(self, ckpt_path: str):
        """加载模型权重"""
        logger.info(f"加载 Layer2 checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=self.device)
        
        # 处理不同的 checkpoint 格式
        if "model" in state:
            # checkpoint 包含完整的训练状态（model, optimizer, scheduler 等）
            if isinstance(state["model"], dict):
                # model 是 state_dict
                state_dict = state["model"]
            else:
                # model 是模型对象，尝试获取 state_dict
                state_dict = state["model"].state_dict() if hasattr(state["model"], "state_dict") else state["model"]
        elif "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
        
        # 移除可能的 "module." 前缀（DDP 训练产生的）
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"加载权重时缺少的键: {missing_keys[:5]}...")
            if len(missing_keys) > 5:
                logger.warning(f"  ... 还有 {len(missing_keys) - 5} 个键")
            # 检查是否缺少关键层
            critical_layers = ["yield_bin_head", "yield_reg_head", "encoder"]
            missing_critical = [k for k in missing_keys if any(crit in k for crit in critical_layers)]
            if missing_critical:
                logger.error(f"❌ 缺少关键层: {missing_critical[:3]}...，模型可能无法正常工作")
        if unexpected_keys:
            logger.warning(f"加载权重时意外的键: {unexpected_keys[:5]}...")
            if len(unexpected_keys) > 5:
                logger.warning(f"  ... 还有 {len(unexpected_keys) - 5} 个键")
        
        # 验证模型是否已正确加载（检查 yield_bin_head 权重是否非零）
        if hasattr(self.model, 'yield_bin_head'):
            head_weight = self.model.yield_bin_head.weight.data
            if torch.allclose(head_weight, torch.zeros_like(head_weight), atol=1e-6):
                logger.error("❌ yield_bin_head 权重全为零，模型可能未正确加载")
            else:
                logger.info(f"✅ yield_bin_head 权重已加载 (shape: {head_weight.shape}, 非零元素: {(head_weight != 0).sum().item()}/{head_weight.numel()})")

    def predict(
        self,
        reactant_smiles: Union[str, List[str]],
        gvp_embedding: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        amount_info: Optional[Union[Dict[str, float], List[Dict[str, float]]]] = None,
    ) -> Dict[str, Any]:
        """
        预测反应产率和生成任务相关 embedding。
        
        Args:
            reactant_smiles: 反应物 SMILES 字符串或列表（支持多个反应物）
            gvp_embedding: 可选的 GVP embedding（如果为 None，将使用内部 gvp_encoder）
                         可以是单个 tensor 或列表（对应多个反应物）
            amount_info: 可选的量信息，格式: 
                        - 单个: {"moles": float, "mass": float, "volume": float}
                        - 多个: [{"moles": float, ...}, ...] (对应每个反应物)
        
        Returns:
            {
                'yield_bin': int,  # 0-9，产率区间
                'yield_reg': float,  # 0-1，产率回归值
                'embedding': torch.Tensor,  # 任务相关 embedding (CLS token 的 embedding)
            }
        """
        import torch.nn.functional as F
        from .amount_utils import build_amount_feature
        
        # 1. 处理多个反应物的情况
        if isinstance(reactant_smiles, str):
            reactant_smiles_list = [reactant_smiles]
            is_single = True
        else:
            reactant_smiles_list = reactant_smiles
            is_single = False
        
        # 2. 获取 GVP embeddings
        if gvp_embedding is None:
            if self.gvp_encoder is None:
                raise ValueError("需要提供 gvp_embedding 或初始化时提供 gvp_encoder")
            gvp_embeddings = []
            for smi in reactant_smiles_list:
                gvp_emb_list = smiles_to_embedding(smi, self.gvp_encoder, str(self.device))
                if gvp_emb_list is None:
                    raise ValueError(f"无法从 SMILES 生成 embedding: {smi}")
                gvp_embeddings.append(torch.tensor(gvp_emb_list, device=self.device, dtype=torch.float32))
        else:
            if isinstance(gvp_embedding, list):
                gvp_embeddings = [emb.to(self.device) if isinstance(emb, torch.Tensor) else torch.tensor(emb, device=self.device, dtype=torch.float32) for emb in gvp_embedding]
            else:
                gvp_embeddings = [gvp_embedding.to(self.device)]
        
        # 确保数量匹配
        if len(gvp_embeddings) != len(reactant_smiles_list):
            if len(gvp_embeddings) == 1 and len(reactant_smiles_list) > 1:
                # 如果只有一个 embedding，复制给所有反应物
                gvp_embeddings = gvp_embeddings * len(reactant_smiles_list)
            else:
                raise ValueError(f"GVP embedding 数量 ({len(gvp_embeddings)}) 与反应物数量 ({len(reactant_smiles_list)}) 不匹配")
        
        # 3. 处理 amount_info
        if amount_info is None:
            amount_info_list = [{"moles": 1.0, "mass": 0.0, "volume": 0.0}] * len(reactant_smiles_list)
        elif isinstance(amount_info, dict):
            amount_info_list = [amount_info] * len(reactant_smiles_list)
        else:
            amount_info_list = amount_info
            if len(amount_info_list) != len(reactant_smiles_list):
                if len(amount_info_list) == 1:
                    amount_info_list = amount_info_list * len(reactant_smiles_list)
                else:
                    raise ValueError(f"amount_info 数量 ({len(amount_info_list)}) 与反应物数量 ({len(reactant_smiles_list)}) 不匹配")
        
        # 4. 构建 tokens（每个反应物一个 token）
        # 注意：键名必须与 collate_layer2 期望的格式一致
        tokens = []
        for i, (smi, gvp_emb, amt_info) in enumerate(zip(reactant_smiles_list, gvp_embeddings, amount_info_list)):
            # 构建 amount 特征（10 维）
            amt_feat = build_amount_feature(
                moles=amt_info.get("moles", 1.0),
                mass=amt_info.get("mass", 0.0),
                volume=amt_info.get("volume", 0.0),
                data_mask=[False, False, False],  # 数据是否缺失
                pred_mask=[False, False, False],  # 是否预测
                volume_includes_solutes=False,
            )
            # 将 amt_feat 拆分为各个字段（与 collate_layer2 期望的格式一致）
            # amt_feat 格式: [moles_log, moles_data_mask, moles_pred_mask,
            #                 mass_log, mass_data_mask, mass_pred_mask,
            #                 vol_log, vol_data_mask, vol_pred_mask,
            #                 volume_includes_solutes]
            tokens.append({
                "emb": gvp_emb.cpu().tolist(),
                "reaction_role": "REACTANT",  # 注意：使用 reaction_role 而不是 role
                "token_type": "INPUT",
                # amount 相关字段（从 amt_feat 中提取）
                "amt_moles_log": amt_feat[0] if amt_feat[0] != 0.0 else None,
                "amt_moles_mask": int(amt_feat[1]),
                "amt_moles_pred_mask": int(amt_feat[2]),
                "amt_mass_log": amt_feat[3] if amt_feat[3] != 0.0 else None,
                "amt_mass_mask": int(amt_feat[4]),
                "amt_mass_pred_mask": int(amt_feat[5]),
                "amt_volume_log": amt_feat[6] if amt_feat[6] != 0.0 else None,
                "amt_volume_mask": int(amt_feat[7]),
                "amt_volume_pred_mask": int(amt_feat[8]),
                "volume_includes_solutes": bool(amt_feat[9]),
            })
        
        # 构建样本
        sample = {
            "tokens": tokens,
            "has_yield": False,  # 推理时不需要 yield label
        }
        
        # 3. 使用 collate_layer2 构建 batch
        from .collate import collate_layer2
        batch = collate_layer2([sample])
        
        # 4. 模型前向传播
        with torch.no_grad():
            out = self.model(batch)
        
        # 5. 提取结果
        # CLS token (pos=0) 的输出用于 yield 预测
        cls_embedding = out["pred_emb"][0, 0, :]  # [D]
        pred_yield_bin_logits = out["pred_yield_bin"][0]  # [10]
        pred_yield_reg = out["pred_yield_reg"][0].item()  # scalar
        
        # 获取 yield_bin（argmax）
        yield_bin = int(pred_yield_bin_logits.argmax().item())
        
        # 检查是否需要反标准化 yield_reg
        # 从 checkpoint 中读取标准化参数（如果存在）
        yield_reg_mean = self.cfg.get("yield_reg_mean", 0.0)
        yield_reg_std = self.cfg.get("yield_reg_std", 1.0)
        
        # 如果 yield_reg 被标准化了，需要反标准化
        if yield_reg_std > 0:
            yield_reg = float(pred_yield_reg * yield_reg_std + yield_reg_mean)
        else:
            yield_reg = float(pred_yield_reg)
        
        # clamp to [0, 1]（产率范围）
        yield_reg = max(0.0, min(1.0, yield_reg))
        
        # 计算 yield_bin 对应的产率范围（0-9 对应 0%-100%，每10%一个区间）
        yield_bin_percent_min = yield_bin * 10
        yield_bin_percent_max = (yield_bin + 1) * 10
        yield_bin_range = f"{yield_bin_percent_min}%-{yield_bin_percent_max}%"
        
        # 调试信息（显示 logits 和概率分布）
        bin_logits_list = pred_yield_bin_logits.cpu().tolist()
        bin_probs = torch.softmax(pred_yield_bin_logits, dim=0).cpu().tolist()
        max_prob_idx = int(torch.argmax(pred_yield_bin_logits).item())
        
        # 检查 logits 是否过于相似（可能模型未正确加载或输入有问题）
        logits_std = float(torch.std(pred_yield_bin_logits).item())
        logits_range = float(pred_yield_bin_logits.max().item() - pred_yield_bin_logits.min().item())
        
        if logits_std < 0.1 or logits_range < 0.5:
            logger.warning(
                f"⚠️  Layer2 yield_bin logits 变化很小 (std={logits_std:.4f}, range={logits_range:.4f})，"
                f"可能模型未正确加载或输入数据有问题"
            )
        
        logger.info(
            f"Layer2 yield 预测: yield_bin={yield_bin} (区间: {yield_bin_range}, 概率: {bin_probs[yield_bin]:.3f}), "
            f"yield_reg={yield_reg:.3f} ({yield_reg*100:.1f}%)"
        )
        logger.debug(
            f"Layer2 yield 预测详情:\n"
            f"  bin_logits={[f'{x:.2f}' for x in bin_logits_list]}\n"
            f"  bin_probs={[f'{x:.3f}' for x in bin_probs]}\n"
            f"  yield_reg_raw={pred_yield_reg:.4f}, yield_reg_final={yield_reg:.4f} ({yield_reg*100:.1f}%)\n"
            f"  标准化参数: mean={yield_reg_mean:.4f}, std={yield_reg_std:.4f}\n"
            f"  logits统计: std={logits_std:.4f}, range={logits_range:.4f}"
        )
        
        return {
            'yield_bin': yield_bin,
            'yield_reg': yield_reg,
            'embedding': cls_embedding,  # 原始维度（256），需要外部通过 mol_adapter 映射到 LLM 维度
            'logits': bin_logits_list,  # 添加 logits 用于调试
            'probs': bin_probs,  # 添加概率分布用于调试
            'logits_std': logits_std,  # 添加 logits 标准差用于调试
            'logits_range': logits_range,  # 添加 logits 范围用于调试
        }
