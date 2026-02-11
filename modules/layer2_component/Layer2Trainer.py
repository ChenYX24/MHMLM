"""
Layer2 训练组件 (Stage 2)

支持功能:
- LR scheduler (cosine annealing + linear warmup)
- Validation loop with best model saving
- Gradient accumulation
- Resume from Stage 1 checkpoint
- Multi-GPU DDP
- IndexedDataset + DistributedSampler
- AMP (Mixed Precision)
- Yield class weights (inverse-frequency)
- Original eval-compatible checkpoint format

用法示例:
    cd /data1/chenyuxuan/MHMLM/

    # Stage 2: 8 GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 scripts/layer2/train_layer2.py \\
        --config scripts/layer2/layer2_train_config_stage2.yaml
"""

from __future__ import annotations

import os
import json
import math
import random
import logging
from pathlib import Path
from glob import glob
from time import time
from collections import OrderedDict
from copy import deepcopy
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .model import ModelConfig, Layer2PretrainModel, compute_losses
from .dataset import Layer2JsonlIterable, Layer2JsonlIndexed
from .collate import collate_layer2
from .masking import MaskingConfig

logger = logging.getLogger(__name__)


def create_logger(logging_dir: str | None, rank: int) -> logging.Logger:
    """创建日志器"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt") if logging_dir else logging.NullHandler()
            ]
        )
        return logging.getLogger(__name__)
    else:
        _logger = logging.getLogger(__name__)
        _logger.addHandler(logging.NullHandler())
        return _logger


def compute_yield_stats_from_jsonl(jsonl_path: str):
    """
    从原始JSONL文件统计yield分布（不使用masking）。
    用于计算类别权重和回归标准化参数。

    Returns:
        tuple: (class_weights, reg_mean, reg_std)
        - class_weights: torch.Tensor[10] or None（10个bin的权重）
        - reg_mean: float（yield_reg的均值）
        - reg_std: float（yield_reg的标准差）
    """
    bin_counts = torch.zeros(10, dtype=torch.float32)
    reg_values = []

    print(f"[INFO] 从原始JSONL统计yield分布: {jsonl_path}")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            yield_bin_val = ex.get("yield_bin")
            if yield_bin_val is not None:
                try:
                    bin_id = int(yield_bin_val)
                    if 0 <= bin_id < 10:
                        bin_counts[bin_id] += 1.0
                except (ValueError, TypeError):
                    continue

            yield_reg_val = ex.get("yield_reg")
            if yield_reg_val is not None:
                try:
                    reg_val = float(yield_reg_val)
                    if math.isfinite(reg_val):
                        reg_values.append(reg_val)
                except (ValueError, TypeError):
                    continue

    # 计算class weights (inverse-frequency)
    total = bin_counts.sum()
    if total > 0:
        bin_counts = torch.clamp(bin_counts, min=1.0)
        class_weights = total / (10.0 * bin_counts)
        print(f"[INFO] Yield bin统计: 总计={total:.0f}, 分布={bin_counts.int().tolist()}")
        print(f"[INFO] Yield class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")
    else:
        class_weights = None
        print("[WARN] 未找到yield_bin样本，不使用类别权重")

    # 计算reg normalization
    if reg_values:
        reg_array = np.array(reg_values)
        reg_mean = float(np.mean(reg_array))
        reg_std = float(np.std(reg_array))
        if reg_std < 1e-6:
            reg_std = 1.0
        print(f"[INFO] Yield reg统计: 样本数={len(reg_values)}, mean={reg_mean:.4f}, std={reg_std:.4f}")
    else:
        reg_mean = 0.0
        reg_std = 1.0
        print("[WARN] 未找到yield_reg样本，使用默认normalization（mean=0, std=1）")

    return class_weights, reg_mean, reg_std


class Layer2Trainer:
    """Layer2 训练器 (Stage 2 with validation, scheduler, grad accum, resume, AMP)"""

    def __init__(self, config: dict):
        self.config = config

        # 启用 TF32
        torch.backends.cuda.matmul.allow_tf32 = config.get('tf32', True)
        torch.backends.cudnn.allow_tf32 = config.get('tf32', True)

        assert torch.cuda.is_available(), "Training requires CUDA"

        dist.init_process_group("nccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = self.rank % torch.cuda.device_count()

        seed = config.get('global_seed', 0) * self.world_size + self.rank
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.set_device(self.device)

        if self.rank == 0:
            print(f"Initialized DDP: world_size={self.world_size}, seed={seed}")

        # 结果目录
        results_dir = config.get('results_dir', './training_output/layer2')
        if self.rank == 0:
            os.makedirs(results_dir, exist_ok=True)
            experiment_index = len(glob(f"{results_dir}/*"))
            self.experiment_dir = f"{results_dir}/{experiment_index:03d}-layer2-stage2"
            self.checkpoint_dir = f"{self.experiment_dir}/checkpoints"
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger = create_logger(self.experiment_dir, self.rank)
            self.logger.info(f"Experiment directory: {self.experiment_dir}")
        else:
            self.experiment_dir = None
            self.checkpoint_dir = None
            self.logger = create_logger(None, self.rank)

        # 模型配置
        model_cfg = ModelConfig(
            mol_emb_dim=config.get('mol_emb_dim', 256),
            hidden_dim=config.get('hidden_dim', 512),
            n_layers=config.get('n_layers', 6),
            n_heads=config.get('n_heads', 8),
            dropout=config.get('dropout', 0.1),
            num_roles=config.get('num_roles', 11),
            num_token_types=config.get('num_token_types', 2),
            tau=config.get('tau', 0.07),
            learnable_tau=config.get('learnable_tau', False),
            symmetric_ince=config.get('symmetric_ince', False),
            use_projection_head=config.get('use_projection_head', False),
            head_dropout=float(config.get('head_dropout', 0.0)),
        )
        self.model_cfg = model_cfg

        # 创建模型
        self.model = Layer2PretrainModel(model_cfg).to(self.device)

        # Resume from Stage 1 checkpoint
        resume_from = config.get('resume_from')
        if resume_from and os.path.exists(resume_from):
            self._load_stage1_checkpoint(resume_from)

        # Freeze backbone (with optional partial unfreeze)
        self.freeze_backbone = config.get('freeze_backbone', False)
        self._backbone_params = []
        self._head_params = []
        if self.freeze_backbone:
            # Phase A: full freeze (no backbone unfreeze), Phase B unfreezes later
            phase_b_step_init = int(config.get('phase_b_step', 0))
            if phase_b_step_init > 0:
                # Phase A: override to 0 unfrozen layers
                saved = config.get('unfreeze_top_layers', 0)
                config['unfreeze_top_layers'] = 0
                self._backbone_params, self._head_params = self._setup_freeze()
                config['unfreeze_top_layers'] = saved  # restore for Phase B
            else:
                self._backbone_params, self._head_params = self._setup_freeze()

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=True)

        # Optimizer — differential LR when partial unfreeze is used
        if self.freeze_backbone and self._backbone_params:
            backbone_lr = float(config.get('backbone_lr', 3e-5))
            head_lr = float(config.get('head_lr', 1e-3))
            param_groups = [
                {'params': self._backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
                {'params': self._head_params, 'lr': head_lr, 'name': 'heads'},
            ]
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=head_lr,
                weight_decay=float(config.get('weight_decay', 0.05)),
            )
            if self.rank == 0:
                print(f"[optimizer] Differential LR: backbone={backbone_lr:.1e}, heads={head_lr:.1e}")
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            # Use head_lr if available (Phase A heads-only), else learning_rate
            lr = float(config.get('head_lr', config.get('learning_rate', 3e-4)))
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=lr,
                weight_decay=float(config.get('weight_decay', 0.05)),
            )
            if self.rank == 0 and config.get('head_lr'):
                print(f"[optimizer] Phase A heads-only LR: {lr:.1e}")

        # Grad accumulation
        self.grad_accumulation_steps = config.get('grad_accumulation_steps', 1)

        # AMP (Mixed Precision)
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        if self.rank == 0:
            print(f"AMP (Mixed Precision): {'enabled' if self.use_amp else 'disabled'}")

        # 数据加载
        self._setup_data(config)

        # Yield class weights + reg normalization
        self.yield_class_weights = None
        self.yield_reg_mean = 0.0
        self.yield_reg_std = 1.0
        if config.get('yield_class_weights', False):
            data_path = config.get('data_path')
            if data_path and os.path.exists(data_path):
                if self.rank == 0:
                    cw, rm, rs = compute_yield_stats_from_jsonl(data_path)
                    # Broadcast to all ranks
                    stats = {'class_weights': cw, 'reg_mean': rm, 'reg_std': rs}
                else:
                    stats = None

                # Broadcast from rank 0
                if self.world_size > 1:
                    import pickle
                    if self.rank == 0:
                        stats_bytes = pickle.dumps(stats)
                        stats_tensor = torch.ByteTensor(list(stats_bytes)).to(self.device)
                        size_tensor = torch.tensor([len(stats_bytes)], dtype=torch.long, device=self.device)
                    else:
                        size_tensor = torch.tensor([0], dtype=torch.long, device=self.device)

                    dist.broadcast(size_tensor, src=0)
                    if self.rank != 0:
                        stats_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device=self.device)
                    dist.broadcast(stats_tensor, src=0)

                    if self.rank != 0:
                        stats_bytes = bytes(stats_tensor.cpu().tolist())
                        stats = pickle.loads(stats_bytes)

                self.yield_class_weights = stats['class_weights']
                self.yield_reg_mean = stats['reg_mean']
                self.yield_reg_std = stats['reg_std']

        # LR Scheduler
        self._setup_scheduler(config)

        # 训练参数
        self.num_epochs = config.get('num_epochs', 100)
        self.save_steps = config.get('save_steps', 2000)
        self.log_steps = config.get('log_steps', 50)
        self.eval_steps = config.get('eval_steps', 1000)

        # Loss 权重
        self.emb_lambda = config.get('emb_lambda', 1.0)
        self.amt_lambda = config.get('amt_lambda', 0.5)
        self.yield_weight = config.get('yield_weight', 2.0)
        self.yield_reg_lambda = config.get('yield_reg_lambda', 1.0)
        self.yield_mode = config.get('yield_mode', 'soft_bin_only')
        self.yield_soft_bin_temperature = config.get('yield_soft_bin_temperature', 0.1)

        # Best validation tracking
        self.best_val_loss = float('inf')
        self.best_composite = -float('inf')
        self.best_yield_acc = -float('inf')
        self.best_selection = self.config.get('best_selection', 'composite')  # 'val_loss', 'yield', 'composite'
        self.composite_alpha = float(self.config.get('composite_alpha', 0.3))  # weight for emb in composite

        # Yield label smoothing (v4)
        self.yield_label_smoothing = float(config.get('yield_label_smoothing', 0.0))

        # EMA (Exponential Moving Average) for smoother generalization
        self.ema_decay = float(config.get('ema_decay', 0.0))
        self.ema_shadow = None
        phase_b_step = int(config.get('phase_b_step', 0))
        if self.ema_decay > 0 and phase_b_step <= 0:
            # No phasing: initialize EMA immediately
            raw_model = self.model.module if hasattr(self.model, 'module') else self.model
            self.ema_shadow = deepcopy(raw_model.state_dict())
            if self.rank == 0:
                print(f"[EMA] Enabled with decay={self.ema_decay}")
        elif self.ema_decay > 0:
            if self.rank == 0:
                print(f"[EMA] Deferred to Phase B (decay={self.ema_decay})")

        # Early stopping
        self.patience = int(config.get('patience', 0))
        self._no_improve_count = 0
        self._should_stop = False
        if self.patience > 0 and self.rank == 0:
            print(f"[EarlyStop] Patience={self.patience} evals")

        # Two-phase training: Phase A (frozen head warmup) -> Phase B (partial unfreeze)
        self.phase_b_step = int(self.config.get('phase_b_step', 0))  # 0 = no phasing
        self._in_phase_b = (self.phase_b_step <= 0)  # if no phasing, start in "phase B" (normal)
        if self.phase_b_step > 0 and self.rank == 0:
            print(f"[TwoPhase] Phase A: {self.phase_b_step} steps (frozen backbone, no EMA)")
            print(f"[TwoPhase] Phase B starts at step {self.phase_b_step} (unfreeze top layers + EMA)")

    def _load_stage1_checkpoint(self, ckpt_path: str):
        """Load Stage 1 checkpoint, skip missing keys (new heads initialized randomly)"""
        if self.rank == 0:
            print(f"Loading Stage 1 checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=f'cuda:{self.device}', weights_only=False)

        # Handle various checkpoint formats
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        # Load with strict=False to allow new heads (emb_proj, etc.) to be randomly initialized
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        if self.rank == 0:
            if missing:
                print(f"  Missing keys (randomly initialized): {missing}")
            if unexpected:
                print(f"  Unexpected keys (ignored): {unexpected}")
            print(f"  Stage 1 checkpoint loaded successfully")

    def _setup_freeze(self):
        """Freeze backbone with optional partial unfreeze of top encoder layers.

        When unfreeze_top_layers=0: same as old _freeze_backbone (heads only).
        When unfreeze_top_layers=N: top N encoder layers + final_ln are also trainable.

        Returns two lists for differential LR:
            backbone_params: unfrozen encoder layers + final_ln (slow LR)
            head_params: all heads + log_tau (fast LR)
        """
        unfreeze_top = self.config.get('unfreeze_top_layers', 0)
        n_layers = self.model_cfg.n_layers

        # Names of head modules (always trainable)
        head_names = {'emb_head', 'emb_proj', 'amt_head', 'yield_bin_head', 'yield_reg_head', 'head_drop'}

        # Which encoder layer indices to unfreeze
        unfrozen_layer_ids = set(range(n_layers - unfreeze_top, n_layers)) if unfreeze_top > 0 else set()

        backbone_params = []  # unfrozen encoder layers + final_ln (slow LR)
        head_params = []      # heads + log_tau (fast LR)
        frozen_count = 0

        for name, param in self.model.named_parameters():
            is_head = any(name.startswith(h) for h in head_names)
            is_tau = ('log_tau' in name)
            is_final_ln = name.startswith('final_ln')

            # Check if param is in an unfrozen encoder layer
            is_unfrozen_layer = False
            if name.startswith('encoder.layers.'):
                parts = name.split('.')
                if len(parts) >= 3:
                    try:
                        layer_idx = int(parts[2])
                        if layer_idx in unfrozen_layer_ids:
                            is_unfrozen_layer = True
                    except ValueError:
                        pass

            if is_head or is_tau:
                param.requires_grad = True
                head_params.append(param)
            elif is_unfrozen_layer or (is_final_ln and unfreeze_top > 0):
                param.requires_grad = True
                backbone_params.append(param)
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        backbone_count = sum(p.numel() for p in backbone_params)
        head_count = sum(p.numel() for p in head_params)
        total = frozen_count + backbone_count + head_count

        if self.rank == 0:
            print(f"[setup_freeze] Frozen: {frozen_count:,} params")
            if unfreeze_top > 0:
                print(f"[setup_freeze] Unfrozen backbone (layers {sorted(unfrozen_layer_ids)} + final_ln): "
                      f"{backbone_count:,} params")
            print(f"[setup_freeze] Trainable heads: {head_count:,} params")
            print(f"[setup_freeze] Total trainable: {backbone_count + head_count:,} / {total:,} "
                  f"({(backbone_count + head_count) / total * 100:.1f}%)")

        return backbone_params, head_params

    def _setup_data(self, config: dict):
        """设置训练和验证数据加载"""
        data_path = config.get('data_path')
        if not data_path:
            raise ValueError("需要指定 data_path")

        use_indexed = config.get('use_indexed_dataset', False)

        # Build MaskingConfig from config
        masking_cfg = MaskingConfig(
            p_forward=config.get('p_forward', 0.25),
            p_retro=config.get('p_retro', 0.20),
            p_condition=config.get('p_condition', 0.15),
            p_random=config.get('p_random', 0.05),
            p_yield_full=config.get('p_yield_full', 0.15),
            p_yield_with_product=config.get('p_yield_with_product', 0.20),
        )

        if self.rank == 0:
            print(f"MaskingConfig: {masking_cfg}")

        if use_indexed:
            self.train_dataset = Layer2JsonlIndexed(
                data_path,
                masking=True,
                masking_cfg=masking_cfg,
            )
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config.get('batch_size', 64),
                sampler=self.train_sampler,
                collate_fn=collate_layer2,
                num_workers=config.get('num_workers', 4),
                pin_memory=True,
                drop_last=True,
            )
        else:
            self.train_dataset = Layer2JsonlIterable(
                data_path,
                masking=True,
                masking_cfg=masking_cfg,
            )
            self.train_sampler = None
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config.get('batch_size', 64),
                collate_fn=collate_layer2,
                num_workers=config.get('num_workers', 4),
                pin_memory=True,
            )

        # Validation dataset (always indexed for random access)
        val_data_path = config.get('val_data_path')
        if val_data_path and os.path.exists(val_data_path):
            # Use no masking for validation, or eval masking
            from .masking import EvalMaskingConfig
            eval_masking_cfg = EvalMaskingConfig()

            self.val_dataset = Layer2JsonlIndexed(
                val_data_path,
                masking=True,
                masking_cfg=eval_masking_cfg,
            )
            self.val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.get('batch_size', 64),
                sampler=self.val_sampler,
                collate_fn=collate_layer2,
                num_workers=config.get('num_workers', 4),
                pin_memory=True,
                drop_last=False,
            )
            if self.rank == 0:
                print(f"Validation dataset: {len(self.val_dataset)} samples")
        else:
            self.val_dataset = None
            self.val_loader = None
            if self.rank == 0:
                print("No validation dataset configured")

    def _setup_scheduler(self, config: dict):
        """Setup cosine annealing with linear warmup"""
        warmup_steps = int(config.get('warmup_steps', 500))
        min_lr = float(config.get('min_lr', 1e-6))
        # For differential LR, use head_lr as reference for warmup factor
        if self.freeze_backbone and self._backbone_params:
            base_lr = float(config.get('head_lr', 1e-3))
        else:
            base_lr = float(config.get('learning_rate', 3e-4))

        # Estimate total steps
        if hasattr(self.train_dataset, '__len__'):
            steps_per_epoch = len(self.train_dataset) // (
                config.get('batch_size', 64) * self.world_size
            )
        else:
            # Estimate from data: ~27958 train samples
            steps_per_epoch = 27958 // (config.get('batch_size', 64) * self.world_size)

        # Account for gradient accumulation
        steps_per_epoch = max(1, steps_per_epoch // self.grad_accumulation_steps)
        total_steps = steps_per_epoch * config.get('num_epochs', 100)

        if self.rank == 0:
            print(f"Scheduler: warmup={warmup_steps}, total={total_steps}, "
                  f"steps_per_epoch={steps_per_epoch}, min_lr={min_lr}")

        # Linear warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-8 / max(base_lr, 1e-8),
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Cosine annealing
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=min_lr,
        )

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    def _transition_to_phase_b(self):
        """Transition from Phase A (frozen backbone) to Phase B (partial unfreeze + EMA).

        Called once when global_step reaches phase_b_step.
        - Unfreezes top N encoder layers + final_ln
        - Creates new optimizer with differential LR
        - Rebuilds scheduler from current step
        - Initializes EMA shadow weights
        """
        if self.rank == 0:
            self.logger.info("=== Transitioning to Phase B ===")

        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        # Unfreeze top layers
        unfreeze_top = self.config.get('unfreeze_top_layers', 0)
        n_layers = self.model_cfg.n_layers
        head_names = {'emb_head', 'emb_proj', 'amt_head', 'yield_bin_head', 'yield_reg_head', 'head_drop'}
        unfrozen_layer_ids = set(range(n_layers - unfreeze_top, n_layers)) if unfreeze_top > 0 else set()

        backbone_params = []
        head_params = []
        frozen_count = 0

        for name, param in raw_model.named_parameters():
            is_head = any(name.startswith(h) for h in head_names)
            is_tau = ('log_tau' in name)
            is_final_ln = name.startswith('final_ln')
            is_unfrozen_layer = False
            if name.startswith('encoder.layers.'):
                parts = name.split('.')
                if len(parts) >= 3:
                    try:
                        layer_idx = int(parts[2])
                        if layer_idx in unfrozen_layer_ids:
                            is_unfrozen_layer = True
                    except ValueError:
                        pass

            if is_head or is_tau:
                param.requires_grad = True
                head_params.append(param)
            elif is_unfrozen_layer or (is_final_ln and unfreeze_top > 0):
                param.requires_grad = True
                backbone_params.append(param)
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        backbone_count = sum(p.numel() for p in backbone_params)
        head_count = sum(p.numel() for p in head_params)

        if self.rank == 0:
            self.logger.info(f"  [Phase B] Unfrozen backbone: {backbone_count:,}, heads: {head_count:,}, frozen: {frozen_count:,}")

        # New optimizer with differential LR
        backbone_lr = float(self.config.get('backbone_lr', 5e-5))
        head_lr_b = float(self.config.get('phase_b_head_lr', self.config.get('head_lr', 1e-3)))
        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
            {'params': head_params, 'lr': head_lr_b, 'name': 'heads'},
        ]
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=head_lr_b,
            weight_decay=float(self.config.get('weight_decay', 0.05)),
        )
        if self.rank == 0:
            self.logger.info(f"  [Phase B] New optimizer: backbone_lr={backbone_lr:.1e}, head_lr={head_lr_b:.1e}")

        # Rebuild scaler (existing one is fine, just reset)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Rebuild scheduler for remaining steps
        warmup_b = int(self.config.get('phase_b_warmup', 200))
        min_lr = float(self.config.get('min_lr', 1e-6))

        if hasattr(self.train_dataset, '__len__'):
            steps_per_epoch = len(self.train_dataset) // (
                self.config.get('batch_size', 64) * self.world_size
            )
        else:
            steps_per_epoch = 27958 // (self.config.get('batch_size', 64) * self.world_size)
        steps_per_epoch = max(1, steps_per_epoch // self.grad_accumulation_steps)
        remaining_epochs = self.num_epochs - (self.phase_b_step // max(steps_per_epoch, 1))
        remaining_steps = max(1, remaining_epochs * steps_per_epoch)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-8 / max(head_lr_b, 1e-8),
            end_factor=1.0,
            total_iters=warmup_b,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, remaining_steps - warmup_b),
            eta_min=min_lr,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_b],
        )

        if self.rank == 0:
            self.logger.info(f"  [Phase B] Scheduler: warmup={warmup_b}, remaining_steps={remaining_steps}")

        # Initialize EMA from current (well-trained) heads
        if self.ema_decay > 0:
            self.ema_shadow = deepcopy(raw_model.state_dict())
            if self.rank == 0:
                self.logger.info(f"  [Phase B] EMA initialized (decay={self.ema_decay})")

        # Enable label smoothing for Phase B
        phase_b_ls = float(self.config.get('phase_b_label_smoothing', self.config.get('yield_label_smoothing', 0.0)))
        self.yield_label_smoothing = phase_b_ls
        if self.rank == 0:
            self.logger.info(f"  [Phase B] label_smoothing={phase_b_ls}")

        self._in_phase_b = True

    def _ema_update(self):
        """Update EMA shadow weights after each optimizer step."""
        if self.ema_shadow is None:
            return
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        decay = self.ema_decay
        with torch.no_grad():
            for key, param in raw_model.state_dict().items():
                if key in self.ema_shadow:
                    self.ema_shadow[key].mul_(decay).add_(param, alpha=1 - decay)

    def _ema_swap(self):
        """Swap model weights with EMA shadow weights. Call before/after validation."""
        if self.ema_shadow is None:
            return
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        current = raw_model.state_dict()
        raw_model.load_state_dict(self.ema_shadow)
        self.ema_shadow = current

    def train(self):
        """训练主循环"""
        self.model.train()
        global_step = 0
        accum_loss = 0.0
        accum_emb = 0.0
        accum_amt = 0.0
        accum_yield = 0.0
        accum_count = 0

        start_time = time()

        for epoch in range(self.num_epochs):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            if self.rank == 0:
                self.logger.info(f"=== Epoch {epoch + 1}/{self.num_epochs} ===")

            for micro_step, batch in enumerate(self.train_loader):
                # 前向传播 (with AMP autocast)
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    out = self.model(batch)

                    # Get underlying model for learnable tau
                    raw_model = self.model.module if hasattr(self.model, 'module') else self.model

                    # 计算损失
                    losses = compute_losses(
                        out,
                        batch,
                        tau=self.config.get('tau', 0.07),
                        emb_lambda=self.emb_lambda,
                        amt_lambda=self.amt_lambda,
                        yield_weight=self.yield_weight,
                        yield_reg_lambda=self.yield_reg_lambda,
                        yield_mode=self.yield_mode,
                        yield_soft_bin_temperature=self.yield_soft_bin_temperature,
                        symmetric_ince=self.model_cfg.symmetric_ince,
                        model=raw_model,
                        yield_class_weights=self.yield_class_weights,
                        yield_reg_mean=self.yield_reg_mean,
                        yield_reg_std=self.yield_reg_std,
                        yield_label_smoothing=self.yield_label_smoothing,
                    )

                    loss = losses['loss_total'] / self.grad_accumulation_steps

                # 反向传播 (with AMP scaler)
                self.scaler.scale(loss).backward()

                # Track losses for logging
                accum_loss += losses['loss_total'].detach().item()
                accum_emb += losses['loss_emb'].item()
                accum_amt += losses['loss_amt'].item()
                accum_yield += losses['loss_yield'].item()
                accum_count += 1

                # Gradient accumulation step
                if (micro_step + 1) % self.grad_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # EMA update
                    self._ema_update()

                    global_step += 1

                    # Phase transition check
                    if self.phase_b_step > 0 and not self._in_phase_b and global_step >= self.phase_b_step:
                        self._transition_to_phase_b()

                    # 日志
                    if global_step % self.log_steps == 0 and self.rank == 0:
                        avg_loss = accum_loss / max(accum_count, 1)
                        avg_emb = accum_emb / max(accum_count, 1)
                        avg_amt = accum_amt / max(accum_count, 1)
                        avg_yield = accum_yield / max(accum_count, 1)
                        lr = self.optimizer.param_groups[0]['lr']
                        elapsed = time() - start_time

                        # Get tau value
                        tau_val = self.config.get('tau', 0.07)
                        if hasattr(raw_model, 'log_tau') and raw_model.log_tau is not None:
                            tau_val = torch.exp(raw_model.log_tau).item()

                        self.logger.info(
                            f"Step {global_step} | loss={avg_loss:.4f} "
                            f"emb={avg_emb:.4f} amt={avg_amt:.4f} yield={avg_yield:.4f} | "
                            f"lr={lr:.2e} tau={tau_val:.4f} | "
                            f"epoch={epoch+1}/{self.num_epochs} | "
                            f"elapsed={elapsed:.0f}s"
                        )
                        accum_loss = 0.0
                        accum_emb = 0.0
                        accum_amt = 0.0
                        accum_yield = 0.0
                        accum_count = 0

                    # Validation
                    if global_step % self.eval_steps == 0 and self.val_loader is not None:
                        self._ema_swap()  # use EMA weights for validation
                        val_metrics = self._validate()
                        if self.rank == 0:
                            y_acc = val_metrics.get('yield_bin_acc', 0)
                            e_top1 = val_metrics.get('emb_top1', 0)
                            composite = y_acc + self.composite_alpha * e_top1

                            self.logger.info(
                                f"  [Val] loss={val_metrics['val_loss']:.4f} "
                                f"emb={val_metrics['val_emb']:.4f} "
                                f"amt={val_metrics['val_amt']:.4f} "
                                f"yield={val_metrics['val_yield']:.4f} "
                                f"emb_top1={e_top1:.4f} "
                                f"yield_acc={y_acc:.4f} "
                                f"composite={composite:.4f}"
                            )

                            # Always track val_loss
                            if val_metrics['val_loss'] < self.best_val_loss:
                                self.best_val_loss = val_metrics['val_loss']

                            # Save best_by_yield separately
                            if y_acc > self.best_yield_acc:
                                self.best_yield_acc = y_acc
                                self._save_checkpoint(global_step, epoch=epoch, tag='best_yield')

                            # Primary best selection
                            improved = False
                            if self.best_selection == 'val_loss':
                                if val_metrics['val_loss'] <= self.best_val_loss:
                                    improved = True
                            elif self.best_selection == 'yield':
                                if y_acc >= self.best_yield_acc:
                                    improved = True
                            else:  # composite (default)
                                if composite > self.best_composite:
                                    self.best_composite = composite
                                    improved = True

                            if improved:
                                self._no_improve_count = 0
                                self._save_checkpoint(global_step, epoch=epoch, is_best=True)
                                self.logger.info(
                                    f"  New best! yield_acc={y_acc:.4f} "
                                    f"composite={composite:.4f} "
                                    f"val_loss={val_metrics['val_loss']:.4f}")
                            else:
                                self._no_improve_count += 1
                                if self.patience > 0:
                                    self.logger.info(
                                        f"  No improvement ({self._no_improve_count}/{self.patience})")
                        self._ema_swap()  # swap back to training weights
                        self.model.train()

                        # Early stopping check (broadcast from rank 0)
                        if self.patience > 0:
                            should_stop = torch.tensor(
                                [1 if self._no_improve_count >= self.patience else 0],
                                device=self.device)
                            if self.world_size > 1:
                                dist.broadcast(should_stop, src=0)
                            if should_stop.item():
                                self._should_stop = True
                                if self.rank == 0:
                                    self.logger.info(
                                        f"  Early stopping at step {global_step} "
                                        f"(no improvement for {self.patience} evals)")
                                break

                    # 保存 checkpoint
                    if global_step % self.save_steps == 0 and self.rank == 0:
                        self._save_checkpoint(global_step, epoch=epoch)

            # Early stopping: check if we broke out of inner loop
            if self._should_stop:
                break

        # 最终保存
        if self.rank == 0:
            self._save_checkpoint(global_step, epoch=epoch, is_final=True)

    @torch.no_grad()
    def _validate(self) -> dict:
        """Validation loop"""
        self.model.eval()

        total_loss = 0.0
        total_emb = 0.0
        total_amt = 0.0
        total_yield = 0.0
        total_emb_correct = 0
        total_emb_count = 0
        total_yield_correct = 0
        total_yield_count = 0
        num_batches = 0

        raw_model = self.model.module if hasattr(self.model, 'module') else self.model

        for batch in self.val_loader:
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                out = self.model(batch)
                losses = compute_losses(
                    out,
                    batch,
                    tau=self.config.get('tau', 0.07),
                    emb_lambda=self.emb_lambda,
                    amt_lambda=self.amt_lambda,
                    yield_weight=self.yield_weight,
                    yield_reg_lambda=self.yield_reg_lambda,
                    yield_mode=self.yield_mode,
                    yield_soft_bin_temperature=self.yield_soft_bin_temperature,
                    symmetric_ince=self.model_cfg.symmetric_ince,
                    model=raw_model,
                    yield_class_weights=self.yield_class_weights,
                    yield_reg_mean=self.yield_reg_mean,
                    yield_reg_std=self.yield_reg_std,
                    yield_label_smoothing=0.0,  # no label smoothing during validation
                )

            total_loss += losses['loss_total'].item()
            total_emb += losses['loss_emb'].item()
            total_amt += losses['loss_amt'].item()
            total_yield += losses['loss_yield'].item()

            # Embedding top-1 accuracy (cast to float32 to avoid AMP dtype mismatch)
            if batch.emb_query_pos.numel() > 0:
                device = out["pred_emb"].device
                qp = batch.emb_query_pos.to(device)
                pos = batch.emb_pos.to(device).float()
                # Use linear head for eval (not projection)
                q = out["pred_emb"][qp[:, 0], qp[:, 1], :].float()
                q = F.normalize(q, p=2, dim=-1)
                pos_norm = F.normalize(pos, p=2, dim=-1)

                tau_val = self.config.get('tau', 0.07)
                if hasattr(raw_model, 'log_tau') and raw_model.log_tau is not None:
                    tau_val = torch.exp(raw_model.log_tau).clamp(min=0.01, max=1.0).item()

                logits = (q @ pos_norm.t()) / tau_val
                preds = logits.argmax(dim=-1)
                targets = torch.arange(pos.size(0), device=device)
                total_emb_correct += (preds == targets).sum().item()
                total_emb_count += pos.size(0)

            # Yield bin accuracy
            y_mask = batch.yield_pred_mask.to(out["pred_yield_bin"].device)
            idx = (y_mask > 0.5).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() > 0:
                y_bin = batch.yield_bin.to(out["pred_yield_bin"].device)[idx]
                pred_bin = out["pred_yield_bin"][idx]
                pred_class = pred_bin.argmax(dim=-1)
                total_yield_correct += (pred_class == y_bin).sum().item()
                total_yield_count += idx.numel()

            num_batches += 1

        # Gather across GPUs
        metrics = torch.tensor([total_loss, total_emb, total_amt, total_yield,
                               total_emb_correct, total_emb_count, num_batches,
                               total_yield_correct, total_yield_count],
                              device=self.device)
        if self.world_size > 1:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        n = max(metrics[6].item(), 1)
        emb_count = max(metrics[5].item(), 1)
        yield_count = max(metrics[8].item(), 1)

        return {
            'val_loss': metrics[0].item() / n,
            'val_emb': metrics[1].item() / n,
            'val_amt': metrics[2].item() / n,
            'val_yield': metrics[3].item() / n,
            'emb_top1': metrics[4].item() / emb_count,
            'yield_bin_acc': metrics[7].item() / yield_count,
        }

    def _save_checkpoint(self, step: int, epoch: int = 0, is_final: bool = False, is_best: bool = False, tag: str = ''):
        """
        保存 checkpoint (original eval-compatible format).

        Uses 'model' and 'cfg' keys to match the original eval script:
            ckpt["model"] -> model.state_dict()
            ckpt["cfg"]   -> ModelConfig as dict

        For best_model: saves EMA weights if EMA is enabled.
        """
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Build cfg dict from ModelConfig (for original eval compatibility)
        cfg_dict = {
            'mol_emb_dim': self.model_cfg.mol_emb_dim,
            'hidden_dim': self.model_cfg.hidden_dim,
            'n_layers': self.model_cfg.n_layers,
            'n_heads': self.model_cfg.n_heads,
            'dropout': self.model_cfg.dropout,
            'num_roles': self.model_cfg.num_roles,
            'num_token_types': self.model_cfg.num_token_types,
            'tau': self.model_cfg.tau,
            'learnable_tau': self.model_cfg.learnable_tau,
            'symmetric_ince': self.model_cfg.symmetric_ince,
            # Note: use_projection_head excluded for original eval compatibility
        }

        # Use EMA weights for best model checkpoint if EMA is active
        if is_best and self.ema_shadow is not None:
            model_weights = deepcopy(self.ema_shadow)
        else:
            model_weights = model.state_dict()

        state = {
            'model': model_weights,                  # original eval uses ckpt["model"]
            'cfg': cfg_dict,                          # original eval uses ckpt["cfg"]
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'step': step,
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'yield_reg_mean': self.yield_reg_mean,
            'yield_reg_std': self.yield_reg_std,
            'config': self.config,
        }

        if tag:
            ckpt_path = f"{self.checkpoint_dir}/{tag}.pt"
        elif is_best:
            ckpt_path = f"{self.checkpoint_dir}/best_model.pt"
        elif is_final:
            ckpt_path = f"{self.checkpoint_dir}/checkpoint_final.pt"
        else:
            ckpt_path = f"{self.checkpoint_dir}/checkpoint_step_{step}.pt"

        torch.save(state, ckpt_path)
        self.logger.info(f"Saved checkpoint: {ckpt_path}")


def main():
    """训练入口"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 训练
    trainer = Layer2Trainer(config)
    trainer.train()

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
