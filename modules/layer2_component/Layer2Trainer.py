"""
Layer2 训练组件

用法示例:
    # cd 至项目根目录
    cd /data1/chenyuxuan/MHMLM/
    
    # 训练
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m modules.layer2_component.Layer2Trainer \\
        --config modules/layer2_component/layer2_train_config.yaml
"""

from __future__ import annotations

import os
import random
import logging
from pathlib import Path
from glob import glob
from time import time
from collections import OrderedDict
import yaml

import torch
import torch.nn as nn
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


class Layer2Trainer:
    """Layer2 训练器"""
    
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
            self.experiment_dir = f"{results_dir}/{experiment_index:03d}-layer2"
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
        )
        
        # 创建模型
        self.model = Layer2PretrainModel(model_cfg).to(self.device)
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.device])
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
        )
        
        # 数据加载
        self._setup_data(config)
        
        # 训练参数
        self.num_epochs = config.get('num_epochs', 10)
        self.save_steps = config.get('save_steps', 1000)
        self.log_steps = config.get('log_steps', 100)
        
        # Loss 权重
        self.emb_lambda = config.get('emb_lambda', 1.0)
        self.amt_lambda = config.get('amt_lambda', 1.0)
        self.yield_weight = config.get('yield_weight', 1.0)
        self.yield_reg_lambda = config.get('yield_reg_lambda', 1.0)
        self.yield_mode = config.get('yield_mode', 'both')
        
    def _setup_data(self, config: dict):
        """设置数据加载"""
        data_path = config.get('data_path')
        if not data_path:
            raise ValueError("需要指定 data_path")
        
        # 使用 IterableDataset 或 IndexedDataset
        use_indexed = config.get('use_indexed_dataset', False)
        
        masking_cfg = MaskingConfig(
            emb_mask_prob=config.get('emb_mask_prob', 0.15),
            amt_mask_prob=config.get('amt_mask_prob', 0.15),
        )
        
        if use_indexed:
            self.train_dataset = Layer2JsonlIndexed(
                data_path,
                masking=True,
                masking_cfg=masking_cfg,
            )
        else:
            self.train_dataset = Layer2JsonlIterable(
                data_path,
                masking=True,
                masking_cfg=masking_cfg,
            )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.get('batch_size', 32),
            collate_fn=collate_layer2,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
        )
        
        if self.world_size > 1:
            # DDP 模式下，IterableDataset 会自动分片
            pass
    
    def train(self):
        """训练主循环"""
        self.model.train()
        global_step = 0
        
        for epoch in range(self.num_epochs):
            if self.rank == 0:
                self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            for batch in self.train_loader:
                # 前向传播
                out = self.model(batch)
                
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
                    model=self.model.module if hasattr(self.model, 'module') else self.model,
                )
                
                loss = losses['loss_total']
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                global_step += 1
                
                # 日志
                if global_step % self.log_steps == 0 and self.rank == 0:
                    self.logger.info(
                        f"Step {global_step}: loss={loss.item():.4f}, "
                        f"emb={losses['loss_emb'].item():.4f}, "
                        f"amt={losses['loss_amt'].item():.4f}, "
                        f"yield={losses['loss_yield'].item():.4f}"
                    )
                
                # 保存 checkpoint
                if global_step % self.save_steps == 0 and self.rank == 0:
                    self._save_checkpoint(global_step)
        
        # 最终保存
        if self.rank == 0:
            self._save_checkpoint(global_step, is_final=True)
    
    def _save_checkpoint(self, step: int, is_final: bool = False):
        """保存 checkpoint"""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        state = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        suffix = 'final' if is_final else f'step_{step}'
        ckpt_path = f"{self.checkpoint_dir}/checkpoint_{suffix}.pt"
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
