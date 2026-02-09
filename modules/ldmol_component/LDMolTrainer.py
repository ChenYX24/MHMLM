"""
LDMol 训练组件
"""

from __future__ import annotations

import os
import random
import logging
from pathlib import Path
from glob import glob
from time import time
from copy import deepcopy
from collections import OrderedDict
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from einops import repeat
from transformers import AutoModelForCausalLM, AutoTokenizer
from rdkit import Chem, RDLogger
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

from .dataset import smi_txt_dataset
from .diffusion import create_diffusion
from .DiT.download import find_model
from .DiT.models import DiTWithTextProj, DiT_models
from .autoencoder.train_autoencoder import ldmol_autoencoder
from .utils import create_logger, AE_SMILES_encode, AE_SMILES_decode, qwen3_encode, regexTokenizer

# 禁用 RDKit 日志
RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)


# =============================================================================
# 工具函数
# =============================================================================

@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.9999) -> None:
    """
    更新 EMA (Exponential Moving Average) 模型参数。
    
    Args:
        ema_model: EMA 模型
        model: 源模型
        decay: 衰减系数，越接近 1 则 EMA 更新越慢
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: nn.Module, flag: bool = True) -> None:
    """设置模型所有参数是否需要计算梯度"""
    for p in model.parameters():
        p.requires_grad = flag

# =============================================================================
# 训练器
# =============================================================================

class LDMolTrainer:
    def __init__(self, config: dict, rank:int, world_size:int, local_rank:int, results_dir:str):
        
        self.config = config
        self.rank = rank 
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        seed = config.global_seed * world_size + rank
        
        if self.rank == 0:
            print(f"Initialized DDP: world_size={self.world_size}, seed={seed}")
        
        # Assets_dir
        assets_dir = Path(__file__).parent / "assets"

        # Results_dir 
        # self.results_dir = results_dir
        # experiment_name = f"{len(glob(f'{results_dir}/*')):03d}-{config.get('dit_name', 'LDMol').replace('/', '-')}"
        # self.experiment_dir = Path(results_dir) / experiment_name

        self.results_dir = Path(results_dir)
        exp_name_container = [None] 

        if self.rank == 0:
            os.makedirs(self.results_dir, exist_ok=True)
            experiment_index = len(glob(f"{self.results_dir}/*"))
            model_name = config.get('dit_name', 'LDMol').replace("/", "-")
            exp_name_container[0] = f"{experiment_index:03d}-{model_name}"
            
            full_path = self.results_dir / exp_name_container[0]
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"Rank 0 created experiment dir: {full_path}")
        dist.broadcast_object_list(exp_name_container, src=0)
        
        self.experiment_dir = self.results_dir / exp_name_container[0]
        if self.rank == 0:
            self.logger = create_logger(self.experiment_dir, self.rank)
            self.logger.info(f"Experiment directory : {self.experiment_dir}")
        else:
            self.logger = create_logger(None, self.rank)
        dist.barrier()

        # Configs
        self.logger.info(f"Initializing LDMolTrainer with config: {config}")
        dit_config = config.dit
        smiles_encoder_config = config.smiles_encoder
        text_encoder_config = config.text_encoder
        assert dit_config.name == "LDMol", f"Unsupported DiT model: {dit_config.name}"
        assert smiles_encoder_config.name in ["autoencoder", "gvpencoder"], f"Unsupported SMILES encoder: {smiles_encoder_config.name}"
        assert text_encoder_config.name == "qwen3_8b", f"Unsupported text encoder: {text_encoder_config.name}"
        
        # Initialize DiT model
        self.logger.info("Initializing DiT model...")
        self.latent_size = dit_config.latent_size
        self.in_channels = dit_config.in_channels
        self.cross_attn_dim = dit_config.cross_attn_dim
        self.condition_dim = dit_config.condition_dim
        base_model = DiT_models[dit_config.name](
            input_size=self.latent_size,
            in_channels=self.in_channels,
            cross_attn=self.cross_attn_dim,
            condition_dim=self.condition_dim,
        )
        
        text_proj = nn.Linear(text_encoder_config.hidden_dim, self.condition_dim)
        model = DiTWithTextProj(base_model, text_proj=text_proj)
        
        self.ema = deepcopy(model).to(self.device)
        requires_grad(self.ema, False)
        self.ema.eval()
        
        self.model = DDP(
            model.to(self.device), 
            device_ids=[self.local_rank], 
            find_unused_parameters=True
        )
        update_ema(self.ema, self.model.module, decay=0)
        self.logger.info(f"DiT parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Initialize Diffusion
        self.diffusion = create_diffusion(timestep_respacing="")
        
        # Initialize SMILES Encoder
        self.logger.info("Initializing SMILES Encoder...")
        
        if smiles_encoder_config.name == "autoencoder":
            ae_tokenizer = regexTokenizer(
                vocab_path=str(assets_dir / "vocab_bpe_300_sc.txt"), 
                max_len=self.latent_size  # 与 latent_size 一致
            )
            ae_config = {
                "bert_config_decoder": str(assets_dir / "config_decoder.json"),
                "bert_config_encoder": str(assets_dir / "config_encoder.json"),
                "embed_dim": 256,
            }
            
            self.smi_encoder = ldmol_autoencoder(
                config=ae_config, 
                no_train=True, 
                tokenizer=ae_tokenizer,
                use_linear=True
            )
        
            if smiles_encoder_config.ckpt:
                state_dict = find_model(smiles_encoder_config.ckpt)
                msg = self.smi_encoder.load_state_dict(state_dict, strict=False)
                self.logger.info(f"Loaded SMILES encoder checkpoint: {smiles_encoder_config.ckpt}, {msg}")
            
            requires_grad(self.smi_encoder, False)
            # del self.smi_encoder.text_encoder  # 训练时不需要文本编码器
            self.smi_encoder = self.smi_encoder.to(self.device)
            self.smi_encoder.eval()
        elif smiles_encoder_config.name == "gvpencoder":
            # TODO: Initialize GVP Encoder
            pass
        else:
            raise ValueError(f"Unsupported SMILES encoder: {smiles_encoder_config.name}")
        
        
        # Initialize Text Encoder
        self.logger.info("Initializing Text Encoder...")
        
        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            text_encoder_config.ckpt,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(self.device)
        
        requires_grad(self.text_encoder, False)
        self.text_encoder.eval()
        
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_config.ckpt,
            use_fast=True,
            model_max_length=text_encoder_config.description_length,
        )
        
        self.logger.info(f"Text Encoder parameters: {sum(p.numel() for p in self.text_encoder.parameters()):,}")
        
    def train(self):

        config = self.config
        train_config = config.train
        assert train_config is not None
        
        # Train config
        epochs = train_config.epochs
        unconditional_prob = train_config.unconditional_prob
        description_length = train_config.text_encoder.description_length
        log_every = train_config.log_every
        ckpt_every = train_config.ckpt_every
        ema_decay = train_config.ema_decay
        learning_rate = train_config.learning_rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0
        )
        
        # Checkpoint dir
        if self.rank == 0:
            checkpoint_dir = f"{self.experiment_dir}/checkpoints"
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        else:
            checkpoint_dir = None
            self.logger = create_logger(None, self.rank)
 
        # base ckpt
        if train_config.ckpt:
            state_dict = find_model(train_config.ckpt)
            msg = self.model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Loaded DiT checkpoint: {train_config.ckpt}, {msg}")

        # Dataset
        data_paths = train_config.data_paths
        self.dataset = smi_txt_dataset(
            data_path=[data_paths] if isinstance(data_paths, str) else data_paths, 
            data_length=None,
            shuffle=True,
            unconditional=False,
            raw_description=True,
        )
        
        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=dist.get_world_size(),
            rank=self.rank,
            shuffle=True,
            seed=config.global_seed,
        )
        batch_size = config.global_batch_size // dist.get_world_size()
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=self.sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        self.model.train()
        self.training_steps = 0
        running_loss = 0.0
        log_steps = 0
        start_time = time()

        
        if self.rank == 0:
            self.logger.info("Traininng Start!")
            self.logger.info(f"Dataset: {len(self.dataset):,} samples, batch_size_per_gpu: {batch_size}")
            self.logger.info(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.sampler.set_epoch(epoch)
            self.logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            for smiles_batch, text_batch in self.loader:
                
                with torch.no_grad():
                    # 1. SMILES → VAE latent
                    if config.smiles_encoder.name == "autoencoder":
                        x = AE_SMILES_encode(smiles_batch, self.smi_encoder)
                        x = x.permute((0, 2, 1)).unsqueeze(-1)  # (B, C, L, 1)
                    elif config.smiles_encoder.name == "gvpencoder":
                        # TODO: GVP Encoder
                        pass
                    else:
                        raise ValueError(f"Unsupported SMILES encoder: {config.smiles_encoder.name}")
                    
                    
                    # 2. 随机 dropout 文本条件（用于 Classifier-Free Guidance）
                    text_batch = [
                        t if random.random() > unconditional_prob else self.dataset.null_text 
                        for t in text_batch
                    ]
                    
                    # 3. 文本 → LLM embedding
                    y, pad_mask = qwen3_encode(
                        text_batch, 
                        self.text_encoder, 
                        self.text_tokenizer,
                        description_length, 
                        self.device
                    )
                    y = y.detach().float()
                    pad_mask = pad_mask.bool()
                
                
                # 随机采样时间步
                t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=self.device)
                
                # 计算损失
                model_kwargs = dict(y=y, pad_mask=pad_mask)
                loss_dict = self.diffusion.training_losses(self.model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                update_ema(self.ema, self.model.module, decay=ema_decay)
                
                running_loss += loss.item()
                log_steps += 1
                self.training_steps += 1
                
                # Log
                if self.training_steps % log_every == 0:
                    torch.cuda.synchronize()
                    elapsed = time() - start_time
                    steps_per_sec = log_steps / elapsed
                    
                    # 跨 GPU 聚合损失
                    avg_loss = torch.tensor(running_loss / log_steps, device=self.device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / self.world_size
                    
                    self.logger.info(
                        f"Step {self.training_steps:07d} | Loss: {avg_loss:.4f} | "
                        f"Steps/sec: {steps_per_sec:.2f}"
                    )
                    
                    # 重置计数器
                    running_loss = 0.0
                    log_steps = 0
                    start_time = time()
                
                # Save Checkpoint
                if self.training_steps % ckpt_every == 0 and self.training_steps > 0:
                    if self.rank == 0:
                        checkpoint = {
                            "model": self.model.module.state_dict(),
                            "ema": self.ema.state_dict(),
                            "opt": self.optimizer.state_dict(),
                            "step": self.training_steps,
                            "config": OmegaConf.to_container(self.config),
                        }
                        path = f"{checkpoint_dir}/{self.training_steps:07d}.pt"
                        torch.save(checkpoint, path)
                        self.logger.info(f"Saved checkpoint: {path}")
                    
                    dist.barrier()
        
        end_time = time()
        if self.rank == 0:
            self.logger.info("Training complete!")
            self.logger.info(f"Training time: {end_time - start_time:.2f} seconds")
        dist.destroy_process_group()

        

    @torch.no_grad()
    def generate_smi_t2m(self, ckpt: str, data_paths: list[str] | str, *,batch_size=32, using_cfg=True, cfg_scale=2.5):
        config = self.config
        rank = self.rank

        # Results dir
        results_dir = self.experiment_dir / "infer"
        if rank == 0:
            results_dir.mkdir(parents=True, exist_ok=True)

        # Load ckpt
        state_dict = find_model(ckpt)
        msg = self.model.module.load_state_dict(state_dict, strict=False)
        self.logger.info(f"Loaded DiT checkpoint: {ckpt}, {msg}")

        for module in [self.model, self.smi_encoder, self.text_encoder]:
            requires_grad(module, False)
            module.eval()
            if rank == 0:
                self.logger.info(f"{module.__class__.__name__}  #parameters: {sum(p.numel() for p in module.parameters())}, #trainable: {sum(p.requires_grad for p in module.parameters())}")

        dataset = smi_txt_dataset(
            data_path=[data_paths] if isinstance(data_paths, str) else data_paths, 
            data_length=None,
            shuffle=True,
            unconditional=False,
            raw_description=True,
        )
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=self.rank,
            shuffle=True,
            seed=config.global_seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        if self.rank == 0:
            self.logger.info("Generating SMILES Start!")
            self.logger.info(f"Dataset: {len(dataset):,} samples, batch_size_per_gpu: {batch_size}")
            self.logger.info(f"Dataset[0]: {dataset[0]}")
    
        dist.barrier()

        # infer 
        prompt_null = "no dsecription."
        description_length = 512
        biot5_embed_null, mask_null = qwen3_encode([prompt_null], self.text_encoder, self.text_tokenizer, description_length, self.device)

        biot5_embed_null = biot5_embed_null.to(self.device).to(torch.float32)
        mask_null = mask_null.to(self.device).bool()

        rank_file_path = results_dir / f"rank_{self.rank}.txt"
        steps = 0
        total_steps = len(loader)
        log_every = 10
        
        with open(rank_file_path, 'w') as f_rank:
            f_rank.write("orig_smiles\tdescription\tpred_smiles\n")
            for x, y in loader:

                # print(f"x:{x}\ny:{y}\nlen:{len(x)}")

                # Sample inputs:
                z = torch.randn(len(x), self.model.module.in_channels, self.latent_size, 1, device=self.device)

                biot5_embed, pad_mask = qwen3_encode(y, self.text_encoder, self.text_tokenizer, description_length, self.device)

                y_cond = biot5_embed.to(self.device).type(torch.float32)
                pad_mask_cond = pad_mask.to(self.device).bool()
            
                y_null = repeat(biot5_embed_null, '1 L D -> B L D', B=len(x))
                pad_mask_null = repeat(mask_null, '1 L -> B L', B=len(x))

                # Setup classifier-free guidance:
                if using_cfg:
                    z = torch.cat([z, z], 0)
                    y_c = torch.cat([y_cond, y_null], 0)
                    pad_mask = torch.cat([pad_mask_cond, pad_mask_null], 0)
                    model_kwargs = dict(y=y_c, pad_mask=pad_mask, cfg_scale=cfg_scale)
                    sample_fn = self.model.module.forward_with_cfg
                else:
                    model_kwargs = dict(y=y_cond, pad_mask=pad_mask)
                    sample_fn = self.model.module.forward

                # Sample images:
                samples = self.diffusion.p_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=self.device
                )
                if using_cfg:
                    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

                samples = samples.squeeze(-1).permute((0, 2, 1))
                samples = AE_SMILES_decode(samples, self.smi_encoder, stochastic=False, k=1)

                # Save samples to disk as individual .png files
                assert len(samples) == len(x)
                for i, s in enumerate(samples):
                    f_rank.write(x[i].replace('[CLS]', '')+'\t'+y[i]+'\t'+s+'\n')
                    f_rank.flush()
                
                steps += 1
                if steps % log_every == 0:
                    self.logger.info(f"Step {steps}/{total_steps}")

                # print(f"samples: {samples}\n\n")
                   
        self.logger.info(f"Rank {self.rank} generation finished.")

        dist.barrier()
        if self.rank == 0:
            final_output_path = results_dir / 'generated_molecules_t2m.txt'
            self.logger.info("Merging files...")
            
            with open(final_output_path, 'w') as f_out:
                for r in range(self.world_size):
                    part_file = results_dir / f'rank_{r}.txt'
                    if part_file.exists():
                        with open(part_file, 'r') as f_in:
                            f_out.write(f_in.read())
                        
                        os.remove(part_file)
            self.logger.info(f"All done! Saved to {final_output_path}")

            # TODO:score


        

# def main():
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#     import argparse
    
#     parser = argparse.ArgumentParser(description="LDMol Trainer")
#     parser.add_argument(
#         "--config", 
#         type=str, 
#         default=str(Path(__file__).parent / "assets" / "ldmol_train.yaml"),
#         help="训练配置文件路径 (yaml)"
#     )
#     args = parser.parse_args()
    
#     # 禁用 tokenizer 并行（避免多进程死锁）
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
#     config = OmegaConf.load(args.config)
#     print(f"Loaded config from: {args.config}")
    
#     trainer = LDMolTrainer(config)
#     trainer.train()


# if __name__ == "__main__":
#     main()
