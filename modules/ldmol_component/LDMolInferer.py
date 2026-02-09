"""
LDMol 推理组件
通过同目录 `ldmol_config.yaml` 管理配置；外部只需实例化并调用方法。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Final
import yaml
import logging
import os
import glob

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from einops import repeat
from rdkit import Chem
from transformers import AutoModelForCausalLM, AutoTokenizer

from .diffusion import create_diffusion
from .DiT.download import find_model
from .DiT.models import DiTWithTextProj, DiT_models
from .autoencoder.train_autoencoder import ldmol_autoencoder
from .utils import create_logger, AE_SMILES_encode, AE_SMILES_decode, qwen3_encode, regexTokenizer


logger = logging.getLogger(__name__)

class LDMolInferer:

    def __init__(self,config: dict): 
        """
        初始化 LDMolInferer。
        
        Args:
            config: 配置文件路径，默认使用同目录下的 ldmol-config.yaml
        """
        #TODO
        # raise NotImplementedError("不再使用")

        self.config = config
        
        # 启用 TF32 加速（Ampere 及更新架构的 GPU）
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
        
        # # Assets_dir
        # assets_dir = Path(__file__).parent / "assets"
        
        # # Result_dir
        # results_dir = config.get('results_dir', './training_output')
        
        # if self.rank == 0:
        #     os.makedirs(results_dir, exist_ok=True)
        #     experiment_index = len(glob(f"{results_dir}/*"))
        #     model_name = config.get('dit_name', 'LDMol').replace("/", "-")
        #     self.experiment_dir = f"{results_dir}/{experiment_index:03d}-{model_name}"
        #     self.checkpoint_dir = f"{self.experiment_dir}/checkpoints"
        #     os.makedirs(self.checkpoint_dir, exist_ok=True)
        #     self.logger = create_logger(self.experiment_dir, self.rank)
        #     self.logger.info(f"Experiment directory: {self.experiment_dir}")
        # else:
        #     self.experiment_dir = None
        #     self.checkpoint_dir = None
        #     self.logger = create_logger(None, self.rank)
        
        # Configs
        data_config = config.data
        dit_config = config.dit
        smiles_encoder_config = config.smiles_encoder
        text_encoder_config = config.text_encoder
        assert dit_config.name == "LDMol", f"Unsupported DiT model: {dit_config.name}"
        assert smiles_encoder_config.name in ["autoencoder", "gvpencoder"], f"Unsupported SMILES encoder: {smiles_encoder_config.name}"
        assert text_encoder_config.name == "qwen3_8b", f"Unsupported text encoder: {text_encoder_config.name}"
        
        # Initialize DiT model
        self.logger.info("Initializing DiT model...")
        base_model = DiT_models[dit_config.name](
            input_size=dit_config.latent_size,
            in_channels=dit_config.in_channels,
            cross_attn=dit_config.cross_attn_dim,
            condition_dim=dit_config.condition_dim,
        )
        
        text_proj = nn.Linear(text_encoder_config.hidden_dim, dit_config.condition_dim)
        model = DiTWithTextProj(base_model, text_proj=text_proj)
        
        if dit_config.ckpt:
            state_dict = find_model(dit_config.ckpt)
            msg = model.load_state_dict(state_dict, strict=False)
            self.logger.info(f"Loaded DiT checkpoint: {dit_config.ckpt}, {msg}")
        
        self.ema = deepcopy(model).to(self.device)
        requires_grad(self.ema, False)
        self.ema.eval()
        
        self.model = DDP(
            model.to(self.device), 
            device_ids=[self.rank], 
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
                max_len=dit_config.latent_size  # 与 latent_size 一致
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
            del self.smi_encoder.text_encoder  # 训练时不需要文本编码器
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
        
        # Dataset
        data_paths = data_config.data_paths
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
        
        self.logger.info(f"Dataset: {len(self.dataset):,} samples, batch_size_per_gpu: {batch_size}")
 
    
    @torch.no_grad()
    def _ensure_null_condition(self) -> None:
        """缓存 CFG 的 null condition embedding。"""
        if self._null_y is not None:
            return

        null_y, null_pad_mask = qwen3_encode(
            [self.prompt_null],
            self.text_encoder,
            self.text_tokenizer,
            self.description_length,
            self.device,
        )

        self._null_y = null_y.to(device=self.device, dtype=torch.float32)
        self._null_pad_mask = null_pad_mask.to(device=self.device, dtype=torch.bool)
        logger.info(
            "CFG null condition cached: prompt_null=%r null_y=%s null_pad_mask=%s",
            self.prompt_null,
            tuple(self._null_y.shape),
            tuple(self._null_pad_mask.shape),
        )

    @torch.no_grad()
    def _encode_text(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        编码文本为 embedding。
        
        Args:
            texts: 文本列表
            
        Returns:
            y_cond: (B, L, hidden_dim)
            pad_mask: (B, L) bool
        """
        y_cond, pad_mask = qwen3_encode(
            texts,
            self.text_encoder,
            self.text_tokenizer,
            self.description_length,
            self.device,
        )
        return y_cond.to(device=self.device, dtype=torch.float32), pad_mask.to(device=self.device, dtype=torch.bool)

    @torch.no_grad()
    def _sample_latents(self, y_cond: torch.Tensor, pad_mask_cond: torch.Tensor) -> torch.Tensor:
        """
        从条件 embedding 采样 latent。
        
        Args:
            y_cond: (B, L, hidden_dim)
            pad_mask_cond: (B, L) bool

        Returns:
            latents: (B, 127, 64)
        """
        assert self._null_y is not None
        assert self._null_pad_mask is not None

        batch_size, seq_len, _ = y_cond.shape
        assert seq_len == self._null_y.shape[1]
        assert seq_len == self._null_pad_mask.shape[1]

        y_null = repeat(self._null_y, '1 L D -> B L D', B=batch_size)
        pad_mask_null = repeat(self._null_pad_mask, '1 L -> B L', B=batch_size)

        using_cfg = self.cfg_scale > 1.0
        z = torch.randn(batch_size, self.in_channels, self.latent_size, 1, device=self.device)

        if using_cfg:
            z = torch.cat([z, z], 0)
            y = torch.cat([y_cond, y_null], 0)
            pad_mask = torch.cat([pad_mask_cond, pad_mask_null], 0)
            model_kwargs = dict(y=y, pad_mask=pad_mask, cfg_scale=self.cfg_scale)
            sample_fn = self.model.forward_with_cfg
        else:
            model_kwargs = dict(y=y_cond, pad_mask=pad_mask_cond)
            sample_fn = self.model.forward

        samples = self.diffusion.p_sample_loop(
            sample_fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=self.device,
        )

        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        return samples.squeeze(-1).permute((0, 2, 1))

    @torch.no_grad()
    def generate_smi_t2m(self, description: str | list[str]) -> str:
        """
        Text-to-Molecule：从文本描述生成分子 SMILES。

        Args:
            description: 分子的文本描述

        Returns:
            smiles: 生成的 SMILES 字符串
        """
        description = [description] if isinstance(description, str) else description
        y_cond, attention_mask = self._encode_text(description)
        latents = self._sample_latents(y_cond, attention_mask)

        smiles_list = AE_SMILES_decode(latents, self.ae_model, stochastic=False, k=1)
        smiles = self.canonicalize_smiles(smiles_list[0])
        return smiles

    @torch.no_grad()
    def generate_smi_dds(
        self,
        input_smiles: str,
        source_text: str,
        target_text: str,
    ) -> str:
        """
        DDS (Diffusion-based Drug Steering)：属性定向优化。
        
        通过迭代优化，将输入分子从满足 source 属性转换为满足 target 属性。

        Args:
            input_smiles: 输入分子的 SMILES
            source_text: 当前属性描述（如 "This molecule has low permeability."）
            target_text: 目标属性描述（如 "This molecule has improved permeability."）

        Returns:
            output_smiles: 优化后的分子 SMILES
        """
        # 标准化输入 SMILES
        input_smiles = self.canonicalize_smiles(input_smiles)
        logger.info(f"DDS: {input_smiles} | {source_text} => {target_text}")

        # 编码输入分子为 latent
        x_source = AE_SMILES_encode([input_smiles], self.ae_model)  # (1, 127, 64)
        x_source = x_source.permute((0, 2, 1)).unsqueeze(-1)  # (1, 64, 127, 1)
        x_target = x_source.clone()

        # 编码文本条件
        y_cond_s, pad_mask_s = self._encode_text([source_text])
        y_cond_t, pad_mask_t = self._encode_text([target_text])
        y_cond_n, pad_mask_n = self._encode_text([self.prompt_null])

        model_kwargs_s = dict(y=y_cond_s, pad_mask=pad_mask_s)
        model_kwargs_t = dict(y=y_cond_t, pad_mask=pad_mask_t)
        model_kwargs_n = dict(y=y_cond_n, pad_mask=pad_mask_n)

        # DDS 迭代优化
        diffusion = self.diffusion_full
        cfg = self.dds_cfg_scale

        for step in range(self.dds_n_iter):
            # 随机采样时间步
            t = random.randint(self.dds_t_min, self.dds_t_max)
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.int)

            # 对 source 和 target 添加相同噪声
            noise = torch.randn_like(x_target)
            x_target_t = diffusion.q_sample(x_target, t_tensor, noise=noise)
            x_source_t = diffusion.q_sample(x_source, t_tensor, noise=noise)

            # 预测去噪输出（使用 CFG）
            # Source 条件
            model_output_s = self.model(x_source_t, t_tensor, **model_kwargs_s)
            model_output_s, _ = torch.split(model_output_s, x_source_t.shape[1], dim=1)
            model_output_sn = self.model(x_source_t, t_tensor, **model_kwargs_n)
            model_output_sn, _ = torch.split(model_output_sn, x_source_t.shape[1], dim=1)

            # Target 条件
            model_output_t = self.model(x_target_t, t_tensor, **model_kwargs_t)
            model_output_t, _ = torch.split(model_output_t, x_target_t.shape[1], dim=1)
            model_output_tn = self.model(x_target_t, t_tensor, **model_kwargs_n)
            model_output_tn, _ = torch.split(model_output_tn, x_target_t.shape[1], dim=1)

            # 应用 CFG
            model_output_s = model_output_sn + cfg * (model_output_s - model_output_sn)
            model_output_t = model_output_tn + cfg * (model_output_t - model_output_tn)

            # 计算梯度并更新
            grad = (model_output_t - model_output_s).detach()
            x_target = x_target - grad * self.dds_loss_scale

            # 每 50 步打印中间结果
            if step % 50 == 0:
                output = x_target.squeeze(-1).permute((0, 2, 1))
                output_smiles = AE_SMILES_decode(output, self.ae_model, stochastic=False, k=1)
                logger.info(f"DDS step {step}: {output_smiles[0]}")

        # 解码最终结果
        output = x_target.squeeze(-1).permute((0, 2, 1))
        output_smiles_list = AE_SMILES_decode(output, self.ae_model, stochastic=False, k=1)
        output_smiles = self.canonicalize_smiles(output_smiles_list[0])

        logger.info(f"DDS result: {input_smiles} => {output_smiles}")
        return output_smiles

    @staticmethod
    def canonicalize_smiles(smiles: str) -> str:
        """
        标准化 SMILES 字符串。
        
        Args:
            smiles: 输入 SMILES
            
        Returns:
            canonical_smiles: 标准化后的 SMILES
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Fail to canonicalize smiles: {smiles}, return origin smiles")
            return smiles
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    # =========================================================================
    # 联合推理接口（用于 LLM + Diffusion 联合训练/推理）
    # =========================================================================

    @torch.no_grad()
    def generate_smi_from_hidden(
        self,
        y_cond: torch.Tensor,
        pad_mask: torch.Tensor,
        num_samples: int = 1,
    ) -> list[str]:
        """
        从外部 LLM 的 hidden states 生成分子 SMILES。
        
        用于联合推理场景：
        1. 外部 Qwen 生成 rationale，同时获取 hidden states
        2. 将 hidden states 传给 LDMol 生成 SMILES
        
        Args:
            y_cond: LLM hidden states，shape (B, L, D)，D=4096 for Qwen3-8B
            pad_mask: attention mask，shape (B, L)
            num_samples: 每个样本生成的 SMILES 数量
        
        Returns:
            smiles_list: 生成的 SMILES 列表
        """
        # 确保在正确的设备上
        y_cond = y_cond.to(device=self.device, dtype=torch.float32)
        pad_mask = pad_mask.to(device=self.device, dtype=torch.bool)
        
        batch_size, seq_len, hidden_dim = y_cond.shape
        
        # 检查 hidden_dim 是否匹配
        expected_dim = self.cfg.get("llm_hidden_dim", 4096)
        if hidden_dim != expected_dim:
            logger.warning(
                f"Hidden dim mismatch: got {hidden_dim}, expected {expected_dim}. "
                f"Make sure you're using the correct LLM."
            )
        
        # 如果序列长度不匹配，进行截断或填充
        target_len = self.description_length
        if seq_len > target_len:
            # 截断（保留前 target_len 个 token）
            y_cond = y_cond[:, :target_len, :]
            pad_mask = pad_mask[:, :target_len]
        elif seq_len < target_len:
            # 填充
            pad_len = target_len - seq_len
            y_cond = torch.cat([
                y_cond,
                torch.zeros(batch_size, pad_len, hidden_dim, device=self.device, dtype=y_cond.dtype)
            ], dim=1)
            pad_mask = torch.cat([
                pad_mask,
                torch.zeros(batch_size, pad_len, device=self.device, dtype=pad_mask.dtype)
            ], dim=1)
        
        # 通过 text_proj 投影（DiT 模型内部的投影层）
        # 注意：y_cond 会在 model.forward 中通过 text_proj 投影
        # 这里不需要手动调用 text_proj，因为 _sample_latents 会处理
        
        # 确保 null condition 已缓存且维度匹配
        if self._null_y is None or self._null_y.shape[1] != target_len:
            self._null_y = None
            self._null_pad_mask = None
            self._ensure_null_condition()
        
        # 采样 latent
        latents = self._sample_latents(y_cond, pad_mask)
        
        # 解码为 SMILES
        smiles_list = AE_SMILES_decode(latents, self.ae_model, stochastic=False, k=num_samples)
        
        # 标准化
        return [self.canonicalize_smiles(s) for s in smiles_list]

    @torch.no_grad()
    def generate_molecule(
        self,
        description: str,
        qwen: torch.nn.Module | None = None,
        qwen_tokenizer = None,
        use_external_encoder: bool | None = None,
    ) -> str:
        """
        统一的分子生成接口。
        
        支持两种模式：
        1. 使用内置的 text_encoder（默认，需要 skip_text_encoder=False）
        2. 使用外部 Qwen（用于联合推理，复用 MolAwareCausalLM 的 LLM）
        
        Args:
            description: 分子描述文本
            qwen: 外部 Qwen 模型（可选）
            qwen_tokenizer: 外部 tokenizer（可选）
            use_external_encoder: 是否使用外部编码器
                - None: 自动判断（如果提供了 qwen 和 qwen_tokenizer 则使用外部；
                        如果 skip_text_encoder=True 但没提供外部 Qwen，则报错）
                - True: 强制使用外部编码器
                - False: 强制使用内置编码器
        
        Returns:
            smiles: 生成的 SMILES
        """
        # 自动判断是否使用外部编码器
        if use_external_encoder is None:
            if qwen is not None and qwen_tokenizer is not None:
                use_external_encoder = True
            elif self.skip_text_encoder:
                # skip_text_encoder=True 但没有提供外部 Qwen，必须报错
                raise ValueError(
                    "LDMolInferer was initialized with skip_text_encoder=True, "
                    "but no external qwen/qwen_tokenizer was provided. "
                    "Please provide both qwen and qwen_tokenizer arguments."
                )
            else:
                use_external_encoder = False
        
        if use_external_encoder:
            if qwen is None or qwen_tokenizer is None:
                raise ValueError(
                    "use_external_encoder=True but qwen or qwen_tokenizer is None. "
                    "Please provide both."
                )
            
            # 使用外部 Qwen 编码文本
            y_cond, pad_mask = qwen3_encode(
                [description],
                qwen,
                qwen_tokenizer,
                self.description_length,
                self.device,
            )
            
            # 调用 generate_smi_from_hidden
            smiles_list = self.generate_smi_from_hidden(y_cond, pad_mask)
            return smiles_list[0] if smiles_list else ""
        else:
            # 使用内置的 text_encoder
            if self.text_encoder is None:
                raise ValueError(
                    "Cannot use internal text_encoder because skip_text_encoder=True. "
                    "Please provide external qwen and qwen_tokenizer."
                )
            return self.generate_smi_t2m(description)

    @torch.no_grad()
    def batch_generate_from_hidden(
        self,
        y_cond_list: list[torch.Tensor],
        pad_mask_list: list[torch.Tensor],
    ) -> list[str]:
        """
        批量从 hidden states 生成 SMILES。
        
        Args:
            y_cond_list: hidden states 列表，每个 shape (1, L, D)
            pad_mask_list: attention mask 列表，每个 shape (1, L)
        
        Returns:
            smiles_list: 生成的 SMILES 列表
        """
        results = []
        for y_cond, pad_mask in zip(y_cond_list, pad_mask_list):
            smiles = self.generate_smi_from_hidden(y_cond, pad_mask)
            results.extend(smiles)
        return results

    @torch.no_grad()
    def edit_molecule(self, prompt: str, src_smiles: str) -> str:
        """
        Edit molecule using the prompt and source SMILES.
        TODO: 讨论好内部的实现
        """
        raise NotImplementedError("Not Implement")
        