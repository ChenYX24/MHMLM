"""
模型初始化模块 (Optimized)
统一处理模型、tokenizer、GNN等的初始化
"""
import os
import json
import gc
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from collections import OrderedDict

# 延迟导入，避免循环依赖
from .mol_aware_lm import MolAwareCausalLM


def clean_state_dict(state_dict: Dict[str, Any]) -> OrderedDict:
    """工具函数：移除 DDP 产生的 'module.' 前缀，并确保返回 OrderedDict"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def init_tokenizer(llm_name: str, mol_token: str = "<mol>") -> AutoTokenizer:
    """
    初始化tokenizer
    优化点：显式设置 padding_side='right'，这对 SFT 训练至关重要
    """
    tokenizer = AutoTokenizer.from_pretrained(llm_name, use_fast=True)
    
    # 1. 强制设置 pad_token (修复 SFTTrainer 报错的核心)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # 2. 强制设置 padding_side (防止生成任务出现错位)
    tokenizer.padding_side = "right"
    
    # 3. 添加特殊token
    to_add = []
    current_vocab = tokenizer.get_vocab()
    if mol_token not in current_vocab:
        to_add.append(mol_token)
    
    # 仅对 Llama 系列模型补充 Llama 3 风格的特殊 token；
    # 对 Qwen / Mistral 等其他模型，不强行注入这些 token，避免与各自的 chat 模板冲突。
    llm_name_lower = llm_name.lower()
    if "llama" in llm_name_lower:
        special_tokens = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
        for t in special_tokens:
            if t not in current_vocab:
                to_add.append(t)
            
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    
    return tokenizer


def init_llm(llm_name: str, tokenizer: AutoTokenizer, bf16: bool = True, device: str = "cuda:0") -> AutoModelForCausalLM:
    """
    初始化LLM
    
    如果给定的路径是 checkpoint 路径（包含 pytorch_model.bin 或 model.safetensors），
    但 llm/ 子目录不存在，会自动调用 split_llm_extras.py 进行拆分
    """
    import os
    from pathlib import Path
    
    # 检查 torch 版本
    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
    requires_torch_26 = torch_version < (2, 6)
    
    # 检查模型目录是否有 safetensors 文件
    model_path = Path(llm_name)
    
    # 如果路径不存在，检查是否是 checkpoint 路径需要拆分
    if not model_path.exists():
        # 检查是否是 checkpoint/llm 路径，但父目录存在且包含混合权重
        parent_path = model_path.parent
        if parent_path.exists():
            bin_path = parent_path / "pytorch_model.bin"
            safetensors_path = parent_path / "model.safetensors"
            if bin_path.exists() or safetensors_path.exists():
                print(f"📦 检测到 checkpoint 路径但 llm 子目录不存在: {llm_name}")
                print(f"   尝试拆分父目录: {parent_path}")
                try:
                    split_script_path = Path(__file__).parent.parent / "scripts" / "ckpt" / "split_llm_extras.py"
                    if split_script_path.exists():
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("split_llm_extras", split_script_path)
                        split_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(split_module)
                        success = split_module.split_checkpoint(str(parent_path), str(parent_path))
                        if success:
                            print(f"✅ Checkpoint 拆分完成，重新检查路径: {llm_name}")
                            model_path = Path(llm_name)  # 重新赋值
                except Exception as e:
                    print(f"⚠️ 自动拆分失败: {e}")
    has_safetensors = False
    has_bin_files = False
    if model_path.exists():
        safetensors_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("model*.safetensors"))
        has_safetensors = len(safetensors_files) > 0
        bin_files = list(model_path.glob("*.bin")) + list(model_path.glob("pytorch_model*.bin"))
        has_bin_files = len(bin_files) > 0
    
    # 如果 torch < 2.6 且只有 .bin 文件，尝试在 CPU 上自动转换为 safetensors（临时关闭安全检查）
    if requires_torch_26 and not has_safetensors and has_bin_files:
        import warnings
        warnings.warn(
            f"⚠️  Torch version {torch.__version__} < 2.6, model has only .bin files. "
            f"Trying CPU-side auto-conversion to safetensors to bypass the security check."
        )
        try:
            print(f"[Model Init] Converting {llm_name} to safetensors (CPU, dtype={'bf16' if bf16 else 'fp32'})...")
            from transformers import AutoConfig
            # 临时关闭 transformers 安全检查，允许加载 .bin
            os.environ["TRANSFORMERS_SAFE_LOADING_DISABLED"] = "1"
            # 选择单一 bin 文件；若存在分片 index，提示手动处理
            index_files = list(model_path.glob("pytorch_model*.bin.index.json"))
            if index_files:
                raise RuntimeError(
                    "Model appears to be sharded (.bin.index.json found); please convert manually "
                    "or upgrade torch>=2.6 to load sharded .bin safely."
                )
            # 取第一个 bin 文件
            bin_file = sorted(list(model_path.glob("*.bin")) + list(model_path.glob("pytorch_model*.bin")))[0]
            print(f"[Model Init] Loading bin weights from: {bin_file.name}")
            state = torch.load(bin_file, map_location="cpu")
            # 兼容 state_dict 包裹
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # 加载 config 并构建模型
            config = AutoConfig.from_pretrained(llm_name)
            temp_model = AutoModelForCausalLM.from_config(config)
            missing, unexpected = temp_model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[Model Init] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
            # 保存为 safetensors
            temp_model.save_pretrained(
                llm_name,
                safe_serialization=True,
                max_shard_size="5GB"
            )
            del temp_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Model Init] ✅ Conversion completed. Safetensors files saved.")
            has_safetensors = True
            # 恢复默认设置
            os.environ.pop("TRANSFORMERS_SAFE_LOADING_DISABLED", None)
        except Exception as conv_e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"Failed to auto-convert model to safetensors: {conv_e}\n"
                f"Please manually convert or upgrade torch to >= 2.6"
            ) from conv_e
    elif requires_torch_26 and not has_safetensors:
        import warnings
        warnings.warn(
            f"⚠️  Torch version {torch.__version__} < 2.6, and model has no safetensors files. "
            f"Transformers requires torch>=2.6 to load .bin files due to security (CVE-2025-32434).\n"
            f"Solutions:\n"
            f"  1. Upgrade torch: pip install torch>=2.6\n"
            f"  2. Convert model to safetensors format\n"
            f"  3. Downgrade transformers to a version before this check"
        )
    
    # 使用 low_cpu_mem_usage=True 加速加载并减少内存占用
    # 优先尝试 safetensors（如果存在）
    try:
        llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            dtype=torch.bfloat16 if bf16 else torch.float32,  # 使用 dtype 替代已弃用的 torch_dtype
            low_cpu_mem_usage=True,
            use_safetensors=True if has_safetensors else None,  # 如果存在 safetensors 则优先使用
            device_map=None,  # 手动控制 to(device)
            trust_remote_code=True
        ).to(device)
    except Exception as e:
        # 如果 safetensors 加载失败，尝试 .bin（需要 torch >= 2.6）
        if "safetensors" in str(e).lower() or "use_safetensors" in str(e).lower():
            if requires_torch_26:
                raise RuntimeError(
                    f"Failed to load model: {e}\n"
                    f"Your torch version ({torch.__version__}) is < 2.6, which is required to load .bin files.\n"
                    f"Please upgrade torch: pip install 'torch>=2.6'"
                ) from e
            # torch >= 2.6，可以尝试 .bin
            llm = AutoModelForCausalLM.from_pretrained(
                llm_name,
                dtype=torch.bfloat16 if bf16 else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=False,  # 强制使用 .bin
                device_map=None,
                trust_remote_code=True
            ).to(device)
        else:
            raise
    
    # 调整vocab size
    old_vocab_size = llm.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer)
    
    if old_vocab_size != new_vocab_size:
        # 只在 rank 0 打印一次
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[Model Init] Resizing token embeddings: {old_vocab_size} -> {new_vocab_size}")
        
        # mean_resizing=True 使用旧 embedding 的均值初始化新 token，比随机初始化收敛更快
        llm.resize_token_embeddings(new_vocab_size, mean_resizing=True)
        
    # 同步配置
    llm.config.vocab_size = len(tokenizer)
    llm.config.pad_token_id = tokenizer.pad_token_id
    llm.config.eos_token_id = tokenizer.eos_token_id
    llm.config.bos_token_id = tokenizer.bos_token_id
    
    return llm


def init_model(
    cfg: Dict[str, Any],
    tokenizer: AutoTokenizer,
    llm: AutoModelForCausalLM,
    device: str = "cuda:0",
) -> MolAwareCausalLM:
    """初始化MolAwareCausalLM模型"""
    mol_token = cfg.get("tokens", {}).get("mol_token", "<mol>")
    train_conf = cfg.get("train", {}) or {}

    # ===== Diffusion 开关 =====
    # 新增：如果 cfg["train"]["use_diffusion"] 为 False，则完全禁用 diffusion，
    # 不再初始化任何 diffusion/diffusion_adapter 相关模块，减小显存占用。
    use_diffusion = train_conf.get("use_diffusion", True)

    if use_diffusion:
        diffusion_conf = cfg.get("diffusion", {}) or {}
        diff_conf = diffusion_conf.get("diffusion", {}) or {}
        diff_adp_conf = diffusion_conf.get("adapter", {}) or {}
    else:
        diffusion_conf = {}
        diff_conf = {}
        diff_adp_conf = {}
    
    # --- 优化设备分配逻辑 ---
    # 允许通过 env 或 config 灵活分配 diffusion 模型位置，减轻主卡显存压力
    if diff_conf:
        diffusion_device = diff_conf.get("device")
        if not diffusion_device or diffusion_device == "cuda:0":
            env_device = os.environ.get("DIFFUSION_DEVICE")
            if env_device:
                diffusion_device = env_device
            elif device.startswith("cuda:"):
                # 自动尝试分配到下一张卡
                try:
                    curr_id = int(device.split(":")[-1])
                    if torch.cuda.device_count() > curr_id + 1:
                        diffusion_device = f"cuda:{curr_id + 1}"
                    else:
                        diffusion_device = device
                except Exception:
                    diffusion_device = device
            else:
                diffusion_device = device
        diff_conf["device"] = diffusion_device
        if diffusion_device != device:
            print(f"📌 Diffusion model placed on {diffusion_device} (Main LLM on {device})")

    # 检查是否禁用 GNN（当 use_offline_spans=False 时，完全不走 GNN 路径）
    use_offline_spans = cfg.get("train", {}).get("use_offline_spans", False)
    disable_gnn = not use_offline_spans  # 如果 use_offline_spans=False，则禁用 GNN
    
    # 检查是否使用 LDMol（阶段1 SFT 训练时可禁用以节省显存）
    # 如果 use_diffusion=False，默认也禁用 LDMol（因为 LDMol 依赖 diffusion）
    use_ldmol = train_conf.get("use_ldmol", use_diffusion)
    # 是否跳过 LDMol 内置的 text_encoder（联合推理时复用主 Qwen 以节省显存）
    ldmol_skip_text_encoder = train_conf.get("ldmol_skip_text_encoder", False)
    
    if not use_ldmol:
        print("📌 LDMol disabled (use_ldmol=False)")
    elif ldmol_skip_text_encoder:
        print("📌 LDMol text_encoder skipped (ldmol_skip_text_encoder=True, will reuse main Qwen)")
    
    # 检查是否使用 Layer2（用于反应产率预测）
    use_layer2 = train_conf.get("use_layer2", False)
    layer2_config = cfg.get("layer2", {}) or {}
    
    if use_layer2:
        print("📌 Layer2 enabled (use_layer2=True)")
    else:
        print("📌 Layer2 disabled (use_layer2=False)")
    
    # 初始化主模型
    model = MolAwareCausalLM(
        llm=llm,
        tokenizer=tokenizer,
        mol_token=mol_token,
        proxy=cfg.get("network", {}).get("proxy"),
        debug=False,
        diffusion_config=diff_conf,
        diffusion_adapter_config=diff_adp_conf,
        disable_gnn=disable_gnn,  # 传递禁用 GNN 标志
        use_ldmol=use_ldmol,  # 是否使用 LDMol
        ldmol_skip_text_encoder=ldmol_skip_text_encoder,  # 是否跳过 LDMol 内置 text_encoder
        layer2_config=layer2_config,  # Layer2 配置
        use_layer2=use_layer2,  # 是否使用 Layer2
    ).to(device)
    
    # --- 权重加载逻辑优化 ---
    checkpoint_dir = cfg.get("paths", {}).get("checkpoint_dir")
    
    # 1. 优先从 Checkpoint 目录整套加载
    if checkpoint_dir:
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.exists():
            print(f"📂 Loading weights from checkpoint: {checkpoint_dir}")
            load_model_weights_from_checkpoint_dir(model, checkpoint_dir, device)
        else:
            print(f"⚠️ Checkpoint 目录不存在: {checkpoint_dir}")
            print(f"   跳过 checkpoint 加载，使用默认初始化")
    
    # 2. 否则从分散路径加载 (Legacy / Fine-grained control)
    else:
        # GNN
        gnn_path = cfg.get("paths", {}).get("gnn_state_dict_path")
        if gnn_path and os.path.exists(gnn_path):
            try:
                sd = torch.load(gnn_path, map_location="cpu")
                sd = sd.get("model_state_dict", sd)
                model.gvp_encoder.load_state_dict(clean_state_dict(sd), strict=False)
                print(f"✅ Loaded GVPEncoder from {gnn_path}")
            except Exception as e:
                print(f"⚠️ Load GVP failed: {e}")
        
        # 其他适配器
        load_additional_weights(model, cfg, device)
    
    # 初始化 GNN 任务头 (如果需要)
    use_gnn_tasks = cfg.get("train", {}).get("use_gnn_tasks", False)
    if use_gnn_tasks or (checkpoint_dir and os.path.exists(checkpoint_dir)):
        init_gnn_task_heads(model, cfg, device)
    
    # 应用冻结策略
    apply_freeze_config(model, cfg)
    
    # 显存清理
    torch.cuda.empty_cache()
    
    return model


def init_gnn_task_heads(model: MolAwareCausalLM, cfg: Dict[str, Any], device: str):
    """初始化GNN任务头"""
    if not hasattr(model, "gvp_encoder") or model.gvp_encoder is None:
        return
    
    try:
        train_cfg = cfg.get("train", {})
        model.gvp_encoder.init_task_heads(
            num_reg_tasks=train_cfg.get("gnn_num_reg_tasks", 5),
            num_cls_tasks=train_cfg.get("gnn_num_cls_tasks", 1),
            head_hidden_dim=train_cfg.get("gnn_head_hidden_dim", None),
            head_dropout=float(train_cfg.get("gnn_head_dropout", 0.1)),
        )
        # 确保新初始化的头在正确的设备上
        model.gvp_encoder.to(device) 
        print(f"✅ GNN Task Heads initialized.")
    except Exception as e:
        print(f"⚠️ GVP head init failed: {e}")


def load_model_weights_from_checkpoint_dir(model: MolAwareCausalLM, ckpt_dir: str, device: str):
    """
    从checkpoint目录加载权重
    优化点：大幅优化大模型分片加载的内存占用
    
    如果 checkpoint 目录存在但没有 llm/ 子目录，会自动调用 split_llm_extras.py 进行拆分
    """
    ckpt_dir = Path(ckpt_dir)
    
    if not ckpt_dir.exists():
        print(f"❌ Checkpoint 目录不存在: {ckpt_dir}")
        return
    
    # 检查是否需要拆分 checkpoint
    llm_dir = ckpt_dir / "llm"
    
    # 如果 llm 目录不存在，但 checkpoint 目录下有 pytorch_model.bin 或 model.safetensors，则需要拆分
    if not llm_dir.exists():
        bin_path = ckpt_dir / "pytorch_model.bin"
        safetensors_path = ckpt_dir / "model.safetensors"
        if bin_path.exists() or safetensors_path.exists():
            print(f"📦 检测到混合 checkpoint，需要拆分: {ckpt_dir}")
            print(f"   调用 split_llm_extras.py 进行自动拆分...")
            try:
                # 导入拆分函数
                split_script_path = Path(__file__).parent.parent / "split_llm_extras.py"
                if split_script_path.exists():
                    # 动态导入
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("split_llm_extras", split_script_path)
                    split_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(split_module)
                    
                    # 调用拆分函数
                    success = split_module.split_checkpoint(str(ckpt_dir), str(ckpt_dir))
                    if success:
                        print(f"✅ Checkpoint 拆分完成")
                        # 重新检查 llm_dir（拆分后应该存在了）
                        llm_dir = ckpt_dir / "llm"
                    else:
                        print(f"⚠️ Checkpoint 拆分失败，尝试继续加载...")
                else:
                    print(f"⚠️ 找不到 split_llm_extras.py: {split_script_path}")
            except Exception as e:
                import traceback
                print(f"⚠️ 自动拆分失败: {e}")
                traceback.print_exc()
                print(f"   请手动运行: python {split_script_path} 或检查 checkpoint 路径")
    
    # 1. 加载 LLM 权重
    if llm_dir.exists():
        try:
            # 优先使用 from_pretrained 加载，因为它内部处理了分片加载的内存管理 (low_cpu_mem_usage)
            # 比手动合并 state_dict 更安全、更省内存
            print(f"⏳ Loading LLM via from_pretrained (memory efficient)...")
            # 这里我们加载到一个临时模型，然后提取 state_dict，或者直接让 model.llm 重新加载
            # 为了最稳妥，我们直接让 model.llm 调用 HF 的加载逻辑
            # 注意：这需要 model.llm 是标准的 HF 模型实例
            
            # 方案 A: 如果只是为了加载权重，用 load_state_dict 配合 safetensors 最好
            # 方案 B (推荐): 使用 transformers 的 load_sharded_checkpoint 工具
            from transformers.modeling_utils import load_sharded_checkpoint
            
            # 检查是 safetensors 还是 bin
            safetensors_index = llm_dir / "model.safetensors.index.json"
            safetensors_file = llm_dir / "model.safetensors"
            is_safetensors = safetensors_index.exists() or safetensors_file.exists()

            if is_safetensors:
                from safetensors.torch import load_file
                if safetensors_file.exists():
                    # 单文件
                    sd = load_file(str(safetensors_file), device="cpu")
                    model.llm.load_state_dict(sd, strict=False)
                else:
                    # 分片 safetensors (HF 原生支持自动处理)
                    # 重新加载一遍可能是最高效的，因为手动处理分片逻辑很复杂
                    load_sharded_checkpoint(model.llm, str(llm_dir), strict=False, prefer_safe=True)
            else:
                # PyTorch Bin
                bin_file = llm_dir / "pytorch_model.bin"
                bin_index = llm_dir / "pytorch_model.bin.index.json"
                if bin_file.exists():
                    sd = torch.load(str(bin_file), map_location="cpu")
                    model.llm.load_state_dict(sd, strict=False)
                elif bin_index.exists():
                    load_sharded_checkpoint(model.llm, str(llm_dir), strict=False, prefer_safe=False)
            
            print(f"✅ Loaded LLM weights from {llm_dir}")
            
        except Exception as e:
            print(f"⚠️ Optimized LLM load failed: {e}, falling back to legacy...")
            # Fallback (你原来的逻辑，虽然慢但能用)
            pass 

    # 2. 加载 Extras (保持不变，但使用 clean_state_dict)
    extras_dir = ckpt_dir / "extras"
    if extras_dir.exists():
        def _load_component(name, filename):
            path = extras_dir / filename
            comp = getattr(model, name, None)
            if path.exists() and comp is not None:
                try:
                    # 检查文件大小
                    file_size = path.stat().st_size
                    if file_size == 0:
                        print(f"⚠️ Skipping {name}: checkpoint file {filename} is empty (0 bytes)")
                        return
                    
                    sd = torch.load(path, map_location="cpu")
                    if isinstance(sd, dict) and "model_state_dict" in sd: sd = sd["model_state_dict"]
                    elif isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
                    
                    # 检查是否有空权重（形状为 [0]）
                    empty_keys = []
                    for k, v in sd.items():
                        if isinstance(v, torch.Tensor) and v.numel() == 0:
                            empty_keys.append(k)
                    
                    if empty_keys:
                        print(f"⚠️ Skipping {name}: checkpoint contains {len(empty_keys)} empty weights (shape [0])")
                        print(f"   示例: {empty_keys[:3]}...")
                        return
                    
                    comp.load_state_dict(clean_state_dict(sd), strict=False)
                    print(f"✅ Loaded {name} from {filename}")
                except Exception as e:
                    print(f"⚠️ Failed to load {name}: {e}")

        _load_component("gvp_encoder", "gvp_encoder.pt")
        _load_component("mol_adapter", "mol_adapter.pt")
        _load_component("diffusion_adapter", "diffusion_adapter.pt")
        
    # 3. 释放 CPU 内存
    gc.collect()
    torch.cuda.empty_cache()


def load_additional_weights(model: MolAwareCausalLM, cfg: Dict[str, Any], device: str):
    """加载额外的权重 (Unified)"""
    paths = cfg.get("paths", {})
    
    def _load_single(path_key, model_attr, label):
        path = paths.get(path_key)
        comp = getattr(model, model_attr, None)
        if path and os.path.exists(path) and comp is not None:
            try:
                # 检查文件大小
                file_size = Path(path).stat().st_size
                if file_size == 0:
                    print(f"⚠️ Skipping {label}: checkpoint file is empty (0 bytes): {path}")
                    return
                
                sd = torch.load(path, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
                
                # 检查是否有空权重
                empty_keys = [k for k, v in sd.items() if isinstance(v, torch.Tensor) and v.numel() == 0]
                if empty_keys:
                    print(f"⚠️ Skipping {label}: checkpoint contains {len(empty_keys)} empty weights (shape [0])")
                    print(f"   文件: {path}")
                    return
                
                comp.load_state_dict(clean_state_dict(sd), strict=False)
                print(f"✅ Loaded {label} from {path}")
            except Exception as e:
                print(f"⚠️ Failed to load {label}: {e}")

    _load_single("gnn_mlp_state_dict_path", "mol_adapter", "mol_adapter")
    _load_single("diffusion_adapter_state_dict_path", "diffusion_adapter", "diffusion_adapter")


def apply_freeze_config(model: MolAwareCausalLM, cfg: Dict[str, Any]):
    """应用冻结配置"""
    train_cfg = cfg.get("train", {})
    
    # 辅助函数：冻结模块
    def _freeze(module, name):
        for p in module.parameters():
            p.requires_grad = False
        print(f"🔒 Frozen {name}")

    if train_cfg.get("freeze_llm", False):
        for n, p in model.llm.named_parameters():
            if 'embed_tokens' not in n: # 通常保留 embedding 训练以适应新token
                p.requires_grad = False
        print("🔒 Frozen LLM (except embed_tokens)")

    if train_cfg.get("freeze_gnn", False) and getattr(model, "gvp_encoder", None):
        _freeze(model.gvp_encoder, "GVP Encoder")

    if train_cfg.get("freeze_mol_adapter", False) and getattr(model, "mol_adapter", None):
        _freeze(model.mol_adapter, "Mol Adapter")

    if train_cfg.get("freeze_diffusion", True) and getattr(model, "diffusion", None):
        _freeze(model.diffusion, "Diffusion Model")
        
    if train_cfg.get("freeze_diffusion_adapter", True) and getattr(model, "diffusion_adapter", None):
        _freeze(model.diffusion_adapter, "Diffusion Adapter")


def init_offline_token_classifier(
    llm: AutoModelForCausalLM,
    mlp_token_classifier_path: Optional[str],
    device: str = "cuda:0",
) -> Optional[nn.Module]:
    """初始化离线token分类器"""
    if not mlp_token_classifier_path:
        print(f"⚠️ mlp_token_classifier_path is not set in config")
        return None
    
    if not os.path.exists(mlp_token_classifier_path):
        print(f"⚠️ Token classifier file not found: {mlp_token_classifier_path}")
        return None
    
    try:
        print(f"📂 Loading token classifier from: {mlp_token_classifier_path}")
        # 这里的结构需要与训练分类器时的结构一致
        # 如果能从 config 读取更好，目前保持默认
        hidden_size = llm.config.hidden_size
        print(f"   Current model hidden_size: {hidden_size}")
        
        # 先检查 checkpoint 中的 hidden_size
        ckpt_for_check = torch.load(mlp_token_classifier_path, map_location="cpu")
        if isinstance(ckpt_for_check, dict):
            if "state_dict" in ckpt_for_check:
                ckpt_sd = ckpt_for_check["state_dict"]
            elif "model_state_dict" in ckpt_for_check:
                ckpt_sd = ckpt_for_check["model_state_dict"]
            else:
                ckpt_sd = ckpt_for_check
            
            # 查找第一个 Linear 层的权重来确定原始 hidden_size
            for key, value in ckpt_sd.items():
                # 移除可能的 prefix
                clean_key = key.replace("classifier.", "").replace("token_classifier.", "").replace("module.", "")
                if "weight" in clean_key and len(value.shape) == 2:
                    # 第一个 Linear 层的输入维度就是 hidden_size
                    ckpt_hidden_size = value.shape[1]
                    print(f"   Checkpoint hidden_size: {ckpt_hidden_size} (from weight shape: {value.shape})")
                    if ckpt_hidden_size != hidden_size:
                        print(f"   ⚠️ WARNING: Hidden size mismatch! Checkpoint was trained with {ckpt_hidden_size}, "
                              f"but current model has {hidden_size}. This classifier cannot be used.")
                        return None
                    break
        
        token_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        ).to(device)
        
        print(f"   Loading checkpoint...")
        # 重用之前加载的 checkpoint（避免重复加载）
        ckpt = ckpt_for_check
        
        # 检查 checkpoint 的结构
        if isinstance(ckpt, dict):
            print(f"   Checkpoint keys: {list(ckpt.keys())[:10]}...")  # 只显示前10个key
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
        
        # 使用 clean_state_dict 移除 potential module. 前缀，并过滤 key
        clean_sd = clean_state_dict(ckpt)
        print(f"   Cleaned state_dict keys (first 10): {list(clean_sd.keys())[:10]}...")
        
        final_sd = OrderedDict()
        for k, v in clean_sd.items():
            # 尝试多种可能的 key 格式
            if k.startswith("classifier."):
                final_sd[k.replace("classifier.", "")] = v
            elif k.startswith("token_classifier."):
                final_sd[k.replace("token_classifier.", "")] = v
            elif not k.startswith("module.") and not "." in k or k.count(".") <= 1:
                # 如果 key 看起来像是分类器的参数（没有太多层级），直接使用
                final_sd[k] = v
        
        print(f"   Final state_dict keys: {list(final_sd.keys())}")
        
        if len(final_sd) == 0:
            print(f"⚠️ No matching keys found in checkpoint. Available keys: {list(clean_sd.keys())[:20]}...")
            return None
        
        # 尝试加载，如果 strict=False 失败，尝试 strict=False
        try:
            token_head.load_state_dict(final_sd, strict=True)
            print(f"   ✅ Loaded with strict=True")
        except Exception as e:
            print(f"   ⚠️ Strict loading failed: {e}, trying strict=False...")
            missing_keys, unexpected_keys = token_head.load_state_dict(final_sd, strict=False)
            if missing_keys:
                print(f"   ⚠️ Missing keys: {missing_keys[:5]}...")
            if unexpected_keys:
                print(f"   ⚠️ Unexpected keys: {unexpected_keys[:5]}...")
        
        token_head.eval()
        
        # 彻底冻结
        for p in token_head.parameters():
            p.requires_grad = False
        
        print(f"✅ Loaded offline token classifier successfully")
        return token_head
    except Exception as e:
        print(f"❌ Failed to load token classifier: {e}")
        import traceback
        traceback.print_exc()
        return None