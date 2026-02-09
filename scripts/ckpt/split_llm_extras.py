import torch
import os
import json
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
import sys


def split_checkpoint(checkpoint_path: str, output_dir: str = None):
    """
    拆分 checkpoint 中的混合权重为 LLM 和 Extras
    
    Args:
        checkpoint_path: checkpoint 目录路径（包含 pytorch_model.bin）
        output_dir: 输出目录，默认为 checkpoint_path
    """
    if output_dir is None:
        output_dir = checkpoint_path
    
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    
    # 检查权重文件：支持多种格式
    bin_path = checkpoint_path / "pytorch_model.bin"
    safetensors_path = checkpoint_path / "model.safetensors"
    
    # 检查是否有分片的 safetensors 文件（model-00001-of-00005.safetensors 等）
    safetensors_index_path = checkpoint_path / "model.safetensors.index.json"
    sharded_safetensors = False
    if safetensors_index_path.exists():
        try:
            with open(safetensors_index_path, 'r') as f:
                index_data = json.load(f)
                if "weight_map" in index_data:
                    sharded_safetensors = True
                    print(f"🔄 检测到分片 safetensors 文件")
        except:
            pass
    
    # 检查 global_step 目录（DeepSpeed ZeRO checkpoint）
    global_step_dirs = sorted([d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("global_step")])
    
    if not bin_path.exists() and not safetensors_path.exists() and not sharded_safetensors and not global_step_dirs:
        print(f"❌ 找不到权重文件: {checkpoint_path}")
        print(f"   期望找到: pytorch_model.bin, model.safetensors, 分片safetensors, 或 global_step* 目录")
        return False
    
    # 优先使用 safetensors（单个或分片）
    if sharded_safetensors:
        print(f"🔄 正在加载分片 safetensors 权重...")
        from safetensors.torch import load_file
        full_state_dict = {}
        with open(safetensors_index_path, 'r') as f:
            index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            # 收集所有需要加载的文件
            shard_files = set(weight_map.values())
            for shard_file in sorted(shard_files):
                shard_path = checkpoint_path / shard_file
                if shard_path.exists():
                    print(f"   加载分片: {shard_file} ...")
                    shard_dict = load_file(str(shard_path))
                    full_state_dict.update(shard_dict)
                else:
                    print(f"   ⚠️ 警告: 分片文件不存在: {shard_file}")
    elif safetensors_path.exists():
        print(f"🔄 正在加载混合权重 (safetensors): {safetensors_path} ...")
        from safetensors.torch import load_file
        full_state_dict = load_file(str(safetensors_path))
    elif global_step_dirs:
        # 尝试从最新的 global_step 目录恢复
        latest_global_step = global_step_dirs[-1]
        print(f"🔄 检测到 DeepSpeed ZeRO checkpoint，尝试从 {latest_global_step.name} 恢复...")
        
        # 查找模型状态文件
        model_state_files = list(latest_global_step.glob("*model_states.pt"))
        if not model_state_files:
            print(f"❌ 在 {latest_global_step} 中找不到 model_states.pt 文件")
            return False
        
        # 加载所有分片的模型状态
        full_state_dict = {}
        for model_state_file in sorted(model_state_files):
            print(f"   加载模型状态: {model_state_file.name} ...")
            state = torch.load(model_state_file, map_location="cpu")
            # DeepSpeed ZeRO 格式：state 可能包含 'module' 键
            if isinstance(state, dict):
                if 'module' in state:
                    state = state['module']
                elif 'model' in state:
                    state = state['model']
                # 合并到 full_state_dict
                if isinstance(state, dict):
                    full_state_dict.update(state)
                else:
                    print(f"   ⚠️ 警告: {model_state_file.name} 格式未知，跳过")
        
        if not full_state_dict:
            print(f"❌ 无法从 {latest_global_step} 加载模型权重")
            return False
        print(f"✅ 成功从 {latest_global_step.name} 加载 {len(full_state_dict)} 个权重")
    else:
        print(f"🔄 正在加载混合权重 (bin): {bin_path} ...")
        full_state_dict = torch.load(bin_path, map_location="cpu")
    
    llm_sd = {}
    extras_sd = {}
    
    print("🔄 正在拆分权重...")
    keys = list(full_state_dict.keys())
    
    # 定义 extras 的关键词（用于识别非 LLM 权重）
    extras_keywords = ["gvp_encoder", "mol_adapter", "diffusion_adapter", "diffusion"]
    
    for k in keys:
        # 检查是否是 extras 权重
        is_extras = any(x in k for x in extras_keywords)
        
        if is_extras:
            extras_sd[k] = full_state_dict[k]
        elif k.startswith("llm."):
            # 有 llm. 前缀，去掉前缀后加入 llm_sd
            new_k = k[4:] 
            llm_sd[new_k] = full_state_dict[k]
        else:
            # 没有前缀，可能是直接的 LLM 权重（未被包装的情况）
            # 检查是否包含 LLM 常见的层名
            llm_keywords = ["embed", "layers", "norm", "lm_head", "model.", "transformer."]
            if any(x in k for x in llm_keywords):
                llm_sd[k] = full_state_dict[k]
            else:
                # 不确定的 key，默认当作 LLM 权重（更安全）
                llm_sd[k] = full_state_dict[k]
    
    print(f"📊 拆分结果: LLM权重={len(llm_sd)} keys, Extras权重={len(extras_sd)} keys")
            
    # ================= 保存 LLM (带自动修复 Config 功能) =================
    llm_save_dir = output_dir / "llm"
    llm_save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 正在保存 LLM 到 {llm_save_dir} ...")
    try:
        # 1. 尝试加载 Config（优先级：llm/子目录 > checkpoint根目录 > 父目录）
        config = None
        llm_config_path = checkpoint_path / "llm" / "config.json"
        root_config_path = checkpoint_path / "config.json"
        parent_config_path = checkpoint_path.parent / "config.json"
        
        if llm_config_path.exists():
            print(f"📂 从 llm/ 子目录加载 config: {llm_config_path}")
            config = AutoConfig.from_pretrained(str(llm_config_path))
        elif root_config_path.exists():
            print(f"📂 从 checkpoint 根目录加载 config: {root_config_path}")
            config = AutoConfig.from_pretrained(str(checkpoint_path))
        elif parent_config_path.exists():
            print(f"📂 从父目录加载 config: {parent_config_path}")
            config = AutoConfig.from_pretrained(str(checkpoint_path.parent))
        else:
            print("⚠️ 未找到 config.json，尝试从 checkpoint 目录自动检测...")
            try:
                config = AutoConfig.from_pretrained(str(checkpoint_path))
            except Exception as e:
                print(f"⚠️ 无法从 checkpoint 加载 config: {e}")
                # 尝试从 llm 子目录加载（即使路径不存在，AutoConfig 可能会自动查找）
                try:
                    llm_dir = checkpoint_path / "llm"
                    if llm_dir.exists():
                        config = AutoConfig.from_pretrained(str(llm_dir))
                        print(f"✅ 从 llm/ 子目录成功加载 config")
                except:
                    pass
                
                if config is None:
                    print("⚠️ 无法加载 config，将使用权重中的实际 vocab_size（代码会自动修正）")
                    # 创建一个临时 config，稍后会被权重中的实际值覆盖
                    # 注意：AutoConfig 已在文件顶部导入，不需要再次导入
                    # 尝试从模型名称推断（如果路径包含模型名）
                    model_name = str(checkpoint_path)
                    if "llama" in model_name.lower():
                        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
                    elif "mistral" in model_name.lower():
                        config = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.1")
                    else:
                        # 默认使用 LLaMA（最常见）
                        config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
                    print(f"⚠️ 使用默认 config（稍后会被权重中的实际值修正）")

        # 2. 【核心修复】检测权重里的词表大小，并修正 Config
        # 智能查找嵌入层：支持多种可能的 key 格式
        embed_weight = None
        possible_keys = [
            "model.embed_tokens.weight",  # LLaMA/Mistral 标准格式
            "embed_tokens.weight",  # 无 model 前缀
            "transformer.wte.weight",  # GPT-2 格式
            "model.embedding.weight",  # 其他可能格式
        ]
        
        # 先尝试已知的 key
        for key in possible_keys:
            if key in llm_sd:
                embed_weight = llm_sd[key]
                print(f"🔍 找到嵌入层: {key}, shape={embed_weight.shape}")
                break
        
        # 如果没找到，搜索所有包含 "embed" 的 key
        if embed_weight is None:
            for k, v in llm_sd.items():
                if "embed" in k.lower() and "weight" in k.lower() and len(v.shape) == 2:
                    embed_weight = v
                    print(f"🔍 自动检测到嵌入层: {k}, shape={embed_weight.shape}")
                    break
        
        if embed_weight is not None:
            real_vocab_size = embed_weight.shape[0]
            if config.vocab_size != real_vocab_size:
                print(f"🔧 检测到词表大小变更: Config({config.vocab_size}) -> Weights({real_vocab_size})")
                print(f"🔧 自动修正 config.vocab_size = {real_vocab_size}")
                config.vocab_size = real_vocab_size
        else:
            print("⚠️ 警告: 未找到嵌入层权重，无法自动修正 vocab_size")
            print(f"   可用的 key 示例: {list(llm_sd.keys())[:5]}...")
        
        # 3. 使用修正后的 Config 初始化模型并加载权重
        model = AutoModelForCausalLM.from_config(config)
        
        # 加载权重 (此时形状应该匹配了)
        model.load_state_dict(llm_sd, strict=True)
        
        # 4. 保存为 HF 格式 (safetensors + config.json)
        model.save_pretrained(str(llm_save_dir), safe_serialization=True)
        
        # 同时保存 tokenizer (如果 checkpoint 目录下有 tokenizer 文件)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
            tokenizer.save_pretrained(str(llm_save_dir))
            print("✅ Tokenizer 也已复制并保存")
        except Exception as e:
            print(f"⚠️ 未找到 Tokenizer 文件: {e}")
        
        print("✅ LLM 保存成功 (safetensors格式)")
        
    except Exception as e:
        import traceback
        print(f"⚠️ LLM save_pretrained 失败: {e}")
        traceback.print_exc()
        print("尝试仅保存 pytorch_model.bin ...")
        torch.save(llm_sd, str(llm_save_dir / "pytorch_model.bin"))
    
    # ================= 保存 Extras =================
    extras_save_dir = output_dir / "extras"
    extras_save_dir.mkdir(parents=True, exist_ok=True)
    
    gvp_only = {k.replace("gvp_encoder.", ""): v for k, v in extras_sd.items() if "gvp_encoder" in k}
    mol_only = {k.replace("mol_adapter.", ""): v for k, v in extras_sd.items() if "mol_adapter" in k}
    diff_only = {k.replace("diffusion_adapter.", ""): v for k, v in extras_sd.items() if "diffusion_adapter" in k}

    if gvp_only: 
        torch.save(gvp_only, str(extras_save_dir / "gvp_encoder.pt"))
        print(f"✅ 保存 GVP encoder ({len(gvp_only)} params)")
    if mol_only: 
        torch.save(mol_only, str(extras_save_dir / "mol_adapter.pt"))
        print(f"✅ 保存 Mol adapter ({len(mol_only)} params)")
    if diff_only: 
        torch.save(diff_only, str(extras_save_dir / "diffusion_adapter.pt"))
        print(f"✅ 保存 Diffusion adapter ({len(diff_only)} params)")

    print(f"\n🎉 拆分完成！输出目录: {output_dir}")
    return True


if __name__ == "__main__":
    # 默认配置（用于命令行直接运行）
    # epoch2_1 训练完成后的最后一个 checkpoint
    checkpoint_path = "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/name_conversion/checkpoint-535"
    output_dir = ""
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = checkpoint_path
    split_checkpoint(checkpoint_path, output_dir)

