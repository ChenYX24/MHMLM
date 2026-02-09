# train_sft.py - 简化版，使用模块化初始化
import os
import re
import glob
import json
import time
import yaml
import random
import numpy as np
from typing import Optional
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, TrainerCallback
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
try:
    from trl import SFTConfig
except ImportError:
    # 如果 SFTConfig 不可用，使用 None 作为标记
    SFTConfig = None
from dataclasses import dataclass
from typing import List, Dict, Any

# ---------------------------------------------------------------------
# 安全反序列化（高版本 PyTorch 支持 add_safe_globals，低版本直接跳过）——允许 numpy ndarray
import numpy
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([numpy.dtype])

# ---------------------------------------------------------------------
# 环境与日志
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 设置 PyTorch CUDA 内存分配器以减少碎片化
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# SwanLab 仅在 rank0 初始化
def _env_rank():
    r = os.environ.get("RANK")
    try:
        return int(r) if r is not None else 0
    except Exception:
        return 0

_IS_RANK0_ENV = (_env_rank() == 0)
import swanlab
swanlab.init(
    project="mol-sft-simple",
    experiment_name="exp-001",
    description="SFT with <mol> embedding-append",
    mode="online" if _IS_RANK0_ENV else "offline"
)

# 导入模块
from modules.model_init import (
    init_tokenizer, init_llm, init_model, init_offline_token_classifier
)
from modules.data_loader import (
    load_training_data, compute_qm9_stats_from_dataset
)
from modules.mol_aware_lm import MolAwareCausalLM

# LDMol 支持（仅用于推理，训练时不需要）
# 注意：trainer.py 中的 init_ldmol_components 和 compute_ldmol_loss 在训练时不需要
# LDMol 组件在推理时通过 sft_tester.py 和 mol_aware_lm.py 初始化
LDMOL_AVAILABLE = True  # 标记为可用，但训练时不使用


# ======================== 工具函数 ========================
def safe_barrier():
    try:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    except Exception:
        pass

def latest_checkpoint(output_dir: str) -> Optional[str]:
    """获取最新的checkpoint"""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else -1
    )
    return checkpoints[-1]

def _ensure_root_weight_link(ckpt_dir: str, llm_dir: str):
    """确保 checkpoint 根目录有一份 HF 认可的权重"""
    src = Path(llm_dir) / "model.safetensors"
    dst = Path(ckpt_dir) / "model.safetensors"
    if not src.exists():
        return
    if dst.exists():
        return
    try:
        os.link(src, dst)
        print(f"[SaveMolAwareCallback] 🔗 hardlink {dst} -> {src}")
    except Exception:
        import shutil
        shutil.copy2(src, dst)
        print(f"[SaveMolAwareCallback] 📄 copied {dst} from {src}")

def _cleanup_old_weights(ckpt_dir: str):
    """清理冗余旧文件"""
    ckpt = Path(ckpt_dir)
    extras_dir = ckpt / "extras"
    llm_dir = ckpt / "llm"

    keep_names = {
        "trainer_state.json", "optimizer.pt", "scheduler.pt",
        "training_args.bin", "config.json", "tokenizer.json",
        "tokenizer.model", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "molaware_metadata.json",
        "model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json",
    }

    patterns = ["*.bin", "*.pt", "*.pth", "*.safetensors"]
    removed = 0
    for pat in patterns:
        for fp in ckpt.glob(pat):
            name = fp.name
            if name in keep_names:
                continue
            if llm_dir in fp.parents or extras_dir in fp.parents:
                continue
            if fp.is_file():
                try:
                    fp.unlink()
                    removed += 1
                except Exception as e:
                    print(f"[SaveMolAwareCallback] WARN: remove {fp} failed: {e}")
    if removed:
        print(f"[SaveMolAwareCallback] 🗑 Cleaned {removed} stale file(s)")

def _cleanup_old_checkpoints(output_dir: str, keep_last_n: int = 3):
    """清理旧的checkpoint目录，只保留最后N个"""
    if not os.path.isdir(output_dir):
        return
    
    # 找到所有checkpoint目录
    checkpoints = []
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, item)):
            try:
                # 提取step数字
                step = int(item.split("-")[1])
                checkpoints.append((step, os.path.join(output_dir, item)))
            except (ValueError, IndexError):
                continue
    
    if len(checkpoints) <= keep_last_n:
        return
    
    # 按step排序，保留最后N个
    checkpoints.sort(key=lambda x: x[0])
    to_remove = checkpoints[:-keep_last_n]
    
    removed = 0
    for step, ckpt_path in to_remove:
        try:
            import shutil
            shutil.rmtree(ckpt_path)
            removed += 1
            print(f"[SaveMolAwareCallback] 🗑 Removed old checkpoint: {os.path.basename(ckpt_path)} (step {step})")
        except Exception as e:
            print(f"[SaveMolAwareCallback] WARN: failed to remove {ckpt_path}: {e}")
    
    if removed > 0:
        print(f"[SaveMolAwareCallback] ✅ Cleaned {removed} old checkpoint(s), kept last {keep_last_n}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def break_gvp_shared_parameters(model: nn.Module):
    """
    克隆 gvp_encoder 的参数，打破潜在的共享存储，避免 safetensors 保存报错。
    """
    gvp = getattr(model, "gvp_encoder", None)
    if gvp is None:
        return
    try:
        sd = gvp.state_dict()
        cloned = {k: v.clone() for k, v in sd.items()}
        gvp.load_state_dict(cloned, strict=False)
        print("[Init] Cloned gvp_encoder state_dict to break shared storage.")
    except Exception as e:
        print(f"[Init] WARN: failed to clone gvp_encoder params: {e}")


def infer_response_template_from_chat_template(tokenizer) -> str:
    """
    根据 tokenizer 的 chat_template 自动推断 response_template：
    - 构造一个带占位符的对话，通过 apply_chat_template 得到完整 prompt
    - 取占位符之后的那一段作为 response_template
    """
    dummy_user = "<DUMMY_USER_CONTENT_FOR_TEMPLATE>"
    system_msg = "You are a helpful chemist."
    try:
        formatted = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": dummy_user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        pos = formatted.rfind(dummy_user)
        if pos != -1:
            # 从占位符结束位置到字符串末尾即为用于提示 assistant 回复的模板前缀
            tpl = formatted[pos + len(dummy_user) :]
            tpl = tpl.lstrip()  # 去掉前置空白
            # 避免返回空字符串
            if tpl:
                return tpl
    except Exception:
        pass

    # Fallback：如果无法从 chat_template 推断，则退回到简单规则
    vocab = tokenizer.get_vocab()
    if "<|start_header_id|>" in vocab and "<|end_header_id|>" in vocab:
        return "<|start_header_id|>assistant<|end_header_id|>"
    return "Assistant:"

# ======================== 回调 ========================
class BarrierCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        safe_barrier()
    def on_evaluate(self, args, state, control, **kwargs):
        safe_barrier()
    def on_train_begin(self, args, state, control, **kwargs):
        safe_barrier()
    def on_train_end(self, args, state, control, **kwargs):
        safe_barrier()
        
class SaveMolAwareCallback(TrainerCallback):
    """
    精简版：只保存 metadata，不保存模型权重
    - config 和 tokenizer 由 CopyConfigCallback 保存
    - metadata 由本 callback 保存
    """
    
    def _save_metadata(self, ckpt_dir: str, model):
        """保存 molaware_metadata.json"""
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            metadata = {
                "class": "MolAwareCausalLM",
                "version": 1,
                "mol_token": getattr(model, "mol_token", "<mol>"),
                "llm_dir": "llm/",
                "extras": {
                    "gvp_encoder": "extras/gvp_encoder.pt" if getattr(model, "gvp_encoder", None) else None,
                    "mol_adapter": "extras/mol_adapter.pt" if getattr(model, "mol_adapter", None) else None,
                    "diffusion_adapter": "extras/diffusion_adapter.pt" if getattr(model, "diffusion_adapter", None) else None,
                }
            }
            meta_path = os.path.join(ckpt_dir, "molaware_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"[Save] ✔ Metadata saved → {meta_path}")

    def on_save(self, args, state, control, **kwargs):
        """只保存 metadata，不保存模型权重"""
        model = kwargs.get("model")
        if model is None:
            # 尝试从 trainer 获取
            trainer = kwargs.get("trainer")
            if trainer is not None:
                model = getattr(trainer, "model", None)
        
        if model is None:
            return
        
        # checkpoint dir
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # 保存 metadata
        self._save_metadata(ckpt_dir, model)
        
        # 清理旧的checkpoint，只保留最新的N个（在rank0执行，避免多进程重复删除）
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() == 0:
                keep_last_n = getattr(args, "save_total_limit", 3)
                _cleanup_old_checkpoints(args.output_dir, keep_last_n=keep_last_n)
            dist.barrier()
        else:
            # 单卡训练时直接清理
            keep_last_n = getattr(args, "save_total_limit", 3)
            _cleanup_old_checkpoints(args.output_dir, keep_last_n=keep_last_n)
        

class SwanLabCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            if logs is not None:
                swanlab.log(logs, step=state.global_step)

class CopyConfigCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            model = kwargs["model"]
            tok = kwargs.get("tokenizer", None)
            if getattr(model, "config", None) is not None:
                model.config.save_pretrained(args.output_dir)
            if tok is not None:
                tok.save_pretrained(args.output_dir)


# ---------------- DataCollator with meta ----------------
class DataCollatorForCompletionOnlyLMWithMeta(DataCollatorForCompletionOnlyLM):
    """保留meta信息的DataCollator"""
    def __init__(self, response_template, tokenizer, instruction_template=None, mlm=False, ignore_index=-100, padding_free=False, **kwargs):
        super().__init__(
            response_template=response_template,
            instruction_template=instruction_template,
            tokenizer=tokenizer,
            mlm=mlm,
            ignore_index=ignore_index,
            padding_free=padding_free,
            **kwargs
        )
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 提取 meta 信息（在调用 torch_call 之前）
        meta_info = []
        for f in features:
            meta_info.append({
                "meta": f.get("meta", None),
                "dataset": f.get("dataset", None),
                "task_type": f.get("task_type", None),
                "smiles": f.get("smiles", None),
                "class_label": f.get("class_label", None),
                "all_targets": f.get("all_targets", None),
            })
        
        # 调用 torch_call 处理 tensor 相关的内容
        batch = self.torch_call(features)
        
        # 将 meta 信息添加回 batch（这些字段不会传递给 model，只用于 loss 计算）
        batch["meta"] = [m["meta"] for m in meta_info]
        batch["dataset"] = [m["dataset"] for m in meta_info]
        batch["task_type"] = [m["task_type"] for m in meta_info]
        batch["smiles"] = [m["smiles"] for m in meta_info]
        batch["class_label"] = [m["class_label"] for m in meta_info]
        batch["all_targets"] = [m["all_targets"] for m in meta_info]
        
        return batch
    
    def torch_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # --- 运行时自检：防止 tokenizer 在多进程中丢失 pad_token ---
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # -------------------------------------------------------

        cleaned_examples: List[Dict[str, Any]] = []

        for ex in features:
            ex = dict(ex)

            # 1. 剥离所有 Meta 信息
            for k in ["meta", "dataset", "task_type", "smiles", "class_label", "all_targets"]:
                ex.pop(k, None)

            # 2. 核心数据清洗
            if "input_ids" in ex:
                cleaned_ex = {}
                # ⚠️ 注意：这里移除了 "labels"，防止 tokenizer.pad 因为无法 pad labels 而报错
                valid_keys = ["input_ids", "attention_mask", "special_tokens_mask"] 
                
                for k in valid_keys:
                    if k in ex:
                        v = ex[k]
                        # 强制解开嵌套列表 [[1,2,3]] -> [1,2,3]
                        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                            v = v[0]
                        # 强制转 int
                        if k == "input_ids" and isinstance(v, list):
                            v = [int(x) for x in v]
                        
                        cleaned_ex[k] = v
                
                if "input_ids" in cleaned_ex:
                    cleaned_examples.append(cleaned_ex)
                continue

            # 处理纯文本的情况
            if "text" in ex:
                text = ex["text"]
                # 确保 text 是字符串，处理各种可能的格式
                if isinstance(text, list):
                    # 如果是列表，尝试提取第一个元素或连接
                    if len(text) > 0:
                        if isinstance(text[0], str):
                            text = text[0]  # 取第一个字符串元素
                        else:
                            text = " ".join(str(t) for t in text)  # 连接所有元素
                    else:
                        text = ""
                elif not isinstance(text, str):
                    text = str(text) if text is not None else ""
                
                if not text or not text.strip(): 
                    continue
                
                # Tokenize（不在这里做 padding，padding 会在父类的 torch_call 中完成）
                tokenized = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                    padding=False,  # 明确不在这里 padding
                    return_tensors=None,  # 返回列表而不是 tensor
                )
                
                # 确保 input_ids 是扁平列表
                input_ids = tokenized["input_ids"]
                if isinstance(input_ids, list):
                    # 如果已经是列表，确保是扁平列表
                    if len(input_ids) > 0 and isinstance(input_ids[0], list):
                        input_ids = input_ids[0]
                    # 确保所有元素都是整数
                    input_ids = [int(x) for x in input_ids if isinstance(x, (int, float, str))]
                else:
                    # 如果不是列表，转换为列表
                    input_ids = [int(x) for x in [input_ids] if isinstance(x, (int, float, str))]
                
                if not input_ids:
                    continue
                
                cleaned_ex = {"input_ids": input_ids}
                
                # 处理 attention_mask
                if "attention_mask" in tokenized:
                    attn = tokenized["attention_mask"]
                    if isinstance(attn, list):
                        if len(attn) > 0 and isinstance(attn[0], list):
                            attn = attn[0]
                        attn = [int(x) for x in attn if isinstance(x, (int, float, str))]
                    else:
                        attn = [int(attn)] if isinstance(attn, (int, float, str)) else []
                    
                    # 确保 attention_mask 长度与 input_ids 一致
                    if len(attn) != len(input_ids):
                        attn = [1] * len(input_ids)  # 如果长度不匹配，使用全1
                    cleaned_ex["attention_mask"] = attn
                else:
                    # 如果没有 attention_mask，创建一个全1的
                    cleaned_ex["attention_mask"] = [1] * len(input_ids)

                cleaned_examples.append(cleaned_ex)
                continue

        # 3. 调用父类进行 Batch Padding
        # 父类 DataCollatorForCompletionOnlyLM 会自动根据 response_template 生成 labels
        # 只要这里不传参差不齐的 labels，tokenizer.pad 就能正常工作
        batch = super().torch_call(cleaned_examples)

        return batch
    
# ---------------- Multi-task SFTTrainer ----------------
class MultiTaskSFTTrainer(SFTTrainer):
    """支持GNN任务的SFTTrainer"""
    QM9_TASKS = ["mu", "alpha", "homo", "lumo", "gap"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # 初始化用于记录 loss 的实例变量（累积值，用于计算平均值）
        self._loss_lm_sum = 0.0
        self._loss_count = 0

    def build_qm9_targets(self, all_targets_list: List[Optional[Dict[str, float]]], device: torch.device) -> torch.Tensor:
        rows = []
        for d in all_targets_list:
            if d is None:
                rows.append([0.0] * len(self.QM9_TASKS))
            else:
                rows.append([float(d.get(k, 0.0)) for k in self.QM9_TASKS])
        return torch.tensor(rows, dtype=torch.float32, device=device)
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        重写 log 方法，添加自定义的 GNN loss 值到日志中
        这样 SwanLab 就能记录这些值了
        """
        # 在调用父类 log 之前，添加自定义的 loss 值（使用累积的平均值）
        if hasattr(self, '_loss_count') and self._loss_count > 0:
            logs['train/loss_lm'] = self._loss_lm_sum / self._loss_count
            
            # 重置累积值，准备下一个 logging 周期
            self._loss_lm_sum = 0.0
            self._loss_count = 0
        
        # 调用父类的 log 方法（这会触发 SwanLab 记录）
        # 传递 start_time 参数以匹配父类签名
        super().log(logs, start_time=start_time)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Loss计算逻辑：
        L_total = L_lm
        
        注意：GVP和diffusion不需要单独的loss，只保留LM loss
        """
        # 移除不需要的字段（保留用于数据传递，但不用于loss计算）
        inputs.pop("dataset", None)
        inputs.pop("task_type", None)
        inputs.pop("smiles", None)
        inputs.pop("class_label", None)
        inputs.pop("all_targets", None)
        inputs.pop("meta", None)

        # 只计算语言模型的SFT loss
        loss_lm, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # 检查 loss_lm 是否为 None
        if loss_lm is None:
            raise ValueError("LM loss is None. This may indicate an issue with the model forward pass or loss computation.")
        
        # 总loss就是LM loss（GVP和diffusion不需要单独的loss）
        loss = loss_lm
        
        # 累积 loss 值，用于在 logging step 时计算平均值
        loss_lm_val = loss_lm.detach().item() if isinstance(loss_lm, torch.Tensor) else float(loss_lm)
        self._loss_lm_sum += loss_lm_val
        self._loss_count += 1

        if return_outputs:
            if hasattr(outputs, "loss"):
                outputs.loss = loss
            outputs.loss_lm = loss_lm
            return loss, outputs
        return loss

# ---------------- Main ----------------
def main_worker(world_size, cfg):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    print(f"[rank {local_rank}] using {device}")

    set_seed(cfg["seed"] + local_rank)

    # 初始化tokenizer和LLM
    llm_name = cfg["paths"]["llm_name_or_path"]
    mol_token = cfg["tokens"]["mol_token"]
    
    print(f"[{local_rank}] Initializing tokenizer...")
    tokenizer = init_tokenizer(llm_name, mol_token)
    # ================= 修复开始 =================
    # 1. 强制设置 pad_token。Llama 3 默认没有 pad_token，这是报错的根源。
    if tokenizer.pad_token is None:
        print(f"[{local_rank}] ⚠️ Tokenizer.pad_token is None, setting to eos_token_id: {tokenizer.eos_token_id}")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 2. 确保 padding_side 为 right (SFTTrainer 需要)
    tokenizer.padding_side = "right"
    
    # 3. 设置最大长度
    tokenizer.model_max_length = int(cfg["train"]["max_seq_length"])
    # ================= 修复结束 =================

    print(f"[{local_rank}] Initializing LLM...")
    llm = init_llm(llm_name, tokenizer, cfg["train"]["bf16"], device)

    # 初始化模型
    print(f"[{local_rank}] Initializing model...")
    model = init_model(cfg, tokenizer, llm, device)
    # 打破 gvp_encoder 共享参数，避免 safetensors 保存时因 shared tensors 报错
    break_gvp_shared_parameters(model)
    
    # 注意：LDMol组件可以在推理时使用，但训练时不需要单独的loss
    # LDMol直接使用LLM的embedding，不需要adapter
    
    # ✅ 必须禁用缓存，否则梯度检查点无效（激活显存暴涨 3–4x）
    llm.config.use_cache = False
    model.config.use_cache = False
    
    # ✅ 强制启用 gradient checkpointing（同时对包装模型与底层 LLM 开启）
    if cfg["train"].get("gradient_checkpointing", False):
        # 先确保底层 llm 关闭缓存并开启 GC
        if hasattr(llm, "config"):
            llm.config.use_cache = False
        if hasattr(llm, "gradient_checkpointing_enable"):
            llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # 再对包装后的整体模型开启 GC
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Sanity check
        gc_enabled = getattr(model, "is_gradient_checkpointing", False)
        if hasattr(model, "module") and hasattr(model.module, "is_gradient_checkpointing"):
            gc_enabled = model.module.is_gradient_checkpointing
        print(f"[{local_rank}] GC enabled: {gc_enabled}")
        if not gc_enabled:
            print(f"[{local_rank}] ⚠️  WARNING: Gradient checkpointing may not be enabled properly!")
    else:
        print(f"[{local_rank}] ⚠️  Gradient checkpointing is disabled in config")

    # 初始化离线token分类器（如果需要）
    use_offline_spans = cfg.get("train", {}).get("use_offline_spans", False)
    offline_token_head = None
    if use_offline_spans:
        mlp_token_classifier_path = cfg["paths"].get("mlp_token_classifier_path")
        offline_token_head = init_offline_token_classifier(llm, mlp_token_classifier_path, device)
        if offline_token_head is None:
            print(f"[{local_rank}] ⚠️ use_offline_spans=True but token classifier not loaded, will use <mol> tags only")

    # 参数统计
    if local_rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[{local_rank}] Trainable params: {trainable_params} / Total: {total_params}")

    # 加载数据
    print(f"[{local_rank}] Loading training data...")
    train_dataset, eval_dataset = load_training_data(
        cfg, tokenizer, llm, offline_token_head, local_rank
    )
    
    # 打印train_dataset的第一条数据（在train_sft.py中）
    if local_rank == 0 and len(train_dataset) > 0:
        print("\n" + "="*80)
        print("📋 First sample from train_dataset (in train_sft.py, after load_training_data):")
        print("="*80)
        first_sample = train_dataset[0]
        print(f"Type: {type(first_sample)}")
        print(f"Content: {first_sample}")
        if isinstance(first_sample, dict):
            print(f"Keys: {list(first_sample.keys())}")
            for key, value in first_sample.items():
                if key == "text":
                    print(f"  {key}: type={type(value)}, length={len(str(value))}, preview={str(value)[:200]}...")
                else:
                    print(f"  {key}: type={type(value)}, value={str(value)[:200]}...")
        print("="*80 + "\n")

    # 计算QM9统计信息（如果需要）
    qm9_means, qm9_stds = None, None
    use_gnn_tasks = cfg.get("train", {}).get("use_gnn_tasks", False)
    if use_gnn_tasks:
        # 尝试从文件加载
        qm9_stats_file = cfg.get("data", {}).get("qm9_stats_file")
        if qm9_stats_file and os.path.exists(qm9_stats_file):
            with open(qm9_stats_file, 'r') as f:
                stats = json.load(f)
                qm9_means = stats.get("means")
                qm9_stds = stats.get("stds")
        else:
            # 从数据集计算
            qm9_means, qm9_stds = compute_qm9_stats_from_dataset(train_dataset)
        
        if local_rank == 0:
            if qm9_means is None:
                print(f"[{local_rank}] ⚠️ QM9 stats not found")
            else:
                print(f"[{local_rank}] ✅ QM9 stats: means={qm9_means}, stds={qm9_stds}")

    # DataCollator
    # 优先根据 tokenizer 的 chat_template 自动推断 response_template；
    # 如果失败，再根据 vocab 中的特殊 token 退回到简单规则。
    response_template = infer_response_template_from_chat_template(tokenizer)
    
    # ✅ 统一使用自定义的 Collator。
    # 即使 use_gnn_tasks=False，我们也需要它来清洗 'text' 列，防止报错。
    # 它的 torch_call 方法里的清洗逻辑是通用的。
    data_collator = DataCollatorForCompletionOnlyLMWithMeta(
        response_template=response_template, 
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # (原来的 if/else 逻辑删除，只保留上面这一段)

    # TrainingArguments
    # 检查是否使用 DeepSpeed
    deepspeed_config_path = cfg["paths"].get("deepspeed_config")
    if deepspeed_config_path:
        # 如果是相对路径，转换为绝对路径（相对于代码目录）
        if not os.path.isabs(deepspeed_config_path):
            # 获取代码目录（train_sft.py所在目录）
            code_dir = os.path.dirname(os.path.abspath(__file__))
            deepspeed_config_path = os.path.join(code_dir, deepspeed_config_path)
            deepspeed_config_path = os.path.abspath(deepspeed_config_path)
        use_deepspeed = os.path.exists(deepspeed_config_path)
        if use_deepspeed and local_rank == 0:
            print(f"🚀 DeepSpeed config found: {deepspeed_config_path}")
    else:
        use_deepspeed = False
    
    # ================= 修复 DeepSpeed ZeRO-3 与 frozen 参数的兼容性问题 =================
    # DeepSpeed 的 count_used_parameters_in_backward 会遍历所有参数，包括 frozen 参数
    # 如果 frozen 参数的 grad_fn 是 None，访问 .next_functions 会报错
    # 解决方案：monkey-patch PyTorch 的 _get_grad_fn_or_grad_acc 函数，跳过 frozen 参数
    if use_deepspeed:
        try:
            # 使用已经导入的 torch 模块
            from torch.autograd.graph import _get_grad_fn_or_grad_acc as original_get_grad_fn
            
            # 保存原始函数
            _original_get_grad_fn_ds = original_get_grad_fn
            
            def safe_get_grad_fn_or_grad_acc(param):
                """安全版本的 _get_grad_fn_or_grad_acc，跳过 frozen 参数"""
                if not getattr(param, "requires_grad", False):
                    # 如果是 frozen 参数，返回 None，避免 DeepSpeed 尝试访问 grad_fn.next_functions
                    return None
                try:
                    return _original_get_grad_fn_ds(param)
                except (AttributeError, TypeError) as e:
                    # 如果访问 grad_fn.next_functions 失败，返回 None
                    if "NoneType" in str(e) or "next_functions" in str(e):
                        return None
                    raise
            
            # Monkey-patch PyTorch 的 _get_grad_fn_or_grad_acc
            # 使用已经导入的 torch 模块，避免作用域问题
            torch.autograd.graph._get_grad_fn_or_grad_acc = safe_get_grad_fn_or_grad_acc
            
            if local_rank == 0:
                print("[Fix] Patched PyTorch's _get_grad_fn_or_grad_acc to skip frozen parameters for DeepSpeed ZeRO-3 compatibility")
        except Exception as e:
            if local_rank == 0:
                print(f"[Fix] Warning: Failed to patch PyTorch for DeepSpeed frozen params compatibility: {e}")
    # ================= 修复结束 =================
    
    if use_deepspeed:
        if local_rank == 0:
            print(f"🚀 Using DeepSpeed config: {deepspeed_config_path}")
        optim_name = "adamw_torch"
    else:
        optim_name = "paged_adamw_8bit"
    
    args = TrainingArguments(
        output_dir=cfg["paths"]["output_dir"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["train"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        learning_rate=float(cfg["train"]["learning_rate"]),
        num_train_epochs=cfg["train"]["num_train_epochs"],
        logging_steps=cfg["train"]["logging_steps"],
        save_strategy="steps",
        save_steps=cfg["train"]["save_steps"],
        eval_strategy="steps",
        eval_steps=cfg["train"]["eval_steps"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        bf16=cfg["train"]["bf16"],
        gradient_checkpointing=cfg["train"]["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
        report_to="none",
        optim=optim_name,
        dataloader_pin_memory=cfg["train"].get("dataloader_pin_memory", True),
        dataloader_num_workers=cfg["train"].get("dataloader_num_workers", 0),
        max_grad_norm=cfg["train"].get("max_grad_norm", 1.0),
        weight_decay=cfg["train"].get("weight_decay", 0.01),
        deepspeed=deepspeed_config_path if use_deepspeed else None,
        save_safetensors=False,
        save_total_limit=cfg["train"].get("save_total_limit", 2),  # 只保留最新的3个checkpoint
        disable_tqdm=False,  # 启用进度条显示（在 rank 0 上会显示）
    )

    # Trainer
    callbacks = [
        SaveMolAwareCallback(),
        SwanLabCallback(),
        CopyConfigCallback(),
        BarrierCallback()
    ]

    # 准备 SFTTrainer 的参数
    # 注意：当前版本的 trl 可能不支持 SFTConfig 参数，或者参数名不是 sft_config
    # 根据错误信息，SFTTrainer 不接受 sft_config 参数
    # 所以我们使用旧的方式，虽然会显示警告但不影响功能
    sft_kwargs = {
        "dataset_text_field": "text",
        "max_seq_length": int(cfg["train"]["max_seq_length"]),
        "packing": cfg["train"]["packing"],
    }
    # 注意：这些参数在新版本中已被弃用，会显示警告，但不影响功能
    # 如果未来 trl 库更新支持 SFTConfig，可以在这里添加相应的逻辑

    # 创建 Trainer（统一使用 MultiTaskSFTTrainer，只计算LM loss）
    # 注意：GVP和diffusion不需要单独的loss，只保留LM loss
    trainer = MultiTaskSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        **sft_kwargs,  # 传递 SFT 相关参数
    )

    # 训练（不使用checkpoint恢复训练状态）
    # 注意：模型权重已经在model_init.py中根据config加载了
    trainer.train(resume_from_checkpoint=None)


def main(cfg_path="configs/config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    main_worker(world_size, cfg)


if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    main(cfg_path)

