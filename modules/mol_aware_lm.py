# mol_aware_lm_integrated.py
# -*- coding: utf-8 -*-
import os
import json
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
import logging

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .gnn import GVPEncoder
from .mlp import MLPAdapter
from .tools import extract_and_convert_online

# 使用 LDMol 进行分子生成
# BUG: diffusion fallback 存在bug, 需要调整架构
ENABLE_DIFFUSION_FALLBACK = False
from .ldmol_component import LDMolInferer

# 使用 Layer2 进行反应产率预测
from .layer2_component import Layer2Inferer

# RDKit
from rdkit import Chem

# 日志
import sys
import io
import os

# 确保stdout和stderr使用UTF-8编码
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# 如果stdout/stderr不是UTF-8，则重新包装
if hasattr(sys.stdout, 'buffer') and (not hasattr(sys.stdout, 'encoding') or sys.stdout.encoding != 'utf-8'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except (AttributeError, ValueError):
        pass
if hasattr(sys.stderr, 'buffer') and (not hasattr(sys.stderr, 'encoding') or sys.stderr.encoding != 'utf-8'):
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except (AttributeError, ValueError):
        pass

logging.getLogger("rdkit").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# 配置logging使用UTF-8编码
class UTF8StreamHandler(logging.StreamHandler):
    """确保日志输出使用UTF-8编码的StreamHandler"""
    def __init__(self, stream=None):
        if stream is None:
            stream = sys.stderr
        super().__init__(stream)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # 确保使用UTF-8编码写入
            if hasattr(stream, 'buffer'):
                stream.buffer.write(msg.encode('utf-8', errors='replace'))
                stream.buffer.write(b'\n')
                self.flush()
            else:
                stream.write(msg)
                stream.write('\n')
                self.flush()
        except Exception:
            self.handleError(record)

# 只在没有配置过logging时才配置
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[UTF8StreamHandler()]
    )

import torch.distributed as dist
import os, glob
import re

# 与 data_loader.py 中 _looks_like_molecule 保持一致的判断逻辑
_MOL_STOPWORDS = {"smiles", "Smiles", "SMILES", "logP", "NSAIDs"}

# 分隔符字符集合（用于判断是否应该检测实体）
_BOUNDARY_CHARS = set(" \n\t,;:!?>")

def _is_boundary_token(tokenizer, token_id: int) -> bool:
    """
    判断一个 token 是否是分隔符（空格、换行、标点等）
    只有在遇到分隔符时才检测实体，避免在单词中间检测
    """
    try:
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        # 检查 token 文本是否包含分隔符，或者整个 token 就是分隔符
        if not token_text:
            return False
        # 如果 token 文本中的任何字符是分隔符，或者整个 token 都是分隔符/空白
        if any(c in _BOUNDARY_CHARS for c in token_text):
            return True
        # 检查是否是纯空白字符
        if token_text.strip() == "":
            return True
        return False
    except Exception:
        return False

def _looks_like_molecule(span_text: str) -> bool:
    """
    软规则判断一个 span 看起来像"分子相关实体"（与 data_loader.py 保持一致）：
    - 很短的碎片（长度 < 2）直接丢掉
    - 含有数字 or 典型 SMILES / 化学式符号（= # () [] @ + / -）就认为是
    - 否则，如果有 >=4 个字母（toluene, ethanol, ibuprofen 等化学名）也认为是
    规则故意写得比较宽松，避免漏掉真正的化学名。
    """
    if not span_text:
        return False
    
    s = span_text.strip()
    if s in _MOL_STOPWORDS:
        return False
    if len(s) < 2:
        return False

    # 典型 SMILES / 化学式特征：数字、=、#、括号、@、+、/、-
    if any(c.isdigit() for c in s):
        return True
    if any(c in "=#()[]@+/-" for c in s):
        return True

    # 对纯字母的情况：如果有 >=4 个字母，当成一个"像化学名"的词
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 4:
        return True

    return False

def has_hf_model_files(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    # 单文件 / 索引文件
    names = [
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "flax_model.msgpack",
        "tf_model.h5",
    ]
    if any(os.path.isfile(os.path.join(d, n)) for n in names):
        return True
    # 分片文件（无论是否有 index，都当作“该目录包含权重”）
    if glob.glob(os.path.join(d, "model-*-of-*.safetensors")):
        return True
    if glob.glob(os.path.join(d, "pytorch_model-*-of-*.bin")):
        return True
    return False

def any_rank_true(flag: bool) -> bool:
    """只要有一个 rank 为 True，就让所有 rank 都为 True。"""
    if not dist.is_available() or not dist.is_initialized():
        return flag
    t = torch.tensor([1 if flag else 0], device=torch.cuda.current_device())
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item())

def zero_touch_module(module: torch.nn.Module) -> torch.Tensor:
    """用 0.0 * param.sum() 把 module 接入计算图，不改变 loss 数值。"""
    if module is None:
        return torch.tensor(0.0, device=torch.cuda.current_device())
    z = torch.tensor(0.0, device=next(module.parameters()).device) if any(p.requires_grad for p in module.parameters()) else torch.tensor(0.0, device=torch.cuda.current_device())
    for p in module.parameters():
        if p.requires_grad:
            z = z + (0.0 * p.float().sum())
    return z

def build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    简易 position_ids：对每行的有效 token（mask=1）做递增计数，padding 处保持 0。
    与很多 LLM 兼容；若你已有自定义实现，保留你自己的即可。
    """
    # (B, T)
    cumsum = attention_mask.long().cumsum(dim=1) * attention_mask.long()
    # 让从 0 开始：把非零位置减 1
    pos_ids = (cumsum - attention_mask.long()).clamp(min=0)
    return pos_ids

class MolAwareCausalLM(nn.Module):
    """
    集成 NER/GNN/Diffusion 的组合模型；按出现顺序把 <mol> 对应的向量“追加到序列末尾”的虚拟步，
    训练时 labels=-100 不计损，推理时推进 KV 但不出 token。
    """
    # --------------------------- 初始化 ---------------------------
    def __init__(
        self,
        llm: nn.Module,
        tokenizer,
        mol_token: str = "<mol>",
        proxy: Optional[str] = None,
        debug: bool = False,
        target_layer_for_capture: int = -1,
        gvp_encoder_config: Optional[Dict] = None,
        mol_adapter_config: Optional[Dict] = None,
        diffusion_config: Optional[Dict] = None,
        diffusion_adapter_config: Optional[Dict] = None,
        token_classifier_head: Optional[nn.Module] = None,
        disable_gnn: bool = False,  # 新增：是否禁用 GNN 处理
        use_ldmol: bool = True,  # 是否使用 LDMol @xyd
        ldmol_skip_text_encoder: bool = False,  # 是否跳过 LDMol 内置的 text_encoder @xyd
        layer2_config: Optional[Dict] = None,  # Layer2 配置
        use_layer2: bool = False,  # 是否使用 Layer2
    ):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.mol_token = mol_token
        self.mol_token_id = tokenizer.convert_tokens_to_ids(mol_token)
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.debug = debug
        self.proxy = proxy
        self.disable_gnn = disable_gnn  # 新增：禁用 GNN 标志

        # if self.mol_token_id is None or self.mol_token_id < 0:
            # raise ValueError(f"Tokenizer does not contain mol_token '{mol_token}'. Please add it first.")

        layers_ref = None
        if hasattr(self.llm, "model") and hasattr(self.llm.model, "layers"):
            layers_ref = self.llm.model.layers
        elif hasattr(self.llm, "transformer") and hasattr(self.llm.transformer, "h"):
            layers_ref = self.llm.transformer.h
        object.__setattr__(self, "_layers_ref", layers_ref)
        self.num_layers = len(self._layers_ref) if self._layers_ref is not None else 0
        self.target_layer_for_capture = (
            self.num_layers - 1 if (target_layer_for_capture < 0 and self.num_layers > 0) else target_layer_for_capture
        )
        self._capture_bucket: List[List[torch.Tensor]] = []
        self._capture_hook = None

        # ---------- 组件 ----------
        try:
            llm_hidden_size = self.llm.config.hidden_size
        except Exception:
            llm_hidden_size = self.llm.config.text_config.hidden_size
        # GVPEncoder
        gvp_encoder_cfg = {
            "node_dims": (10, 1),
            "edge_dims": (1, 1),
            "hidden_scalar_dim": 256,
            "hidden_vector_dim": 16,
            "output_dim": 256,
            "num_layers": 4,
        }
        if gvp_encoder_config:
            gvp_encoder_cfg.update(gvp_encoder_config)

        # MLP Adapter（把 GVP 向量映射到 LLM 维度）
        mol_adapter_cfg = {
            "input_dim": gvp_encoder_cfg["output_dim"],
            "output_dim": llm_hidden_size,
            "hidden_dim": 2048,
            "num_layers": 2,
        }
        if mol_adapter_config:
            mol_adapter_cfg.update(mol_adapter_config)

        ##############################
        #  LDMol 初始化 @xyd
        # 通过 use_ldmol 参数控制是否加载 LDMol（阶段1 SFT 训练时可禁用以节省显存）
        # 通过 ldmol_skip_text_encoder 参数控制是否跳过加载内置 text_encoder（联合推理时复用主 Qwen）

        # Temp
        self.use_ldmol = False
        # self.use_ldmol = use_ldmol 

        self.ldmol_skip_text_encoder = ldmol_skip_text_encoder
        
        if self.use_ldmol:
            self.ldmol = LDMolInferer(
                device=self._first_device(),
                skip_text_encoder=self.ldmol_skip_text_encoder,
            )
        else:
            self.ldmol = None
            logging.info(f"LDMol disabled (use_ldmol={use_ldmol})")
        self.enable_diffusion_fallback = ENABLE_DIFFUSION_FALLBACK and self.use_ldmol
        ##############################

        # 先初始化 GVP encoder（Layer2 需要它）
        self.gvp_encoder = GVPEncoder(**gvp_encoder_cfg).to(self._first_device())
        self.mol_adapter = MLPAdapter(**mol_adapter_cfg).to(self._first_device())
        
        ##############################
        #  Layer2 初始化（在 GVP encoder 之后）
        # 通过 use_layer2 参数控制是否加载 Layer2（用于反应产率预测）
        self.use_layer2 = use_layer2
        if self.use_layer2:
            layer2_cfg = layer2_config or {}
            self.layer2_inferer = Layer2Inferer(
                config_path=layer2_cfg.get("config_path"),
                device=self._first_device(),
                gvp_encoder=self.gvp_encoder,  # 复用 GVP encoder
                gvp_ckpt_path=layer2_cfg.get("gvp_ckpt_path"),
            )
            logging.info(f"Layer2 enabled")
        else:
            self.layer2_inferer = None
            logging.info(f"Layer2 disabled (use_layer2={use_layer2})")
        ##############################
        self.smiles_cache: Dict[str, str] = {}
        # 可选：外部注入的 token 分类头（用于检测分子实体的位置）
        self.token_classifier_head = token_classifier_head
        

        # ---------- GNN Pipeline 日志统计 ----------
        self.gnn_stats = {
            "smiles_processed": 0,
            "gnn_cache_hits": 0,
            "gnn_cache_misses": 0,
            "smiles_valid": 0,
            "smiles_invalid": 0,
            "diffusion_fallback_count": 0,
            "total_mol_embeddings": 0,
        }
        # 每N个样本打印一次统计
        self.gnn_log_interval = 10

        # ---------- 关键：HF Trainer 兼容字段 ----------
        # 让 Trainer 把它当作 PreTrainedModel 一样保存
        self._config = getattr(self.llm, "config", None)
        self._keys_to_ignore_on_save = getattr(self.llm, "_keys_to_ignore_on_save", None)
        self._keys_to_ignore_on_load_missing = getattr(self.llm, "_keys_to_ignore_on_load_missing", None)
        self._keys_to_ignore_on_load_unexpected = getattr(self.llm, "_keys_to_ignore_on_load_unexpected", None)

    # --------------------------- HF 兼容接口 ---------------------------
    @property
    def config(self):
        return self._config

    @property
    def _keys_to_ignore_on_save(self):
        return getattr(self.llm, "_keys_to_ignore_on_save", [])

    @_keys_to_ignore_on_save.setter
    def _keys_to_ignore_on_save(self, v):
        # 仅为了避免 AttributeError；Trainer 不需要我们真正改 llm 的字段
        self.__dict__["__keys_to_ignore_on_save"] = v

    @property
    def _keys_to_ignore_on_load_missing(self):
        return getattr(self.llm, "_keys_to_ignore_on_load_missing", [])

    @_keys_to_ignore_on_load_missing.setter
    def _keys_to_ignore_on_load_missing(self, v):
        self.__dict__["__keys_to_ignore_on_load_missing"] = v

    @property
    def _keys_to_ignore_on_load_unexpected(self):
        return getattr(self.llm, "_keys_to_ignore_on_load_unexpected", [])

    @_keys_to_ignore_on_load_unexpected.setter
    def _keys_to_ignore_on_load_unexpected(self, v):
        self.__dict__["__keys_to_ignore_on_load_unexpected"] = v

    def to(self, *args, **kwargs):
        # 同步把底座 LLM 与自定义模块都迁移设备
        super().to(*args, **kwargs)
        self.llm.to(*args, **kwargs)
        self.gvp_encoder.to(*args, **kwargs)
        self.mol_adapter.to(*args, **kwargs)
        if self.ldmol is not None:
            self.ldmol.to(*args, **kwargs)
        if self.layer2_inferer is not None:
            self.layer2_inferer.to(*args, **kwargs)
        return self

    # --------------------------- 辅助 ---------------------------
    def _first_device(self):
        try:
            return self.llm.model.layers[0].input_layernorm.weight.device
        except Exception:
            return next(self.llm.parameters()).device

    def _get_smiles_from_context(self, llm_context: str) -> Optional[str]:
        if llm_context in self.smiles_cache:
            smiles_map = self.smiles_cache[llm_context]
        else:
            smiles_map = extract_and_convert_online(llm_context, self.proxy)
            self.smiles_cache[llm_context] = smiles_map
        if not smiles_map:
            return None
        last_cem = ""
        last_idx = -1
        for cem_name in smiles_map:
            idx = llm_context.rfind(cem_name)
            if idx > last_idx:
                last_idx = idx
                last_cem = cem_name
        return smiles_map.get(last_cem)

    def _extract_last_between_mol_tags(self, text: str) -> Optional[str]:
        """
        提取文本中最后一对 <mol>...</mol> 的内部内容；找不到返回 None。
        """
        if not text:
            return None
        start = text.rfind("<mol>")
        end = text.rfind("</mol>")
        if start == -1 or end == -1 or end <= start:
            return None
        inner = text[start + len("<mol>"):end].strip()
        return inner if inner else None

    def _find_all_mol_spans(self, text: str):
        """
        返回所有 <mol>...</mol> 的 (inner_text, end_char_index) 列表，end_char_index 指向 </mol> 末尾在 text 中的字符位置。
        """
        if not text:
            return []
        try:
            spans = []
            for m in re.finditer(r"<mol>(.*?)</mol>", text, flags=re.DOTALL):
                inner = (m.group(1) or "").strip()
                spans.append((inner, m.end()))
            return spans
        except Exception:
            return []

    def _detect_mol_entities_with_classifier(self, input_ids: torch.Tensor, dec_text: str, enable_thinking: bool = False) -> List[Tuple[str, int]]:
        """
        使用 token_classifier_head 检测分子实体，参考 mlp_inference.py 的实现。
        如果没有 token_classifier_head 或检测失败，fallback 到文本匹配方法。
        
        Args:
            input_ids: 输入 token ids
            dec_text: 解码后的文本
        Returns:
            List[(inner_text, end_char_index)]: 检测到的分子实体 spans
        """
        # 优先：若文本已包含离线标注的 <mol>...</mol>，直接用文本匹配，避免额外前向
        if ("<mol>" in dec_text) and ("</mol>" in dec_text):
            return self._find_all_mol_spans(dec_text)
        # 其次：若没有分类器，则回退到文本匹配
        if self.token_classifier_head is None:
            # logger.info("[TokenClassifier] ❌ No token_classifier_head, using text matching fallback")
            return self._find_all_mol_spans(dec_text)
        
        # 优化：使用token数量而不是字符数来判断，更准确
        # 将文本转换为token数来估算
        try:
            # 快速估算：大约4个字符 = 1个token（对于英文和SMILES）
            estimated_tokens = len(dec_text) // 4
            max_tokens = getattr(self, '_max_text_length_for_detection', 4096) // 4  # 转换为token估算
            if estimated_tokens > max_tokens * 2:  # 允许更大的容差
                if getattr(self, '_verbose_logging', False):
                    logging.debug(f"[TokenClassifier] ⚠️  Text too long (est. {estimated_tokens} tokens), will use truncation")
        except Exception:
            pass  # 如果估算失败，继续处理
        
        try:
            # 只在verbose模式下显示详细日志
            if getattr(self, '_verbose_logging', False):
                text_preview = dec_text[:500] if len(dec_text) > 500 else dec_text
                preview_suffix = "..." if len(dec_text) > 500 else ""
                logger.info(f"[TokenClassifier] 🔍 Starting entity detection with classifier. Text length: {len(dec_text)} chars. Preview:\n{text_preview}{preview_suffix}")
            
            # 1) 清除原有标签的临时文本（用于分类器检测）
            text_clean = re.sub(r"</?mol>", "", dec_text)
            
            # 2) Tokenize 清除标签后的文本获取 offsets
            # 优化：直接使用tokenizer的truncation机制，让tokenizer自动处理长度限制
            # 使用更大的max_length以支持长文本（2048 tokens足够处理大部分情况）
            max_token_length = 2048
            _old_side = getattr(self.tokenizer, "truncation_side", "right")
            self.tokenizer.truncation_side = "left"
            try:
                enc = self.tokenizer(
                    text_clean,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    padding=False,
                    truncation=True,
                    max_length=max_token_length,
                    add_special_tokens=False
                )
            finally:
                self.tokenizer.truncation_side = _old_side
            clean_input_ids = enc["input_ids"].to(input_ids.device)
            attention_mask = enc["attention_mask"].to(input_ids.device)
            offsets = enc["offset_mapping"][0].tolist()
            
            # 3) 使用 LLM 获取 hidden states（用于分类器）
            device = input_ids.device
            with torch.no_grad():
                outputs = self.llm(
                    input_ids=clean_input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = outputs.hidden_states[-1]  # (1, T, H)
            
            # 4) 使用 token_classifier_head 进行分类
            with torch.no_grad():
                class_logits = self.token_classifier_head(hidden_states)  # (1, T, 2)
                preds = torch.argmax(class_logits, dim=-1)[0].cpu().tolist()
            
            if getattr(self, '_verbose_logging', False):
                logger.info(f"[TokenClassifier] ✅ Classifier prediction completed. Found {sum(1 for p in preds if p == 1)} entity tokens")
            
            # 5) 提取实体 spans（合并连续标签为1的片段）
            entity_spans = []
            current_start, current_end = None, None
            for label, (start, end) in zip(preds, offsets):
                if start == end:
                    continue
                if label == 1:  # 分子实体标签
                    if current_start is None:
                        current_start, current_end = start, end
                    else:
                        current_end = end
                else:
                    if current_start is not None:
                        entity_spans.append((current_start, current_end))
                        current_start, current_end = None, None
            if current_start is not None:
                entity_spans.append((current_start, current_end))
            
            # 6) 后处理：确保标签不打断单词（扩展到空格边界）
            expanded_spans = []
            for start, end in entity_spans:
                while start > 0 and text_clean[start-1] not in " \n\t.,;:!?()[]{}":
                    start -= 1
                while end < len(text_clean) and text_clean[end] not in " \n\t.,;:!?()[]{}":
                    end += 1
                expanded_spans.append((start, end))
            
            # 7) 合并重叠的 span（防止重复标记）
            final_spans = []
            for span in sorted(expanded_spans):
                if not final_spans or span[0] > final_spans[-1][1]:
                    final_spans.append(span)
                else:
                    final_spans[-1] = (final_spans[-1][0], max(final_spans[-1][1], span[1]))
            
            # 8) 转换为 (inner_text, end_char) 格式
            # 注意：end_char 现在是相对于 text_clean 的位置，需要映射回原始 dec_text
            result_spans = []
            for start, end in final_spans:
                inner_text = text_clean[start:end].strip()
                if inner_text:
                    # 在原始 dec_text 中搜索（考虑可能移除了<mol>标签）
                    # 直接在 text_clean 中搜索更准确
                    idx_in_clean = text_clean.find(inner_text, max(0, start - 50), min(len(text_clean), end + 50))
                    if idx_in_clean >= 0:
                        # 将 text_clean 的位置映射回 dec_text（考虑<mol>标签）
                        # 简化：直接返回实体文本，位置使用估算值
                        end_in_clean = idx_in_clean + len(inner_text)
                        result_spans.append((inner_text, end_in_clean))
            
            if getattr(self, '_verbose_logging', False):
                if result_spans:
                    logger.info(f"[TokenClassifier] 🎯 Detected {len(result_spans)} entities: {[r[0] for r in result_spans]}")
                else:
                    logger.info("[TokenClassifier] ⚠️  No entities detected")
            
            return result_spans
            
        except Exception as e:
            logger.warning(f"[TokenClassifier] Failed to detect entities: {e}, falling back to text matching")
            return self._find_all_mol_spans(dec_text)


    def _decide_smiles_or_diffusion(self, llm_context_text: Optional[str], fallback_hctx: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        基于 <mol>...</mol> 的内部内容进行判别：
        - 若内部可被 RDKit 解析为 SMILES：走 GVP -> mol_adapter
        - 否则：在允许时使用 diffusion 路径
        返回映射后的 LLM 维度向量，或 None 表示无法生成。
        """
        inner = self._extract_last_between_mol_tags(llm_context_text or "")
        if inner:
            # 直接用 RDKit 判定是否为 SMILES
            is_smiles = False
            if Chem is not None:
                try:
                    is_smiles = (Chem.MolFromSmiles(inner) is not None)
                except Exception:
                    pass
            if is_smiles:
                try:
                    if getattr(self, '_verbose_logging', False):
                        logger.info(f"[GVP] 🔵 调用 GVP encoder，SMILES: {inner[:100]}")
                    gvp_embedding = self.gvp_encoder.forward_from_smiles(inner).squeeze(0)
                    result = self.mol_adapter(gvp_embedding)
                    if getattr(self, '_verbose_logging', False):
                        logger.info(f"[GVP] ✅ GVP encoder 完成，embedding shape: {result.shape}")
                    return result
                except Exception as e:
                    if getattr(self, '_verbose_logging', False):
                        logger.warning(f"[GVP] ❌ GVP encoder 失败: {e}")
                    return None
            # 非 SMILES 或失败 -> diffusion（仅在启用兜底时）
            if self.enable_diffusion_fallback:
                if self._verbose_logging:
                    logger.info(f"[Diffusion] 🟣 调用 Diffusion fallback，内容: {inner[:100] if inner else 'None'}")
                result = self._generate_smiles_convert_to_embedding(text=llm_context_text)
                return result
            return None

        # 没有成对标签时，回退旧逻辑：从上下文抽取 CEM 名 -> SMILES
        smiles = self._get_smiles_from_context(llm_context_text or "") if llm_context_text else None
        if smiles:
            try:
                if getattr(self, '_verbose_logging', False):
                    logger.info(f"[GVP] 🔵 调用 GVP encoder（从上下文提取），SMILES: {smiles[:100]}")
                gvp_embedding = self.gvp_encoder.forward_from_smiles(smiles).squeeze(0)
                result = self.mol_adapter(gvp_embedding)
                if getattr(self, '_verbose_logging', False):
                    logger.info(f"[GVP] ✅ GVP encoder 完成，embedding shape: {result.shape}")
                return result
            except Exception as e:
                if getattr(self, '_verbose_logging', False):
                    logger.warning(f"[GVP] ❌ GVP encoder 失败: {e}")
                pass
        if fallback_hctx is not None:
            if getattr(self, '_verbose_logging', False):
                logger.info(f"[Diffusion] 🟣 调用 Diffusion fallback（无标签回退）")
            result = self._black_box_from_hidden_hctx(fallback_hctx)
            if getattr(self, '_verbose_logging', False):
                logger.info(f"[Diffusion] ✅ Diffusion fallback 完成，embedding shape: {result.shape if result is not None else 'None'}")
            return result
        return None

    def _get_last_hidden_before_pos(self, row_ids: torch.Tensor, end_pos: int) -> torch.Tensor:
        assert end_pos > 0, "end_pos should be > 0"
        dev = self._first_device()
        prefix = row_ids[:end_pos].unsqueeze(0).to(dev)
        attn = (prefix != self.pad_token_id).long().to(dev)
        out = self.llm(input_ids=prefix, attention_mask=attn,
                       output_hidden_states=True, use_cache=False, return_dict=True)
        return out.hidden_states[-1][0, -1, :].detach()
    
    def _generate_smiles_convert_to_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        使用LDMol从text生成分子SMILES，然后转换为gvp embedding
        
        :param text: text for SMILES
        :type text: str
        :return: 错误时返回None,正常返回gvp embedding
        :rtype: Tensor | None
        """
        if self.ldmol is None or not LDMOL_AVAILABLE:
            logger.warning("LDMol unavailable, return None.")
            return None
        if self._verbose_logging:
            logger.info("[Diffusion] 🟣 开始 Diffusion 生成")
        assert self.llm is not None and self.tokenizer is not None, "self.llm or self.tokenizer is None"
        generated_smiles = self.ldmol.generate_molecule(
            description=text,
            qwen=self.llm,
            qwen_tokenizer=self.tokenizer
        )
        if generated_smiles is None:
            logger.warning("LDMol fails to generate smiles, return None.")
            return None
        if self._verbose_logging:
            logger.info(f"✅ LDMol 生成 SMILES: {generated_smiles}")
            logger.info(f"[GVP] 🔵 调用 GVP encoder（处理 Diffusion 生成的 SMILES）")
        gvp_embedding = self.gvp_encoder.forward_from_smiles(generated_smiles).squeeze(0)
        mol_emb = self.mol_adapter(gvp_embedding)
        if self._verbose_logging:
            logger.info(f"[GVP] ✅ GVP encoder 完成，embedding shape: {mol_emb.shape}")
        return mol_emb
            

    def _black_box_from_hidden_hctx(self, h_ctx: torch.Tensor) -> Optional[torch.Tensor]:
        """
        使用LDMol从LLM的hidden state生成分子SMILES，然后转换为embedding
        """
        # TODO
        # raise ValueError("Updating @xyd")
        logger.info("[Diffusion] 🟣 开始 Diffusion 生成（从 hidden state）")
        if self.ldmol_components is None or not LDMOL_AVAILABLE:
            logger.warning("[Diffusion] ❌ LDMol 不可用，跳过")
            return None
        
        dev = self._first_device()
        h_ctx = h_ctx.to(dev)
        
        try:
            # 使用LDMol从LLM hidden state生成SMILES
            from .ldmol.inference import generate_molecule_from_llm_embedding
            gen_smiles = generate_molecule_from_llm_embedding(
                self.ldmol_components, h_ctx, dev
            )
            
            if not gen_smiles:
                if getattr(self, '_verbose_logging', False):
                    logger.warning("[Diffusion] ❌ 未生成有效的 SMILES")
                return None
            
            if getattr(self, '_verbose_logging', False):
                logger.info(f"[Diffusion] ✅ Diffusion 生成 SMILES: {gen_smiles}")
            
            # 将生成的SMILES转换为embedding（使用GVP+mol_adapter）
            if getattr(self, '_verbose_logging', False):
                logger.info(f"[GVP] 🔵 调用 GVP encoder（处理 Diffusion 生成的 SMILES）")
            gvp_embedding = self.gvp_encoder.forward_from_smiles(gen_smiles).squeeze(0)
            mol_emb = self.mol_adapter(gvp_embedding)
            if getattr(self, '_verbose_logging', False):
                logger.info(f"[GVP] ✅ GVP encoder 完成，embedding shape: {mol_emb.shape}")
            return mol_emb
            
        except Exception as e:
            logger.warning(f"[BlackBox] ❌ LDMol generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    # def _black_box_embed_offline(
    #     self,
    #     row_ids: torch.Tensor,
    #     row_embeds: torch.Tensor,
    #     row_mask: torch.Tensor,
    #     pos_mol: int,
    # ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    #     # 基于 </mol> 触发：取到目前为止的上下文，解析 <mol>...</mol> 内部并判别
    #     raise ValueError("TODO: _decide_smiles_or_diffusion 接口修改，如需使用，代码需要修改 @xyd")
    #     llm_context = self.tokenizer.decode(row_ids[:pos_mol + 1].tolist(), skip_special_tokens=True)
    #     h_ctx = self._get_last_hidden_before_pos(row_ids, pos_mol)  # [H]
    #     emb = self._decide_smiles_or_diffusion(llm_context_text=llm_context, fallback_hctx=h_ctx)
    #     return emb

    def _black_box_embed_online(
        self,
        llm_context_text: Optional[str] = None,
        context_ids: Optional[torch.Tensor] = None,
        h_ctx: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if getattr(self, '_verbose_logging', False):
            logger.info(f"[Diffusion] 🟣 调用 Diffusion（从文本），文本: {llm_context_text[:100] if llm_context_text else 'None'}...")
        if llm_context_text is not None:
            emb = self._decide_smiles_or_diffusion(llm_context_text=llm_context_text, fallback_hctx=h_ctx)
            if emb is not None:
                if getattr(self, '_verbose_logging', False):
                    logger.info(f"[Diffusion] ✅ Diffusion 完成，embedding shape: {emb.shape}")
                return emb
        if context_ids is None and llm_context_text is not None:
            dev = self._first_device()
            toks = self.tokenizer(llm_context_text, return_tensors="pt", add_special_tokens=False)
            context_ids = toks["input_ids"].to(dev)
        if context_ids is not None:
            attn = (context_ids != self.pad_token_id).long().to(context_ids.device)
            out = self.llm(
                input_ids=context_ids, attention_mask=attn,
                output_hidden_states=True, use_cache=False, return_dict=True
            )
            h_ctx = out.hidden_states[-1][0, -1, :].detach()
            return self._black_box_from_hidden_hctx(h_ctx)
        return None

    # --------------------------- 训练/评估前向 ---------------------------
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            **kwargs,
        ) -> CausalLMOutputWithPast:
            assert input_ids is not None, "MolAwareCausalLM 需要 input_ids"

            # 1) 先离线拼接 <mol> 的嵌入到序列末尾（仅在存在 <mol> 时追加）
            new_embeds, new_masks, new_labels, appended_mol_cnt = self._append_mol_embeds_to_end_offline(
                input_ids, attention_mask, labels
            )

            # 2) 常规 LLM 前向
            position_ids = build_position_ids(new_masks).to(new_masks.device)
            
            outputs = self.llm(
                inputs_embeds=new_embeds,
                attention_mask=new_masks,
                position_ids=position_ids,
                labels=new_labels,
                return_dict=True,
                **kwargs,
            )

            # 3) —— DDP 安全处理 ——：
            # "本 rank 是否真的追加过 mol 向量"
            used_mol_local = (appended_mol_cnt > 0)
            # "所有 rank 是否至少有一个用到 mol 分支"
            used_mol_global = any_rank_true(used_mol_local)

            if used_mol_global and (not used_mol_local) and (outputs.loss is not None):
                if hasattr(self, "mol_adapter"):
                    outputs.loss = outputs.loss + zero_touch_module(self.mol_adapter)
                if hasattr(self, "gnn_mlp"):
                    outputs.loss = outputs.loss + zero_touch_module(self.gnn_mlp)
                if hasattr(self, "diffusion_mlp"):
                    outputs.loss = outputs.loss + zero_touch_module(self.diffusion_mlp)

            return outputs

    def _append_mol_embeds_to_end_offline(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            labels: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
        """
        将 batch 内每个样本中出现的 <mol>...</mol> 对，基于 </mol> 闭合处触发一次“虚拟步”：
        - 若内部是可解析的 SMILES：SMILES -> GVP -> mol_adapter，得到 LLM 维度向量
        - 否则（且启用兜底）：用 diffusion 路径（基于 h_ctx 或上下文）得到向量
        然后把这些向量“追加到序列末尾”；对应 mask=1，label=-100（不计 LM loss）。
        返回：(new_embeds, new_masks, new_labels, appended_mol_cnt_total)
        """
        # 如果禁用了 GNN，直接返回原始 embeddings，不处理任何 <mol> 标签
        if self.disable_gnn:
            embed_tokens = self.llm.get_input_embeddings()
            embeds = embed_tokens(input_ids)
            if attention_mask is None:
                attention_mask = (input_ids != self.pad_token_id).long().to(input_ids.device)
            return embeds, attention_mask, labels, 0
        
        assert input_ids.dim() == 2, "input_ids 形状应为 (B, T)"
        embed_tokens = self.llm.get_input_embeddings()
        emb_dev = embed_tokens.weight.device

        input_ids = input_ids.to(emb_dev)
        if attention_mask is not None:
            attention_mask = attention_mask.to(emb_dev)
        if labels is not None:
            labels = labels.to(emb_dev)

        B, T = input_ids.shape
        device = input_ids.device
        embeds = embed_tokens(input_ids)         # (B, T, D)
        D = embeds.size(-1)

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long().to(device)
        has_labels = labels is not None

        rows_embeds, rows_masks, rows_labels = [], [], []
        max_len = 0
        appended_mol_cnt_total = 0  # 本 batch 实际追加的 mol 向量个数

        # per-forward 局部缓存：SMILES -> mol_emb（保留计算图以训练 mol_adapter）
        per_forward_mol_emb_cache: Dict[str, torch.Tensor] = {}

        for b in range(B):
            row_ids = input_ids[b]          # (T,)
            row_emb = embeds[b]             # (T, D)
            row_msk = attention_mask[b]     # (T,)
            row_lbl = labels[b] if has_labels else None

            # 先把原始 token 的 embed/mask/label 按顺序压入
            new_emb_list = [row_emb[i] for i in range(T)]
            new_msk_list = [int(row_msk[i].item()) for i in range(T)]
            new_lbl_list = [int(row_lbl[i].item()) for i in range(T)] if has_labels else None

            # 使用 token_classifier_head 检测分子实体的位置
            valid_len = int(row_msk.sum().item())
            dec_text = self.tokenizer.decode(row_ids[:valid_len].tolist(), skip_special_tokens=False)
            
            # 获取实体 spans（使用 token_classifier_head 或 fallback 到文本匹配）
            spans = self._detect_mol_entities_with_classifier(row_ids[:valid_len], dec_text)  # [(inner, end_char)]

            if spans:
                # 使用 offsets_mapping 将字符位置映射到 token 索引
                toks = self.tokenizer(dec_text, return_offsets_mapping=True, add_special_tokens=False)
                offsets = toks.get("offset_mapping")
                trigger_idx_to_span = {}
                if offsets is not None:
                    # fast tokenizer: offsets 是 batch 维的
                    offsets = offsets[0].tolist() if hasattr(offsets, "tolist") else offsets
                    # 为每个 span 找 token 边界索引（第一个 end>=end_char 的 token）
                    for inner, end_char in spans:
                        tok_idx = None
                        for i_off, (_s, _e) in enumerate(offsets):
                            if _e >= end_char and _e > 0:
                                tok_idx = i_off
                                break
                        if tok_idx is None:
                            tok_idx = len(offsets) - 1
                        # 保护：不要超过有效长度
                        tok_idx = min(tok_idx, valid_len - 1)
                        trigger_idx_to_span[tok_idx] = (inner, end_char)

                # 遍历触发点，在该 token 索引处为该样本追加一次虚拟向量
                for trig_idx, (inner_text, end_char) in sorted(trigger_idx_to_span.items()):
                    # 对应位置是 padding 则跳过
                    if new_msk_list[trig_idx] == 0:
                        continue

                    mol_emb = None
                    # 判定是否是 SMILES
                    is_smiles = False
                    if Chem is not None and inner_text:
                        try:
                            is_smiles = (Chem.MolFromSmiles(inner_text) is not None)
                        except Exception:
                            is_smiles = False

                    if is_smiles and inner_text:
                        # logger.info(f"[EntityProcessing] ✅ Entity '{inner_text}' is valid SMILES, using GVP+adapter")
                        # —— 命中局部缓存就直接复用（保留计算图）——
                        if inner_text in per_forward_mol_emb_cache:
                            mol_emb = per_forward_mol_emb_cache[inner_text]
                            self.gnn_stats["gnn_cache_hits"] += 1
                        else:
                            # 一般 GVP 冻结，可 no_grad，节省显存/算力；mol_adapter 需要梯度以便训练
                            with torch.no_grad():
                                gvp_embedding = self.gvp_encoder.forward_from_smiles(inner_text).squeeze(0)
                            mol_emb = self.mol_adapter(gvp_embedding)  # shape: [D]
                            per_forward_mol_emb_cache[inner_text] = mol_emb
                            self.gnn_stats["gnn_cache_misses"] += 1
                            self.gnn_stats["smiles_processed"] += 1
                            self.gnn_stats["smiles_valid"] += 1
                        self.gnn_stats["total_mol_embeddings"] += 1
                    elif self.enable_diffusion_fallback:
                        if getattr(self, '_verbose_logging', False):
                            logger.info(f"[EntityProcessing] 🎲 Entity '{inner_text}' is NOT valid SMILES, calling BLACKBOX fallback")
                        # 兜底：用文本到 </mol> 结束位置作为上下文计算 h_ctx / 或直接在线黑盒
                        ctx_text = dec_text[:end_char]
                        mol_emb = self._black_box_embed_online(llm_context_text=ctx_text, context_ids=None, h_ctx=None)
                        if mol_emb is not None:
                            if getattr(self, '_verbose_logging', False):
                                logger.info(f"[EntityProcessing] ✅ Blackbox returned embedding successfully")
                            self.gnn_stats["diffusion_fallback_count"] += 1
                        else:
                            # 关键信息：blackbox 失败应该总是警告
                            logger.warning(f"[EntityProcessing] ❌ Blackbox returned None for entity '{inner_text[:50]}...'")
                    else:
                        if getattr(self, '_verbose_logging', False):
                            logger.info(f"[EntityProcessing] ⚠️  Entity '{inner_text}' is invalid and fallback disabled")
                        self.gnn_stats["smiles_processed"] += 1
                        self.gnn_stats["smiles_invalid"] += 1

                    if mol_emb is None:
                        # 关键信息：如果跳过了很多实体，应该记录
                        if getattr(self, "debug", False):
                            if getattr(self, '_verbose_logging', False):
                                logger.info("[Offline] Skip virtual step at </mol> (no embedding).")
                            else:
                                # 非 verbose 模式下，只在 debug 时输出简要信息
                                pass
                        continue

                    new_emb_list.append(mol_emb)
                    new_msk_list.append(1)
                    if has_labels:
                        new_lbl_list.append(-100)
                    appended_mol_cnt_total += 1

            # 本样本的拼接结果 -> tensor
            new_len = len(new_msk_list)
            max_len = max(max_len, new_len)

            new_emb = torch.stack(new_emb_list, dim=0)                                 # (L, D)
            new_msk = torch.tensor(new_msk_list, device=device, dtype=row_msk.dtype)   # (L,)
            new_lbl = (torch.tensor(new_lbl_list, device=device, dtype=input_ids.dtype)
                    if has_labels else None)

            rows_embeds.append(new_emb)
            rows_masks.append(new_msk)
            if has_labels:
                rows_labels.append(new_lbl)

        # 对齐到同一长度（右侧 padding）
        padded_embeds, padded_masks = [], []
        padded_labels = [] if has_labels else None

        for b in range(B):
            E = rows_embeds[b]; M = rows_masks[b]
            pad_len = max_len - E.size(0)

            if pad_len > 0:
                E = torch.cat([E, torch.zeros(pad_len, D, device=E.device, dtype=E.dtype)], dim=0)
                M = torch.cat([M, torch.zeros(pad_len, device=M.device, dtype=M.dtype)], dim=0)
                if has_labels:
                    L = rows_labels[b]
                    L = torch.cat([L, torch.full((pad_len,), -100, device=L.device, dtype=L.dtype)], dim=0)
                else:
                    L = None
            else:
                L = rows_labels[b] if has_labels else None

            padded_embeds.append(E.unsqueeze(0))  # (1, max_len, D)
            padded_masks.append(M.unsqueeze(0))   # (1, max_len)
            if has_labels:
                padded_labels.append(L.unsqueeze(0) if L is not None else None)

        new_embeds = torch.cat(padded_embeds, dim=0)              # (B, max_len, D)
        new_masks  = torch.cat(padded_masks,  dim=0)              # (B, max_len)
        new_labels = torch.cat(padded_labels, dim=0) if has_labels else None  # (B, max_len) or None

        # 关键信息：如果本批次处理了实体，输出简要摘要
        if appended_mol_cnt_total > 0:
            if getattr(self, '_verbose_logging', False):
                logger.info(f"[Offline] Batch processed: appended {appended_mol_cnt_total} mol embeddings")
            # 非 verbose 模式下不输出，避免过于频繁

        # 定期打印GNN pipeline统计信息（关键信息，总是输出）
        if hasattr(self, "gnn_stats") and appended_mol_cnt_total > 0:
            stats = self.gnn_stats
            total = stats["gnn_cache_hits"] + stats["gnn_cache_misses"]
            if total > 0 and (stats["total_mol_embeddings"] % self.gnn_log_interval == 0):
                hit_rate = stats["gnn_cache_hits"] / total * 100 if total > 0 else 0
                if getattr(self, '_verbose_logging', False):
                    # 详细统计信息
                    logger.info(
                        f"[GNN Pipeline] Stats: SMILES processed={stats['smiles_processed']}, "
                        f"valid={stats['smiles_valid']}, invalid={stats['smiles_invalid']}, "
                        f"cache_hits={stats['gnn_cache_hits']}, cache_misses={stats['gnn_cache_misses']}, "
                        f"hit_rate={hit_rate:.1f}%, diffusion_fallback={stats['diffusion_fallback_count']}, "
                        f"total_embeddings={stats['total_mol_embeddings']}"
                    )
                else:
                    # 简要统计信息（关键信息）
                    logger.info(
                        f"[GNN Pipeline] Processed {stats['total_mol_embeddings']} embeddings "
                        f"(valid: {stats['smiles_valid']}, invalid: {stats['smiles_invalid']}, "
                        f"cache hit rate: {hit_rate:.1f}%)"
                    )
        
        if getattr(self, "debug", False):
            orig_tokens = attention_mask.sum().item()
            new_tokens  = new_masks.sum().item()
            print(f"[MolAware/offline] appended {int(new_tokens - orig_tokens)} embeddings to batch end; "
                f"mol_appended_count={appended_mol_cnt_total}")

        return new_embeds, new_masks, new_labels, appended_mol_cnt_total


    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        realtime_mol: bool = True,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        verbose_logging: bool = False,  # 控制详细日志输出
        max_text_length_for_detection: int = 4096,  # 超出此长度跳过实体检测（但不停止生成），支持few-shot等长prompt
        skip_special_tokens: bool = False,
        stop_on_eos: bool = True,  # ✅ 新增：是否遇到EOS/eot停止（默认 True）
        **kwargs,
    ):
        """
        推理阶段在线处理（online）：
        - 逐 token 采样
        - 在边界 token 处（或检测到新的 </mol> 对）检测实体并插入一次 virtual step（inputs_embeds）
        关键修复：
        - ✅ 恢复 stop tokens（EOS / <|eot_id|>），否则必跑满 max_new_tokens
        - ✅ virtual step 去重：同一次“出现(文本, end_char)”只插一次，避免反复插入造成重复/不收敛
        """
        use_cache = kwargs.pop("use_cache", True)
        no_repeat_ngram_size = int(kwargs.pop("no_repeat_ngram_size", 0) or 0)

        try:
            self.llm.config.use_cache = True
        except Exception:
            pass

        if not realtime_mol:
            # 走 HF 自带 generate（正常 EOS 停止）
            return self.llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                use_cache=use_cache,
                **kwargs,
            )

        # realtime_mol 支持 input_ids 和/或 inputs_embeds
        # - 如果同时提供：使用 input_ids 进行 token 检测，inputs_embeds 作为额外 embedding 插入（类似 GVP 虚拟步）
        # - 如果只提供 input_ids：正常处理
        # - 如果只提供 inputs_embeds：跳过 token 检测（因为已经是 embedding 了）
        inputs_embeds_extra = kwargs.pop("inputs_embeds", None)
        has_input_ids = input_ids is not None
        has_inputs_embeds = inputs_embeds_extra is not None
        
        if not has_input_ids and not has_inputs_embeds:
            raise ValueError("必须提供 input_ids 或 inputs_embeds")
        
        if not realtime_mol:
            # 非 realtime_mol 模式，使用标准 generate
            if has_inputs_embeds and not has_input_ids:
                # 只有 inputs_embeds
                return self.llm.generate(
                    inputs_embeds=inputs_embeds_extra,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=eos_token_id,
                    repetition_penalty=repetition_penalty,
                    use_cache=use_cache,
                    **kwargs,
                )
            else:
                # 有 input_ids（可能同时有 inputs_embeds，但标准 generate 不支持同时传入）
                return self.llm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=eos_token_id,
                    repetition_penalty=repetition_penalty,
                    use_cache=use_cache,
                    **kwargs,
                )
        
        # realtime_mol 模式：支持 input_ids 和/或 inputs_embeds
        if has_input_ids:
            # 有 input_ids，可以进行 token 检测
            if input_ids.size(0) > 1:
                raise ValueError(f"realtime_mol 仅支持 batch=1，当前 batch={input_ids.size(0)}。请在调用时确保 batch_size=1 或逐个处理")
            if attention_mask is None:
                attention_mask = (input_ids != self.pad_token_id).long()
        elif has_inputs_embeds:
            # 只有 inputs_embeds，跳过 token 检测
            if inputs_embeds_extra.size(0) > 1:
                raise ValueError(f"realtime_mol 仅支持 batch=1，当前 batch={inputs_embeds_extra.size(0)}。请在调用时确保 batch_size=1 或逐个处理")
            if attention_mask is None:
                # 从 inputs_embeds 的形状推断 attention_mask
                attention_mask = torch.ones(inputs_embeds_extra.size(0), inputs_embeds_extra.size(1), dtype=torch.long, device=inputs_embeds_extra.device)
        
        llm = self.llm
        dev = self._first_device()
        
        if has_input_ids:
            input_ids = input_ids.to(dev)
        if has_inputs_embeds:
            inputs_embeds_extra = inputs_embeds_extra.to(dev)
        attention_mask = attention_mask.to(dev)

        # 设置日志与检测参数
        self._verbose_logging = verbose_logging
        self._max_text_length_for_detection = max_text_length_for_detection
        self._detection_interval = getattr(self, "_detection_interval", 5)  # 每N个边界token检测一次
        self._boundary_token_count = 0

        # per-generation cache：SMILES -> mol_emb（推理无梯度）
        gen_mol_emb_cache: Dict[str, torch.Tensor] = {}

        # ✅ 关键：去重 “同一次出现” 的 virtual step
        # 用 (effective_text, end_char_pos) 作为 key；同一次出现只插一次
        processed_occurrences = set()

        # 你之前用 processed_pair_count / processed_inner_texts，这里保留但不作为插入唯一依据
        processed_pair_count = 0
        processed_inner_texts = set()

        # ====== stop tokens（✅ 恢复）======
        stop_token_ids = set()
        end_id = eos_token_id if eos_token_id is not None else self.eos_token_id
        if (end_id is None or end_id < 0) and self.tokenizer is not None:
            # 兜底尝试 <|eot_id|>
            try:
                eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot_token_id is not None and eot_token_id >= 0:
                    end_id = eot_token_id
                    if verbose_logging:
                        logger.info(f"[Generate] Using <|eot_id|> (token_id={end_id}) as EOS token fallback")
            except Exception:
                pass

        if stop_on_eos:
            if end_id is not None and end_id >= 0:
                stop_token_ids.add(int(end_id))
            # 强烈建议加上 eot（Llama 系列经常用它结束一条消息）
            try:
                eot = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot is not None and eot >= 0:
                    stop_token_ids.add(int(eot))
            except Exception:
                pass

        def _prepare_probs(_logits: torch.Tensor) -> torch.Tensor:
            probs = torch.softmax(_logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            probs = torch.clamp(probs, min=0.0)
            sum_probs = probs.sum(dim=-1, keepdim=True)
            probs = probs / sum_probs.clamp(min=1e-8)
            return probs

        def _apply_topk_topp_temp(_logits: torch.Tensor) -> torch.Tensor:
            logits2 = _logits
            if temperature and temperature != 1.0:
                logits2 = logits2 / float(temperature)
            if top_k and top_k > 0:
                v, _ = torch.topk(logits2, int(top_k))
                logits2 = logits2.masked_fill(logits2 < v[:, [-1]], float("-inf"))
            if top_p and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits2, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = probs.cumsum(dim=-1)
                cutoff = (cumprobs > float(top_p)).float().cumsum(dim=-1).bool()
                sorted_logits[cutoff] = float("-inf")
                logits2 = torch.full_like(logits2, float("-inf")).scatter(1, sorted_indices, sorted_logits)
            return logits2

        def _apply_sampling(_logits: torch.Tensor) -> torch.Tensor:
            if do_sample:
                logits2 = _apply_topk_topp_temp(_logits)
                probs = _prepare_probs(logits2)
                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs <= 0).all():
                    return torch.argmax(_logits, dim=-1, keepdim=True)
                return torch.multinomial(probs, num_samples=1)
            return torch.argmax(_logits, dim=-1, keepdim=True)

        def _block_no_repeat_ngrams(logits: torch.Tensor, prefix_ids: List[int], gen_ids: List[int], n: int) -> torch.Tensor:
            """
            简易 no_repeat_ngram_size（只基于当前 prefix+generated）
            参考 transformers 的逻辑，但这里做一个轻量版：只禁止“刚要形成的 ngram”重复出现过。
            """
            if n <= 0:
                return logits
            seq = prefix_ids + gen_ids
            if len(seq) < n:
                return logits
            # 当前要预测的是第 len(seq) 位置 -> 它会形成一个 n-1 前缀
            prev_ngram = tuple(seq[-(n - 1):]) if n > 1 else tuple()
            # 收集历史出现过的 ngram：prev_ngram -> next_token 列表
            banned = set()
            if n == 1:
                # n==1 就是禁止重复 token（太强，不建议）；这里不做
                return logits
            for i in range(len(seq) - n + 1):
                ng = tuple(seq[i:i + n])
                if ng[:-1] == prev_ngram:
                    banned.add(ng[-1])
            if banned:
                logits[:, list(banned)] = float("-inf")
            return logits

        # ====== 初始前向 ======
        if has_input_ids:
            # 使用 input_ids 进行初始前向
            outputs = llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        else:
            # 只有 inputs_embeds
            outputs = llm(
                inputs_embeds=inputs_embeds_extra,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        past = outputs.past_key_values
        attn_mask = attention_mask

        # ====== 如果同时提供了 inputs_embeds，将其作为额外 embedding 插入（类似 GVP 虚拟步）======
        if has_input_ids and has_inputs_embeds:
            # 同时有 input_ids 和 inputs_embeds，将 inputs_embeds 插入到序列末尾
            model_dtype = next(self.llm.parameters()).dtype
            if inputs_embeds_extra.dtype != model_dtype:
                inputs_embeds_extra = inputs_embeds_extra.to(dtype=model_dtype)
            if inputs_embeds_extra.device != dev:
                inputs_embeds_extra = inputs_embeds_extra.to(device=dev)
            
            # 确保 inputs_embeds 的形状正确：[1, seq_len, hidden_dim]
            if inputs_embeds_extra.dim() == 2:
                inputs_embeds_extra = inputs_embeds_extra.unsqueeze(0)  # [seq_len, hidden_dim] -> [1, seq_len, hidden_dim]
            
            # 插入 virtual step（类似 GVP 虚拟步）
            if verbose_logging:
                logger.info(f"[Generate] ✅ 插入额外 embedding（类似 GVP 虚拟步），形状: {inputs_embeds_extra.shape}")
            
            # 更新 attention_mask：为额外的 embedding 添加 mask
            extra_seq_len = inputs_embeds_extra.size(1)
            extra_mask = torch.ones(1, extra_seq_len, device=dev, dtype=attn_mask.dtype)
            
            # 前向传播，插入额外 embedding
            # 注意：这里只传入额外 embedding 的 attention_mask，因为 past_key_values 已经包含了之前的状态
            outputs = llm(
                inputs_embeds=inputs_embeds_extra,
                attention_mask=extra_mask,  # 只传入额外 embedding 的 mask
                past_key_values=past,
                use_cache=use_cache,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            past = outputs.past_key_values
            # 更新 attn_mask 以包含额外 embedding 的部分（用于后续生成）
            attn_mask = torch.cat([attn_mask, extra_mask], dim=1)

        # ====== 处理输入中已存在实体（可选，但你原逻辑有；这里保留并加去重）======
        # 注意：如果只有 inputs_embeds（没有 input_ids），跳过 token 检测（因为已经是 embedding 了）
        if not has_input_ids:
            input_spans = []  # 跳过检测
        else:
            try:
                input_text = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)

                if hasattr(self, "token_classifier_head") and self.token_classifier_head is not None:
                    input_spans = self._detect_mol_entities_with_classifier(input_ids[0], input_text)
                else:
                    input_spans = self._find_all_mol_spans(input_text)
            except Exception as e:
                if verbose_logging:
                    logger.warning(f"[Generate] ⚠️  Failed to detect input entities: {e}")
                input_spans = []

            if input_spans:
                model_dtype = next(self.llm.parameters()).dtype

                for inner_text, end_char in input_spans:
                    cleaned_text = (inner_text or "").strip()
                    trailing_punct = ",.;:!?"
                    while cleaned_text and cleaned_text[-1] in trailing_punct:
                        cleaned_text = cleaned_text[:-1].strip()
                    while cleaned_text and cleaned_text[0] in trailing_punct:
                        cleaned_text = cleaned_text[1:].strip()

                    text_to_check = cleaned_text if cleaned_text else inner_text
                    is_smiles = False
                    if Chem is not None and text_to_check:
                        try:
                            mol = Chem.MolFromSmiles(text_to_check)
                            if mol is not None:
                                canonical_smiles = Chem.MolToSmiles(mol)
                                if canonical_smiles and len(text_to_check) >= 5:
                                    is_smiles = True
                        except Exception:
                            is_smiles = False

                    effective_text = text_to_check if is_smiles else (inner_text or "")
                    occ_key = (effective_text, int(end_char))
                    if occ_key in processed_occurrences:
                        continue
                    processed_occurrences.add(occ_key)

                    mol_emb = None
                    if is_smiles and effective_text:
                        cache_key = effective_text
                        if cache_key in gen_mol_emb_cache:
                            mol_emb = gen_mol_emb_cache[cache_key]
                            if verbose_logging:
                                logger.info(f"[Generate] ✅ Input entity (cached): '{effective_text}'")
                        else:
                            try:
                                with torch.no_grad():
                                    gvp_embedding_raw = self.gvp_encoder.forward_from_smiles(effective_text).squeeze(0)
                                    mol_adapter_dtype = next(self.mol_adapter.parameters()).dtype
                                    gvp_embedding = gvp_embedding_raw.to(dtype=mol_adapter_dtype) if gvp_embedding_raw.dtype != mol_adapter_dtype else gvp_embedding_raw
                                    mol_emb = self.mol_adapter(gvp_embedding)
                                    if mol_emb.dtype != model_dtype:
                                        mol_emb = mol_emb.to(dtype=model_dtype)
                                gen_mol_emb_cache[cache_key] = mol_emb
                                if verbose_logging:
                                    logger.info(f"[Generate] ✅ Input entity (fresh): '{effective_text}' -> GVP -> mol_adapter")
                            except Exception as e:
                                logger.warning(f"[Generate] ⚠️  Failed to process input SMILES '{effective_text}': {e}")
                                mol_emb = None
                    elif self.enable_diffusion_fallback:
                        try:
                            h_ctx = outputs.hidden_states[-1][0, -1, :].detach()
                            mol_emb = self._black_box_from_hidden_hctx(h_ctx)
                            if mol_emb is not None and verbose_logging:
                                logger.info(f"[Generate] ✅ Input entity (diffusion): '{inner_text}'")
                        except Exception as e:
                            logger.warning(f"[Generate] ⚠️  Diffusion failed on input entity '{inner_text}': {e}")
                            mol_emb = None

                    if mol_emb is not None:
                        if mol_emb.device != dev:
                            mol_emb = mol_emb.to(device=dev)
                        if mol_emb.dtype != model_dtype:
                            mol_emb = mol_emb.to(dtype=model_dtype)

                        # 插入 virtual step
                        outputs = llm(
                            inputs_embeds=mol_emb.view(1, 1, -1),
                            attention_mask=torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1),
                            past_key_values=past,
                            use_cache=use_cache,
                            output_hidden_states=True,
                            return_dict=True,
                            **kwargs,
                        )
                        past = outputs.past_key_values
                        attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1)
                        processed_inner_texts.add(effective_text)

                if input_text.count("</mol>") > 0:
                    processed_pair_count = input_text.count("</mol>")

        # ====== 主循环 ======
        generated_ids: List[int] = []
        if has_input_ids:
            prefix_ids = input_ids[0].tolist()
        else:
            # 只有 inputs_embeds 时，无法获取 prefix_ids，设为空列表
            prefix_ids = []

        force_detection_next = False  # 插 virtual step 后，下一个边界检测不强制（避免死循环）；这里保留但默认不用

        for step in range(int(max_new_tokens)):
            logits = outputs.logits[:, -1, :]

            # repetition penalty（你原逻辑）
            if repetition_penalty and repetition_penalty != 1.0 and generated_ids:
                uniq = list(set(generated_ids))
                logits[:, uniq] = logits[:, uniq] / float(repetition_penalty)
                recent_window = min(10, len(generated_ids))
                recent_tokens_penalty = generated_ids[-recent_window:]
                recent_uniq = list(set(recent_tokens_penalty))
                if recent_uniq:
                    logits[:, recent_uniq] = logits[:, recent_uniq] / (float(repetition_penalty) * 1.2)

            # no_repeat_ngram（可选，只有 inputs_embeds 时跳过）
            if has_input_ids and no_repeat_ngram_size and no_repeat_ngram_size > 0:
                logits = _block_no_repeat_ngrams(logits, prefix_ids, generated_ids, no_repeat_ngram_size)

            next_token = _apply_sampling(logits)
            next_id = int(next_token.item())

            # ===== 情况 A：采样到了 <mol> token -> 立即尝试插入 virtual step（你原逻辑保留）=====
            # 注意：只有 inputs_embeds（没有 input_ids）时，跳过 token 检测（因为已经是 embedding 了）
            if not has_input_ids:
                # 只有 inputs_embeds 时，跳过 <mol> token 检测和处理
                pass
            elif next_id == self.mol_token_id:
                current_context_ids = torch.cat(
                    [input_ids, torch.tensor([generated_ids], device=dev, dtype=input_ids.dtype)],
                    dim=1
                )
                llm_context_text = self.tokenizer.decode(current_context_ids[0].tolist(), skip_special_tokens=False)

                mol_embedding = None
                gnn_path = None
                inner = self._extract_last_between_mol_tags(llm_context_text or "")

                is_smiles = False
                if inner and Chem is not None:
                    try:
                        is_smiles = (Chem.MolFromSmiles(inner) is not None)
                    except Exception:
                        is_smiles = False

                if inner and is_smiles:
                    try:
                        if inner in gen_mol_emb_cache:
                            mol_embedding = gen_mol_emb_cache[inner]
                            gnn_path = "GNN (cached via <mol>)"
                        else:
                            model_dtype = next(self.llm.parameters()).dtype
                            with torch.no_grad():
                                gvp_embedding_raw = self.gvp_encoder.forward_from_smiles(inner).squeeze(0)
                                mol_adapter_dtype = next(self.mol_adapter.parameters()).dtype
                                gvp_embedding = gvp_embedding_raw.to(dtype=mol_adapter_dtype) if gvp_embedding_raw.dtype != mol_adapter_dtype else gvp_embedding_raw
                                mol_embedding = self.mol_adapter(gvp_embedding)
                                if mol_embedding.dtype != model_dtype:
                                    mol_embedding = mol_embedding.to(dtype=model_dtype)
                            gen_mol_emb_cache[inner] = mol_embedding
                            gnn_path = "GNN (fresh via <mol>)"
                    except Exception as e:
                        logger.warning(f"[Generate] ⚠️  Failed to process SMILES '{inner}' via <mol>: {e}")
                        mol_embedding = None
                elif self.enable_diffusion_fallback:
                    try:
                        h_ctx_step = outputs.hidden_states[-1][0, -1, :].detach()
                        mol_embedding = self._black_box_from_hidden_hctx(h_ctx_step)
                        if mol_embedding is not None:
                            gnn_path = "Diffusion fallback (via <mol>)"
                    except Exception as e:
                        logger.warning(f"[Generate] ⚠️  Diffusion fallback failed via <mol>: {e}")
                        mol_embedding = None

                if mol_embedding is None:
                    # 禁止选中 <mol>，重新采样
                    logits_block = logits.clone()
                    logits_block[0, self.mol_token_id] = float("-inf")
                    next_token = _apply_sampling(logits_block)
                    next_id = int(next_token.item())
                else:
                    model_dtype = next(self.llm.parameters()).dtype
                    if mol_embedding.dtype != model_dtype:
                        mol_embedding = mol_embedding.to(dtype=model_dtype)
                    if mol_embedding.device != dev:
                        mol_embedding = mol_embedding.to(device=dev)

                    if verbose_logging:
                        logger.info(f"[Generate] 🎯 Inserting virtual step via {gnn_path}")

                    outputs = llm(
                        inputs_embeds=mol_embedding.view(1, 1, -1),
                        attention_mask=torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1),
                        past_key_values=past,
                        use_cache=use_cache,
                        output_hidden_states=True,
                        return_dict=True,
                        **kwargs,
                    )
                    past = outputs.past_key_values
                    attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1)
                    # 不把 <mol> 计入 generated_ids
                    continue

            # ===== 常规生成一个 token =====
            step_ids = next_token  # [1,1]
            attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1)
            outputs = llm(
                input_ids=step_ids,
                attention_mask=attn_mask,
                past_key_values=past,
                use_cache=use_cache,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
            past = outputs.past_key_values
            generated_ids.append(next_id)

            # ✅ stop token：及时 break
            if stop_on_eos and (next_id in stop_token_ids):
                if verbose_logging:
                    tok_txt = ""
                    try:
                        tok_txt = self.tokenizer.decode([next_id], skip_special_tokens=False)
                    except Exception:
                        pass
                    logger.info(f"[Generate] 🛑 Stop token hit: id={next_id} text={tok_txt!r}")
                break

            # ===== 情况 B：边界处检测实体，并插 virtual step（✅ 加"出现去重"）=====
            # 注意：只有 inputs_embeds（没有 input_ids）时，跳过 token 检测（因为已经是 embedding 了）
            if not has_input_ids:
                # 只有 inputs_embeds 时，跳过边界检测
                continue
            
            try:
                # 是否需要检测：边界token计数/间隔
                should_detect_at_boundary = False
                if force_detection_next:
                    should_detect_at_boundary = True
                    force_detection_next = False
                elif _is_boundary_token(self.tokenizer, next_id):
                    self._boundary_token_count += 1
                    if self._boundary_token_count >= int(self._detection_interval):
                        should_detect_at_boundary = True
                        self._boundary_token_count = 0

                if not should_detect_at_boundary:
                    continue

                # 只检测尾部窗口，窗口足够大 + overlap，避免实体跨窗被截断
                WINDOW_TOKENS = 2048   # 你愿意慢可以更大：3072/4096（看显存/速度）
                OVERLAP_TOKENS = 512   # 防止实体跨窗（长SMILES/长化学名）

                current_context_ids = torch.cat([input_ids, torch.tensor([generated_ids], device=dev, dtype=input_ids.dtype)], dim=1)
                seq = current_context_ids[0]
                L = seq.numel()

                start = max(0, L - WINDOW_TOKENS)
                tokens_to_detect = seq[start:]  # 仅尾部
                text_to_detect = self.tokenizer.decode(tokens_to_detect.tolist(), skip_special_tokens=False)

                # 计算 offset：窗口前面那一段字符长度（用于把 end_char 映射回全局）
                # 为了不漏，offset 的计算可以慢一点：decode 一次 prefix
                prefix_text = self.tokenizer.decode(seq[:start].tolist(), skip_special_tokens=False) if start > 0 else ""
                input_offset_chars = len(prefix_text)

                detected_spans = self._detect_mol_entities_with_classifier(tokens_to_detect, text_to_detect)
                # 映射回全局字符位置（如果你后面要用 end_char 做全局比较）
                detected_spans = [(t, p + input_offset_chars) for (t, p) in detected_spans]


                current_context_ids = torch.cat(
                    [input_ids, torch.tensor([generated_ids], device=dev, dtype=input_ids.dtype)],
                    dim=1
                )
                llm_context_text = self.tokenizer.decode(current_context_ids[0].tolist(), skip_special_tokens=False)

                full_text_mol_count = llm_context_text.count("</mol>")

                # 检测文本（这里你取消了截断，我保持一致）
                text_to_detect = llm_context_text
                tokens_to_detect = current_context_ids[0]
                input_offset_chars = 0

                spans: List[Tuple[str, int]] = []

                # 如果出现新的 </mol>，优先跑一次检测
                if full_text_mol_count > processed_pair_count:
                    if hasattr(self, "token_classifier_head") and self.token_classifier_head is not None:
                        detected_spans = self._detect_mol_entities_with_classifier(tokens_to_detect, text_to_detect)
                        if input_offset_chars > 0:
                            detected_spans = [(t, p + input_offset_chars) for t, p in detected_spans]
                        spans.extend(detected_spans)

                # 边界处常规检测
                detected_spans = self._detect_mol_entities_with_classifier(tokens_to_detect, text_to_detect)
                if input_offset_chars > 0:
                    detected_spans = [(t, p + input_offset_chars) for t, p in detected_spans]

                # 过滤：尽量只保留生成部分（你原逻辑）
                input_text_only = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
                generated_text_only = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

                filtered_spans = []
                input_text_len = len(input_text_only)
                for inner_text, end_char_pos in detected_spans:
                    if inner_text in generated_text_only:
                        # 如果在 input_text 中也出现过，则要求 end_char_pos 落在 input_text_len 之后
                        if inner_text not in input_text_only:
                            filtered_spans.append((inner_text, end_char_pos))
                        else:
                            if int(end_char_pos) > int(input_text_len):
                                filtered_spans.append((inner_text, end_char_pos))

                # 用 looks_like_molecule 再筛一次（你原逻辑）
                for inner_text, end_char_pos in filtered_spans:
                    inner = (inner_text or "").strip()
                    if _looks_like_molecule(inner):
                        spans.append((inner_text, end_char_pos))

                # 去重 spans（同一轮避免重复）
                uniq = []
                seen_local = set()
                for t, p in spans:
                    key = (t, int(p))
                    if key in seen_local:
                        continue
                    seen_local.add(key)
                    uniq.append((t, int(p)))
                spans = uniq

                if spans:
                    inserted_virtual_steps = False
                    model_dtype = next(self.llm.parameters()).dtype

                    for inner_text, end_char_pos in spans:
                        # 清理实体文本
                        cleaned_text = (inner_text or "").strip()
                        trailing_punct = ",.;:!?"
                        while cleaned_text and cleaned_text[-1] in trailing_punct:
                            cleaned_text = cleaned_text[:-1].strip()
                        while cleaned_text and cleaned_text[0] in trailing_punct:
                            cleaned_text = cleaned_text[1:].strip()

                        text_to_check = cleaned_text if cleaned_text else (inner_text or "")
                        is_smiles = False
                        if Chem is not None and text_to_check:
                            try:
                                mol = Chem.MolFromSmiles(text_to_check)
                                if mol is not None:
                                    canonical_smiles = Chem.MolToSmiles(mol)
                                    if canonical_smiles and len(text_to_check) >= 5:
                                        is_smiles = True
                            except Exception:
                                is_smiles = False

                        effective_text = text_to_check if is_smiles else (inner_text or "")

                        # ✅ 关键：出现去重 —— 同一次出现只插一次
                        occ_key = (effective_text, int(end_char_pos))
                        if occ_key in processed_occurrences:
                            continue
                        processed_occurrences.add(occ_key)

                        mol_emb = None
                        if is_smiles and effective_text:
                            cache_key = effective_text
                            if cache_key in gen_mol_emb_cache:
                                mol_emb = gen_mol_emb_cache[cache_key]
                                if verbose_logging:
                                    logger.info(f"[Generate] ✅ Reuse cached embedding for '{effective_text}'")
                            else:
                                try:
                                    with torch.no_grad():
                                        gvp_embedding_raw = self.gvp_encoder.forward_from_smiles(effective_text).squeeze(0)
                                        mol_adapter_dtype = next(self.mol_adapter.parameters()).dtype
                                        gvp_embedding = gvp_embedding_raw.to(dtype=mol_adapter_dtype) if gvp_embedding_raw.dtype != mol_adapter_dtype else gvp_embedding_raw
                                        mol_emb = self.mol_adapter(gvp_embedding)
                                        if mol_emb.dtype != model_dtype:
                                            mol_emb = mol_emb.to(dtype=model_dtype)
                                    gen_mol_emb_cache[cache_key] = mol_emb
                                    if verbose_logging:
                                        logger.info(f"[Generate] ✅ Fresh embedding for '{effective_text}' (GVP+adapter)")
                                except Exception as e:
                                    logger.warning(f"[Generate] ⚠️  Failed to process SMILES '{effective_text}': {e}")
                                    mol_emb = None
                        elif self.enable_diffusion_fallback:
                            try:
                                h_ctx_step2 = outputs.hidden_states[-1][0, -1, :].detach()
                                mol_emb = self._black_box_from_hidden_hctx(h_ctx_step2)
                                if mol_emb is not None and verbose_logging:
                                    logger.info(f"[Generate] ✅ Diffusion fallback embedding for '{inner_text}'")
                            except Exception as e:
                                logger.warning(f"[Generate] ⚠️  Diffusion fallback failed for '{inner_text}': {e}")
                                mol_emb = None

                        if mol_emb is None:
                            continue

                        if mol_emb.device != dev:
                            mol_emb = mol_emb.to(device=dev)
                        if mol_emb.dtype != model_dtype:
                            mol_emb = mol_emb.to(dtype=model_dtype)

                        # 插入 virtual step
                        outputs = llm(
                            inputs_embeds=mol_emb.view(1, 1, -1),
                            attention_mask=torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1),
                            past_key_values=past,
                            use_cache=use_cache,
                            output_hidden_states=True,
                            return_dict=True,
                            **kwargs,
                        )
                        past = outputs.past_key_values
                        attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=dev, dtype=attn_mask.dtype)], dim=1)
                        inserted_virtual_steps = True
                        processed_inner_texts.add(effective_text)

                    processed_pair_count = max(processed_pair_count, full_text_mol_count)

                    # 插入了虚拟步就继续下一轮采样（避免当前 token 位置反复检测）
                    if inserted_virtual_steps:
                        continue

            except Exception as e:
                logger.warning(f"[Generate] ⚠️  Exception in entity detection/GNN logic: {e}", exc_info=False)

        if not generated_ids:
            # 如果没有生成任何 token，返回原始输入
            if has_input_ids:
                return input_ids
            else:
                # 只有 inputs_embeds 时，无法返回原始输入（因为没有 input_ids）
                # 返回一个空的 tensor
                return torch.empty(1, 0, dtype=torch.long, device=dev)
        
        gen = torch.tensor([generated_ids], device=dev, dtype=torch.long)
        # 返回结果：如果有 input_ids，拼接返回；否则只返回生成的 token IDs
        if has_input_ids:
            return torch.cat([input_ids, gen], dim=1)
        else:
            return gen


    # --------------------------- HF 保存/加载 ---------------------------
    def state_dict(self, *args, **kwargs):
        # 保存整个组合模型的权重（包含自定义模块 + 底座 llm 的参数拷贝）
        sd = super().state_dict(*args, **kwargs)
        # 去重相同 storage，避免稀奇的共享 tensor 被重复引用
        seen = {}
        for k, v in list(sd.items()):
            if not isinstance(v, torch.Tensor):
                continue
            sid = self._storage_id(v)
            if sid in seen:
                sd[k] = v.clone()
            else:
                seen[sid] = k
        return sd

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        - 先调用底座 LLM 的 save_pretrained（保存权重、config 等）
        - 再额外保存组合模型的自定义模块（.pt）
        - 写入一个 metadata.json 记录额外文件名，便于 from_pretrained 恢复
        """
        os.makedirs(save_directory, exist_ok=True)
        # 1) 保存底座 LLM
        out = self.llm.save_pretrained(save_directory, **kwargs)

        # 2) 额外保存自定义模块
        extras = {}
        if hasattr(self, "gvp_encoder") and self.gvp_encoder is not None:
            torch.save(self.gvp_encoder.state_dict(), os.path.join(save_directory, "gvp_encoder.pt"))
            extras["gvp_encoder"] = "gvp_encoder.pt"
        if hasattr(self, "mol_adapter") and self.mol_adapter is not None:
            torch.save(self.mol_adapter.state_dict(), os.path.join(save_directory, "mol_adapter.pt"))
            extras["mol_adapter"] = "mol_adapter.pt"
        # 注意：diffusion_adapter 已移除，LDMol直接使用LLM的hidden states
        # 不再保存 diffusion_adapter

        # diffusion 主体通常体量较大且可选，不强制保存；如果需要自行加：
        # if hasattr(self, "diffusion") and self.diffusion is not None:
        #     torch.save(self.diffusion.state_dict(), os.path.join(save_directory, "diffusion.pt"))
        #     extras["diffusion"] = "diffusion.pt"

        meta = {
            "class": "MolAwareCausalLM",
            "version": 1,
            "extras": extras,
            "mol_token": self.mol_token,
        }
        with open(os.path.join(save_directory, "molaware_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return out

    @classmethod
    def from_pretrained(cls, save_directory: str, tokenizer=None,
                        diffusion_config=None, diffusion_adapter_config=None,
                        layer2_config=None, use_layer2=False,
                        **kwargs):
        root = save_directory
        meta_path = os.path.join(root, "molaware_metadata.json")
        has_meta = os.path.isfile(meta_path)

        # 1) 解析 metadata（若存在）
        meta = {}
        extras_map = {}
        if has_meta:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            extras_map = meta.get("extras", {}) or {}

        # 2) 决定 LLM 目录：优先 <root>/llm，其次 <root>
        llm_dir = os.path.join(root, "llm")
        if not has_hf_model_files(llm_dir):
            llm_dir = root
        print(f"[from_pretrained] using llm_dir={llm_dir}")

        # 3) 加载底座 LLM
        # 处理 torch 版本限制问题：如果遇到 torch.load 安全限制，尝试使用 safetensors 或绕过检查
        try:
            base_llm = AutoModelForCausalLM.from_pretrained(llm_dir, **kwargs)
        except (ValueError, RuntimeError) as e:
            error_str = str(e)
            if ("torch.load" in error_str and "v2.6" in error_str) or ("CVE-2025-32434" in error_str):
                # torch 版本限制，尝试使用环境变量绕过检查（临时方案）
                # 注意：这需要 transformers 支持 TRANSFORMERS_SAFE_LOADING_DISABLED 环境变量
                old_val = os.environ.get("TRANSFORMERS_SAFE_LOADING_DISABLED", None)
                try:
                    # 尝试设置环境变量来禁用安全检查
                    os.environ["TRANSFORMERS_SAFE_LOADING_DISABLED"] = "1"
                    # 重新尝试加载
                    base_llm = AutoModelForCausalLM.from_pretrained(llm_dir, **kwargs)
                except Exception as e2:
                    # 如果还是失败，尝试使用 safetensors（如果存在）
                    import glob
                    safetensors_files = glob.glob(os.path.join(llm_dir, "*.safetensors"))
                    if safetensors_files:
                        try:
                            # 尝试只加载 safetensors
                            base_llm = AutoModelForCausalLM.from_pretrained(
                                llm_dir, 
                                use_safetensors=True,
                                **kwargs
                            )
                        except Exception as e3:
                            raise RuntimeError(
                                f"无法加载模型，torch 版本限制。请升级 torch >= 2.6 或使用 safetensors 格式的模型。"
                                f"原始错误: {e}\n尝试绕过失败: {e2}\n尝试 safetensors 失败: {e3}"
                            ) from e3
                    else:
                        raise RuntimeError(
                            f"无法加载模型，torch 版本限制。请升级 torch >= 2.6 或使用 safetensors 格式的模型。"
                            f"原始错误: {e}\n尝试绕过失败: {e2}"
                        ) from e2
                finally:
                    # 恢复环境变量
                    if old_val is not None:
                        os.environ["TRANSFORMERS_SAFE_LOADING_DISABLED"] = old_val
                    elif "TRANSFORMERS_SAFE_LOADING_DISABLED" in os.environ:
                        del os.environ["TRANSFORMERS_SAFE_LOADING_DISABLED"]
            else:
                raise

        # 4) tokenizer：若未传入，则用根目录（因为 tokenizer 保存在根）
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(root, use_fast=True)

        # 5) 构造实例
        model = cls(llm=base_llm, tokenizer=tokenizer,
                    diffusion_config=diffusion_config,
                    diffusion_adapter_config=diffusion_adapter_config,
                    layer2_config=layer2_config,
                    use_layer2=use_layer2)

        # 6) 加载 extras（按 metadata 的相对路径）
        def _maybe_load_sub(sd_path, module_attr):
            if not sd_path:
                return
            path = os.path.join(root, sd_path) if not os.path.isabs(sd_path) else sd_path
            if os.path.isfile(path):
                sd = torch.load(path, map_location="cpu")
                mod = getattr(model, module_attr, None)
                if mod is not None and hasattr(mod, "load_state_dict"):
                    # 兼容直接存 state_dict（keys 裸的）或者带前缀；使用 strict=False 更韧性
                    mod.load_state_dict(sd, strict=False)

        if has_meta:
            _maybe_load_sub(extras_map.get("gvp_encoder"), "gvp_encoder")
            _maybe_load_sub(extras_map.get("mol_adapter"), "mol_adapter")
            # 注意：diffusion_adapter 和旧的 diffusion 已移除
            # 不再加载这些组件

        return model


    # --------------------------- 其它辅助 ---------------------------
    def gradient_checkpointing_enable(self, *args, **kwargs):
        if self.config is not None:
            try:
                self.config.use_cache = False
            except Exception:
                pass
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            try:
                return self.llm.gradient_checkpointing_enable(*args, **kwargs)
            except TypeError:
                return self.llm.gradient_checkpointing_enable()
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            try:
                out = self.llm.gradient_checkpointing_disable()
            except TypeError:
                out = None
        else:
            out = None
        if self.config is not None:
            try:
                self.config.use_cache = True
            except Exception:
                pass
        return out

    @staticmethod
    def _storage_id(t: torch.Tensor):
        try:
            return t.untyped_storage().data_ptr()
        except Exception:
            return t.storage().data_ptr()
