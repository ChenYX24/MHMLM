"""
数据加载模块
统一处理数据加载和格式化
"""
import os
import json
import re
import hashlib
from typing import Optional, List, Dict, Any, Callable, Tuple
from datasets import load_dataset, Dataset
import torch


def safe_to_str(x):
    """安全转换为字符串"""
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return "\n".join(safe_to_str(xx) for xx in x)
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x)

WORD_BOUNDARY_CHARS = set(" \n\t.,;:!?()[]{}")

# ==== 新增：mol span 处理相关的通用工具 ====

# “单词边界”字符：
#   - 这些字符会阻止 span 向左/向右继续扩展
#   - 注意：这里特意 **不把 '.'、'['、']' 放进去**，
#     方便把 [Na+].[Cl-]、CCO.CS(=O)C 这种离子/混合物扩成一整块
_MOL_BOUNDARY_CHARS = " \n\t,;:!?{}"
# 去掉 [] 和 {}，避免把 [Na+].[Cl-] 里的方括号剪掉
_MOL_TRIM_CHARS = "'\"`“”‘’()"

_MOL_STOPWORDS = {"smiles", "Smiles", "SMILES", "logP", "NSAIDs"}

def _looks_like_molecule(span_text: str) -> bool:
    """
    软规则判断一个 span 看起来像“分子相关实体”：
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

    # 对纯字母的情况：如果有 >=4 个字母，当成一个“像化学名”的词
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 4:
        return True

    return False


def _expand_and_merge_mol_spans(text: str, spans):
    """
    对初始的 (start, end) spans 做统一后处理：

    1) 向左/向右扩展到“单词边界”（_MOL_BOUNDARY_CHARS）
    2) 去掉两端的引号/括号（_MOL_TRIM_CHARS）
    3) 合并重叠或相邻的 spans
    4) 只保留看起来像“分子相关”的 span（_looks_like_molecule）

    Args:
        text: 原始字符串（已经去掉旧的 <mol> 标签）
        spans: List[(start, end)]，来自 token offset + 预测 label=1

    Returns:
        List[(start, end)]：处理后的 spans（按起始位置排序，不含空 span）
    """
    if not spans:
        return []

    expanded = []
    n = len(text)

    for s, e in spans:
        if s is None or e is None:
            continue
        if s >= e:
            continue

        # 1) 向左扩展，直到遇到边界字符
        while s > 0 and text[s - 1] not in _MOL_BOUNDARY_CHARS:
            s -= 1

        # 2) 向右扩展，直到遇到边界字符
        while e < n and text[e] not in _MOL_BOUNDARY_CHARS:
            e += 1

        # 3) 修剪前后的引号、括号等
        while s < e and text[s] in _MOL_TRIM_CHARS:
            s += 1
        while e > s and text[e - 1] in _MOL_TRIM_CHARS:
            e -= 1
                # 3.5) 如果最后一个字符是句尾的小数点，且后面就是空白 / 换行 / 结束 / 特殊 token，去掉这个点

        while s < e and text[e - 1] == '.':
            # e == n：就是字符串末尾
            # e < n 且后面是空白 or 换行 or '<'（通常是 <|eot_id|> 前面）就认为是句尾
            if e == n or text[e] in " \n\t<":
                e -= 1
            else:
                break

        if s < e:
            expanded.append((s, e))

    if not expanded:
        return []

    # 4) 合并重叠/相邻 spans
    expanded.sort()
    merged = []
    for s, e in expanded:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    # 5) 最后筛一遍“像分子”的 spans
    final_spans = []
    for s, e in merged:
        span_text = text[s:e]
        if _looks_like_molecule(span_text):
            final_spans.append((s, e))

    return final_spans

def _expand_spans_to_word_boundaries(
    text: str,
    spans: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    将字符级 span 扩展到“词边界”（参考独立 MLP 推理脚本的逻辑）：
    - 向左扩展直到遇到空白或标点
    - 向右扩展直到遇到空白或标点
    并对扩展后的 span 做一次合并，防止重叠/嵌套
    """
    if not spans or not isinstance(text, str):
        return spans

    expanded: List[Tuple[int, int]] = []
    n = len(text)

    for start, end in spans:
        s, e = start, end
        # 防御越界
        if s < 0:
            s = 0
        if e > n:
            e = n

        # 向左扩展，直到遇到“边界字符”
        while s > 0 and text[s - 1] not in WORD_BOUNDARY_CHARS:
            s -= 1
        # 向右扩展，直到遇到“边界字符”
        while e < n and text[e] not in WORD_BOUNDARY_CHARS:
            e += 1

        expanded.append((s, e))

    # 对扩展后的 span 再合并一次，防止重叠
    expanded.sort()
    merged: List[List[int]] = []
    for s, e in expanded:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    return [tuple(x) for x in merged]

def _save_dataset_to_jsonl(dataset: Dataset, file_path: str, is_tagged: bool = False):
    """保留旧接口以兼容调用方，但不再用于缓存。"""
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


def load_preprocessed_data(
    data_file: str,
    cache_dir: str = "./cache",
    use_cache: bool = True,
    max_samples: Optional[int] = None,
    max_message_chars: Optional[int] = None,
):
    """加载预处理后的数据（不使用缓存与重试，直接读取源文件）
    
    Args:
        data_file: 原始数据路径
        max_message_chars: 如果指定，对 messages 的内容总字符数超限的样本进行过滤
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    file_size = os.path.getsize(data_file)
    print(f"📂 Loading data from: {data_file} (size: {file_size / 1024 / 1024:.2f} MB)")
    if max_samples is not None:
        print(f"🔍 DEBUG MODE: Limiting to {max_samples} samples")
    
    data_list = []
    
    def normalize_content(content):
        """将content统一转换为字符串格式"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        text_parts.append(str(item["text"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts) if text_parts else ""
        if isinstance(content, dict):
            if "text" in content:
                return str(content["text"])
            return json.dumps(content, ensure_ascii=False)
        return str(content) if content is not None else ""
    
    # 手动读取 JSON / JSONL，避免 load_dataset 的重试与缓存
    if data_file.endswith('.jsonl'):
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "messages" in data and isinstance(data["messages"], list):
                        for msg in data["messages"]:
                            if "content" in msg:
                                msg["content"] = normalize_content(msg["content"])
                    data_list.append(data)
                except json.JSONDecodeError as je:
                    print(f"⚠️  Skipping invalid JSON at line {line_num}: {je}")
                except Exception as ex:
                    print(f"⚠️  Error processing line {line_num}: {ex}")
    else:
        with open(data_file, 'r', encoding='utf-8') as f:
            try:
                loaded = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load JSON file: {e}") from e
            if isinstance(loaded, list):
                data_iter = loaded
            else:
                data_iter = [loaded]
            for idx, data in enumerate(data_iter, 1):
                if isinstance(data, dict) and "messages" in data and isinstance(data["messages"], list):
                    for msg in data["messages"]:
                        if "content" in msg:
                            msg["content"] = normalize_content(msg["content"])
                data_list.append(data)
    
    if not data_list:
        raise ValueError(f"No valid data loaded from {data_file}")
    
    try:
        raw = Dataset.from_list(data_list)
    except Exception as e2:
        print(f"⚠️  Dataset.from_list failed: {e2}")
        normalized_list = []
        for item in data_list:
            normalized_item = {}
            for k, v in item.items():
                if k == "messages" and isinstance(v, list):
                    normalized_item[k] = v
                elif isinstance(v, (dict, list)) and k != "messages":
                    normalized_item[k] = json.dumps(v, ensure_ascii=False)
                else:
                    normalized_item[k] = v
            normalized_list.append(normalized_item)
        raw = Dataset.from_list(normalized_list)
    
    print(f"📊 Loaded {len(raw)} raw samples")
    
    if max_samples is not None and len(raw) > max_samples:
        raw = raw.select(range(max_samples))
    
    def _parse_chatml_text_to_messages(text: str):
        """
        将 ChatML/Qwen 风格 text:
        <|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n...\n<|im_end|>\n
        解析成 messages=[{role, content}, ...]
        """
        if not isinstance(text, str) or not text.strip():
            return None

        pattern = r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>"
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if not matches:
            return None

        msgs = []
        for role, content in matches:
            msgs.append({"role": role, "content": content.strip()})

        # 至少要有 user+assistant 才能做 SFT（否则没有监督信号）
        has_user = any(m["role"] == "user" and m["content"] for m in msgs)
        has_asst = any(m["role"] == "assistant" and m["content"] for m in msgs)
        if not (has_user and has_asst):
            return None
        return msgs

    def check_and_preserve_messages(example):
        """
        兼容两种输入：
        1) messages 格式（你当前 loader 的默认格式）
        2) 仅 text（ChatML/Qwen 风格），会尝试解析回 messages

        目标：
        - 尽可能保证 example["messages"] 是 list[dict]，并且含 user+assistant
        - text 统一置为 "__MESSAGES_PLACEHOLDER__"，让 load_training_data 里用 tokenizer.apply_chat_template 生成最终 text
        """
        msgs = example.get("messages", None)
        text = example.get("text", "")

        # -------- case 1: 已有 messages --------
        if isinstance(msgs, list):
            has_valid_content = False
            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                    msg["content"] = content
                if content and content.strip():
                    has_valid_content = True

            example["text"] = "__MESSAGES_PLACEHOLDER__" if has_valid_content else ""
            return example

        # -------- case 2: 没有 messages，但有 text（兼容你 text+meta 数据）--------
        if isinstance(text, str) and text.strip():
            parsed = _parse_chatml_text_to_messages(text)
            if parsed is not None:
                example["messages"] = parsed
                example["text"] = "__MESSAGES_PLACEHOLDER__"
                return example

            # 如果解析不了（比如不是 ChatML），这里不要 raise，直接置空让后续 filter 掉
            example["text"] = ""
            return example

        # -------- case 3: 两者都没有/为空 --------
        example["text"] = ""
        return example
    
    raw = raw.map(check_and_preserve_messages, num_proc=min(4, os.cpu_count() or 1))
    
    def is_valid(example):
        t = example.get("text", "")
        if t == "__MESSAGES_PLACEHOLDER__":
            return True
        return isinstance(t, str) and len(t.strip()) > 0
    
    processed = raw.filter(is_valid, num_proc=min(4, os.cpu_count() or 1))
    
    # 过滤过长对话（按 messages 总字符数）
    if max_message_chars is not None:
        def message_length_ok(example):
            msgs = example.get("messages", [])
            if not isinstance(msgs, list):
                return False
            total = 0
            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                total += len(content)
                if total > max_message_chars:
                    return False
            return True
        
        before = len(processed)
        processed = processed.filter(message_length_ok, num_proc=min(4, os.cpu_count() or 1))
        after = len(processed)
        print(f"✂️  Filtered long messages by max_message_chars={max_message_chars}: {before} -> {after}")
    
    print(f"✅ After filtering: {len(processed)} valid samples")
    
    if len(processed) == 0:
        raise ValueError(
            f"❌ No valid samples found in {data_file}!\n"
            f"   Please check:\n"
            f"   1. Data file format (should be JSONL with 'text' field)\n"
            f"   2. Text field should not be empty"
        )
    
    def ensure_text_is_string(example):
        text = example.get("text", "")
        if not isinstance(text, str):
            if isinstance(text, list):
                example["text"] = text[0] if len(text) > 0 and isinstance(text[0], str) else ""
            else:
                example["text"] = str(text) if text is not None else ""
        else:
            example["text"] = text
        return example
    
    processed = processed.map(ensure_text_is_string, num_proc=16)
    
    return processed


def format_dataset_with_offline_spans(
    batch: Dict[str, List],
    tag_text_with_classifier: Optional[Callable[[str], str]] = None,
) -> Dict[str, List]:
    """格式化数据集，可选使用离线标注"""
    texts = []
    inputs = batch.get("input", [])
    outputs = batch.get("output", [])
    
    for i in range(len(inputs)):
        user = safe_to_str(inputs[i]).strip()
        assistant = safe_to_str(outputs[i]).strip()
        # 使用通用的 "User / Assistant" 文本格式，不再绑定任何特定模型的特殊 token。
        # 具体的 chat_template 由 tokenizer 在训练/推理时决定。
        concat = f"User: {user}\n\nAssistant: {assistant}"
        
        if tag_text_with_classifier is not None:
            tagged = tag_text_with_classifier(concat)
            texts.append(tagged)
        else:
            texts.append(concat)
    
    result = {"text": texts}
    
    # 保留meta信息（如果存在）
    meta_keys = [
        "id", "dataset", "source", "task_type", "smiles", "class_label",
        "property_name", "property_symbol", "property_description",
        "unit", "target_value", "all_targets"
    ]
    for k in meta_keys:
        if k in batch:
            result[k] = batch[k]
    
    return result


def create_tag_text_function(
    tokenizer,
    llm,
    offline_token_head,
    local_rank: int,
    max_length: int = 512,
) -> Optional[Callable[[str], str]]:
    """创建文本标注函数（单文本版本，用于兼容）"""
    if offline_token_head is None:
        return None
    
    def tag_text_with_classifier(text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        try:
            # 先清除已有 <mol> 标签，避免重复嵌套
            clean = re.sub(r"</?mol>", "", text)
            enc = tokenizer(
                clean,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            input_ids = enc["input_ids"].to(local_rank)
            attn = enc["attention_mask"].to(local_rank)
            offsets = enc["offset_mapping"][0].tolist()
            
            with torch.no_grad():
                out = llm(
                    input_ids=input_ids,
                    attention_mask=attn,
                    output_hidden_states=True,
                    return_dict=True
                )
                hs = out.hidden_states[-1]  # (1, T, H)
                # 确保 dtype 匹配：获取 offline_token_head 的 dtype
                try:
                    head_dtype = next(offline_token_head.parameters()).dtype
                    if head_dtype != hs.dtype:
                        hs = hs.to(head_dtype)
                except (StopIteration, AttributeError):
                    # 如果没有参数或无法获取dtype，使用默认的float32
                    if hs.dtype != torch.float32:
                        hs = hs.to(torch.float32)
                logits = offline_token_head(hs)  # (1, T, 2)
                preds = torch.argmax(logits, dim=-1)[0].tolist()
            
            # 收集连续实体字符 span（先按 token offset 拼起来）
            spans = []
            cur = None
            for p, (s, e) in zip(preds, offsets):
                if s == e:
                    continue
                if p == 1:
                    if cur is None:
                        cur = [s, e]
                    else:
                        cur[1] = e
                else:
                    if cur is not None:
                        spans.append(tuple(cur))
                        cur = None
            if cur is not None:
                spans.append(tuple(cur))

            if not spans:
                return clean
            
            # 新增：用统一的规则扩展 + 合并 + 修剪 + 过滤
            spans = _expand_and_merge_mol_spans(clean, spans)
            if not spans:
                return clean
            
            # === special token 保护逻辑 ===
            # 这里只保护我们自己插入的 <mol> 标签，避免重复嵌套；
            # 不再依赖任何特定模型（如 Llama）的对话 header token。
            special_tokens = [
                "<mol>", "</mol>",
            ]
            
            # 找到所有特殊 token 的位置
            special_token_ranges = []
            for st in special_tokens:
                start = 0
                while True:
                    pos = clean.find(st, start)
                    if pos == -1:
                        break
                    special_token_ranges.append((pos, pos + len(st)))
                    start = pos + 1
            
            # 检查特殊 token 对之间的范围（如 <|start_header_id|>...<|end_header_id|>）
            header_pairs = []
            start_pos = 0
            while True:
                start_header = clean.find("<|start_header_id|>", start_pos)
                if start_header == -1:
                    break
                end_header = clean.find("<|end_header_id|>", start_header)
                if end_header != -1:
                    header_pairs.append((start_header, end_header + len("<|end_header_id|>")))
                    start_pos = end_header + len("<|end_header_id|>")
                else:
                    break
            
            # 过滤掉与特殊 token 重叠或在特殊 token 对之间的 spans
            filtered_spans = []
            for s, e in spans:
                is_special = False
                
                # 检查是否与任何特殊 token 重叠
                for st_start, st_end in special_token_ranges:
                    if not (e <= st_start or s >= st_end):
                        is_special = True
                        break
                
                # 检查是否在任何 header 对之间（包括边界）
                if not is_special:
                    for pair_start, pair_end in header_pairs:
                        if s >= pair_start and e <= pair_end:
                            is_special = True
                            break
                
                if not is_special:
                    filtered_spans.append((s, e))
            
            if not filtered_spans:
                return clean
            
            # 从后往前插 <mol></mol>，避免索引偏移
            tagged = clean
            for s, e in reversed(filtered_spans):
                tagged = tagged[:e] + "</mol>" + tagged[e:]
                tagged = tagged[:s] + "<mol>" + tagged[s:]
            return tagged
        except Exception:
            # 出错时保底返回原始文本
            return text
    
    return tag_text_with_classifier




def tag_text_with_smiles(text: str, smiles: Optional[str]) -> str:
    """
    基于 SMILES 匹配在文本中添加 <mol></mol> 标签
    
    Args:
        text: 原始文本
        smiles: SMILES 字符串（如果为 None，则返回原文本）
    
    Returns:
        标注后的文本
    """
    if not isinstance(text, str) or not text:
        return text
    
    if not smiles or not isinstance(smiles, str):
        return text
    
    # 移除已有的 <mol> 标签
    clean_text = re.sub(r"</?mol>", "", text)
    
    # 在文本中查找 SMILES 字符串
    # 使用正则表达式匹配，确保匹配完整的 SMILES（前后不是字母数字）
    # SMILES 可能包含特殊字符，需要转义
    escaped_smiles = re.escape(smiles)
    
    # 查找所有匹配位置
    matches = list(re.finditer(escaped_smiles, clean_text))
    
    if not matches:
        # 如果没有找到精确匹配，尝试不区分大小写
        matches = list(re.finditer(re.escape(smiles), clean_text, re.IGNORECASE))
    
    if not matches:
        return clean_text
    
    # 从后往前插入标签，避免索引偏移
    tagged_text = clean_text
    for match in reversed(matches):
        start, end = match.span()
        # 检查是否在特殊 token 内部
        # 避免在特殊 token 中插入标签
        special_tokens = [
            "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
            "<|user|>", "<|assistant|>",  # 兼容旧格式
        ]
        
        is_special = False
        for st in special_tokens:
            # 检查匹配位置是否与特殊 token 重叠
            st_start = clean_text.find(st, max(0, start - len(st)), min(len(clean_text), end + len(st)))
            if st_start != -1:
                st_end = st_start + len(st)
                # 如果匹配位置与特殊 token 有任何重叠，跳过
                if not (end <= st_start or start >= st_end):
                    is_special = True
                    break
        
        if not is_special:
            # 插入标签
            tagged_text = tagged_text[:end] + "</mol>" + tagged_text[end:]
            tagged_text = tagged_text[:start] + "<mol>" + tagged_text[start:]
    
    return tagged_text


def create_batch_tag_text_function(
    tokenizer,
    llm,
    offline_token_head,
    local_rank: int,
    max_length: int = 512,
    batch_size: int = 32,  # 默认值从 32 改为更小的值
) -> Optional[Callable[[List[str]], List[str]]]:
    """创建批量文本标注函数（更快）"""
    if offline_token_head is None:
        return None
    
    # 确保 LLM 在 eval 模式（节省内存）
    original_training_mode = llm.training
    llm.eval()
    # 保存原始 use_cache 设置
    original_use_cache = None
    if hasattr(llm.config, 'use_cache'):
        original_use_cache = llm.config.use_cache
        llm.config.use_cache = False
    
    def tag_texts_batch(texts: List[str]) -> List[str]:
        """批量处理文本标注"""
        if not texts:
            return texts
        
        results = []
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        
        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # 先清理已有的 <mol> 标签
            batch_cleaned = [re.sub(r"</?mol>", "", t) if isinstance(t, str) else "" for t in batch_texts]
            
            try:
                # 清理内存
                torch.cuda.empty_cache()
                
                # 批量编码（使用padding）
                enc = tokenizer(
                    batch_cleaned,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=max_length,
                    padding=True,
                )
                input_ids = enc["input_ids"].to(device)
                attn = enc["attention_mask"].to(device)
                offsets_list = enc["offset_mapping"]
                
                # 立即释放 CPU 上的编码结果
                del enc
                
                with torch.no_grad():
                    # 使用更节省内存的推理方式
                    out = llm(
                        input_ids=input_ids,
                        attention_mask=attn,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,  # 禁用缓存以节省内存
                    )
                    hs = out.hidden_states[-1]  # (B, T, H)
                    # 确保 dtype 匹配
                    try:
                        head_dtype = next(offline_token_head.parameters()).dtype
                        if head_dtype != hs.dtype:
                            hs = hs.to(head_dtype)
                    except (StopIteration, AttributeError):
                        if hs.dtype != torch.float32:
                            hs = hs.to(torch.float32)
                    logits = offline_token_head(hs)  # (B, T, 2)
                    preds = torch.argmax(logits, dim=-1).cpu().tolist()  # (B, T)
                
                # 清理 GPU 内存
                del out, hs, logits, input_ids, attn
                # offsets_list 保持在 CPU
                offsets_list_cpu = offsets_list
                torch.cuda.empty_cache()
                
                # 对每个样本处理
                for j, (clean_text, pred, offsets) in enumerate(zip(batch_cleaned, preds, offsets_list_cpu)):
                    if not clean_text:
                        results.append(batch_texts[j])
                        continue
                    
                    # 收集连续实体字符 span（token offset 层）
                    spans = []
                    cur = None
                    offsets_items = offsets if isinstance(offsets, list) else offsets.tolist()
                    for p, (s, e) in zip(pred, offsets_items):
                        if s == e:
                            continue
                        if p == 1:
                            if cur is None:
                                cur = [s, e]
                            else:
                                cur[1] = e
                        else:
                            if cur is not None:
                                spans.append(tuple(cur))
                                cur = None
                    if cur is not None:
                        spans.append(tuple(cur))
                    
                    if not spans:
                        results.append(clean_text)
                        continue
                    
                    # 新增：统一做扩展 + 合并 + 修剪 + 过滤
                    spans = _expand_and_merge_mol_spans(clean_text, spans)
                    if not spans:
                        results.append(clean_text)
                        continue
                    
                    # === special token 保护逻辑（与单样本版本保持一致） ===
                    special_tokens = [
                        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
                        "<|user|>", "<|assistant|>",  # 兼容旧格式
                    ]
                    
                    # 找到所有特殊 token 的位置
                    special_token_ranges = []
                    for st in special_tokens:
                        start = 0
                        while True:
                            pos = clean_text.find(st, start)
                            if pos == -1:
                                break
                            special_token_ranges.append((pos, pos + len(st)))
                            start = pos + 1
                    
                    # 检查特殊 token 对之间的范围（如 <|start_header_id|>...<|end_header_id|>）
                    header_pairs = []
                    start_pos = 0
                    while True:
                        start_header = clean_text.find("<|start_header_id|>", start_pos)
                        if start_header == -1:
                            break
                        end_header = clean_text.find("<|end_header_id|>", start_header)
                        if end_header != -1:
                            header_pairs.append((start_header, end_header + len("<|end_header_id|>")))
                            start_pos = end_header + len("<|end_header_id|>")
                        else:
                            break
                    
                    # 过滤掉与特殊 token 重叠或在特殊 token 对之间的 spans
                    filtered_spans = []
                    for s, e in spans:
                        is_special = False
                        
                        # 检查是否与任何特殊 token 重叠
                        for st_start, st_end in special_token_ranges:
                            if not (e <= st_start or s >= st_end):
                                is_special = True
                                break
                        
                        # 检查是否在任何 header 对之间（包括边界）
                        if not is_special:
                            for pair_start, pair_end in header_pairs:
                                if s >= pair_start and e <= pair_end:
                                    is_special = True
                                    break
                        
                        if not is_special:
                            filtered_spans.append((s, e))
                    
                    if not filtered_spans:
                        results.append(clean_text)
                        continue
                    
                    tagged = clean_text
                    for s, e in reversed(filtered_spans):
                        tagged = tagged[:e] + "</mol>" + tagged[e:]
                        tagged = tagged[:s] + "<mol>" + tagged[s:]
                    results.append(tagged)
                
                # 每个 batch 成功处理后清理内存
                torch.cuda.empty_cache()
                    
            except Exception as e:
                # 检查是否是内存相关错误
                error_msg = str(e).lower()
                is_memory_error = any(keyword in error_msg for keyword in [
                    "cuda out of memory",
                    "out of memory",
                    "cublas",
                    "cudnn",
                    "memory",
                ])
                
                if is_memory_error:
                    # 内存错误直接抛出，不进行fallback
                    print(f"❌ CUDA memory error during batch tagging (batch {i//batch_size}): {e}")
                    print(f"   Batch size: {batch_size}, Max length: {max_length}")
                    print(f"   Suggestion: Reduce offline_tagging_batch_size in config or reduce max_seq_length")
                    # 清理内存
                    torch.cuda.empty_cache()
                    raise RuntimeError(f"CUDA out of memory during offline tagging. Original error: {e}") from e
                else:
                    # 其他类型的错误也直接抛出（不再fallback）
                    print(f"❌ Batch tagging failed for batch {i//batch_size}: {e}")
                    torch.cuda.empty_cache()
                    raise RuntimeError(f"Batch tagging failed. Original error: {e}") from e
        
        # 恢复原始设置（虽然函数结束时可能不需要，但为了安全）
        if original_training_mode:
            llm.train()
        if original_use_cache is not None:
            llm.config.use_cache = original_use_cache
        
        return results
    
    return tag_texts_batch




def load_training_data(
    cfg: Dict[str, Any],
    tokenizer,
    llm,
    offline_token_head: Optional[torch.nn.Module],
    local_rank: int,
) -> tuple:
    """
    加载训练数据
    
    Returns:
        train_dataset, eval_dataset
    """
    data_cfg = cfg.get("data", {})
    dataset_path = data_cfg.get("dataset_path") or cfg.get("train", {}).get("dataset_path")
    
    if not dataset_path:
        raise ValueError("dataset_path not found in config")
    
    # 如果是相对路径，转换为绝对路径（相对于代码目录）
    if not os.path.isabs(dataset_path):
        # 获取代码目录（train_sft.py所在目录）
        code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(code_dir, dataset_path)
        dataset_path = os.path.abspath(dataset_path)
    
    print(f"📂 Using dataset path: {dataset_path}")
    
    # 加载预处理后的数据（已经是格式化后的，包含text字段）
    use_cache = cfg.get("data", {}).get("use_cache", True)
    # 支持调试模式：限制数据量（在这里做「带 seed 的随机采样」）
    max_samples = cfg.get("data", {}).get("debug_max_samples", None)
    max_tokens = cfg.get("data", {}).get("max_tokens", None)  # 按 token 数过滤超长样本
    if max_samples is not None:
        print(f"🔍 DEBUG MODE ENABLED: max_samples={max_samples}")
    # 注意：这里不再把 max_samples 传给 load_preprocessed_data，以避免总是取前 N 条
    max_message_chars = cfg.get("data", {}).get("max_message_chars", None)
    if max_message_chars is not None:
        print(f"⛔ Max message chars: {max_message_chars}")
    processed_dataset = load_preprocessed_data(
        dataset_path,
        cache_dir="./cache",
        use_cache=use_cache,
        max_samples=None,
        max_message_chars=max_message_chars,
    )

    # 如果配置了 debug_max_samples，并且数据量大于该值，则使用 seed 做一次可复现的随机采样
    if max_samples is not None and len(processed_dataset) > max_samples:
        base_seed = int(cfg.get("seed", 42))
        rank = int(os.environ.get("RANK", 0))
        # 不同 rank 使用不同 seed，避免多卡采样完全重合；但同一配置/节点下是可复现的
        shuffle_seed = base_seed + rank
        if rank == 0:
            print(f"🔀 Shuffling dataset with seed={shuffle_seed} and selecting first {max_samples} samples")
        processed_dataset = processed_dataset.shuffle(seed=shuffle_seed)
        processed_dataset = processed_dataset.select(range(max_samples))
        if rank == 0:
            print(f"✅ After debug sampling: {len(processed_dataset)} samples")
    
    # 如果数据包含 messages 字段，使用 tokenizer.apply_chat_template 转换为 text
    # 这比简单的字符串拼接更准确，能使用模型特定的 chat template
    if len(processed_dataset) > 0 and "messages" in processed_dataset[0]:
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print("🔄 Converting messages format to text using tokenizer.apply_chat_template...")
        
        def convert_messages_with_template(example):
            """使用 tokenizer.apply_chat_template 将 messages 转换为 text。
            无论 example 里是否已有 text，只要存在 messages，就统一按 chat_template 重新渲染，
            以确保格式与当前 tokenizer（例如 Mistral 的 [INST] 模板）严格一致。
            """
            if "messages" in example and isinstance(example["messages"], list):
                messages = example["messages"]
                try:
                    # 使用 tokenizer 的 chat_template 生成完整对话文本
                    formatted_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,          # 只格式化为字符串
                        add_generation_prompt=False,  # 训练时不加生成提示
                    )
                    example["text"] = formatted_text
                    # 可选：如果想节省内存，可以删除 messages 字段
                    # del example["messages"]
                except Exception as e:
                    # 如果 apply_chat_template 失败，回退到简单的 User/Assistant 拼接
                    if rank == 0 and len(str(e)) < 200:
                        print(f"⚠️  apply_chat_template failed for one sample: {e}, using fallback")
                    text_parts = []
                    for msg in messages:
                        role = msg.get("role", "").lower()
                        content = msg.get("content", "")
                        if content:
                            if role == "system":
                                continue
                            elif role == "user":
                                text_parts.append(f"User: {content}")
                            elif role == "assistant":
                                text_parts.append(f"Assistant: {content}")
                    example["text"] = "\n\n".join(text_parts) if text_parts else ""
            return example
        
        try:
            processed_dataset = processed_dataset.map(
                convert_messages_with_template,
                num_proc=min(4, os.cpu_count() or 1),
                desc="Converting messages to text with chat template"
            )
            if rank == 0:
                print("✅ Messages converted to text using chat template")
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Failed to convert messages with template: {e}")
                print("   Using data as-is (may already have text field)")
            # 如果转换失败，继续使用现有数据
    
    # 过滤超长 token 的样本（基于生成后的 text）
    if max_tokens is not None:
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print(f"✂️  Filtering samples longer than {max_tokens} tokens (tokenizer-based)")
        def token_length_ok(example):
            t = example.get("text", "")
            if not isinstance(t, str):
                return False
            # 不截断，完整计算长度
            ids = tokenizer.encode(t, add_special_tokens=True, truncation=False)
            return len(ids) <= max_tokens
        before = len(processed_dataset)
        processed_dataset = processed_dataset.filter(token_length_ok, num_proc=1)
        after = len(processed_dataset)
        if rank == 0:
            print(f"✂️  Token-length filter: {before} -> {after} samples (max_tokens={max_tokens})")
    
    # 打印第一条数据的结构
    rank = int(os.environ.get("RANK", 0))
    if rank == 0 and len(processed_dataset) > 0:
        print("\n" + "="*80)
        print("📋 First sample from dataset (after load_preprocessed_data):")
        print("="*80)
        first_sample = processed_dataset[0]
        print(f"Type: {type(first_sample)}")
        print(f"Content: {first_sample}")
        if isinstance(first_sample, dict):
            print(f"Keys: {list(first_sample.keys())}")
            for key, value in first_sample.items():
                print(f"  {key}: type={type(value)}, value={str(value)[:200]}...")
        print("="*80 + "\n")
    
    # 判断是否是 epoch1（通过检查 dataset_path 是否包含 "epoch1"）
    is_epoch1 = "epoch1" in dataset_path.lower()
    
    # 只通过实际数据判断是否已包含标签
    has_mol_tags = False
    if len(processed_dataset) > 0:
        sample = processed_dataset[0]
        sample_text = sample.get("text", "")
        has_mol_tags = ("<mol>" in sample_text) and ("</mol>" in sample_text)
        if has_mol_tags:
            print(f"✅ Data contains <mol> tags, but cache metadata not found")

    # 判断是否需要标注
    is_already_tagged = has_mol_tags
    need_tagging = not is_already_tagged
    tagged_cache_file = None
    rank = int(os.environ.get("RANK", 0))

    if is_already_tagged:
        # 数据已标注：直接使用，不需要重新标注
        if rank == 0:
            print("✅ Data is already tagged, skipping tagging step")
        need_tagging = False
    else:
        # 数据未标注：删除所有可能存在的 <mol> / </mol> 标签，准备标注
        if rank == 0:
            print("🧹 Removing any existing <mol></mol> tags before tagging")

        def strip_mol_tags(ex):
            try:
                t = ex.get("text", "")
                if isinstance(t, str):
                    ex["text"] = re.sub(r"</?mol>", "", t)
                elif isinstance(t, list):
                    # 如果是列表，处理每个元素
                    ex["text"] = [re.sub(r"</?mol>", "", str(item)) if isinstance(item, str) else str(item) for item in t]
                else:
                    # 其他类型，转换为字符串后处理
                    ex["text"] = re.sub(r"</?mol>", "", str(t)) if t is not None else ""
                return ex
            except Exception as e:
                print(f"⚠️  Error in strip_mol_tags: {e}, example keys: {list(ex.keys()) if isinstance(ex, dict) else 'N/A'}")
                return ex

        try:
            processed_dataset = processed_dataset.map(
                strip_mol_tags,
                num_proc=min(16, os.cpu_count() or 1),
                desc="Stripping any existing <mol> tags",
            )
            if rank == 0:
                print(f"✅ Stripped <mol> tags, dataset size: {len(processed_dataset)}")
        except Exception as e:
            print(f"❌ Failed to strip <mol> tags: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 不再使用 tagged 缓存
        tagged_cache_file = None
    
    if not need_tagging:
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print("✅ Data already contains <mol> tags, skipping tagging")
    else:
        # 对于 epoch1，使用 SMILES 匹配方法（更快，不需要 LLM 推理）
        if is_epoch1:
            rank = int(os.environ.get("RANK", 0))
            if rank == 0:
                print("🔄 Applying SMILES-based tagging for epoch1 data...")
            
            def tag_with_smiles(example):
                """使用 SMILES 匹配添加标签"""
                text = example.get("text", "")
                smiles = example.get("smiles", None)
                if text and smiles:
                    example["text"] = tag_text_with_smiles(text, smiles)
                return example
            
            # 批量处理（使用 map，可以并行）
            processed_dataset = processed_dataset.map(
                tag_with_smiles,
                num_proc=min(4, os.cpu_count() or 1),
                desc="Tagging with SMILES"
            )
            
            if rank == 0:
                print("✅ SMILES-based tagging completed")
                # 保存tagged数据到缓存（标记为已标注）
                if use_cache and tagged_cache_file:
                    print(f"💾 Saving tagged data to cache: {tagged_cache_file}")
                    os.makedirs(os.path.dirname(tagged_cache_file) if os.path.dirname(tagged_cache_file) else ".", exist_ok=True)
                    _save_dataset_to_jsonl(processed_dataset, tagged_cache_file, is_tagged=True)
                    print(f"✅ Tagged cache saved ({len(processed_dataset)} samples, is_tagged=True)")
        
        # 对于 epoch2 或其他情况，使用离线标注（LLM + token classifier）
        elif cfg.get("train", {}).get("use_offline_spans", False):
            if offline_token_head is None:
                if rank == 0:
                    print("⚠️  use_offline_spans=True but offline_token_head is None")
                    print("   This might happen if token classifier failed to load")
                    print("   Will skip offline tagging and use data as-is")
                # 如果 offline_token_head 为 None，跳过标注，直接使用数据
                need_tagging = False
            else:
                # 检查是否在 DDP 环境下
                rank = int(os.environ.get("RANK", 0))
                world_size = int(os.environ.get("WORLD_SIZE", 1))
                is_distributed = world_size > 1
                # 在 DDP 环境下，将数据分片，每个进程处理自己的分片，然后同步
                # 使用更小的 max_length 用于 offline tagging 以节省内存（可以小于训练时的 max_seq_length）
                training_max_length = cfg.get("train", {}).get("max_seq_length", 2048)
                # offline tagging 时使用更小的长度以节省内存（默认是训练长度的一半，最小 512）
                max_length = cfg.get("train", {}).get("offline_tagging_max_length", None)
                if max_length is None:
                    max_length = max(512, training_max_length // 2)  # 默认使用训练长度的一半，最小 512
                batch_size = cfg.get("train", {}).get("offline_tagging_batch_size", 32)
                
                if is_distributed:
                    # 计算每个进程的数据分片
                    total_size = len(processed_dataset)
                    chunk_size = total_size // world_size
                    start_idx = rank * chunk_size
                    end_idx = start_idx + chunk_size if rank < world_size - 1 else total_size
                    
                    print(f"🔄 Applying offline tagging to add <mol> tags... (rank {rank}/{world_size-1}, processing samples {start_idx}-{end_idx-1})")
                    print(f"   Using max_length={max_length} (training max_length={training_max_length}), batch_size={batch_size}")
                    
                    # 确保 LLM 在 eval 模式并清理内存
                    llm.eval()
                    torch.cuda.empty_cache()
                    
                    # 选择当前进程的数据分片
                    processed_dataset_shard = processed_dataset.select(range(start_idx, end_idx))
                    
                    # 使用批量处理函数（真正的批量推理，更快）
                    batch_tag_func = create_batch_tag_text_function(
                        tokenizer, llm, offline_token_head, local_rank, max_length, batch_size
                    )
                    
                    if batch_tag_func is not None:
                        def apply_tagging_batch(batch):
                            """批量处理标注（真正的批量推理）"""
                            texts = batch.get("text", [])
                            if not texts:
                                return batch
                            
                            # 确保 texts 是字符串列表
                            if isinstance(texts, list) and len(texts) > 0:
                                # 检查第一个元素是否是字符串
                                if not isinstance(texts[0], str):
                                    # 如果不是字符串，尝试转换
                                    texts = [str(t) if t is not None else "" for t in texts]
                            elif not isinstance(texts, list):
                                # 如果不是列表，转换为列表
                                texts = [str(texts)] if texts else []
                            
                            # 批量处理
                            tagged_texts = batch_tag_func(texts)
                            # 确保返回的是列表
                            if not isinstance(tagged_texts, list):
                                tagged_texts = [tagged_texts] if tagged_texts else []
                            batch["text"] = tagged_texts
                            return batch
                        
                        print(f"   Using batch size: {batch_size} (batch inference enabled)")
                        processed_dataset_shard = processed_dataset_shard.map(
                            apply_tagging_batch,
                            batched=True,
                            batch_size=batch_size,
                            num_proc=1,  # 单进程避免CUDA问题
                        )
                        # 清理 GPU 内存
                        torch.cuda.empty_cache()
                        print(f"✅ Offline tagging completed for shard {rank} ({len(processed_dataset_shard)} samples)")
                    
                    # 同步所有进程，确保所有分片都处理完成
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.barrier()
                        print(f"✅ All processes completed offline tagging (rank {rank})")
                    
                    # 每个进程保存自己的分片到临时文件，然后让 rank 0 合并保存到缓存
                    if use_cache and tagged_cache_file:
                        # 每个进程保存自己的分片到临时文件
                        shard_cache_file = tagged_cache_file.replace(".jsonl", f"_shard_{rank}.jsonl")
                        os.makedirs(os.path.dirname(shard_cache_file) if os.path.dirname(shard_cache_file) else ".", exist_ok=True)
                        _save_dataset_to_jsonl(processed_dataset_shard, shard_cache_file, is_tagged=True)
                        print(f"💾 Rank {rank}: Saved shard to {shard_cache_file} ({len(processed_dataset_shard)} samples)")
                        
                        # 同步，确保所有分片都已保存
                        if dist.is_initialized():
                            dist.barrier()
                        
                        # rank 0 负责收集所有分片并合并保存到最终的 tagged cache
                        if rank == 0:
                            print(f"💾 Rank 0: Collecting all shards and saving to cache...")
                            all_shards = []
                            for r in range(world_size):
                                shard_file = tagged_cache_file.replace(".jsonl", f"_shard_{r}.jsonl")
                                if os.path.exists(shard_file):
                                    shard_dataset = load_dataset("json", data_files=shard_file, cache_dir="./cache", split="train", streaming=False)
                                    all_shards.append(shard_dataset)
                                    print(f"   Loaded shard {r}: {len(shard_dataset)} samples")
                            
                            if all_shards:
                                # 合并所有分片
                                from datasets import concatenate_datasets
                                merged_dataset = concatenate_datasets(all_shards)
                                print(f"   Merged {len(merged_dataset)} samples from {len(all_shards)} shards")
                                
                                # 保存到最终的 tagged cache
                                _save_dataset_to_jsonl(merged_dataset, tagged_cache_file, is_tagged=True)
                                print(f"✅ Tagged cache saved: {tagged_cache_file} ({len(merged_dataset)} samples, is_tagged=True)")
                                
                                # 清理临时分片文件
                                for r in range(world_size):
                                    shard_file = tagged_cache_file.replace(".jsonl", f"_shard_{r}.jsonl")
                                    if os.path.exists(shard_file):
                                        try:
                                            os.remove(shard_file)
                                            meta_file = shard_file + ".meta"
                                            if os.path.exists(meta_file):
                                                os.remove(meta_file)
                                        except Exception as e:
                                            print(f"⚠️  Failed to remove shard file {shard_file}: {e}")
                        
                        # 再次同步，确保 rank 0 完成保存
                        if dist.is_initialized():
                            dist.barrier()
                        
                        # 如果缓存已保存，所有进程都从缓存加载完整数据集
                        # 这样每个进程都能看到完整的数据，训练步数才会正确
                        if os.path.exists(tagged_cache_file):
                            print(f"📂 Reloading full tagged dataset from cache for all processes... (rank {rank})")
                            cached_full = load_dataset("json", data_files=tagged_cache_file, cache_dir="./cache", split="train", streaming=False)
                            print(f"✅ Loaded full dataset from cache: {len(cached_full)} samples (rank {rank})")
                            processed_dataset = cached_full
                        else:
                            # 如果缓存保存失败，使用分片（fallback）
                            print(f"⚠️  Cache file not found after saving, using shard for rank {rank} ({len(processed_dataset_shard)} samples)")
                            processed_dataset = processed_dataset_shard
                    else:
                        # 如果没有保存缓存，使用分片
                        print(f"✅ Using processed shard for rank {rank} ({len(processed_dataset_shard)} samples)")
                        print(f"   Note: Cache not saved, using shard. DataLoader will handle data distribution.")
                        processed_dataset = processed_dataset_shard
                else:
                    # 单进程模式，处理全部数据
                    print(f"🔄 Applying offline tagging to add <mol> tags...")
                    
                    # 确保 LLM 在 eval 模式并清理内存
                    llm.eval()
                    torch.cuda.empty_cache()
                    
                    # 使用批量处理函数（真正的批量推理，更快）
                    batch_tag_func = create_batch_tag_text_function(
                        tokenizer, llm, offline_token_head, local_rank, max_length, batch_size
                    )
                    
                    if batch_tag_func is not None:
                        def apply_tagging_batch(batch):
                            """批量处理标注（真正的批量推理）"""
                            texts = batch.get("text", [])
                            if not texts:
                                return batch
                            
                            # 确保 texts 是字符串列表
                            if isinstance(texts, list) and len(texts) > 0:
                                # 检查第一个元素是否是字符串
                                if not isinstance(texts[0], str):
                                    # 如果不是字符串，尝试转换
                                    texts = [str(t) if t is not None else "" for t in texts]
                            elif not isinstance(texts, list):
                                # 如果不是列表，转换为列表
                                texts = [str(texts)] if texts else []
                            
                            # 批量处理
                            tagged_texts = batch_tag_func(texts)
                            # 确保返回的是列表
                            if not isinstance(tagged_texts, list):
                                tagged_texts = [tagged_texts] if tagged_texts else []
                            batch["text"] = tagged_texts
                            return batch
                        
                        print(f"   Using batch size: {batch_size} (batch inference enabled)")
                        processed_dataset = processed_dataset.map(
                            apply_tagging_batch,
                            batched=True,
                            batch_size=batch_size,
                            num_proc=1,  # 单进程避免CUDA问题
                        )
                        print("✅ Offline tagging completed")
                        # 保存tagged数据到缓存（标记为已标注）
                        if use_cache and tagged_cache_file:
                            print(f"💾 Saving tagged data to cache: {tagged_cache_file}")
                            os.makedirs(os.path.dirname(tagged_cache_file) if os.path.dirname(tagged_cache_file) else ".", exist_ok=True)
                            _save_dataset_to_jsonl(processed_dataset, tagged_cache_file, is_tagged=True)
                            print(f"✅ Tagged cache saved ({len(processed_dataset)} samples, is_tagged=True)")
    
    # 划分训练集和验证集
    eval_split = cfg.get("train", {}).get("eval_split", 0.05)
    split = processed_dataset.train_test_split(
        test_size=eval_split,
        seed=cfg.get("seed", 42)
    )
    
    train_size = len(split["train"])
    eval_size = len(split["test"])
    print(f"📈 Dataset split: {train_size} train, {eval_size} eval (split={eval_split})")
    
    if train_size == 0:
        raise ValueError(
            f"❌ Training dataset is empty after splitting!\n"
            f"   Total samples: {len(processed_dataset)}\n"
            f"   Eval split: {eval_split}\n"
            f"   This might happen if eval_split is too large or dataset is too small"
        )
    
    # 打印最终返回的train_dataset的第一条数据
    rank = int(os.environ.get("RANK", 0))
    if rank == 0 and len(split["train"]) > 0:
        print("\n" + "="*80)
        print("📋 First sample from train_dataset (final, before return):")
        print("="*80)
        first_train_sample = split["train"][0]
        print(f"Type: {type(first_train_sample)}")
        print(f"Content: {first_train_sample}")
        if isinstance(first_train_sample, dict):
            print(f"Keys: {list(first_train_sample.keys())}")
            for key, value in first_train_sample.items():
                if key == "text":
                    print(f"  {key}: type={type(value)}, length={len(str(value))}, preview={str(value)[:200]}...")
                else:
                    print(f"  {key}: type={type(value)}, value={str(value)[:200]}...")
        print("="*80 + "\n")
    
    return split["train"], split["test"]


def compute_qm9_stats_from_dataset(dataset) -> tuple:
    """从数据集中计算QM9统计信息"""
    tasks = ["mu", "alpha", "homo", "lumo", "gap"]
    sums = [0.0] * len(tasks)
    sqs = [0.0] * len(tasks)
    cnt = 0
    
    for ex in dataset:
        if ex.get("dataset") != "QM9" or ex.get("task_type") != "regression":
            continue
        at = ex.get("all_targets")
        if at is None:
            continue
        cnt += 1
        for i, t in enumerate(tasks):
            val = float(at.get(t, 0.0))
            sums[i] += val
            sqs[i] += val ** 2
    
    if cnt == 0:
        return None, None
    
    means = [s / cnt for s in sums]
    vars_ = [sq / cnt - m ** 2 for sq, m in zip(sqs, means)]
    stds = [max(1e-8, v) ** 0.5 for v in vars_]
    
    return means, stds


def clean_cached_data(cache_file: str, output_file: Optional[str] = None):
    """
    清理缓存数据中的错误标注
    
    修复以下问题：
    1. 移除特殊 token 中的 <mol> 标签（如 <|start_header_id|><mol>assistant</mol><|end_header_id|>）
    2. 清理 "the question is" 和 "the answer is" 前缀，转换为标准格式
    
    Args:
        cache_file: 缓存文件路径
        output_file: 输出文件路径（如果为 None，则覆盖原文件）
    """
    if not os.path.exists(cache_file):
        print(f"❌ Cache file not found: {cache_file}")
        return

    print(f"📂 Loading cache file: {cache_file}")
    dataset = load_dataset("json", data_files=cache_file, cache_dir="./cache", split="train", streaming=False)
    print(f"📊 Loaded {len(dataset)} samples")
    
    def clean_text(text: str) -> str:
        """清理文本中的错误标注"""
        if not isinstance(text, str):
            return text
        
        # 1. 移除特殊 token 中的 <mol> 标签（仅对旧的 Llama 3.2 格式缓存生效）
        # 修复 <|start_header_id|><mol>assistant</mol><|end_header_id|> -> <|start_header_id|>assistant<|end_header_id|>
        if "<|start_header_id|>" in text and "<|end_header_id|>" in text:
            text = re.sub(
                r'<\|start_header_id\|><mol>(assistant|user)</mol><\|end_header_id\|>',
                r'<|start_header_id|>\1<|end_header_id|>',
                text
            )
        
        # 2. 清理 "the question is" 和 "the answer is" 前缀
        # 如果文本中已经包含标准格式，但还有这些前缀，需要移除前缀并保留实际内容
        if "<|start_header_id|>assistant<|end_header_id|>" in text:
            # 如果已经是标准格式，查找 assistant 部分中的 "the question is" 和 "the answer is"
            # 提取 assistant 部分
            assistant_match = re.search(
                r'<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n\s*\n(.*?)(?:\s*<\|eot_id\|>|$)',
                text,
                re.DOTALL
            )
            if assistant_match:
                assistant_content = assistant_match.group(1)
                # 检查是否包含 "the question is" 和 "the answer is"
                pattern = r'the\s+question\s+is\s+(.+?)\s*,\s*the\s+answer\s+is\s+(.+?)(?:\s*<\|eot_id\|>|$)'
                match = re.search(pattern, assistant_content, re.IGNORECASE | re.DOTALL)
                if match:
                    # 提取实际的 answer 内容（忽略 question 部分，因为 question 已经在 user 部分了）
                    answer = match.group(2).strip()
                    # 替换 assistant 部分，移除 "the question is ... , the answer is" 前缀，只保留 answer
                    # 使用字符串替换而不是正则表达式，避免转义问题
                    start_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    end_marker = "<|eot_id|>"
                    start_idx = text.find(start_marker)
                    if start_idx != -1:
                        start_idx += len(start_marker)
                        end_idx = text.find(end_marker, start_idx)
                        if end_idx != -1:
                            # 替换 assistant 内容
                            text = text[:start_idx] + answer + text[end_idx:]
        else:
            # 如果没有标准格式，尝试从 "the question is" 和 "the answer is" 构建标准格式
            pattern = r'the\s+question\s+is\s+(.+?)\s*,\s*the\s+answer\s+is\s+(.+?)(?:\s*<\|eot_id\|>|$)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                # 重新格式化为标准的 Llama 3.2 格式
                text = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        
        return text
    
    def clean_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """清理单个样本"""
        if "text" in example:
            example["text"] = clean_text(example["text"])
        return example
    
    print("🧹 Cleaning cached data...")
    cleaned_dataset = dataset.map(clean_example, num_proc=min(4, os.cpu_count() or 1))
    
    # 保存清理后的数据
    if output_file is None:
        output_file = cache_file
    
    print(f"💾 Saving cleaned data to: {output_file}")
    _save_dataset_to_jsonl(cleaned_dataset, output_file, is_tagged=True)
    print(f"✅ Cleaned cache saved ({len(cleaned_dataset)} samples)")


if __name__ == "__main__":
    """
    命令行工具：清理缓存数据
    
    用法：
        python -m modules.data_loader <cache_file> [output_file]
    """
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m modules.data_loader <cache_file> [output_file]")
        print("Example: python -m modules.data_loader ./cache/epoch2_preprocessed_tagged_offline_fa392044.jsonl")
        sys.exit(1)
    
    cache_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    clean_cached_data(cache_file, output_file)

