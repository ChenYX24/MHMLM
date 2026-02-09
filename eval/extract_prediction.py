#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 raw_output 中提取预测结果的通用函数
支持多种格式：<|im_start|>assistant, <think>, <think> 等
"""

import re
from typing import Optional


# 任务类型定义（需要与评分脚本中的定义一致）
SMILES_TOKEN_RE = re.compile(r"([A-Za-z0-9@+\-\[\]\(\)=#\\/%.]+)")
FORMULA_TOKEN_RE = re.compile(r"([A-Za-z0-9\(\)\.\+\-]+)")
NUMBER_TOKEN_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
BOOL_TOKEN_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _canonical_bool(text: str) -> str:
    """将文本规范化为 Yes/No，支持多种格式"""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    
    # 移除末尾的标点
    text = text.rstrip('.,;:!?')
    
    # 映射表 - 正面值（Yes/True/Positive/Toxic）
    yes_values = {"yes", "y", "true", "t", "1", "positive", "toxic", "unsafe", "harmful"}
    # 映射表 - 负面值（No/False/Negative/Non-toxic）
    no_values = {"no", "n", "false", "f", "0", "negative", "non-toxic", "non toxic", "nontoxic", "safe", "non-harmful"}
    
    # 直接匹配
    if text in yes_values:
        return "Yes"
    elif text in no_values:
        return "No"
    
    # 正则匹配（处理包含额外文本的情况）
    m = BOOL_TOKEN_RE.search(text)
    if m:
        v = m.group(1).lower()
        if v in ("yes", "true", "toxic", "unsafe"):
            return "Yes"
        elif v in ("no", "false", "non-toxic", "safe"):
            return "No"
    
    # 尝试匹配toxic相关词汇
    if "toxic" in text and "non" not in text and "not" not in text:
        return "Yes"
    elif "non-toxic" in text or "nontoxic" in text or ("non" in text and "toxic" in text):
        return "No"
    
    return ""


def _extract_core_answer(text: str, task_name: str, text_tasks: set, smiles_tasks: set, 
                         formula_element_tasks: set, formula_split_tasks: set,
                         number_tasks: set, boolean_tasks: set) -> str:
    """从清理后的文本中提取核心答案"""
    if text is None:
        return ""
    text = str(text)

    if task_name in text_tasks:
        return text.strip()

    if task_name in smiles_tasks:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = SMILES_TOKEN_RE.search(line)
            if m:
                return m.group(1)
        return text.strip()

    if task_name in formula_element_tasks or task_name in formula_split_tasks:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = FORMULA_TOKEN_RE.search(line)
            if m:
                return m.group(1)
        return text.strip()

    if task_name in number_tasks:
        m = NUMBER_TOKEN_RE.search(text)
        if m:
            return m.group(0)
        return text.strip()

    if task_name in boolean_tasks:
        return _canonical_bool(text)

    # 默认：返回第一行非空内容
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


def get_special_tokens_from_tokenizer(tokenizer) -> list:
    """
    从tokenizer获取所有special tokens
    返回: list of special token strings
    """
    special_tokens = []
    
    # 获取常见的special tokens
    for attr in ['eos_token', 'bos_token', 'pad_token', 'unk_token', 'sep_token', 'cls_token']:
        token = getattr(tokenizer, attr, None)
        if token and isinstance(token, str):
            special_tokens.append(token)
    
    # 获取额外的special tokens（如果有）
    if hasattr(tokenizer, 'additional_special_tokens') and tokenizer.additional_special_tokens:
        special_tokens.extend(tokenizer.additional_special_tokens)
    
    # 移除None值和重复
    special_tokens = [t for t in special_tokens if t]
    special_tokens = list(set(special_tokens))
    
    return special_tokens


def get_assistant_marker_from_tokenizer(tokenizer) -> list:
    """
    从tokenizer的chat_template中获取assistant标记
    返回: list of possible assistant markers (e.g., ["<|im_start|>assistant", "[/INST]"])
    """
    markers = []
    
    # 尝试从chat_template推断assistant标记
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            # 测试chat_template，看看assistant标记是什么
            test_messages = [
                {"role": "system", "content": "test"},
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": ""}
            ]
            template_result = tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 查找assistant相关的标记
            if "<|im_start|>assistant" in template_result:
                markers.append("<|im_start|>assistant")
            if "[/INST]" in template_result:
                markers.append("[/INST]")
            if "<|start_header_id|>assistant<|end_header_id|>" in template_result:
                markers.append("<|start_header_id|>assistant<|end_header_id|>")
        except Exception:
            pass
    
    # 如果没有找到，使用常见的默认值
    if not markers:
        markers = ["<|im_start|>assistant", "[/INST]"]
    
    return markers


def remove_special_tokens(text: str, special_tokens: list = None) -> str:
    """
    移除所有special tokens，返回清理后的文本
    
    Args:
        text: 输入文本
        special_tokens: special tokens列表（如果为None，使用默认的常见tokens）
    """
    if not text:
        return ""
    
    text = str(text)
    
    if special_tokens:
        # 使用提供的special tokens
        for token in special_tokens:
            if token:
                # 转义特殊字符以便用于regex
                escaped_token = re.escape(token)
                text = re.sub(escaped_token, "", text)
    else:
        # 如果没有提供，使用默认的常见tokens（向后兼容）
        text = re.sub(r"<\|endoftext\|>", "", text)
        text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text, flags=re.DOTALL)
        text = re.sub(r"<\|im_start\|>", "", text)
        text = re.sub(r"<\|im_end\|>", "", text)
        text = re.sub(r"<\|eot_id\|>", "", text)
        text = re.sub(r"</s>", "", text)
        text = re.sub(r"<s>", "", text)
        text = re.sub(r"\[/INST\]", "", text)
        text = re.sub(r"\[INST\]", "", text)
    
    # 清理多余空白
    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.strip()
    
    return text


def extract_answer_only(raw_output: str, assistant_markers: list = None, special_tokens: list = None) -> str:
    """
    从raw_output中提取只有answer的版本（从assistant部分提取，移除think标签）
    
    Args:
        raw_output: 原始输出文本
        assistant_markers: assistant标记列表（如果为None，使用默认的常见标记）
        special_tokens: special tokens列表（用于清理）
    
    返回: 只有答案的文本，已移除think标签和special tokens
    """
    if not raw_output:
        return ""
    
    text = str(raw_output)
    
    # 如果没有提供assistant_markers，使用默认值
    if not assistant_markers:
        assistant_markers = ["<|im_start|>assistant", "[/INST]", "<|start_header_id|>assistant<|end_header_id|>", "\nassistant\n", "assistant\n"]
    
    # 1. 提取assistant部分
    assistant_text = ""
    found_marker = False
    
    for marker in assistant_markers:
        if marker in text:
            if marker == "[/INST]":
                assistant_text = text.split("[/INST]")[-1]
            elif marker == "<|start_header_id|>assistant<|end_header_id|>":
                # 查找这个标记之后的内容
                marker_start = text.find(marker)
                if marker_start >= 0:
                    after_marker = text[marker_start + len(marker):]
                    # 查找下一个header或结束标记
                    next_header = after_marker.find("<|start_header_id|>")
                    if next_header >= 0:
                        assistant_text = after_marker[:next_header]
                    else:
                        assistant_text = after_marker
            elif marker in ("\nassistant\n", "assistant\n"):
                # 处理简单的 "assistant" 标记（Intern-S1等模型使用）
                # 支持 "\nassistant\n" 和 "assistant\n" 两种格式
                marker_start = text.find(marker)
                if marker_start >= 0:
                    assistant_text = text[marker_start + len(marker):]
            else:
                # 处理类似 "<|im_start|>assistant" 的标记
                marker_start = text.find(marker)
                if marker_start >= 0:
                    after_marker = text[marker_start + len(marker):]
                    after_marker = after_marker.lstrip()
                    # 查找结束标记
                    if "<|im_end|>" in after_marker:
                        assistant_text = after_marker.split("<|im_end|>")[0]
                    elif "<|im_start|>" in after_marker:
                        assistant_text = after_marker.split("<|im_start|>")[0]
                    else:
                        assistant_text = after_marker
            found_marker = True
            break
    
    if not found_marker:
        assistant_text = text
    
    assistant_text = assistant_text.lstrip()
    
    # 2. 移除think标签（包括内容和标签本身）
    # 支持多种think标签格式（这些是从tokenizer无法获取的，所以保留写死）
    think_tags = [
        ("<think>", "</think>"),  # 标准think标签
        ("<thinking>", "</thinking>"),  # thinking标签
    ]
    
    answer_text = assistant_text
    for open_tag, close_tag in think_tags:
        # 先移除所有think标签对（包括内容）
        pattern = re.escape(open_tag) + r"(.*?)" + re.escape(close_tag)
        answer_text = re.sub(pattern, "", answer_text, flags=re.DOTALL)
        # 处理没有闭合标签的情况：移除从open_tag到文本末尾的所有内容（如果close_tag不存在）
        # 但只在open_tag存在且close_tag不存在时执行
        if open_tag in answer_text and close_tag not in answer_text:
            # 找到最后一个open_tag的位置
            last_open = answer_text.rfind(open_tag)
            if last_open >= 0:
                # 移除从open_tag开始到文本末尾的所有内容
                answer_text = answer_text[:last_open]
        # 移除单独的think标签（防止残留）
        answer_text = re.sub(re.escape(open_tag), "", answer_text)
        answer_text = re.sub(re.escape(close_tag), "", answer_text)
    
    # 3. 移除special tokens
    answer_text = remove_special_tokens(answer_text, special_tokens)
    
    return answer_text.strip()


def extract_prediction_from_raw(
    raw_output: str,
    task_name: str,
    text_tasks: set = None,
    smiles_tasks: set = None,
    formula_element_tasks: set = None,
    formula_split_tasks: set = None,
    number_tasks: set = None,
    boolean_tasks: set = None,
    answer_only: str = None,  # 如果提供了answer_only，直接使用它
) -> str:
    """
    从 raw_output 或 answer_only 中提取预测结果
    
    规则:
    1. 如果提供了answer_only，直接使用它（优先）
    2. 否则从raw_output中提取：先提取assistant部分，优先使用think之外的内容
    3. 移除所有无效标签和special tokens
    
    Args:
        raw_output: 原始输出文本
        task_name: 任务名称
        text_tasks: 文本任务集合
        smiles_tasks: SMILES任务集合
        formula_element_tasks: 公式元素匹配任务集合
        formula_split_tasks: 公式分割匹配任务集合
        number_tasks: 数值任务集合
        boolean_tasks: 布尔任务集合
        answer_only: 只包含answer的文本（可选，如果提供则优先使用）
    """
    # 如果提供了answer_only，直接使用它，不需要进一步处理
    if answer_only is not None and answer_only.strip():
        text = answer_only.strip()
        # 提供默认任务集合（如果未指定）
        if text_tasks is None:
            text_tasks = {"molecule_captioning"}
        if smiles_tasks is None:
            smiles_tasks = {"forward_synthesis", "retrosynthesis", "molecule_generation", "name_conversion-i2s"}
        if formula_element_tasks is None:
            formula_element_tasks = {"name_conversion-i2f", "name_conversion-s2f"}
        if formula_split_tasks is None:
            formula_split_tasks = {"name_conversion-s2i"}
        if number_tasks is None:
            number_tasks = {"property_prediction-esol", "property_prediction-lipo"}
        if boolean_tasks is None:
            boolean_tasks = {"property_prediction-bbbp", "property_prediction-clintox", 
                            "property_prediction-hiv", "property_prediction-sider"}
        # 直接对answer_only进行任务类型提取
        return _extract_core_answer(
            text, task_name, text_tasks, smiles_tasks,
            formula_element_tasks, formula_split_tasks,
            number_tasks, boolean_tasks
        )
    
    # 如果没有提供answer_only，从raw_output中提取
    if not raw_output:
        return ""
    
    # 提供默认任务集合（如果未指定）
    if text_tasks is None:
        text_tasks = {"molecule_captioning"}
    if smiles_tasks is None:
        smiles_tasks = {"forward_synthesis", "retrosynthesis", "molecule_generation", "name_conversion-i2s"}
    if formula_element_tasks is None:
        formula_element_tasks = {"name_conversion-i2f", "name_conversion-s2f"}
    if formula_split_tasks is None:
        formula_split_tasks = {"name_conversion-s2i"}
    if number_tasks is None:
        number_tasks = {"property_prediction-esol", "property_prediction-lipo"}
    if boolean_tasks is None:
        boolean_tasks = {"property_prediction-bbbp", "property_prediction-clintox", 
                        "property_prediction-hiv", "property_prediction-sider"}
    
    text = str(raw_output)
    
    # 1. 首先提取 <|im_start|>assistant 之后的内容（或 [/INST] 之后）
    assistant_text = ""
    if "[/INST]" in text:
        assistant_text = text.split("[/INST]")[-1]
    elif "<|im_start|>assistant" in text:
        # 找到assistant开始位置
        assistant_start = text.find("<|im_start|>assistant")
        if assistant_start >= 0:
            # 提取assistant标签之后的内容
            after_assistant = text[assistant_start + len("<|im_start|>assistant"):]
            # 如果后面有换行或空格，跳过它们
            after_assistant = after_assistant.lstrip()
            # 如果遇到 <|im_end|> 或 <|im_start|>user，截断
            if "<|im_end|>" in after_assistant:
                assistant_text = after_assistant.split("<|im_end|>")[0]
            elif "<|im_start|>" in after_assistant:
                assistant_text = after_assistant.split("<|im_start|>")[0]
            else:
                assistant_text = after_assistant
    else:
        # 如果没有找到assistant标签，使用整个文本
        assistant_text = text
    
    # 移除开头的换行和空白
    assistant_text = assistant_text.lstrip()
    
    # 2. 处理 think 相关标签（<think> 或 <think>）
    # 策略：
    #   a) 先移除所有think标签，提取think之外的内容
    #   b) 如果think之外没有内容，才提取think标签内的内容
    think_tags = [
        ("<think>", "</think>"),  # 标准think标签
        ("<think>", "</think>"),  # redacted_reasoning标签
    ]
    
    # 先尝试提取think之外的内容
    text_without_think = assistant_text
    think_contents = []  # 保存think标签内的内容作为备选
    
    for open_tag, close_tag in think_tags:
        if open_tag not in text_without_think:
            continue
            
        # 查找所有think标签对
        pattern = re.escape(open_tag) + r"(.*?)" + re.escape(close_tag)
        matches = list(re.finditer(pattern, text_without_think, re.DOTALL))
        
        if matches:
            # 收集所有think标签内的内容（作为备选）
            for match in matches:
                think_content = match.group(1).strip()
                if think_content:
                    think_contents.append(think_content)
            
            # 移除所有think标签（包括内容和标签本身），保留think之外的内容
            text_without_think = re.sub(pattern, "", text_without_think, flags=re.DOTALL)
        else:
            # 如果没有闭合标签，移除开标签
            text_without_think = re.sub(re.escape(open_tag), "", text_without_think)
    
    # 清理think之外的内容
    text_without_think = text_without_think.strip()
    # 移除终止标签
    if "<|im_end|>" in text_without_think:
        text_without_think = text_without_think.split("<|im_end|>")[0].strip()
    
    # 3. 选择最终文本：优先使用think之外的内容，如果没有则使用think内的内容
    if text_without_think and len(text_without_think) > 0:
        # 使用think之外的内容
        text = text_without_think
    elif think_contents:
        # 如果think之外没有内容，使用最后一个think标签内的内容
        text = think_contents[-1]
    else:
        # 如果都没有，使用原始的assistant部分（已经移除了think标签）
        text = assistant_text
    
    # 3. 移除 <|im_end|> 及其之后的内容
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    
    # 4. 移除所有无效标签
    text = re.sub(r"<\|endoftext\|>", "", text)
    text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text)  # 移除所有 im_start...im_end 块
    text = re.sub(r"<\|im_start\|>", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    # 移除剩余的 think/redacted_reasoning 标签（如果还没提取的话）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>", "", text)
    text = re.sub(r"</think>", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>", "", text)
    text = re.sub(r"</think>", "", text)
    text = re.sub(r"</s>", "", text)
    text = re.sub(r"<s>", "", text)
    
    # 5. 清理多余空白
    text = re.sub(r"\n\s*\n", "\n", text)  # 多个换行合并为一个
    text = text.strip()
    
    # 6. 根据任务类型提取核心答案
    return _extract_core_answer(
        text, task_name, text_tasks, smiles_tasks,
        formula_element_tasks, formula_split_tasks,
        number_tasks, boolean_tasks
    )

