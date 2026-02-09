#!/usr/bin/env python
"""Debug 脚本：查看模型原始输出并测试 SMILES 提取"""
import json
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_smiles_and_thinking(text: str) -> tuple[str, str]:
    """从生成文本中提取 SMILES 和思考过程"""
    thinking = ""
    content = text
    
    # 1. 提取 <think>...</think> 思考过程
    think_pattern = r"<think>(.*?)</think>"
    think_match = re.search(think_pattern, text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        content = text[think_match.end():].strip()
    elif text.startswith("<think>"):
        thinking = text[7:].strip()
        content = ""
    
    # 2. 从 content 提取 SMILES
    smiles = ""
    if content:
        # 尝试匹配 ```smiles ... ```
        pattern = r"```smiles\s*\n?([^\n`]+)\n?```"
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            smiles = match.group(1).strip()
        else:
            # 尝试匹配任意代码块
            pattern = r"```\s*\n?([^\n`]+)\n?```"
            match = re.search(pattern, content)
            if match:
                smiles = match.group(1).strip()
            else:
                # 回退：第一行非空文本
                for line in content.strip().split("\n"):
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("```"):
                        smiles = line
                        break
    
    return smiles, thinking

# 加载模型
ckpt = "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200/llm"
print(f"Loading model from: {ckpt}")

tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    ckpt,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

# 加载测试数据（只取第一条）
data_path = "/data1/chenyuxuan/MHMLM/eval_results/data/ldmol/drug_optim/processed/test_text2smi.jsonl"
with open(data_path) as f:
    sample = json.loads(f.readline())

print("\n" + "="*60)
print("PROMPT (truncated):")
print("="*60)
print(sample["prompt"][:300] + "...")

# 构建消息（启用思考模式）
messages = [{"role": "user", "content": sample["prompt"]}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # 启用思考链
)

# 生成
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,  # 足够长以完成思考
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

# 解码
raw_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("\n" + "="*60)
print("RAW OUTPUT:")
print("="*60)
print(raw_output[:2000] + "..." if len(raw_output) > 2000 else raw_output)

# 提取 SMILES 和 thinking
smiles, thinking = extract_smiles_and_thinking(raw_output)

print("\n" + "="*60)
print("EXTRACTED SMILES:")
print("="*60)
print(smiles if smiles else "(empty)")

print("\n" + "="*60)
print("EXTRACTED THINKING (truncated):")
print("="*60)
print(thinking[:500] + "..." if len(thinking) > 500 else thinking if thinking else "(empty)")

print("\n" + "="*60)
print("EXPECTED SMILES:")
print("="*60)
print(sample["ground_truth"])
