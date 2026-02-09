import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ========== 配置 ==========
model_path = "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200/llm"
data_path = "/data1/chenyuxuan/MHMLM/eval_results/data/ldmol/drug_optim/raw/converted_reasoning_test5_fixed_updated.jsonl"

# ========== 加载模型 ==========
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("Model loaded!")

# ========== 加载一个样本 ==========
with open(data_path, "r", encoding="utf-8") as f:
    line = f.readline()
    sample = json.loads(line)

# 构建 prompt（instruction + input）
user_content = sample["instruction"] + "\n\n" + sample["input"]
print("=" * 60)
print("User prompt :")
print(user_content)
print("=" * 60)

# ========== 生成 ==========
messages = [{"role": "user", "content": user_content}]

# 使用 chat template，禁用思考模式
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # 不启用思考链
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

# 解码生成的部分
generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("=" * 60)
print("Generated output:")
print(generated_text)
print("=" * 60)

# 显示 ground truth
print("Ground truth (from output field):")
print(sample["output"])