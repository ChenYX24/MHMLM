#!/usr/bin/env python3
"""
测试脚本：比较不同模型和推理方式的结果
- 模型：llasmol, intern-s1
- 推理方式：sft_tester, 普通推理（transformers）
- 测试所有jsonl文件的第一个prompt

使用方法：
    python test_prompts_comparison.py --model llasmol --type sft_tester
    python test_prompts_comparison.py --model llasmol --type transformers
    python test_prompts_comparison.py --model intern-s1 --type sft_tester
    python test_prompts_comparison.py --model intern-s1 --type transformers
"""

import json
import os
import sys
import argparse
import inspect
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目路径
BASE_DIR = Path("/data1/chenyuxuan/MHMLM")
sys.path.insert(0, str(BASE_DIR))

from sft_tester import MolAwareGenerator2

# 配置
TEST_DIR = BASE_DIR / "1223results_baseline/LlaSMol-Mistral-7B-merged_fewshot"
MODEL_DIR = Path("/data1/chenyuxuan/base_model")

# 模型路径配置
MODELS = {
    "llasmol": MODEL_DIR / "LlaSMol-Mistral-7B-merged",
    "intern-s1": MODEL_DIR / "Intern-S1-mini",
}

# Intern-S1 需要 base_llm_path
BASE_LLM_PATHS = {
    "llasmol": None,
    "intern-s1": MODEL_DIR / "qwen3_8b",
}

TOKEN_CLS_PATH = "/data1/lvchangwei/LLM/Lora/llama_mlp_token_classifier.pt"
OUTPUT_FILE = BASE_DIR / "1223results_baseline/prompt_comparison_results.jsonl"

# 统一后处理函数
import re

YESNO_RE = re.compile(r"\b(Yes|No)\b", re.IGNORECASE)
FLOAT_RE = re.compile(r"[-+]?\d+(\.\d+)?")

# 排除的 jsonl 文件关键字
EXCLUDE_SUBSTR = ["metrics", "eval_summary", "evaluation", "result", "score"]


def safe_apply_chat_template(tokenizer, messages, extra_kwargs=None):
    """安全地调用 apply_chat_template，只传递支持的参数"""
    extra_kwargs = extra_kwargs or {}
    if not hasattr(tokenizer, "apply_chat_template") or getattr(tokenizer, "chat_template", None) is None:
        return None
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
        supported = set(sig.parameters.keys())
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        for k, v in extra_kwargs.items():
            if k in supported:
                kwargs[k] = v
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        return None


def format_chat_prompt(tokenizer, prompt, model_name, system_msg="You are a careful chemist. Follow the requested output format exactly."):
    """统一格式化 chat prompt，确保两种推理路径使用相同的格式"""
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]
    extra = {}
    if model_name == "intern-s1":
        extra["enable_thinking"] = False
    text = safe_apply_chat_template(tokenizer, messages, extra_kwargs=extra)
    if text is not None:
        return text
    # fallback
    return f"System: {system_msg}\n\nUser: {prompt}\n\nAssistant: "


def infer_input_device(model):
    """推断输入应该放在哪个设备上（处理多卡情况）"""
    if hasattr(model, "device") and str(model.device) not in ("meta", "cpu"):
        return model.device
    if hasattr(model, "hf_device_map"):
        # 优先找 embedding 相关 key
        for k, v in model.hf_device_map.items():
            if any(x in k for x in ["embed", "wte", "word_embeddings", "tok_embeddings"]):
                return v
        # 再找第一个 cuda
        for v in model.hf_device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                return v
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def strip_thinking(t: str) -> str:
    """提取 thinking 标签后的最终答案（多分隔符兜底）"""
    if not t:
        return t
    # 先处理显式 think 标签
    if "<think>" in t and "</think>" in t:
        return t.split("</think>", 1)[1].strip()
    # 再尝试用常见分隔符截取最后一段"像答案"的内容
    THINK_SPLITS = ["</think>", "</think>", "</thinking>", "<|im_end|>", "<|endoftext|>"]
    for sp in THINK_SPLITS:
        if sp in t:
            tail = t.split(sp)[-1].strip()
            if tail:
                return tail
    return t


def clean_output(task: str, text: str) -> str:
    """统一后处理：清理输出格式，提取最终答案"""
    if text is None or not text:
        return ""
    
    # 1) 去掉常见 special token / 尾巴
    t = text.replace("<|im_end|>", "").replace("</s>", "").strip()
    
    # 2) 处理 thinking：提取最终答案
    t = strip_thinking(t)
    
    # 3) 提取标签内内容（如果有）
    #   <SMILES> ... </SMILES> / <SOL> ... </SOL> / <SOLUTION> ... </SOLUTION>
    for tag in ["SMILES", "SOL", "SOLUTION", "MOLFORMULA"]:
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", t, flags=re.S)
        if m:
            t = m.group(1).strip()
            break
    
    # 4) 只取第一行（大多数任务都要求单行）
    t = t.splitlines()[0].strip()
    
    # 5) task-specific 兜底抽取
    if task.startswith("property_prediction-") and task not in ["property_prediction-esol", "property_prediction-lipo"]:
        m = YESNO_RE.search(t)
        if m:
            return "Yes" if m.group(1).lower() == "yes" else "No"
        return t  # 实在不行原样返回，方便排查
    
    if task in ["property_prediction-esol", "property_prediction-lipo"]:
        m = FLOAT_RE.search(t)
        return m.group(0) if m else t
    
    # smiles 类任务：简单去空格
    if task in ["forward_synthesis", "retrosynthesis", "molecule_generation"]:
        return t.replace(" ", "")
    
    return t


def gen_config_for_task(task: str) -> Dict[str, Any]:
    """根据任务类型返回合适的生成配置"""
    if task.startswith("property_prediction-") and task not in ["property_prediction-esol", "property_prediction-lipo"]:
        # Yes/No 分类任务：greedy + 小 max_new_tokens
        return {
            "max_new_tokens": 8,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.06,
            "no_repeat_ngram_size": 3,
        }
    if task in ["property_prediction-esol", "property_prediction-lipo"]:
        # 数值回归任务
        return {
            "max_new_tokens": 32,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.06,
            "no_repeat_ngram_size": 3,
        }
    if task in ["forward_synthesis", "retrosynthesis", "molecule_generation"]:
        # SMILES 生成任务
        return {
            "max_new_tokens": 256,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.06,
            "no_repeat_ngram_size": 3,
        }
    # 其他任务（name_conversion, molecule_captioning 等）
    return {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.9,
        "repetition_penalty": 1.06,
        "no_repeat_ngram_size": 3,
    }


def load_first_prompt_from_jsonl(jsonl_path: Path) -> Dict[str, Any]:
    """从jsonl文件加载第一个prompt"""
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line:
            return json.loads(first_line)
    return None


def find_all_jsonl_files(test_dir: Path) -> List[Path]:
    """查找所有jsonl文件（排除评估结果相关文件）"""
    jsonl_files = []
    
    for file in test_dir.glob("*.jsonl"):
        # 按名字包含关键字排除
        if any(s in file.name.lower() for s in EXCLUDE_SUBSTR):
            continue
        jsonl_files.append(file)
    
    return sorted(jsonl_files)


def load_sft_tester_generator(
    model_name: str,
    model_path: Path,
    base_llm_path: Path = None
) -> MolAwareGenerator2:
    """加载 sft_tester generator（只加载一次）"""
    generator = MolAwareGenerator2()
    
    cfg = {
        "ckpt_dir": str(model_path),
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "dtype": "bf16",
        "debug": False,
        "enable_thinking": False,
        "realtime_mol": False,
    }
    
    if base_llm_path:
        cfg["base_llm_path"] = str(base_llm_path)
    
    if model_name == "llasmol":
        cfg["token_classifier_path"] = TOKEN_CLS_PATH
    
    generator.load(cfg)
    return generator


def inference_with_sft_tester(
    generator: MolAwareGenerator2,
    prompt: str,
    task_name: str,
    model_name: str
) -> str:
    """使用已加载的 sft_tester generator 进行推理"""
    # 根据任务类型选择生成配置
    gen_config = gen_config_for_task(task_name)
    
    # 对于非 SMILES 任务（property_prediction, name_conversion-s2i），
    # llasmol 的 sft_tester 可能会过滤掉非分子内容，建议用 transformers
    # 但这里先尝试，如果返回空再处理
    
    # Intern-S1 需要强制关闭 thinking
    kwargs = {
        "add_dialog_wrapper": True,
        "realtime_mol": False,
        **gen_config
    }
    
    # 如果 generator 支持 enable_thinking 参数，传递它
    # 注意：这取决于 sft_tester 的实现
    try:
        result = generator.generate(prompt, **kwargs)
    except TypeError:
        # 如果不支持某些参数，使用基本参数
        result = generator.generate(
            prompt,
            add_dialog_wrapper=True,
            realtime_mol=False,
            max_new_tokens=gen_config.get("max_new_tokens", 256),
            temperature=gen_config.get("temperature", 0.2),
            top_p=gen_config.get("top_p", 0.9),
            repetition_penalty=gen_config.get("repetition_penalty", 1.06),
            no_repeat_ngram_size=gen_config.get("no_repeat_ngram_size", 3),
        )
    
    return result


def load_transformers_model(
    model_name: str,
    model_path: Path,
    base_llm_path: Path = None
) -> tuple:
    """加载 transformers 模型和 tokenizer（只加载一次）"""
    # 加载 tokenizer
    if base_llm_path and model_name == "intern-s1":
        tokenizer_path = base_llm_path
    else:
        tokenizer_path = model_path
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=True
        )
    except Exception as e:
        raise RuntimeError(f"加载 tokenizer 失败: {e}")
    
    # 加载模型
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,  # 使用 torch_dtype（标准 API）
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        raise RuntimeError(f"加载模型失败: {e}")
    
    return tokenizer, model


def inference_with_transformers(
    model_name: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prompt: str,
    task_name: str
) -> str:
    """使用已加载的 transformers 模型进行推理"""
    # 根据任务类型选择生成配置
    gen_config = gen_config_for_task(task_name)
    
    # 格式化 prompt（使用统一的 format_chat_prompt，确保与 sft_tester 一致）
    system_msg = "You are a careful chemist. Follow the requested output format exactly."
    formatted_prompt = format_chat_prompt(tokenizer, prompt, model_name, system_msg)
    
    # Tokenize
    try:
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_ids_len = inputs["input_ids"].shape[1]  # 保存原始长度，用于后续解码
        
        # 移动输入到模型设备（使用改进的设备推断）
        dev = infer_input_device(model)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
    except Exception as e:
        raise RuntimeError(f"Tokenize 失败: {e}")
    
    # Generate
    try:
        # 设置 eos_token_id 和 pad_token_id
        generate_kwargs = {
            **inputs,
            **gen_config,
        }
        
        # 确保设置了 eos_token_id
        if "eos_token_id" not in generate_kwargs:
            generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if "pad_token_id" not in generate_kwargs:
            generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
        
        with torch.no_grad():
            outputs = model.generate(**generate_kwargs)
    except Exception as e:
        raise RuntimeError(f"生成失败: {e}")
    
    # Decode
    try:
        # 对于 intern-s1，先不 skip special tokens，避免把关键内容跳掉
        # 后续通过 clean_output 统一处理
        skip_special = model_name != "intern-s1"
        generated_text = tokenizer.decode(
            outputs[0][input_ids_len:],
            skip_special_tokens=skip_special
        )
    except Exception as e:
        raise RuntimeError(f"解码失败: {e}")
    
    return generated_text


def collect_all_prompts(jsonl_files: List[Path]) -> List[Dict[str, Any]]:
    """预先收集所有任务的 prompt"""
    tasks = []
    print(f"📋 收集所有任务的 prompt...")
    
    for jsonl_file in jsonl_files:
        task_name = jsonl_file.stem
        print(f"  📝 {task_name}...")
        
        # 加载第一个 prompt
        data = load_first_prompt_from_jsonl(jsonl_file)
        if not data:
            print(f"    ⚠️  跳过（文件为空）")
            continue
        
        prompt = data.get("prompt", "")
        gold = data.get("gold", "")
        input_text = data.get("input", "")
        
        if not prompt:
            print(f"    ⚠️  跳过（没有 prompt）")
            continue
        
        tasks.append({
            "task": task_name,
            "prompt": prompt,
            "input": input_text,
            "gold": gold,
        })
        print(f"    ✓ Prompt 长度: {len(prompt)} 字符")
    
    print(f"\n✅ 共收集到 {len(tasks)} 个任务\n")
    return tasks


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="测试脚本：比较不同模型和推理方式的结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法（需要运行四次，每次不同的 setting）：
  python test_prompts_comparison.py --model llasmol --type sft_tester
  python test_prompts_comparison.py --model llasmol --type transformers
  python test_prompts_comparison.py --model intern-s1 --type sft_tester
  python test_prompts_comparison.py --model intern-s1 --type transformers
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["llasmol", "intern-s1"],
        help="要使用的模型名称"
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["sft_tester", "transformers"],
        help="推理方式"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（默认：根据 model 和 type 自动生成）"
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    model_name = args.model
    inference_type = args.type
    
    print(f"🚀 运行设置: model={model_name}, type={inference_type}\n")
    
    # 检查模型路径
    if model_name not in MODELS:
        print(f"❌ 未知的模型: {model_name}")
        return
    
    model_path = MODELS[model_name]
    if not model_path.exists():
        print(f"❌ 模型路径不存在: {model_path}")
        return
    
    base_llm_path = BASE_LLM_PATHS.get(model_name)
    if base_llm_path and not base_llm_path.exists():
        print(f"❌ Base LLM 路径不存在: {base_llm_path}")
        return
    
    # 步骤1: 查找测试文件并收集所有任务的 prompt
    print(f"🔍 查找测试文件...")
    jsonl_files = find_all_jsonl_files(TEST_DIR)
    print(f"找到 {len(jsonl_files)} 个 jsonl 文件\n")
    
    tasks = collect_all_prompts(jsonl_files)
    
    if not tasks:
        print("❌ 没有找到有效的任务，退出")
        return
    
    # 步骤2: 加载指定的模型和推理方式
    print(f"📦 加载模型: {model_name} ({inference_type})...")
    
    loaded_model = None
    if inference_type == "sft_tester":
        try:
            sft_generator = load_sft_tester_generator(model_name, model_path, base_llm_path)
            loaded_model = sft_generator
            print(f"✅ sft_tester generator 加载完成")
        except Exception as e:
            print(f"❌ sft_tester generator 加载失败: {e}")
            return
    else:  # transformers
        try:
            tokenizer, model = load_transformers_model(model_name, model_path, base_llm_path)
            loaded_model = (tokenizer, model)
            print(f"✅ transformers 模型加载完成")
        except Exception as e:
            print(f"❌ transformers 模型加载失败: {e}")
            return
    
    print(f"\n✅ 模型加载完成，开始处理 {len(tasks)} 个任务...\n")
    
    # 步骤3: 处理所有任务
    results = []
    
    for task_idx, task_data in enumerate(tasks, 1):
        task_name = task_data["task"]
        prompt = task_data["prompt"]
        gold = task_data["gold"]
        input_text = task_data["input"]
        
        print(f"[{task_idx}/{len(tasks)}] 📝 处理任务: {task_name}")
        print(f"  Prompt 长度: {len(prompt)} 字符")
        
        try:
            if inference_type == "sft_tester":
                print(f"    → sft_tester 推理中...")
                result_raw = inference_with_sft_tester(loaded_model, prompt, task_name, model_name)
            else:  # transformers
                print(f"    → transformers 推理中...")
                tokenizer, model = loaded_model
                result_raw = inference_with_transformers(model_name, tokenizer, model, prompt, task_name)
            
            # 统一后处理：清理输出
            result_clean = clean_output(task_name, result_raw)
            
            results.append({
                "task": task_name,
                "model": model_name,
                "type": inference_type,
                "prompt": prompt,
                "input": input_text,
                "gold": gold,
                "prediction_raw": result_raw,  # 保留原始输出用于 debug
                "prediction": result_clean,    # 清理后的输出用于评测
            })
            
            # 如果清理后为空但原始不为空，给出警告
            if not result_clean and result_raw:
                print(f"    ⚠️  警告：清理后输出为空（原始输出长度: {len(result_raw)}）")
            
            print(f"    ✓ 完成 (原始: {len(result_raw)} 字符, 清理后: {len(result_clean)} 字符)")
        except Exception as e:
            print(f"    ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "task": task_name,
                "model": model_name,
                "type": inference_type,
                "prompt": prompt,
                "input": input_text,
                "gold": gold,
                "prediction_raw": f"ERROR: {str(e)}",
                "prediction": f"ERROR: {str(e)}",
            })
    
    # 确定输出文件路径
    if args.output:
        output_file = Path(args.output)
    else:
        # 根据 model 和 type 自动生成文件名
        output_file = OUTPUT_FILE.parent / f"prompt_comparison_{model_name}_{inference_type}.jsonl"
    
    # 保存结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 完成！结果已保存到: {output_file}")
    print(f"   共 {len(results)} 条结果")
    print(f"   设置: model={model_name}, type={inference_type}")


if __name__ == "__main__":
    main()

