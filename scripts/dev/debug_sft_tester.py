#!/usr/bin/env python3
"""
调试脚本：测试 sft_tester.py 的不同配置
支持：
1. 直接生成（不使用 GVP 和 diffusion）
2. +gvp（使用 GVP）
3. +gvp+diffusion（使用 GVP 和 diffusion supplement）
4. 特殊任务用 diffusion（generation）
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sft_tester import MolAwareGenerator2

# 默认配置
DEFAULT_CONFIG = {
    "ckpt_dir": "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200",
    "device": "cuda:0",
    "device_map": None,  # 单卡模式
    "dtype": "bf16",
    "debug": True,
    "token_classifier_path": "/data1/lvchangwei/LLM/Lora/qwen3_mlp_token_head.pt",
}

# # Diffusion 配置
# LDMOL_CONFIG = {
#     "enabled": True,
#     "ckpt_path": "/data1/chenyuxuan/checkpoint/diffusion_pretrained/ours/ldmol/ldmol_chatmol-qwen3_8b.pt",
#     "vae_path": "/data1/chenyuxuan/checkpoint/diffusion_pretrained/official/checkpoint_autoencoder.ckpt",
#     "num_sampling_steps": 100,
#     "cfg_scale": 2.5,
# }

# 测试 prompt
TEST_PROMPTS = {
    "normal": "Describe this molecule: CCCCCCC(O)C/C=C\\CCCCCCCC(=O)[O-]\nPlease only output the answer.",
    "generation": "Generate a molecule that is a potential drug candidate for treating diabetes. Please only output the answer.",
    "synthesis": "Predict a possible product from the listed reactants and reagents. CCN.CN1C=CC=C1C=O\nPlease only output the answer without any explanation or additional text.",
}


def test_mode_1_direct_generation(cfg, prompt):
    """模式1：直接生成（不使用 GVP 和 diffusion）"""
    print("\n" + "="*80)
    print("模式1：直接生成（不使用 GVP 和 diffusion）")
    print("="*80)
    
    gen = MolAwareGenerator2()
    gen.load(cfg)
    
    text = gen.generate(
        prompt,
        add_dialog_wrapper=True,
        realtime_mol=False,  # 不使用 GVP
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.06,  # 提高重复惩罚以减少重复
        no_repeat_ngram_size=3,  # 防止3-gram重复
        skip_special_tokens=True,
    )
    
    print("\n=== Generated Text ===")
    print(text)
    return text


def test_mode_2_with_gvp(cfg, prompt):
    """模式2：+gvp（使用 GVP）"""
    print("\n" + "="*80)
    print("模式2：+gvp（使用 GVP）")
    print("="*80)
    
    gen = MolAwareGenerator2()
    gen.load(cfg)
    
    text = gen.generate(
        prompt,
        add_dialog_wrapper=True,
        realtime_mol=True,  # 使用 GVP
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.06,  # 提高重复惩罚以减少重复
        no_repeat_ngram_size=3,  # 防止3-gram重复
        skip_special_tokens=True,
    )
    
    print("\n=== Generated Text ===")
    print(text)
    return text


def test_mode_3_with_gvp_diffusion(cfg, prompt):
    """模式3：+gvp+diffusion（使用 GVP 和 diffusion supplement）"""
    print("\n" + "="*80)
    print("模式3：+gvp+diffusion（使用 GVP 和 diffusion supplement）")
    print("="*80)
    
    # NOTE:ldmol config直接在 /data1/chenyuxuan/MHMLM/modules/ldmol_component/ldmol_config.yaml
    # cfg_with_ldmol = cfg.copy()
    
    gen = MolAwareGenerator2()
    gen.load(cfg)
    
    text = gen.generate(
        prompt,
        add_dialog_wrapper=True,
        realtime_mol=True,  # 使用 GVP
        use_diffusion_as_smiles_supplement=True,  # 使用 diffusion supplement
        max_new_tokens=256,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.06,  # 提高重复惩罚以减少重复
        no_repeat_ngram_size=3,  # 防止3-gram重复
        skip_special_tokens=True,
    )
    
    print("\n=== Generated Text ===")
    print(text)
    return text


def test_mode_4_diffusion_generation(cfg, prompt):
    """模式4：特殊任务用 diffusion（generation）"""
    print("\n" + "="*80)
    print("模式4：特殊任务用 diffusion（generation）")
    print("="*80)
    
    # NOTE:ldmol config直接在 /data1/chenyuxuan/MHMLM/modules/ldmol_component/ldmol_config.yaml
    # cfg_with_ldmol = cfg.copy()
    gen = MolAwareGenerator2()
    gen.load(cfg)
    
    text = gen.generate(
        prompt,
        add_dialog_wrapper=True,
        realtime_mol=True,  # 不使用 GVP
        task_type="molecule_generation",  # 特殊任务：分子生成（注意：代码中使用 "molecule_generation" 来触发 diffusion）
        max_new_tokens=512,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.06,  # 提高重复惩罚以减少重复
        no_repeat_ngram_size=3,  # 防止3-gram重复
        skip_special_tokens=True,
        verbose_logging=True,  # 启用详细日志以查看 diffusion 生成过程
    )
    
    print("\n=== Generated Text ===")
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="调试 sft_tester.py 的不同配置")
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="测试模式：1=直接生成, 2=+gvp, 3=+gvp+diffusion, 4=特殊任务用diffusion"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=["normal", "generation", "synthesis"],
        default="normal",
        help="Prompt 类型"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Checkpoint 目录路径（覆盖默认配置）"
    )
    parser.add_argument(
        "--token-classifier",
        type=str,
        default=None,
        help="Token classifier 路径（覆盖默认配置）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="设备（默认：cuda:0）"
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        default=None,
        help="自定义 prompt（覆盖默认 prompt）"
    )
    
    args = parser.parse_args()
    
    # 构建配置
    cfg = DEFAULT_CONFIG.copy()
    if args.ckpt_dir:
        cfg["ckpt_dir"] = args.ckpt_dir
    if args.token_classifier:
        cfg["token_classifier_path"] = args.token_classifier
    if args.device:
        cfg["device"] = args.device
    
    # 选择 prompt
    if args.custom_prompt:
        prompt = args.custom_prompt
    else:
        prompt = TEST_PROMPTS[args.prompt_type]
    
    print(f"\n配置信息：")
    print(f"  Checkpoint: {cfg['ckpt_dir']}")
    print(f"  Token Classifier: {cfg.get('token_classifier_path', 'None')}")
    print(f"  Device: {cfg['device']}")
    print(f"  Prompt: {prompt[:100]}...")
    
    # 根据模式执行测试
    if args.mode == 1:
        test_mode_1_direct_generation(cfg, prompt)
    elif args.mode == 2:
        test_mode_2_with_gvp(cfg, prompt)
    elif args.mode == 3:
        test_mode_3_with_gvp_diffusion(cfg, prompt)
    elif args.mode == 4:
        test_mode_4_diffusion_generation(cfg, prompt)
    
    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)


if __name__ == "__main__":
    main()

