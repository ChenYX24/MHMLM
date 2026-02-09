# Drug Optimization 评估

用于评估药物优化模型（LLM 和 Diffusion）的性能。

## 快速开始

```bash
cd /data1/chenyuxuan/MHMLM/eval/drug_optim

# LLM 模型评估
python run_drug_optim_eval.py --config config/llm_cpt_sft.yaml

# Diffusion 模型评估
python run_drug_optim_eval.py --config config/diffusion_base.yaml
```

## 目录结构

```
eval/drug_optim/
├── run_drug_optim_eval.py      # 主入口脚本
├── config/                     # 配置文件
│   ├── llm_base.yaml
│   ├── llm_cpt_sft.yaml
│   └── diffusion_base.yaml
├── testers/                    # 测试器模块
│   ├── base.py
│   ├── llm_tester.py
│   └── diffusion_tester.py
├── scoring/                    # 评分模块
│   ├── scorer.py               # 评分入口
│   ├── admet_reasoning_richness.py
│   ├── filter.py
│   └── test2.py
└── eval_output/                # 输出目录
    └── <model_name>/
        ├── output.txt
        ├── test_log.log
        ├── scoring_summary.json
        └── run_info.json
```

## 配置文件

### LLM 配置示例

```yaml
model_type: llm
model_name: llm_cpt_sft
ckpt: /path/to/checkpoint
input_data: /path/to/test_text2smi.jsonl
algorithm: chat
device: auto              # auto / cuda:0 / 0 / 0,1,2（多卡）
max_new_tokens: 256
temperature: 0.7
```

### Diffusion 配置示例

```yaml
model_type: diffusion
model_name: diffusion_base
ckpt: /path/to/checkpoint.pt
input_data: /path/to/test_dds.txt
algorithm: dds
device: cuda:0            # cuda:0 / 0（仅单卡）
```

### 设备配置说明

| 值 | 说明 |
|----|------|
| `auto` | 自动分配到所有可用 GPU |
| `cuda:0` | 指定单卡 |
| `0` | 等同于 `cuda:0` |
| `0,1,2` | 多卡并行（LLM 专用，自动设置 CUDA_VISIBLE_DEVICES） |

## 评估流程

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  配置文件        │────▶│  测试阶段        │────▶│  评分阶段        │
│  config/*.yaml  │     │  生成 output.txt │     │  ADMET 打分      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**测试阶段：**
- LLM: 使用 chat 接口生成 SMILES，从 `test_text2smi.jsonl` 读取 prompt
- Diffusion: 使用 DDS 算法优化分子，从 `test_dds.txt` 读取输入

**评分阶段：**
- 调用 `scoring/admet_reasoning_richness.py` 进行 ADMET 对比评分
- 输出 `scoring_summary.json`

## 输出格式

### output.txt

Tab 分隔的 6 列：

| 列名 | 说明 |
|------|------|
| row_id | 行号 |
| original_smiles | 原始分子 |
| source_caption | 原始属性描述 |
| target_caption | 目标属性描述 |
| gt_smiles | Ground truth SMILES |
| pred_smiles | 预测的 SMILES |

### scoring_summary.json

包含 ADMET 评分指标：
- `avg_main_reward`: 主奖励均值
- `avg_bonus_f1`: F1 分数
- `validity_rate`: 有效性比例
- 等其他指标...

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--config` | 配置文件路径（必需） |
| `--output-dir` | 自定义输出目录 |
| `--skip-test` | 跳过测试，直接评分 |
| `--skip-score` | 跳过评分 |

## 模型检查点

| 模型 | 路径 |
|------|------|
| llm_base | `/data1/chenyuxuan/base_model/qwen3_8b` |
| llm_cpt_sft | `/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200` |
| diffusion_base | `/data1/chenyuxuan/checkpoint/diffusion_pretrained/ours/ldmol/ldmol_chatmol.pt` |

## 输入数据

| 模型类型 | 输入文件 | 格式 |
|----------|----------|------|
| LLM | `test_text2smi.jsonl` | JSONL，含 prompt/ground_truth |
| Diffusion | `test_dds.txt` | TSV，4 列 |

数据位置：`/data1/chenyuxuan/MHMLM/eval_results/data/ldmol/drug_optim/processed/`
