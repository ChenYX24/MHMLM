# Layer2-LLM 快速开始

## 一键运行完整流程

```bash
cd /data1/chenyuxuan/MHMLM

# 使用默认配置运行
bash scripts/layer2_llm/run_full_pipeline.sh
```

## 自定义配置

```bash
# 设置环境变量
export TRAIN_DATA_INPUT="/path/to/queries.jsonl"
export TRAIN_CONFIG="/path/to/config.yaml"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NUM_GPUS=4

# 运行
bash scripts/layer2_llm/run_full_pipeline.sh
```

## 流程说明

脚本会自动执行三个阶段：

1. **生成训练数据** - 使用 Layer2 pipeline 生成训练数据
2. **训练 LLM** - 使用生成的数据训练模型
3. **评测 ChemBench** - 评测 product, retro, yield 三个任务

## 输出位置

- **训练数据**: `scripts/layer2_llm/data/training_data.jsonl`
- **训练输出**: `/data1/chenyuxuan/checkpoint/qwen3_8b_layer2_llm_YYYYMMDD_HHMMSS/`
- **评测结果**: `eval_chembench_layer2_llm_YYYYMMDD_HHMMSS/`

## 查看结果

```bash
# 查看评测结果
cat eval_chembench_layer2_llm_*/pred_product.jsonl | head -5
cat eval_chembench_layer2_llm_*/pred_retro.jsonl | head -5
cat eval_chembench_layer2_llm_*/pred_yield.jsonl | head -5

# 查看准确率
cat eval_chembench_layer2_llm_*/chembench4k_*_test_summary.json | grep acc
```

## 详细文档

完整指南请参考: `../../LAYER2_LLM_TRAINING_GUIDE.md`
