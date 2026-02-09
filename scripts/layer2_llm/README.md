# Layer2 + LLM 联合训练

本目录包含 Layer2 与 LLM 联合训练的相关脚本。

## 文件说明

- `generate_training_data.py`: 生成 LLM + Layer2 联合训练数据
- `train_layer2_llm.sh`: 训练 LLM + Layer2 模型的脚本
- `quick_test.sh`: 快速测试脚本

## 使用流程

### 1. 生成训练数据

```bash
python generate_training_data.py \
    --input /path/to/queries.jsonl \
    --output /path/to/training_data.jsonl \
    --config /path/to/model_config.yaml \
    --task_type "reaction_prediction"
```

### 2. 训练模型

```bash
bash train_layer2_llm.sh
```

### 3. 快速测试

```bash
bash quick_test.sh
```

## 数据格式

### 输入格式（queries.jsonl）

```json
{"input": "query text"}
```

### 输出格式（training_data.jsonl）

```json
{
    "input": "原始 query",
    "intermediate": "第一轮 JSON 输出",
    "molecules_info": [
        {
            "smiles": "CCO",
            "role": "REACTANT",
            "amount_info": {...}
        }
    ],
    "layer2_info": {
        "yield_bin": 5,
        "yield_reg": 0.75,
        "embedding_shape": [1024]
    },
    "output": "最终 LLM 输出"
}
```

## 详细文档

- 完整指南：`../../LAYER2_GUIDE.md`
- 测试指令：`../../LAYER2_TEST_INSTRUCTIONS.md`
