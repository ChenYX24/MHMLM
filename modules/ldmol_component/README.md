## LDMol Component

封装了 LDMol 的推理和训练接口，配置读取同目录 `ldmol_config.yaml`。

### 功能

| 模块 | 功能 |
|------|------|
| **LDMolInferer** | 推理：T2M（文本生成分子）、DDS（属性定向优化） |
| **LDMolTrainer** | 训练：支持 DDP 分布式训练 |

---

## 目录结构

```
ldmol_component/
├── LDMolInferer.py      # 推理接口
├── LDMolTrainer.py      # 训练接口
├── ldmol_config.yaml    # 推理配置文件
├── __init__.py          # 模块导出
├── __main__.py          # 推理测试入口
├── test_inferer.sh      # 推理测试脚本
├── test_trainer.sh      # 训练测试脚本
├── utils.py             # 工具函数
├── assets/              # 固定资源
│   ├── config_decoder.json
│   ├── config_encoder.json
│   └── vocab_bpe_300_sc.txt
├── diffusion/           # Diffusion 实现
├── DiT/                 # DiT 模型实现
└── autoencoder/         # Autoencoder 实现
```

---

## 推理 (LDMolInferer)

### 配置说明

`ldmol_config.yaml` 中的主要配置项：

| 配置项 | 说明 |
|--------|------|
| `text_encoder_name` | 文本编码器类型，目前仅支持 `qwen` |
| `text_encoder_path` | Text Encoder 模型路径 |
| `ldmol_ckpt_path` | DiT 模型 checkpoint 路径 |
| `vae_ckpt_path` | Autoencoder checkpoint 路径 |
| `num_sampling_steps` | T2M 采样步数（默认 100） |
| `cfg_scale` | T2M 的 CFG 系数（默认 2.5） |
| `dds_*` | DDS 相关参数 |

### API 示例

```python
from ldmol_component import LDMolInferer

# 初始化（自动加载 text_encoder）
ldmol = LDMolInferer(device="cuda:0")

# T2M：文本生成分子
smiles = ldmol.generate_smi_t2m(
    description="a drug-like small molecule with high solubility..."
)

# DDS：属性定向优化
new_smiles = ldmol.generate_smi_dds(
    input_smiles="CN1CCc2nc(O)n3nc(-c4ccccc4Cl)nc3c2C1",
    source_text="This molecule has low permeability.",
    target_text="This molecule has improved permeability."
)
```

### 联合推理接口（LLM + Diffusion）

支持使用外部 Qwen 的 hidden states 生成 SMILES，用于 LLM + Diffusion 联合推理。

```python
from ldmol_component import LDMolInferer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 初始化 LDMol
ldmol = LDMolInferer(device="cuda:0")

# 加载外部 Qwen（用于联合推理）
qwen = AutoModelForCausalLM.from_pretrained("/path/to/qwen", torch_dtype=torch.bfloat16).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("/path/to/qwen")

# 方法 1：使用统一接口（推荐）
smiles = ldmol.generate_molecule(
    description="a drug-like molecule with improved solubility...",
    qwen=qwen,              # 外部 Qwen（可选）
    qwen_tokenizer=tokenizer,  # 外部 tokenizer（可选）
)

# 方法 2：直接从 hidden states 生成
# 假设你已经有了 Qwen 的 hidden states
y_cond = torch.randn(1, 512, 4096, device="cuda:0")  # (B, L, hidden_dim)
pad_mask = torch.ones(1, 512, device="cuda:0")       # (B, L)

smiles_list = ldmol.generate_smi_from_hidden(
    y_cond=y_cond,
    pad_mask=pad_mask,
)
```

详细的联合训练方案请参考：`scripts/drug_optim/code/llm_diffusion_cotrain/README.md`

### 测试推理

```bash
cd LDMol
bash ldmol_component/test_inferer.sh

# 指定设备
DEVICE=cuda:1 bash ldmol_component/test_inferer.sh
```

---

## 训练 (LDMolTrainer)

### 数据格式

训练数据为 TSV 文件，每行格式：

```
SMILES\t描述文本
```

或

```
CID\tSMILES\t描述文本
```

### API 示例

```python
from ldmol_component import LDMolTrainer, TrainConfig

# 创建配置
config = TrainConfig(
    data_path="./data/train.txt",
    text_encoder_path="/path/to/qwen3_8b",
    vae_ckpt_path="/path/to/vae.ckpt",
    epochs=100,
    global_batch_size=64,
)

# 创建训练器并开始训练
trainer = LDMolTrainer(config)
trainer.train()
```

### 命令行训练

```bash
# 单卡训练
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    -m ldmol_component.LDMolTrainer \
    --data_path ./data/train.txt \
    --text_encoder_path /path/to/qwen3_8b \
    --vae_ckpt_path /path/to/vae.ckpt \
    --epochs 100 \
    --global_batch_size 64

# 多卡训练（4 GPU）
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    -m ldmol_component.LDMolTrainer \
    --data_path ./data/train.txt \
    --text_encoder_path /path/to/qwen3_8b \
    --vae_ckpt_path /path/to/vae.ckpt \
    --epochs 100 \
    --global_batch_size 128
```

### 使用测试脚本

```bash
cd LDMol

# 单卡训练（默认配置）
bash ldmol_component/test_trainer.sh

# 多卡训练
GPUS=0,1,2,3 NPROC=4 bash ldmol_component/test_trainer.sh

# 自定义参数
DATA_PATH=./data/my_train.txt \
EPOCHS=50 \
GLOBAL_BATCH_SIZE=128 \
bash ldmol_component/test_trainer.sh
```

### 训练配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_path` | - | 训练数据路径（必需） |
| `vae_ckpt_path` | - | VAE 权重路径（必需） |
| `text_encoder_path` | - | Text Encoder 路径（必需） |
| `ldmol_ckpt_path` | "" | DiT 预训练权重（可选，用于继续训练） |
| `epochs` | 100 | 训练轮数 |
| `global_batch_size` | 64 | 全局 batch size（所有 GPU 总和） |
| `learning_rate` | 1e-4 | 学习率 |
| `description_length` | 256 | 文本最大长度 |
| `results_dir` | "./results" | 输出目录 |
| `log_every` | 100 | 日志频率（步） |
| `ckpt_every` | 5000 | 保存频率（步） |

### 输出目录结构

```
results/
└── 000-LDMol/
    ├── log.txt                # 训练日志
    └── checkpoints/
        ├── 0005000.pt         # step 5000 checkpoint
        ├── 0010000.pt         # step 10000 checkpoint
        └── ...
```

---

## 注意事项

1. **Text Encoder**：目前仅支持 `qwen`，其他类型会抛出 `AssertionError`
2. **路径**：配置文件中支持绝对路径和相对路径（相对于配置文件所在目录）
3. **DDP**：训练需要使用 `torchrun` 启动，即使单卡也需要
4. **显存**：Qwen3_8B 较大，建议使用 80GB 显存的 GPU
