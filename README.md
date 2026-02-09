# MHMLM - Molecular Hierarchical Multi-Language Model

分子层次多语言模型，整合了LDMol的diffusion功能，支持文本生成和分子生成。

## 快速开始

### 环境设置

使用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```bash
# 安装uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
uv venv
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

### 训练

```bash
python train_sft.py configs/config.yaml
```

### 推理

```python
from sft_tester import MolAwareGenerator2

gen = MolAwareGenerator2()
gen.load({
    "ckpt_dir": "/path/to/checkpoint",
    "ldmol": {
        "enabled": True,
        "vae_path": "/path/to/vae.ckpt",
        "ckpt_path": "/path/to/ldmol.pt",
    }
})

# 普通文本生成
text = gen.generate("Describe this molecule: CCO")

# 分子生成任务
smiles = gen.generate(
    "Generate a molecule that can treat headaches",
    task_type="molecule_generation"
)
```

## 主要特性

1. **统一训练**：只计算LM loss，GVP和diffusion不需要单独的loss
2. **直接使用LLM embedding**：LDMol直接使用LLM的hidden states，无需adapter
3. **SMILES补充**：支持`use_diffusion_as_smiles_supplement`参数
4. **分子生成任务**：支持`task_type="molecule_generation"`

## 文件说明

详细文件说明请参考 [FILES.md](FILES.md)

## 依赖

主要依赖：
- PyTorch >= 2.0.1
- Transformers >= 4.30.1
- RDKit >= 2023.3.1
- timm >= 0.9.16
- einops >= 0.7.0

完整依赖列表请参考 `pyproject.toml`
