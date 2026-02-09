#!/bin/bash
# Layer2 快速测试脚本

set -e

PROJECT_ROOT="/data1/chenyuxuan/MHMLM"
cd ${PROJECT_ROOT}

echo "============================================"
echo "Layer2 完整测试流程"
echo "============================================"

# 1. 检查数据
echo ""
echo "1. 检查数据..."
LAYER2_DATA="/data1/chenyuxuan/Layer2/data"
if [ ! -f "${LAYER2_DATA}/ord_layer2/layer2_test.jsonl" ]; then
    echo "⚠️  测试数据不存在: ${LAYER2_DATA}/ord_layer2/layer2_test.jsonl"
    echo "   请确保数据已下载"
else
    echo "✅ 数据检查通过"
fi

# 2. 检查配置文件
echo ""
echo "2. 检查配置文件..."
if [ ! -f "modules/layer2_component/layer2_config.yaml" ]; then
    echo "❌ Layer2 配置文件不存在"
    exit 1
fi
if [ ! -f "scripts/layer2/layer2_train_config.yaml" ]; then
    echo "❌ Layer2 训练配置文件不存在"
    exit 1
fi
echo "✅ 配置文件检查通过"

# 3. 测试 Python 导入
echo ""
echo "3. 测试 Python 导入..."
python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}')

try:
    from modules.layer2_component.Layer2Inferer import Layer2Inferer
    print('✅ Layer2Inferer 导入成功')
except Exception as e:
    print(f'❌ Layer2Inferer 导入失败: {e}')
    sys.exit(1)

try:
    from sft_tester import MolAwareGenerator2
    print('✅ MolAwareGenerator2 导入成功')
except Exception as e:
    print(f'❌ MolAwareGenerator2 导入失败: {e}')
    sys.exit(1)
"

# 4. 测试 JSON 解析
echo ""
echo "4. 测试 JSON 解析..."
python -c "
import json
import re

# 测试 JSON 格式
test_json = '''
{
    \"molecules\": [
        {
            \"smiles\": \"CCO\",
            \"role\": \"REACTANT\",
            \"amount_info\": {
                \"moles\": 1.0
            }
        }
    ]
}
'''

try:
    parsed = json.loads(test_json)
    if 'molecules' in parsed:
        print('✅ JSON 格式验证通过')
    else:
        print('❌ JSON 格式验证失败')
        sys.exit(1)
except Exception as e:
    print(f'❌ JSON 解析失败: {e}')
    sys.exit(1)
"

# 5. 检查依赖
echo ""
echo "5. 检查依赖..."
python -c "
try:
    import json_repair
    print('✅ json-repair 已安装')
except ImportError:
    print('⚠️  json-repair 未安装，建议安装: pip install json-repair')

try:
    import torch
    print(f'✅ torch 已安装: {torch.__version__}')
except ImportError:
    print('❌ torch 未安装')
    sys.exit(1)
"

echo ""
echo "============================================"
echo "✅ 基础检查完成！"
echo "============================================"
echo ""
echo "下一步："
echo "1. 运行 Layer2 训练: bash scripts/layer2/train_layer2.py"
echo "2. 运行 Layer2 评测: bash scripts/run/run_eval_layer2_testset.sh"
echo "3. 生成训练数据: python scripts/layer2_llm/generate_training_data.py"
echo "4. 训练 LLM+Layer2: bash scripts/layer2_llm/train_layer2_llm.sh"
echo ""
echo "详细指令请参考: LAYER2_TEST_INSTRUCTIONS.md"
