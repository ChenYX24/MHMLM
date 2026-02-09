#!/usr/bin/env python3
"""
单独训练 Layer2 模型
用法: python scripts/layer2/train_layer2.py [--config scripts/layer2/layer2_train_config.yaml]
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.layer2_component.Layer2Trainer import main as trainer_main

if __name__ == "__main__":
    # 默认配置路径
    DEFAULT_CONFIG = "/data1/chenyuxuan/MHMLM/scripts/layer2/layer2_train_config.yaml"
    
    parser = argparse.ArgumentParser(description="训练 Layer2 模型")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, 
                       help=f"配置文件路径，默认: {DEFAULT_CONFIG}")
    
    args = parser.parse_args()
    
    # 修改 sys.argv 以传递给 Layer2Trainer.main
    import sys
    sys.argv = ["Layer2Trainer", "--config", args.config]
    
    trainer_main()
