"""
Layer2 Component
用于反应产率预测和任务相关 embedding 生成
"""

from .Layer2Inferer import Layer2Inferer
from .Layer2Trainer import Layer2Trainer

__all__ = ["Layer2Inferer", "Layer2Trainer"]
