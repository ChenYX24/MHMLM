from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


# 单位到基准单位的倍率（基准：mol / g / L）
_MOLES_TO_MOL = {
    "MOLE": 1.0,
    "MILLIMOLE": 1e-3,
    "MICROMOLE": 1e-6,
    "NANOMOLE": 1e-9,
}
_MASS_TO_G = {
    "GRAM": 1.0,
    "MILLIGRAM": 1e-3,
    "MICROGRAM": 1e-6,
}
_VOLUME_TO_L = {
    "LITER": 1.0,
    "MILLILITER": 1e-3,
    "MICROLITER": 1e-6,
}


@dataclass(frozen=True)
class AmountChannels:
    # 三通道：log1p(基准值) + 监督可用性 mask（数据里是否存在该通道的真值）
    amt_moles_log: Optional[float]
    amt_moles_mask: int
    amt_mass_log: Optional[float]
    amt_mass_mask: int
    amt_volume_log: Optional[float]
    amt_volume_mask: int
    volume_includes_solutes: Optional[bool]


def _to_base_value(q: dict[str, Any], unit_map: dict[str, float]) -> Optional[float]:
    try:
        value = float(q.get("value"))
    except Exception:
        return None
    units = q.get("units")
    if not isinstance(units, str) or units not in unit_map:
        return None
    x = value * float(unit_map[units])
    if not math.isfinite(x) or x <= 0.0:
        return None
    return x


def amount_to_channels(amount: Any) -> AmountChannels:
    """
    将提取阶段的 amount dict 转为三通道特征。
    - 不做互转（mass->moles 等），每个通道独立监督
    - unmeasured 直接忽略
    - 仅对 x>0 的数值生成 log1p；否则视为缺失（mask=0）
    """
    if not isinstance(amount, dict):
        return AmountChannels(
            amt_moles_log=None,
            amt_moles_mask=0,
            amt_mass_log=None,
            amt_mass_mask=0,
            amt_volume_log=None,
            amt_volume_mask=0,
            volume_includes_solutes=None,
        )

    moles_x = None
    mass_x = None
    volume_x = None

    if isinstance(amount.get("moles"), dict):
        moles_x = _to_base_value(amount["moles"], _MOLES_TO_MOL)
    if isinstance(amount.get("mass"), dict):
        mass_x = _to_base_value(amount["mass"], _MASS_TO_G)
    if isinstance(amount.get("volume"), dict):
        volume_x = _to_base_value(amount["volume"], _VOLUME_TO_L)

    volume_includes_solutes = None
    if volume_x is not None and "volume_includes_solutes" in amount:
        try:
            volume_includes_solutes = bool(amount["volume_includes_solutes"])
        except Exception:
            volume_includes_solutes = None

    def _log1p_or_none(x: Optional[float]) -> tuple[Optional[float], int]:
        if x is None:
            return None, 0
        return math.log1p(x), 1

    moles_log, moles_mask = _log1p_or_none(moles_x)
    mass_log, mass_mask = _log1p_or_none(mass_x)
    volume_log, volume_mask = _log1p_or_none(volume_x)

    return AmountChannels(
        amt_moles_log=moles_log,
        amt_moles_mask=moles_mask,
        amt_mass_log=mass_log,
        amt_mass_mask=mass_mask,
        amt_volume_log=volume_log,
        amt_volume_mask=volume_mask,
        volume_includes_solutes=volume_includes_solutes,
    )


def build_amount_feature(
    moles: float = 1.0,
    mass: float = 0.0,
    volume: float = 0.0,
    data_mask: list[bool] = None,
    pred_mask: list[bool] = None,
    volume_includes_solutes: bool = False,
) -> list[float]:
    """
    构建 amount 特征向量（10 维）。
    
    特征格式：
    [moles_log, moles_data_mask, moles_pred_mask,
     mass_log, mass_data_mask, mass_pred_mask,
     vol_log, vol_data_mask, vol_pred_mask,
     volume_includes_solutes]
    
    Args:
        moles: 摩尔数（mol）
        mass: 质量（g）
        volume: 体积（L）
        data_mask: [moles_data_mask, mass_data_mask, vol_data_mask]，表示数据是否缺失
        pred_mask: [moles_pred_mask, mass_pred_mask, vol_pred_mask]，表示是否预测
        volume_includes_solutes: 体积是否包含溶质
    
    Returns:
        10 维特征向量（list[float]）
    """
    import math
    
    if data_mask is None:
        data_mask = [False, False, False]
    if pred_mask is None:
        pred_mask = [False, False, False]
    
    # 计算 log1p（如果值为 0 或 None，则使用 0.0）
    moles_log = math.log1p(moles) if moles > 0.0 else 0.0
    mass_log = math.log1p(mass) if mass > 0.0 else 0.0
    vol_log = math.log1p(volume) if volume > 0.0 else 0.0
    
    # 转换为 float
    moles_data_mask = 1.0 if data_mask[0] else 0.0
    mass_data_mask = 1.0 if data_mask[1] else 0.0
    vol_data_mask = 1.0 if data_mask[2] else 0.0
    
    moles_pred_mask = 1.0 if pred_mask[0] else 0.0
    mass_pred_mask = 1.0 if pred_mask[1] else 0.0
    vol_pred_mask = 1.0 if pred_mask[2] else 0.0
    
    vis_f = 1.0 if volume_includes_solutes else 0.0
    
    return [
        moles_log,
        moles_data_mask,
        moles_pred_mask,
        mass_log,
        mass_data_mask,
        mass_pred_mask,
        vol_log,
        vol_data_mask,
        vol_pred_mask,
        vis_f,
    ]

