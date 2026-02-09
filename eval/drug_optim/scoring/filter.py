import json
import os

# 输入和输出路径
input_path = "/data2/liuhaoran/reasoning_process/reasoning_process.jsonl"
output_path = "/data2/liuhaoran/reasoning_process/reasoning_process_cleaned.jsonl"

# 需要保留的ADMET关键字段
KEY_FEATURES = [
    # 吸收
    "Caco-2 Permeability", "MDCK Permeability", "HIA", "F20%", "F30%", "F50%",
    # 分布
    "BBB", "Pgp inhibitor", "Pgp substrate", "BCRP inhibitor", "VDss", "PPB",
    # 代谢
    "CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4", "HLM Stability",
    # 排泄
    "CLplasma", "T1/2",
    # 毒性
    "hERG Blockers", "DILI", "Human Hepatotoxicity", "AMES Toxicity", "Genotoxicity",
    "Rat Oral Acute Toxicity", "Drug-induced Nephrotoxicity",
    "Drug-induced Neurotoxicity", "Ototoxicity", "Hematotoxicity", "Skin Sensitization",
    # 类药性
    "Lipinski Rule", "Pfizer Rule", "GSK Rule", "GoldenTriangle", "QED", "Fsp3",
    # 结构可行性
    "SAscore", "MCE-18"
]

###############################################################################
# 评估阈值（用于 bad feature 判断与阈值奖励）
# - 这些阈值为轻量级启发式配置，便于快速评估与可运行性
# - 如需更严格/真实的阈值，请在此处按需求调整
###############################################################################

# 质性打分的数值化映射（便于比较）
_QUAL_TO_SCORE = {
    "++++": 4,
    "+++": 3,
    "++": 2,
    "+": 1,
    "+/-": 0.5,
    "-": 0,
    "--": -1,
    "---": -2
}


def _to_float(value):
    """将字符串值尽量转为数值，支持 +/++/+++ 等标签。"""
    if value is None:
        return None
    s = str(value).strip()
    if s in _QUAL_TO_SCORE:
        return float(_QUAL_TO_SCORE[s])
    try:
        return float(s)
    except Exception:
        return None


def check_item(item, value) -> bool:
    """
    判断某个指标是否“达到了坏阈值”。
    item: (flag, threshold, msg)
    """
    flag, threshold, _msg = item
    v = _to_float(value)
    t = _to_float(threshold)
    if flag in (">", ">=", "ge", "gte"):
        return v is not None and t is not None and v >= t
    if flag in ("<", "<=", "le", "lte"):
        return v is not None and t is not None and v <= t
    if flag in ("=", "==", "eq"):
        return str(value).strip() == str(threshold).strip()
    if flag in ("!=", "neq"):
        return str(value).strip() != str(threshold).strip()
    if flag == "contains":
        return str(threshold).strip() in str(value)
    return False


# checks 的结构：{category: {feature: (flag, threshold, msg)}}
checks = {
    "toxicity": {
        "AMES Toxicity": (">=", 0.5, "AMES 风险偏高"),
        "Genotoxicity": (">=", 0.5, "基因毒性偏高"),
        "DILI": (">=", 0.5, "DILI 风险偏高"),
        "Human Hepatotoxicity": (">=", 0.5, "肝毒性偏高"),
        "hERG Blockers": (">=", 0.5, "hERG 阻断风险偏高")
    },
    "absorption": {
        "Caco-2 Permeability": ("<=", -5.0, "Caco-2 渗透性偏低"),
        "MDCK Permeability": ("<=", 0.0, "MDCK 渗透性偏低"),
        "HIA": ("<=", 0.0, "HIA 偏低"),
        "F20%": ("<=", 0.0, "F20% 偏低"),
        "F30%": ("<=", 0.0, "F30% 偏低"),
        "F50%": ("<=", 0.0, "F50% 偏低")
    },
    "distribution": {
        "Pgp inhibitor": (">=", "++", "P-gp 抑制风险偏高"),
        "Pgp substrate": (">=", "++", "P-gp 底物风险偏高"),
        "BCRP inhibitor": (">=", "++", "BCRP 抑制风险偏高"),
        "PPB": (">=", 95, "血浆蛋白结合率偏高")
    },
    "metabolism": {
        "CYP1A2 inhibitor": (">=", "++", "CYP1A2 抑制偏强"),
        "CYP2C19 inhibitor": (">=", "++", "CYP2C19 抑制偏强"),
        "CYP2C9 inhibitor": (">=", "++", "CYP2C9 抑制偏强"),
        "CYP2D6 inhibitor": (">=", "++", "CYP2D6 抑制偏强"),
        "CYP3A4 inhibitor": (">=", "++", "CYP3A4 抑制偏强")
    }
}


def clean_admet_profile(admet_text: str) -> str:
    """
    从原始ADMET Profile文本中筛选保留关键信息
    """
    cleaned_lines = []
    for line in admet_text.split("\n"):
        if any(key in line for key in KEY_FEATURES):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def main():
    # 读取并清洗数据
    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            sample = json.loads(line)

            original_input = sample.get("input", "")
            # 将input拆分为SMILES和ADMET Profile部分
            parts = original_input.split("ADMET Profile:")

            if len(parts) == 2:
                smiles = parts[0].strip()
                admet_profile = clean_admet_profile(parts[1])
                # 重新组合
                sample["input"] = f"{smiles}\nADMET Profile:\n{admet_profile}"
            else:
                # 如果格式不标准，保持原样
                sample["input"] = original_input

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"清洗完成！输出文件保存在: {output_path}")


if __name__ == "__main__":
    main()
