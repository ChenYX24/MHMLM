"""
LDMolInferer 测试脚本

用法:
    python -m ldmol_component --device cuda:0 --seed 0
"""

import logging
import argparse

import torch

from .LDMolInferer import LDMolInferer


class BracketColorFormatter(logging.Formatter):
    LEVEL_COLORS = {
        "DEBUG": "\033[37m",    # white
        "INFO": "\033[36m",     # cyan
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",    # red
        "CRITICAL": "\033[35m", # magenta
    }
    RESET = "\033[0m"
    NAME_COLOR = "\033[37m"

    def format(self, record):
        timestamp = self.formatTime(record, self.datefmt)
        level_color = self.LEVEL_COLORS.get(record.levelname, "")
        record.msg = (
            f"[{level_color}{record.levelname}{self.RESET}|"
            f"{self.NAME_COLOR}{record.name}{self.RESET}|"
            f"{self.NAME_COLOR}{timestamp}{self.RESET}] {record.getMessage()}"
        )
        return record.msg


def setup_root_logger(level=logging.INFO):
    """初始化 root logger，全局生效"""
    handler = logging.StreamHandler()
    formatter = BracketColorFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)


setup_root_logger()
logger = logging.getLogger("LDMolInferer")


def _t2m_example() -> tuple[str, str]:
    """T2M 测试用例"""
    ref_smiles = (
        "C1[C@@H]([C@H](OC2=C1C(=C(C(=C2)O)[C@H]3[C@@H]([C@H](OC4=CC(=CC(=C34)O)O)C5=CC(=C(C=C5)O)O)O)O)C6=CC(=C(C=C6)O)O)O"
    )
    description = (
        "The molecule is a proanthocyanidin consisting of two molecules of (+)-catechin joined by a bond between positions 4 and 6' in alpha-configuration. "
        "Procyanidin B6 is isolated from leaves and fruit of cowberry Vaccinium vitis-idaea and other plants. It can also be found in grape seeds and in beer. "
        "It has a role as a metabolite. It is a hydroxyflavan, a proanthocyanidin and a biflavonoid. It derives from a (+)-catechin."
    )
    return ref_smiles, description


def _dds_example() -> tuple[str, str, str]:
    """DDS 测试用例"""
    input_smiles = "CN1CCc2nc(O)n3nc(-c4ccccc4Cl)nc3c2C1"
    source_text = "This molecule has the following properties: low permeability."
    target_text = "This molecule has the following properties: improved permeability."
    return input_smiles, source_text, target_text


def main(args):
    # 初始化 LDMolInferer（自动加载 text_encoder）
    ldmol = LDMolInferer(config_path=args.config, device=args.device)

    # ===== 测试 T2M =====
    print("\n" + "=" * 60)
    print("Testing T2M (Text-to-Molecule)")
    print("=" * 60)
    
    ref_smiles, description = _t2m_example()
    pred_smiles = ldmol.generate_smi_t2m(description)

    ref_smiles = LDMolInferer.canonicalize_smiles(ref_smiles)
    print(f"\nReference SMILES:\n{ref_smiles}")
    print(f"\nDescription:\n{description}")
    print(f"\nGenerated SMILES:\n{pred_smiles}")
    print(f"\nMatch: {ref_smiles == pred_smiles}")

    # ===== 测试 DDS =====
    print("\n" + "=" * 60)
    print("Testing DDS (Diffusion-based Drug Steering)")
    print("=" * 60)

    input_smiles, source_text, target_text = _dds_example()
    output_smiles = ldmol.generate_smi_dds(input_smiles, source_text, target_text)

    print(f"\nInput SMILES: {input_smiles}")
    print(f"Source: {source_text}")
    print(f"Target: {target_text}")
    print(f"Output SMILES: {output_smiles}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDMolInferer 测试脚本")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="配置文件路径（默认使用同目录下的 ldmol_config.yaml）"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0",
        help="推理设备（默认 cuda:0）"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="随机种子（默认 0）"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    main(args)
