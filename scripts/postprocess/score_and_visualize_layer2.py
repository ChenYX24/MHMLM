#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 Layer2 测试集的 LLM 评测结果进行打分和可视化。

用法:
    python scripts/score_and_visualize_layer2_v2.py \
    --results_dir /path/to/results \
    --gvp-root /path/to/gvp-gnn \
    --gvp-ckpt /path/to/gvp/checkpoint.pt
    
    
"""

import sys
import os
import json
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm

# 尝试导入json_repair，如果不可用则使用fallback
try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    print("[WARNING] json_repair not available. Install with: pip install json-repair")

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).parent.parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# GVP encoder缓存
_gvp_encoder = None
_gvp_device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"


def extract_smiles_from_text(text: str, return_all: bool = False) -> Optional[str]:
    """从文本中提取SMILES字符串
    
    策略：
    1. 先尝试匹配 <SMILES>...</SMILES> 格式（可能有多个）
    2. 尝试解析JSON格式，提取products数组
    3. 如果没有，按空格分割文本，过滤掉不符合SMILES规则的token
    4. 处理用 '.' 分隔的多个SMILES（反应式格式）
    5. 优先选择RDKit可解析的候选
    
    Args:
        text: 输入文本
        return_all: 如果True，返回所有找到的SMILES列表（用于多SMILES情况）
    
    Returns:
        如果return_all=False，返回最匹配的单个SMILES
        如果return_all=True，返回所有找到的SMILES列表
    """
    if not text:
        return None if not return_all else []
    
    all_smiles = []
    
    # 1. 优先匹配所有 <SMILES> ... </SMILES> 格式
    smiles_pattern = r'<SMILES>\s*([^<]+)\s*</SMILES>'
    matches = re.findall(smiles_pattern, text, re.IGNORECASE)
    for match in matches:
        cand = match.strip()
        if cand and not re.fullmatch(r"\d+(?:\.\d+)?%?", cand):
            all_smiles.append(cand)
    
    # 2. 尝试解析JSON格式，提取products数组
    try:
        # 先尝试直接解析
        data = json.loads(text)
        if isinstance(data, dict) and "products" in data:
            products = data["products"]
            if isinstance(products, list):
                for prod in products:
                    if isinstance(prod, str) and prod.strip():
                        all_smiles.append(prod.strip())
    except (json.JSONDecodeError, ValueError, TypeError):
        # 如果直接解析失败，尝试使用json_repair
        if HAS_JSON_REPAIR:
            try:
                repaired = json_repair.repair_json(text)
                data = json.loads(repaired)
                if isinstance(data, dict) and "products" in data:
                    products = data["products"]
                    if isinstance(products, list):
                        for prod in products:
                            if isinstance(prod, str) and prod.strip():
                                all_smiles.append(prod.strip())
            except Exception:
                pass
    
    # 3. 如果还没有找到，尝试从文本中提取（可能是纯SMILES字符串）
    if not all_smiles:
        # 按空格分割文本，获取所有token
        tokens = text.split()
        candidates = []
        
        # SMILES基本规则：包含字母、数字、括号、特殊字符等
        smiles_char_pattern = r'^[A-Za-z0-9@+\-\[\]\(\)=#\\/%.]+$'
        # 必须包含至少一个字母（原子符号）
        has_letter_pattern = r'[A-Za-z]'
        
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            
            # 过滤掉明显的非SMILES内容
            if token.lower() in {"smiles", "/smiles", "<smiles>", "</smiles>", "<smiles", "smiles>"}:
                continue
            if token.startswith("</") or (token.startswith("<") and token.endswith(">")):
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?%?", token):
                continue
            if len(token) < 3:
                continue
            if not re.match(smiles_char_pattern, token):
                continue
            if not re.search(has_letter_pattern, token):
                continue
            
            candidates.append(token)
        
        if candidates:
            # 检查是否包含 '.' 分隔的多个SMILES（反应式格式）
            # 如果最长的候选包含 '.'，尝试分割
            longest = max(candidates, key=len)
            if '.' in longest and len(longest) > 20:  # 可能是多个SMILES用.连接
                parts = longest.split('.')
                for part in parts:
                    part = part.strip()
                    if part and len(part) >= 3 and re.search(has_letter_pattern, part):
                        all_smiles.append(part)
            else:
                all_smiles.extend(candidates)
    
    if not all_smiles:
        return None if not return_all else []
    
    # 4. 验证和过滤：优先选择RDKit可解析的
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        valid_smiles = []
        for sm in all_smiles:
            # 如果包含 '.'，可能是多个SMILES，尝试分割
            if '.' in sm and len(sm) > 20:
                parts = sm.split('.')
                for part in parts:
                    part = part.strip()
                    if part and Chem.MolFromSmiles(part) is not None:
                        valid_smiles.append(part)
            else:
                if Chem.MolFromSmiles(sm) is not None:
                    valid_smiles.append(sm)
        RDLogger.EnableLog('rdApp.*')
        
        if valid_smiles:
            if return_all:
                return valid_smiles
            # 返回最长的有效SMILES
            return max(valid_smiles, key=len)
    except Exception:
        pass
    
    # 如果没有RDKit可解析的，返回最长的
    if return_all:
        return all_smiles
    return max(all_smiles, key=len)


def extract_yield_from_text(text: str) -> Optional[float]:
    """从文本中提取yield百分比数值
    
    策略：
    1. 先尝试解析JSON格式，提取yield_percent或yield percent字段
    2. 匹配显式百分比模式（如"85%"）
    3. 匹配"yield"关键词附近的数字
    4. 回退到第一个合理的数字
    """
    if not text:
        return None
    text = str(text).strip()

    # 1) 尝试解析JSON格式（很多预测是JSON）
    def _extract_from_obj(obj: Any) -> Optional[float]:
        if not isinstance(obj, dict):
            return None
        # 优先匹配各种写法的 yield_percent / yield.percent / yield percent
        for k, v in obj.items():
            k_norm = str(k).strip().lower()
            if re.fullmatch(r"yield[_\. ]?percent", k_norm):
                try:
                    fv = float(v)
                    if 0 <= fv <= 100:
                        return fv
                    if 100 < fv <= 10000:
                        return fv / 100.0
                except (TypeError, ValueError):
                    continue
        # 退而求其次：有些旧格式只有 'yield' 或 'percent'
        for key in ("yield", "percent"):
            if key in obj:
                try:
                    fv = float(obj[key])
                    if 0 <= fv <= 100:
                        return fv
                    if 100 < fv <= 10000:
                        return fv / 100.0
                except (TypeError, ValueError):
                    continue
        return None

    try:
        data = json.loads(text)
        v = _extract_from_obj(data)
        if v is not None:
            return v
    except (json.JSONDecodeError, ValueError, TypeError):
        if HAS_JSON_REPAIR:
            try:
                repaired = json_repair.repair_json(text)
                data = json.loads(repaired)
                v = _extract_from_obj(data)
                if v is not None:
                    return v
            except Exception:
                pass
    
    # 2) 尝试从JSON字符串片段中提取（如 {"yield_percent":85.0}）
    json_patterns = [
        r'"yield_percent"\s*:\s*(\d+(?:\.\d+)?)',
        r'"yield\s+percent"\s*:\s*(\d+(?:\.\d+)?)',
        r'"yieldPercent"\s*:\s*(\d+(?:\.\d+)?)',
        r'"yield"\s*:\s*(\d+(?:\.\d+)?)',
    ]
    for pattern in json_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1))
                if 0 <= v <= 100:
                    return v
                if 100 < v <= 10000:
                    return v / 100.0
            except ValueError:
                continue

    # 3) Prefer explicit percent patterns like "85%" or "85 %"
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 100:
                return v
        except ValueError:
            pass

    # 4) Prefer numbers near the keyword "yield"
    m = re.search(r"yield\s*[:=]?\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        try:
            v = float(m.group(1))
            if 0 <= v <= 100:
                return v
        except ValueError:
            pass

    # 5) Fallback: first numeric token, with basic normalization
    number_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    matches = re.findall(number_pattern, text)
    for tok in matches:
        try:
            value = float(tok)
            if 0 <= value <= 100:
                return value
            if 100 < value <= 10000:
                return value / 100.0
        except ValueError:
            continue
    return None


def load_gvp_encoder(gvp_root: Optional[Path] = None, gvp_ckpt: Optional[str] = None, device: str = "cuda:0"):
    """加载GVP encoder"""
    global _gvp_encoder, _gvp_device
    
    if _gvp_encoder is not None:
        return _gvp_encoder
    
    if not gvp_root or not gvp_ckpt:
        return None
    
    try:
        # 添加GVP路径到sys.path
        gvp_root = Path(gvp_root).resolve()
        if str(gvp_root) not in sys.path:
            sys.path.insert(0, str(gvp_root))
        
        from modules.gnn import GVPEncoder
        import torch
        from collections import OrderedDict
        
        gvp_cfg = {
            "node_dims": (10, 1),
            "edge_dims": (1, 1),
            "hidden_scalar_dim": 256,
            "hidden_vector_dim": 16,
            "output_dim": 256,
            "num_layers": 4,
        }
        dev = torch.device(device)
        encoder = GVPEncoder(**gvp_cfg).to(dev)
        encoder.eval()
        
        sd = torch.load(gvp_ckpt, map_location="cpu")
        state = sd.get("model_state_dict", sd)
        state = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
        encoder.load_state_dict(state, strict=False)
        
        _gvp_encoder = encoder
        _gvp_device = device
        return encoder
    except Exception as e:
        print(f"[WARNING] Failed to load GVP encoder: {e}")
        return None


def calculate_gvp_similarity(pred_smiles: str, true_smiles: str, encoder) -> Optional[float]:
    """计算GVP embedding的cos相似度"""
    if not encoder or not pred_smiles or not true_smiles:
        return None
    
    try:
        import torch
        import torch.nn.functional as F
        
        # 获取embedding
        pred_emb = encoder.forward_from_smiles(pred_smiles)
        true_emb = encoder.forward_from_smiles(true_smiles)
        
        if pred_emb is None or true_emb is None:
            return None
        
        if isinstance(pred_emb, torch.Tensor):
            pred_emb = pred_emb.squeeze().detach()
        if isinstance(true_emb, torch.Tensor):
            true_emb = true_emb.squeeze().detach()
        
        # L2归一化
        pred_emb = F.normalize(pred_emb, p=2, dim=-1)
        true_emb = F.normalize(true_emb, p=2, dim=-1)
        
        # 计算cos相似度
        similarity = F.cosine_similarity(pred_emb.unsqueeze(0), true_emb.unsqueeze(0), dim=-1)
        return float(similarity.item())
    except Exception:
        return None


def calculate_smiles_similarity(pred_smiles: str, true_smiles: str, gvp_encoder=None) -> Dict[str, Any]:
    """计算SMILES相似度（同时返回GVP embedding相似度和RDKit指纹相似度）
    
    支持多个SMILES的最优匹配策略：
    1. 预测数量 = 真实值数量：找到最优匹配，计算平均相似度
    2. 预测数量 < 真实值数量：对每个真实值找最相似的预测，未匹配的真实值视为0，算平均
    3. 预测数量 > 真实值数量：只保留与真实值最相似的预测，多余的丢弃，算平均
    """
    result = {
        "valid": False,
        "exact_match": False,
        "gvp_similarity": None,
        "rdkit_similarity": None
    }
    
    if not pred_smiles or not true_smiles:
        return result
    
    # 处理多个SMILES的情况
    def split_smiles(smiles_str: str) -> List[str]:
        """分割可能包含多个SMILES的字符串"""
        if isinstance(smiles_str, list):
            return [str(s).strip() for s in smiles_str if s]
        smiles_str = str(smiles_str).strip()
        # 如果包含 '.' 且长度较长，可能是多个SMILES
        if '.' in smiles_str and len(smiles_str) > 20:
            parts = smiles_str.split('.')
            return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 3]
        return [smiles_str]
    
    pred_list = split_smiles(pred_smiles)
    true_list = split_smiles(true_smiles)
    
    # 如果只有一个SMILES，直接比较
    if len(pred_list) == 1 and len(true_list) == 1:
        pred_s = pred_list[0]
        true_s = true_list[0]
        
        # 先进行简单的字符串匹配
        if pred_s.strip() == true_s.strip():
            result.update({
                "valid": True,
                "exact_match": True,
                "gvp_similarity": 1.0,
                "rdkit_similarity": 1.0
            })
            return result
        
        # 计算GVP embedding相似度
        if gvp_encoder:
            gvp_sim = calculate_gvp_similarity(pred_s, true_s, gvp_encoder)
            if gvp_sim is not None:
                result["gvp_similarity"] = gvp_sim
        
        # 计算RDKit指纹相似度
        try:
            from rdkit import Chem, DataStructs, RDLogger
            
            RDLogger.DisableLog('rdApp.*')
            pred_mol = Chem.MolFromSmiles(pred_s)
            true_mol = Chem.MolFromSmiles(true_s)
            RDLogger.EnableLog('rdApp.*')
            
            if pred_mol is not None and true_mol is not None:
                pred_fp = Chem.RDKFingerprint(pred_mol)
                true_fp = Chem.RDKFingerprint(true_mol)
                rdkit_sim = DataStructs.TanimotoSimilarity(pred_fp, true_fp)
                result["rdkit_similarity"] = float(rdkit_sim)
                result["valid"] = True
        except Exception:
            pass
        
        return result
    
    # 多个SMILES的情况：使用最优匹配策略
    try:
        from rdkit import Chem, DataStructs, RDLogger
        RDLogger.DisableLog('rdApp.*')
        
        # 计算所有配对之间的相似度矩阵
        gvp_sim_matrix = []
        rdkit_sim_matrix = []
        exact_match_matrix = []
        
        for true_s in true_list:
            true_s = true_s.strip()
            gvp_row = []
            rdkit_row = []
            exact_row = []
            
            # 尝试解析真实值SMILES
            true_mol = None
            true_fp = None
            try:
                true_mol = Chem.MolFromSmiles(true_s)
                if true_mol is not None:
                    true_fp = Chem.RDKFingerprint(true_mol)
            except Exception:
                pass
            
            for pred_s in pred_list:
                pred_s = pred_s.strip()
                
                # 检查精确匹配
                if pred_s == true_s:
                    exact_row.append(True)
                    gvp_row.append(1.0)
                    rdkit_row.append(1.0)
                else:
                    exact_row.append(False)
                    
                    # 计算GVP相似度
                    gvp_sim = None
                    if gvp_encoder:
                        gvp_sim = calculate_gvp_similarity(pred_s, true_s, gvp_encoder)
                    gvp_row.append(gvp_sim)
                    
                    # 计算RDKit相似度
                    rdkit_sim = None
                    if true_mol is not None:
                        try:
                            pred_mol = Chem.MolFromSmiles(pred_s)
                            if pred_mol is not None:
                                pred_fp = Chem.RDKFingerprint(pred_mol)
                                rdkit_sim = DataStructs.TanimotoSimilarity(pred_fp, true_fp)
                        except Exception:
                            pass
                    rdkit_row.append(rdkit_sim)
            
            gvp_sim_matrix.append(gvp_row)
            rdkit_sim_matrix.append(rdkit_row)
            exact_match_matrix.append(exact_row)
        
        RDLogger.EnableLog('rdApp.*')
        
        # 最优匹配：对每个真实值，找到最相似的预测（贪心算法）
        # 策略：以真实值为基准，每个真实值匹配一个最相似的预测
        used_pred_indices = set()
        matched_gvp_sims = []
        matched_rdkit_sims = []
        has_exact_match = False
        
        for true_idx in range(len(true_list)):
            best_pred_idx = None
            best_gvp_sim = None
            best_rdkit_sim = None
            best_exact = False
            
            # 找到未使用的最相似预测
            for pred_idx in range(len(pred_list)):
                if pred_idx in used_pred_indices:
                    continue
                
                exact = exact_match_matrix[true_idx][pred_idx]
                gvp_sim = gvp_sim_matrix[true_idx][pred_idx]
                rdkit_sim = rdkit_sim_matrix[true_idx][pred_idx]
                
                # 优先选择精确匹配
                if exact:
                    best_pred_idx = pred_idx
                    best_gvp_sim = 1.0
                    best_rdkit_sim = 1.0
                    best_exact = True
                    break
                
                # 否则选择相似度最高的（优先RDKit，因为更稳定）
                if rdkit_sim is not None:
                    if best_rdkit_sim is None or rdkit_sim > best_rdkit_sim:
                        best_pred_idx = pred_idx
                        best_rdkit_sim = rdkit_sim
                        best_gvp_sim = gvp_sim
                elif gvp_sim is not None:
                    if best_gvp_sim is None or gvp_sim > best_gvp_sim:
                        best_pred_idx = pred_idx
                        best_gvp_sim = gvp_sim
                        best_rdkit_sim = rdkit_sim
            
            if best_exact:
                has_exact_match = True
            
            if best_pred_idx is not None:
                used_pred_indices.add(best_pred_idx)
                # 记录匹配的相似度（如果没有则记录0）
                matched_gvp_sims.append(best_gvp_sim if best_gvp_sim is not None else 0.0)
                matched_rdkit_sims.append(best_rdkit_sim if best_rdkit_sim is not None else 0.0)
            else:
                # 没有找到匹配的预测，视为0相似度
                matched_gvp_sims.append(0.0)
                matched_rdkit_sims.append(0.0)
        
        # 计算平均相似度（至少有一个真实值，所以matched列表不为空）
        if matched_gvp_sims:
            avg_gvp_sim = sum(matched_gvp_sims) / len(matched_gvp_sims)
            # 只有当至少有一个非零相似度时才设置
            if any(s > 0 for s in matched_gvp_sims):
                result["gvp_similarity"] = avg_gvp_sim
        
        if matched_rdkit_sims:
            avg_rdkit_sim = sum(matched_rdkit_sims) / len(matched_rdkit_sims)
            # 只有当至少有一个非零相似度时才设置
            if any(s > 0 for s in matched_rdkit_sims):
                result["rdkit_similarity"] = avg_rdkit_sim
                result["valid"] = True
        
        if has_exact_match:
            result["exact_match"] = True
        
    except Exception as e:
        pass
    
    return result


def calculate_yield_metrics(pred_yields: List[float], true_yields: List[float]) -> Dict[str, float]:
    """计算yield的多项指标"""
    if not pred_yields or not true_yields:
        return {}
    
    pred_yields = np.array(pred_yields)
    true_yields = np.array(true_yields)
    
    # MAE和RMSE
    errors = pred_yields - true_yields
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mean_error = float(np.mean(errors))
    
    # 四舍五入到 10% 档后的准确率（严格和宽松）
    # 例如：67%→70%，74%→70%，使用最近的 10% 档比较
    pred_bins_round = np.round(pred_yields / 10.0).astype(int)  # 0–10
    true_bins_round = np.round(true_yields / 10.0).astype(int)
    
    # 严格准确率：档位完全相同（例如都在 70% 档）
    strict_acc = float(np.mean(pred_bins_round == true_bins_round))
    
    # 宽松准确率：±1 档内（窗口长度为 3 个档：例如 60/70/80 都算对）
    relaxed_acc = float(np.mean(np.abs(pred_bins_round - true_bins_round) <= 1))
    
    # NDCG计算（基于 10% 档的整数编码）
    # 将yield值映射到bin（0-100，每10%一个bin，共10个bin）
    def yield_to_bin(y):
        return int(np.clip(y / 10.0, 0, 9))
    
    true_bins = [yield_to_bin(y) for y in true_yields]
    pred_bins = [yield_to_bin(y) for y in pred_yields]
    
    # 计算NDCG（每个样本的bin作为相关性）
    ndcg_scores = []
    for pred_bin, true_bin in zip(pred_bins, true_bins):
        # 相关性：距离真值bin越近，得分越高
        relevance = 1.0 / (1.0 + abs(pred_bin - true_bin))
        
        # 理想情况：完美排序（真值bin排第一）
        ideal_relevance = 1.0
        idcg = ideal_relevance / np.log2(2)  # rank 1
        
        # 实际排序：按预测bin排序（这里简化为直接使用预测值）
        dcg = relevance / np.log2(2)  # rank 1
        
        ndcg = dcg / (idcg + 1e-8)
        ndcg_scores.append(ndcg)
    
    avg_ndcg = float(np.mean(ndcg_scores))
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mean_error": mean_error,
        "strict_accuracy": strict_acc,
        "relaxed_accuracy": relaxed_acc,
        "ndcg": avg_ndcg
    }


def extract_role_from_text(text: str) -> Optional[str]:
    """从文本中提取role（分类任务的类别）"""
    if not text:
        return None
    # 移除SMILES标签，提取文本内容
    text = re.sub(r'<SMILES>.*?</SMILES>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = text.strip()
    # 如果文本为空或只包含标点，返回None
    if not text or text in ['.', ',', ';', ':', '!', '?']:
        return None
    return text


def calculate_task_metrics(results_dir: Path, gvp_encoder=None) -> Dict[str, Any]:
    """按任务分别计算指标"""
    task_metrics = {}
    prediction_files = list(results_dir.glob("*_predictions.jsonl"))
    
    for pred_file in tqdm(prediction_files, desc="Calculating metrics"):
        task_name = pred_file.stem.replace("_predictions", "")
        
        predictions = []
        ground_truths = []
        valid_samples = 0
        total_samples = 0
        
        with open(pred_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                    total_samples += 1
                    
                    if result.get("prediction") is None or result.get("error"):
                        continue
                    
                    predictions.append(result["prediction"])
                    ground_truths.append(result["ground_truth"])
                    valid_samples += 1
                except Exception:
                    continue
        
        if valid_samples == 0:
            task_metrics[task_name] = {
                "total_samples": total_samples,
                "valid_samples": 0,
                "error": "No valid predictions"
            }
            continue
        
        metrics = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "valid_rate": valid_samples / total_samples if total_samples > 0 else 0.0
        }
        
        # SMILES任务（mask_product, mask_reactant）
        if task_name in ["mask_product", "mask_reactant"]:
            gvp_similarities = []
            rdkit_similarities = []
            exact_matches = 0
            invalid_pred_count = 0
            invalid_true_count = 0
            error_count = 0
            
            for pred, true in zip(predictions, ground_truths):
                pred_smiles = extract_smiles_from_text(pred)
                true_smiles = extract_smiles_from_text(true)
                
                if not pred_smiles:
                    invalid_pred_count += 1
                    continue
                if not true_smiles:
                    invalid_true_count += 1
                    continue
                
                sim_result = calculate_smiles_similarity(pred_smiles, true_smiles, gvp_encoder)
                if sim_result["valid"] or sim_result["exact_match"]:
                    if sim_result["exact_match"]:
                        exact_matches += 1
                    if sim_result["gvp_similarity"] is not None:
                        gvp_similarities.append(sim_result["gvp_similarity"])
                    if sim_result["rdkit_similarity"] is not None:
                        rdkit_similarities.append(sim_result["rdkit_similarity"])
                    if not sim_result["exact_match"] and not (sim_result["gvp_similarity"] or sim_result["rdkit_similarity"]):
                        error_count += 1
                else:
                    error_count += 1
            
            # 总是更新这些基本指标
            metrics.update({
                "exact_matches": exact_matches,
                "exact_match_rate": exact_matches / valid_samples if valid_samples > 0 else 0.0,
                "invalid_pred_count": invalid_pred_count,
                "invalid_true_count": invalid_true_count,
                "parsing_error_count": error_count
            })
            
            if gvp_similarities:
                metrics.update({
                    "gvp_avg_similarity": float(np.mean(gvp_similarities)),
                    "gvp_median_similarity": float(np.median(gvp_similarities)),
                    "gvp_std_similarity": float(np.std(gvp_similarities)) if len(gvp_similarities) > 1 else 0.0,
                    "num_gvp_similarity": len(gvp_similarities)
                })
            
            if rdkit_similarities:
                metrics.update({
                    "rdkit_avg_similarity": float(np.mean(rdkit_similarities)),
                    "rdkit_median_similarity": float(np.median(rdkit_similarities)),
                    "rdkit_std_similarity": float(np.std(rdkit_similarities)) if len(rdkit_similarities) > 1 else 0.0,
                    "num_rdkit_similarity": len(rdkit_similarities)
                })
            
            # 如果没有有效的相似度计算，添加警告
            if not gvp_similarities and not rdkit_similarities and exact_matches == 0:
                metrics["warning"] = "No valid SMILES pairs found for similarity calculation"
        
        # mask_role是分类任务
        elif task_name == "mask_role":
            correct = 0
            total_classified = 0
            invalid_pred_count = 0
            invalid_true_count = 0
            
            for pred, true in zip(predictions, ground_truths):
                pred_role = extract_role_from_text(pred)
                true_role = extract_role_from_text(true)
                
                if not pred_role:
                    invalid_pred_count += 1
                    continue
                if not true_role:
                    invalid_true_count += 1
                    continue
                
                total_classified += 1
                # 简单的字符串匹配（可以后续改进为更复杂的匹配）
                if pred_role.strip().lower() == true_role.strip().lower():
                    correct += 1
            
            if total_classified > 0:
                metrics.update({
                    "accuracy": correct / total_classified,
                    "correct": correct,
                    "total_classified": total_classified,
                    "invalid_pred_count": invalid_pred_count,
                    "invalid_true_count": invalid_true_count
                })
            else:
                metrics.update({
                    "accuracy": 0.0,
                    "correct": 0,
                    "total_classified": 0,
                    "invalid_pred_count": invalid_pred_count,
                    "invalid_true_count": invalid_true_count,
                    "warning": "No valid role predictions found"
                })
        
        # Yield任务
        elif task_name in ["predict_yield_full", "predict_product_and_yield"]:
            pred_yields = []
            true_yields = []
            invalid_pred_count = 0
            invalid_true_count = 0
            
            for pred, true in zip(predictions, ground_truths):
                pred_yield = extract_yield_from_text(pred)
                true_yield = extract_yield_from_text(true)
                
                # 确保yield值在合理范围内（0-100）
                if pred_yield is not None:
                    if pred_yield < 0 or pred_yield > 100:
                        invalid_pred_count += 1
                        continue
                else:
                    invalid_pred_count += 1
                    continue
                
                if true_yield is not None:
                    if true_yield < 0 or true_yield > 100:
                        invalid_true_count += 1
                        continue
                else:
                    invalid_true_count += 1
                    continue
                
                pred_yields.append(pred_yield)
                true_yields.append(true_yield)
            
            if pred_yields:
                yield_metrics = calculate_yield_metrics(pred_yields, true_yields)
                metrics.update(yield_metrics)
                metrics["num_yield_calculated"] = len(pred_yields)
                metrics["invalid_pred_count"] = invalid_pred_count
                metrics["invalid_true_count"] = invalid_true_count
            else:
                metrics.update({
                    "num_yield_calculated": 0,
                    "invalid_pred_count": invalid_pred_count,
                    "invalid_true_count": invalid_true_count,
                    "warning": "No valid yield predictions found"
                })
            
            # predict_product_and_yield还需要计算SMILES相似度
            if task_name == "predict_product_and_yield":
                gvp_similarities = []
                rdkit_similarities = []
                product_exact_matches = 0
                product_error_count = 0
                
                for pred, true in zip(predictions, ground_truths):
                    pred_smiles = extract_smiles_from_text(pred)
                    true_smiles = extract_smiles_from_text(true)
                    
                    if pred_smiles and true_smiles:
                        sim_result = calculate_smiles_similarity(pred_smiles, true_smiles, gvp_encoder)
                        if sim_result["valid"] or sim_result["exact_match"]:
                            if sim_result["exact_match"]:
                                product_exact_matches += 1
                            if sim_result["gvp_similarity"] is not None:
                                gvp_similarities.append(sim_result["gvp_similarity"])
                            if sim_result["rdkit_similarity"] is not None:
                                rdkit_similarities.append(sim_result["rdkit_similarity"])
                        else:
                            product_error_count += 1
                
                if gvp_similarities or rdkit_similarities or product_exact_matches > 0:
                    metrics.update({
                        "product_exact_matches": product_exact_matches,
                        "product_exact_match_rate": product_exact_matches / len(predictions) if predictions else 0.0,
                        "product_error_count": product_error_count
                    })
                    
                    if gvp_similarities:
                        metrics.update({
                            "product_gvp_avg_similarity": float(np.mean(gvp_similarities)),
                            "product_gvp_median_similarity": float(np.median(gvp_similarities)),
                            "num_product_gvp_similarity": len(gvp_similarities)
                        })
                    
                    if rdkit_similarities:
                        metrics.update({
                            "product_rdkit_avg_similarity": float(np.mean(rdkit_similarities)),
                            "product_rdkit_median_similarity": float(np.median(rdkit_similarities)),
                            "num_product_rdkit_similarity": len(rdkit_similarities)
                        })
        
        task_metrics[task_name] = metrics
    
    return task_metrics


def plot_task_metrics(task_metrics: Dict[str, Any], output_dir: Path):
    """生成任务指标的可视化图表"""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    tasks = list(task_metrics.keys())
    
    # 1. SMILES相似度对比图（GVP和RDKit）
    smiles_tasks = [t for t in tasks if t in ["mask_product", "mask_reactant"]]
    if smiles_tasks:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        gvp_similarities = []
        rdkit_similarities = []
        exact_rates = []
        
        for task in smiles_tasks:
            metrics = task_metrics[task]
            gvp_sim = metrics.get("gvp_avg_similarity", 0.0)
            rdkit_sim = metrics.get("rdkit_avg_similarity", 0.0)
            exact_rate = metrics.get("exact_match_rate", 0.0)
            gvp_similarities.append(gvp_sim if gvp_sim > 0 else None)
            rdkit_similarities.append(rdkit_sim if rdkit_sim > 0 else None)
            exact_rates.append(exact_rate)
        
        x = np.arange(len(smiles_tasks))
        width = 0.35
        
        # GVP相似度
        ax = axes[0]
        bars1 = ax.bar(x - width/2, [s if s is not None else 0 for s in gvp_similarities], width, 
                      label='GVP', color='steelblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, [s if s is not None else 0 for s in rdkit_similarities], width,
                      label='RDKit', color='coral', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(smiles_tasks, rotation=45, ha='right')
        ax.set_ylabel('Average Similarity')
        ax.set_title('SMILES Similarity (GVP vs RDKit)')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 精确匹配率
        ax = axes[1]
        ax.bar(range(len(smiles_tasks)), exact_rates, color='green', alpha=0.7)
        ax.set_xticks(range(len(smiles_tasks)))
        ax.set_xticklabels(smiles_tasks, rotation=45, ha='right')
        ax.set_ylabel('Exact Match Rate')
        ax.set_title('Exact Match Rate by Task')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # 对比图
        ax = axes[2]
        all_metrics = []
        all_labels = []
        all_colors = []
        for task in smiles_tasks:
            metrics = task_metrics[task]
            if metrics.get("gvp_avg_similarity"):
                all_metrics.append(metrics["gvp_avg_similarity"])
                all_labels.append(f"{task}\nGVP")
                all_colors.append('steelblue')
            if metrics.get("rdkit_avg_similarity"):
                all_metrics.append(metrics["rdkit_avg_similarity"])
                all_labels.append(f"{task}\nRDKit")
                all_colors.append('coral')
        if all_metrics:
            ax.bar(range(len(all_metrics)), all_metrics, color=all_colors, alpha=0.7)
            ax.set_xticks(range(len(all_metrics)))
            ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Similarity')
            ax.set_title('Detailed Similarity Comparison')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "smiles_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 1.5 mask_role分类准确率
    if "mask_role" in tasks:
        metrics = task_metrics["mask_role"]
        if "accuracy" in metrics:
            fig, ax = plt.subplots(figsize=(6, 5))
            acc = metrics["accuracy"]
            ax.bar(["mask_role"], [acc], color='purple', alpha=0.7)
            ax.set_ylabel('Accuracy')
            ax.set_title(f'mask_role Classification Accuracy\n{acc:.4f}')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / "mask_role_accuracy.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2. Yield预测多指标图
    yield_tasks = [t for t in tasks if "mae" in task_metrics[t]]
    if yield_tasks:
        for task in yield_tasks:
            metrics = task_metrics[task]
            mae = metrics.get("mae", 0)
            rmse = metrics.get("rmse", 0)
            strict_acc = metrics.get("strict_accuracy", 0)
            relaxed_acc = metrics.get("relaxed_accuracy", 0)
            ndcg = metrics.get("ndcg", 0)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # 误差指标
            ax = axes[0]
            bars = ax.bar(["MAE", "RMSE"], [mae, rmse], color=['steelblue', 'coral'], alpha=0.7)
            ax.set_ylabel('Error')
            ax.set_title(f'{task} - Error Metrics\nMAE: {mae:.2f}, RMSE: {rmse:.2f}')
            ax.grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom')
            
            # 准确率和NDCG
            ax = axes[1]
            bars = ax.bar(["Strict\nAcc", "Relaxed\nAcc", "NDCG"], 
                         [strict_acc, relaxed_acc, ndcg], 
                         color=['red', 'orange', 'green'], alpha=0.7)
            ax.set_ylabel('Score')
            ax.set_title(f'{task} - Accuracy & NDCG')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            safe_task_name = task.replace('/', '_')
            plt.savefig(plots_dir / f"yield_metrics_{safe_task_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # 3. 样本数量对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    total_samples = [task_metrics[t].get("total_samples", 0) for t in tasks]
    valid_samples = [task_metrics[t].get("valid_samples", 0) for t in tasks]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    ax.bar(x - width/2, total_samples, width, label='Total', color='lightgray', alpha=0.7)
    ax.bar(x + width/2, valid_samples, width, label='Valid', color='steelblue', alpha=0.7)
    
    ax.set_xlabel('Task')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Sample Count by Task')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "sample_counts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 综合指标汇总表
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ["Task", "Total", "Valid", "Valid Rate", "Main Metric", "Value"]
    
    for task in tasks:
        metrics = task_metrics[task]
        row = [
            task,
            metrics.get("total_samples", 0),
            metrics.get("valid_samples", 0),
            f"{metrics.get('valid_rate', 0)*100:.1f}%"
        ]
        
        # Pick a representative metric per task
        if "accuracy" in metrics:
            row.extend(["Accuracy", f"{metrics.get('accuracy', 0.0):.4f}"])
        elif "mae" in metrics:
            row.extend(["MAE", f"{metrics.get('mae', 0.0):.2f}"])
        else:
            gvp = metrics.get("gvp_avg_similarity", None)
            rd = metrics.get("rdkit_avg_similarity", None)
            if gvp is not None:
                row.extend(["GVP Avg Sim", f"{gvp:.4f}"])
            elif rd is not None:
                row.extend(["RDKit Avg Sim", f"{rd:.4f}"])
            else:
                row.extend(["N/A", "N/A"])
        
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Task Metrics Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(plots_dir / "metrics_summary.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Score and visualize Layer2 testset evaluation results")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing evaluation results (with *_predictions.jsonl files)"
    )
    parser.add_argument(
        "--gvp-root",
        type=str,
        default=None,
        help="Path to GVP-GNN root directory (optional, for GVP embedding similarity)"
    )
    parser.add_argument(
        "--gvp-ckpt",
        type=str,
        default=None,
        help="Path to GVP checkpoint file (optional, for GVP embedding similarity)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for GVP encoder (default: cuda:0)"
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"[ERROR] Results directory does not exist: {results_dir}")
        return 1
    
    # 加载GVP encoder（如果提供）
    gvp_encoder = None
    if args.gvp_root and args.gvp_ckpt:
        print(f"[INFO] Loading GVP encoder from: {args.gvp_ckpt}")
        gvp_encoder = load_gvp_encoder(Path(args.gvp_root), args.gvp_ckpt, args.device)
        if gvp_encoder:
            print("[INFO] GVP encoder loaded successfully")
        else:
            print("[WARNING] Failed to load GVP encoder, will skip GVP similarity calculation")
    else:
        print("[INFO] GVP encoder not provided, will skip GVP similarity calculation")
    
    print(f"[INFO] Calculating metrics for results in: {results_dir}")
    task_metrics = calculate_task_metrics(results_dir, gvp_encoder)
    
    # 保存指标
    metrics_file = results_dir / "task_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(task_metrics, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Metrics saved to: {metrics_file}")
    
    # 打印指标摘要
    print("\n" + "="*60)
    print("Task Metrics Summary")
    print("="*60)
    for task, metrics in task_metrics.items():
        print(f"\n{task}:")
        print(f"  Total samples: {metrics.get('total_samples', 0)}")
        print(f"  Valid samples: {metrics.get('valid_samples', 0)}")
        if "accuracy" in metrics:
            print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        if "gvp_avg_similarity" in metrics or "rdkit_avg_similarity" in metrics or "exact_match_rate" in metrics:
            if metrics.get("gvp_avg_similarity") is not None:
                print(f"  GVP Avg Similarity: {metrics.get('gvp_avg_similarity', 0):.4f} (n={metrics.get('num_gvp_similarity', 0)})")
            if metrics.get("rdkit_avg_similarity") is not None:
                print(f"  RDKit Avg Similarity: {metrics.get('rdkit_avg_similarity', 0):.4f} (n={metrics.get('num_rdkit_similarity', 0)})")
            if "exact_match_rate" in metrics:
                print(f"  Exact Match Rate: {metrics.get('exact_match_rate', 0):.4f} (n={metrics.get('exact_matches', 0)})")
            if "invalid_pred_count" in metrics:
                print(f"  Invalid Predictions: {metrics.get('invalid_pred_count', 0)}")
            if "invalid_true_count" in metrics:
                print(f"  Invalid Ground Truth: {metrics.get('invalid_true_count', 0)}")
            if "parsing_error_count" in metrics:
                print(f"  Parsing Errors: {metrics.get('parsing_error_count', 0)}")
            if "warning" in metrics:
                print(f"  Warning: {metrics.get('warning', '')}")
        if "mae" in metrics:
            mae = metrics.get('mae', 0)
            rmse = metrics.get('rmse', 0)
            mean_err = metrics.get('mean_error', 0)
            strict_acc = metrics.get('strict_accuracy', 0)
            relaxed_acc = metrics.get('relaxed_accuracy', 0)
            ndcg = metrics.get('ndcg', 0)
            num_yield = metrics.get('num_yield_calculated', 0)
            invalid_pred = metrics.get('invalid_pred_count', 0)
            invalid_true = metrics.get('invalid_true_count', 0)
            print(f"  Yield Metrics:")
            print(f"    Valid samples (with prediction): {metrics.get('valid_samples', 0)}")
            print(f"    Successfully parsed yield: {num_yield} (used for MAE/RMSE)")
            if invalid_pred > 0:
                print(f"    Failed to parse prediction yield: {invalid_pred}")
            if invalid_true > 0:
                print(f"    Failed to parse ground truth yield: {invalid_true}")
            print(f"    MAE: {mae:.2f} (only calculated on {num_yield} parsed samples)")
            print(f"    RMSE: {rmse:.2f} (only calculated on {num_yield} parsed samples)")
            print(f"    Mean Error: {mean_err:.2f}")
            print(f"    Strict Accuracy (rounded): {strict_acc:.4f}")
            print(f"    Relaxed Accuracy (±1, rounded): {relaxed_acc:.4f}")
            print(f"    NDCG: {ndcg:.4f}")
    
    # 生成可视化
    print("\n[INFO] Generating visualization plots...")
    plot_task_metrics(task_metrics, results_dir)
    print(f"[INFO] Plots saved to: {results_dir}/plots/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
