from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


ROLE_SOLVENT = "SOLVENT"
ROLE_CATALYST = "CATALYST"
ROLE_REAGENT = "REAGENT"
ROLE_REACTANT = "REACTANT"
ROLE_PRODUCT = "PRODUCT"


@dataclass(frozen=True)
class MaskingConfig:
    seed: int = 42
    p_forward: float = 0.4
    p_retro: float = 0.3
    p_condition: float = 0.2
    p_random: float = 0.1
    p_yield_full: float = 0.0       # task2a: full info, predict yield
    p_yield_with_product: float = 0.0  # task2b: mask product + predict yield

    # retro 任务：随机 mask 的 INPUT token 数量范围（含端点）
    retro_min_mask: int = 1
    retro_max_mask: int = 2

    # random 任务：随机 mask token 的比例
    random_token_ratio: float = 0.15

    # amount mask 的占位值（模型侧再用可学习向量/参数替换）
    mask_amt_value: float = 0.0

    # 是否在 forward/retro/random 中同时 mask amount
    mask_amount_in_forward: bool = True
    mask_amount_in_retro: bool = True
    mask_amount_in_random: bool = True


@dataclass(frozen=True)
class EvalMaskingConfig:
    """
    评测专用 MaskingConfig，与 LLM template 任务完全一致：
    - 任务1 (task1_mask_product): Forward Synthesis - mask产物，预测产物
    - 任务2a (task2a_predict_yield_full): 完整表达式直接预测yield（有yield的数据，根据reaction_id随机分一半）
    - 任务2b (task2b_predict_product_and_yield): mask产物，预测产物和yield（有yield的数据，根据reaction_id随机分一半）
    - 任务3 (task3_mask_role): mask反应条件（溶剂/催化剂/试剂）
    - 任务4 (task4_mask_reactant): Retrosynthesis - mask反应物
    
    注意：任务2a和2b使用reaction_id的hash值决定，确保与LLM template一致
    """
    seed: int = 42
    # 评测任务概率（与LLM template生成逻辑一致）
    # 对于有yield的数据，任务2会被随机分为2a和2b（各占一半）
    p_task1_mask_product: float = 0.25      # Forward Synthesis: mask产物
    p_task4_mask_reactant: float = 0.25     # Retrosynthesis: mask反应物
    p_task3_mask_role: float = 0.2          # mask反应条件
    p_task2_yield: float = 0.3              # yield任务总概率（会被分为2a和2b）

    # retro 任务：随机 mask 的 INPUT token 数量范围（含端点）
    retro_min_mask: int = 1
    retro_max_mask: int = 2

    # amount mask 的占位值（模型侧再用可学习向量/参数替换）
    mask_amt_value: float = 0.0

    # 是否在 forward/retro 中同时 mask amount
    mask_amount_in_forward: bool = True
    mask_amount_in_retro: bool = True


def _stable_rng(seed: int, reaction_id: str, extra: str) -> random.Random:
    h = hashlib.md5(f"{seed}\t{reaction_id}\t{extra}".encode("utf-8")).digest()
    rng_seed = int.from_bytes(h[:8], byteorder="little", signed=False)
    return random.Random(rng_seed)


def _choose_task(rng: random.Random, cfg: MaskingConfig, has_yield: bool = False) -> str:
    ps = [cfg.p_forward, cfg.p_retro, cfg.p_condition, cfg.p_random,
          cfg.p_yield_full, cfg.p_yield_with_product]
    s = sum(ps)
    if s <= 0:
        return "forward"
    r = rng.random() * s
    cum = 0.0
    tasks = ["forward", "retro", "condition", "random",
             "yield_full", "yield_with_product"]
    for p, task in zip(ps, tasks):
        cum += p
        if r < cum:
            # yield tasks require has_yield; fall back to forward if not available
            if task in ("yield_full", "yield_with_product") and not has_yield:
                return "forward"
            return task
    return "random"


def _choose_eval_task(
    rng: random.Random, 
    cfg: EvalMaskingConfig, 
    reaction_id: str,
    has_yield: bool
) -> str:
    """
    评测专用任务选择，与 LLM template 任务完全一致。
    
    任务分配逻辑：
    1. 先选择任务类型（task1/task3/task4/yield总任务）
    2. 如果选择了yield任务且有yield数据，根据reaction_id的hash决定是task2a还是task2b（各占一半）
    """
    ps = [cfg.p_task1_mask_product, cfg.p_task4_mask_reactant, cfg.p_task3_mask_role, cfg.p_task2_yield]
    s = sum(ps)
    if s <= 0:
        return "task1_mask_product"
    
    r = rng.random() * s
    if r < ps[0]:
        return "task1_mask_product"  # Forward Synthesis
    r -= ps[0]
    if r < ps[1]:
        return "task4_mask_reactant"  # Retrosynthesis
    r -= ps[1]
    if r < ps[2]:
        return "task3_mask_role"  # mask reaction conditions
    # 否则是 yield 任务
    if has_yield:
        # 使用 reaction_id 的 hash 决定是 task2a 还是 task2b（与 LLM template 逻辑一致）
        import hashlib
        seed_hash = int(hashlib.md5(reaction_id.encode()).hexdigest(), 16)
        if (seed_hash % 2) == 0:
            return "task2a_predict_yield_full"
        else:
            return "task2b_predict_product_and_yield"
    else:
        # 没有 yield 数据，回退到 task1
        return "task1_mask_product"


def _mask_amount_fields(token: dict[str, Any], cfg: MaskingConfig | EvalMaskingConfig, targets: dict[str, Any]) -> None:
    for ch in ("moles", "mass", "volume"):
        key_v = f"amt_{ch}_log"
        key_m = f"amt_{ch}_mask"
        key_pm = f"amt_{ch}_pred_mask"

        if int(token.get(key_m, 0)) != 1:
            token[key_pm] = 0
            continue

        true_v = token.get(key_v)
        if true_v is None:
            token[key_pm] = 0
            continue

        token[key_pm] = 1
        targets.setdefault("amount", []).append((token["_idx"], ch, true_v))
        token[key_v] = float(cfg.mask_amt_value)


def apply_dynamic_mask(
    example: dict[str, Any], 
    cfg: MaskingConfig | EvalMaskingConfig, 
    *, 
    view_id: int = 0
) -> dict[str, Any]:
    """
    输入：一条 Layer2 reaction 记录（从 layer2_*.jsonl 读入的 dict）
    输出：masked view（包含 pred_mask 与 targets，用于训练）

    约定：
    - 该函数只负责“生成 mask 视图”，不做 tensor 化，也不做 padding。
    - `emb` 被 mask 时设为 None，并记录 `emb_pred_mask=1`，同时把真值存到 targets。
    """
    reaction_id = str(example.get("reaction_id", ""))
    rng = _stable_rng(cfg.seed, reaction_id, f"view:{view_id}")

    # 深拷贝，避免污染原始数据
    ex = copy.deepcopy(example)
    tokens: List[dict[str, Any]] = ex.get("tokens") or []
    if not isinstance(tokens, list):
        raise ValueError("example.tokens 必须是 list")

    # 为 tokens 补充稳定索引，方便 targets 记录
    for i, t in enumerate(tokens):
        if not isinstance(t, dict):
            raise ValueError("tokens[*] 必须是 dict")
        t["_idx"] = i
        t["emb_pred_mask"] = 0
        t["amt_moles_pred_mask"] = 0
        t["amt_mass_pred_mask"] = 0
        t["amt_volume_pred_mask"] = 0

    # 可选：序列顺序增强（shuffle）
    rng.shuffle(tokens)
    # shuffle 后重建 idx（targets 用新的位置）
    for i, t in enumerate(tokens):
        t["_idx"] = i

    # 根据 config 类型选择任务
    has_yield = (ex.get("yield_reg") is not None) or (ex.get("yield_bin") is not None)
    if isinstance(cfg, EvalMaskingConfig):
        task = _choose_eval_task(rng, cfg, reaction_id, has_yield)
    else:
        task = _choose_task(rng, cfg, has_yield=has_yield)
    targets: Dict[str, Any] = {"task": task, "embedding": [], "amount": []}

    # helper：mask embedding
    def _mask_emb(t: dict[str, Any]) -> None:
        emb = t.get("emb")
        if emb is None:
            return
        t["emb_pred_mask"] = 1
        targets["embedding"].append((t["_idx"], emb))
        t["emb"] = None

    # 选择 token 子集
    input_tokens = [t for t in tokens if t.get("token_type") == "INPUT"]
    outcome_tokens = [t for t in tokens if t.get("token_type") == "OUTCOME"]

    # 任务映射：eval任务名称 -> 训练任务逻辑
    # task1_mask_product (Forward Synthesis): mask产物
    # task4_mask_reactant (Retrosynthesis): mask反应物
    # task3_mask_role: mask反应条件
    # task2a_predict_yield_full: 完整表达式预测yield（不mask任何token）
    # task2b_predict_product_and_yield: mask产物，预测产物和yield
    
    if task == "task1_mask_product" or task == "forward":
        # task1_mask_product: Forward Synthesis - mask产物
        for t in outcome_tokens:
            _mask_emb(t)
            if cfg.mask_amount_in_forward:
                _mask_amount_fields(t, cfg, targets)
        ex["yield_pred_mask"] = 0  # task1不预测yield

    elif task == "task4_mask_reactant" or task == "retro":
        # task4_mask_reactant: Retrosynthesis - mask反应物
        k_min = max(1, cfg.retro_min_mask)
        k_max = max(k_min, cfg.retro_max_mask)
        k = min(len(input_tokens), rng.randint(k_min, k_max))
        for t in rng.sample(input_tokens, k=k):
            _mask_emb(t)
            if cfg.mask_amount_in_retro:
                _mask_amount_fields(t, cfg, targets)
        ex["yield_pred_mask"] = 0  # task4不预测yield

    elif task == "task3_mask_role" or task == "condition":
        # task3_mask_role: mask反应条件（溶剂/催化剂/试剂）
        for t in input_tokens:
            role = t.get("reaction_role")
            if role in {ROLE_SOLVENT, ROLE_CATALYST, ROLE_REAGENT}:
                _mask_emb(t)
        ex["yield_pred_mask"] = 0  # task3不预测yield

    elif task == "task2a_predict_yield_full" or task == "yield_full":
        # task2a: 完整表达式直接预测yield（不mask任何token）
        ex["yield_pred_mask"] = 1
        
    elif task == "task2b_predict_product_and_yield" or task == "yield_with_product":
        # task2b: mask产物，预测产物和yield
        for t in outcome_tokens:
            _mask_emb(t)
            if cfg.mask_amount_in_forward:
                _mask_amount_fields(t, cfg, targets)
        ex["yield_pred_mask"] = 1
        
    else:  # random (仅在训练时使用)
        if not isinstance(cfg, EvalMaskingConfig):
            # mask 随机比例 token 的 embedding + 可选 amount
            n = len(tokens)
            k = max(1, int(round(n * cfg.random_token_ratio)))
            for t in rng.sample(tokens, k=min(k, n)):
                _mask_emb(t)
                if cfg.mask_amount_in_random:
                    _mask_amount_fields(t, cfg, targets)
        ex["yield_pred_mask"] = 0

    # 清理内部字段，避免进入模型输入
    for t in tokens:
        t.pop("_idx", None)

    ex["tokens"] = tokens
    ex["targets"] = targets
    return ex

