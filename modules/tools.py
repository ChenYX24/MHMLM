# tools/chem_tools_combined.py
# -*- coding: utf-8 -*-
"""
合并工具：
- 工具1：从文本中提取化学实体名（chemdataextractor）并映射到 SMILES（SQLite），以及直接从文本中识别 SMILES（正则+RDKit校验）
- 工具2：将扩散模型生成的 RDKit Mol 列表，稳定地转换为代表性 SMILES

依赖（可选）：
- RDKit（强烈推荐，用于 SMILES 校验与 canonical 化）
- chemdataextractor（用于文本化合物实体抽取）
- SQLite（本地 compounds/synonyms 数据库）
"""

from typing import List, Dict, Optional, Tuple, Set
import re
import sqlite3

from rdkit import Chem
from rdkit.Chem import AllChem
_HAS_RDKIT = True


# Lazy import to avoid network issues during stanza initialization
cde = None
_HAS_CDE = False
def _ensure_cde():
    """Lazy import of chemdataextractor to avoid network issues"""
    global cde, _HAS_CDE
    if cde is None and not _HAS_CDE:
        try:
            # 设置环境变量禁用自动下载（如果可能）
            import os
            os.environ.setdefault('CDE_DISABLE_DOWNLOADS', '1')
            # 禁用 stanza 自动下载
            os.environ.setdefault('STANZA_RESOURCES_DIR', '/tmp/stanza_resources')
            import chemdataextractor as cde_module
            cde = cde_module
            _HAS_CDE = True
            # 只在成功时打印，避免重复警告
        except Exception as e:
            # 静默处理，避免重复打印警告
            if not _HAS_CDE:
                print(f"Warning: chemdataextractor import failed: {e}. Name-to-SMILES mapping will be disabled.")
            cde = None
            _HAS_CDE = True  # 标记为已尝试，避免重复尝试
    return cde

# 延迟初始化，不在模块导入时立即执行
cde = None


# ======================================================================
#                              工具1：文本 → SMILES
# ======================================================================

# --- 1A. 名称映射（SQLite） ---

def create_high_freq_table(db_path: str) -> None:
    """
    创建/确保高频表存在：high_freq_compounds(name TEXT PRIMARY KEY, smiles TEXT NOT NULL)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS high_freq_compounds (
            name   TEXT PRIMARY KEY,
            smiles TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def query_smiles_by_name(name: str, db_path: str) -> Optional[str]:
    """
    按名称检索 SMILES：优先 high_freq_compounds；否则查 compounds；再查 synonyms。
    若在后两者命中，会回填到 high_freq_compounds（缓存）。
    """
    name = (name or "").strip().lower()
    if not name:
        return None

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 高频表
    cur.execute("SELECT smiles FROM high_freq_compounds WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        conn.close()
        return row[0]

    # compounds 精确匹配
    cur.execute("SELECT smiles FROM compounds WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        cur.execute("INSERT OR REPLACE INTO high_freq_compounds(name, smiles) VALUES(?,?)", (name, row[0]))
        conn.commit()
        conn.close()
        return row[0]

    # synonyms 精确匹配
    cur.execute(
        "SELECT c.smiles FROM compounds c JOIN synonyms s ON c.cid = s.cid WHERE s.synonym = ?",
        (name,)
    )
    row = cur.fetchone()
    if row:
        cur.execute("INSERT OR REPLACE INTO high_freq_compounds(name, smiles) VALUES(?,?)", (name, row[0]))
        conn.commit()
        conn.close()
        return row[0]

    conn.close()
    return None


def extract_entities_to_smiles(text: str, db_path: str) -> Dict[str, str]:
    """
    使用 chemdataextractor 从文本抽取化学实体名，并查询数据库映射到 SMILES。
    返回 {实体名: SMILES}，仅包含查到 SMILES 的条目。
    """
    out: Dict[str, str] = {}
    if not text:
        return out
    
    # Lazy import CDE
    _ensure_cde()
    if cde is None:
        return out

    try:
        doc = cde.Document.from_string(text)
        cems = getattr(doc, "cems", [])
        if not cems:
            return out

        seen: Set[str] = set()
        names: List[str] = []
        for cem in cems:
            t = cem.text.strip()
            if len(t) < 3:
                continue
            if t in seen:
                continue
            seen.add(t)
            names.append(t)

        for name in names:
            smi = query_smiles_by_name(name, db_path=db_path)
            if smi:
                out[name] = smi
        return out
    except Exception:
        return out


# --- 1B. 直接从文本中识别 SMILES（正则 + 可选 RDKit 校验） ---

# 一个较为宽松的 SMILES 允许字符集合（支持多碎片“.”）
_SMILES_CHARS = r"A-Za-z0-9@\+\-\[\]\(\)\\\/=#%\."
_SMILES_CANDIDATE = re.compile(rf"([{_SMILES_CHARS}]+)")

# 为了降低误检，我们要求候选串：
# 1) 含有至少一个“符号类”字符（如 @ + - [ ] ( ) / \ = # %），或
# 2) 含有“元素+数字”的模式（如 C1, N2），或
# 3) 至少包含一个“元素大写字母后可跟小写”的模式（如 Cl, Br），并且长度>=3
_SYMBOLY_RE = re.compile(r"[@\+\-\[\]\(\)\\\/=#%]")
_ELEMNUM_RE = re.compile(r"[A-Z][a-z]?\d")
_ELEMENT_RE = re.compile(r"(?:Br|Cl|Si|Se|Na|Ca|Li|Mg|Al|Sn|Ag|Zn|Cu|Fe|Mn|Co|Ni|Mo|Hf|Ta|Ti|Cr|Pt|Au|Hg|Pb|Bi|I|F|O|N|S|P|B|C)")


def _looks_like_smiles(token: str) -> bool:
    token = token.strip()
    if len(token) < 3:
        return False
    if _SYMBOLY_RE.search(token):
        return True
    if _ELEMNUM_RE.search(token):
        return True
    if _ELEMENT_RE.search(token):
        return True
    return False


def _canonical_if_valid_smiles(token: str) -> Optional[str]:
    # if not _HAS_RDKIT:
    return token if _looks_like_smiles(token) else None
    try:
        m = Chem.MolFromSmiles(token, sanitize=True)
        if m is None:
            return None
        return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        return None


def find_smiles_in_text(text: str, max_hits: int = 16, unique: bool = True) -> List[str]:
    """
    在原始文本中扫描候选 SMILES 片段，并用 RDKit 校验（若可用）。
    返回 canonical SMILES 列表。
    """
    if not text:
        return []

    hits: List[str] = []
    seen: Set[str] = set()
    for cand in _SMILES_CANDIDATE.findall(text):
        token = cand.strip()
        if not _looks_like_smiles(token):
            continue
        smi = _canonical_if_valid_smiles(token)
        if not smi:
            continue
        if unique:
            if smi in seen:
                continue
            seen.add(smi)
        hits.append(smi)
        if len(hits) >= max_hits:
            break
    return hits

SENT_SPLIT_RE = re.compile(r'(?<=[。！？!?\.])\s*|\n+')  # 常见中英文句末标点
def resolve_text_to_smiles(
    text: str,
    db_path: str,
    prefer_names: bool = True,
) -> Dict[str, object]:
    """
    综合器：给定文本 → 返回
    {
      'from_names': {name: smiles},   # chemdataextractor + DB
      'from_smiles': [smiles, ...],   # 文本中直接匹配出的 SMILES
      'union_smiles': [unique list]   # 两者合并去重后的 SMILES 列表
    }
    """
    # 1) 按 <mol> 切分，取最后一个片段
    parts = text.split("<mol>")
    focus_text = parts[-1].strip() if parts else text.strip()

    # 2) 分句，取最后一句
    sents = [s.strip() for s in SENT_SPLIT_RE.split(focus_text) if s.strip()]
    focus_sent = sents[-1] if sents else focus_text
    
    
    # 3) 实体抽取
    from_names = extract_entities_to_smiles(focus_sent, db_path=db_path) if prefer_names else {}
    from_smiles = find_smiles_in_text(focus_sent) or []

    # 4) 根据在句子中的位置选择“最后一个出现的”
    candidates = []

    # 名称实体：保存 (位置, smiles)
    for name, smi in from_names.items():
        for m in re.finditer(re.escape(name), focus_sent):
            candidates.append((m.start(), smi))

    # 文本中直接出现的 SMILES
    for smi in from_smiles:
        for m in re.finditer(re.escape(smi), focus_sent):
            candidates.append((m.start(), smi))

    if not candidates:
        return None

    # 取最后一个出现的（按 start 排序）
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def extract_and_convert_online(text: str, proxy: Optional[str] = None) -> Dict[str, str]:
    """
    简洁版在线 NER -> SMILES
    返回 {token: SMILES}，直接用 find_smiles_in_text
    """
    hits = find_smiles_in_text(text)
    return {smi: smi for smi in hits}
# ======================================================================
#                              工具2：Mol(s) → SMILES
# ======================================================================

def mol_to_canonical_smiles(mol) -> Optional[str]:
    """
    将 RDKit Mol 转为 canonical SMILES；失败返回 None。
    """
    if not _HAS_RDKIT or mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def best_smiles_from_generated_mols(
    gen_mols: List,
    pick_largest_fragment: bool = True
) -> Optional[str]:
    """
    从扩散模型生成的 RDKit Mol 列表中挑选一个“代表性” SMILES：
      - 对每个 mol → canonical smiles；
      - 若 smiles 含 '.' 多碎片，选原子数最多的碎片；
      - 返回第一个成功得到的 smiles（再 canonical 一次），否则 None。
    """
    if not gen_mols or not _HAS_RDKIT:
        return None

    def _largest_fragment_smiles(smi: str) -> str:
        if not pick_largest_fragment or "." not in smi:
            return smi
        frags = smi.split(".")
        scored = []
        for f in frags:
            m = Chem.MolFromSmiles(f)
            if m is not None:
                scored.append((m.GetNumAtoms(), f))
        if not scored:
            return frags[0]
        scored.sort(reverse=True)
        return scored[0][1]

    for m in gen_mols:
        smi = mol_to_canonical_smiles(m)
        if not smi:
            continue
        smi = _largest_fragment_smiles(smi)
        try:
            mm = Chem.MolFromSmiles(smi)
            if mm is None:
                continue
            return Chem.MolToSmiles(mm, canonical=True)
        except Exception:
            continue
    return None


# ======================================================================
#                                  测试
# ======================================================================

if __name__ == "__main__":
    # 你需要根据自己的机器准备：
    # - RDKit（可选，但强烈推荐）
    # - chemdataextractor（可选）
    # - SQLite 数据库文件 `compounds.db`，包含 tables: compounds(name, smiles), synonyms(cid, synonym)
    #   如果没有数据库，名称映射相关测试会返回空，仅 SMILES 正则部分依然可用。
    proxy_url = "http://127.0.0.1:7899"  # 如果需要代理，请设置此变量
    DB_PATH = "compounds.db"
    try:
        create_high_freq_table(DB_PATH)
    except Exception:
        pass

    print("=== 测试：工具1 —— 文本解析（名称+SMILES） ===")
    samples = [
        # 同时含有英文名称与 SMILES
        "We tested aspirin CC(C)CC1=CC=C(C=C1)C(C)C(=O)O, (CC(=O)OC1=CC=CC=C1C(=O)O) and ibuprofen  in this experiment.",
        # 纯 SMILES（多碎片）
        "Mixture: CCO.CN was observed prior to reaction. <mol>",
        # 中文名（需要你的DB有对应映射）
        "在实验中，我们对阿司匹林和对乙酰氨基酚进行了对照。",
        # 嵌入更复杂符号的SMILES
        "Chiral test: C[C@H](O)[C@@H](N)C(=O)O appeared in trace amounts.",
        # 没有任何化学信息
        "It's just a plain sentence without chemicals."
    ]

    for i, text in enumerate(samples, 1):
        res = resolve_text_to_smiles(text, db_path=DB_PATH, prefer_names=True)
        print(f"\n[Case {i}] {text}")

        if isinstance(res, dict):
            print("  from_names :", res.get("from_names", []))
            print("  from_smiles:", res.get("from_smiles", []))
            print("  union      :", res.get("union_smiles", []))
        elif isinstance(res, list):
            # 如果返回列表，则尝试打印每个元素的 from_names/from_smiles
            for r in res:
                if isinstance(r, dict):
                    print("  from_names :", r.get("from_names", []))
                    print("  from_smiles:", r.get("from_smiles", []))
                    print("  union      :", r.get("union_smiles", []))
                else:
                    # r 是字符串
                    print("  from_names :", r)
                    print("  from_smiles:", r)
                    print("  union      :", r)
        else:
            # 直接是字符串
            print("  from_names :", res)
            print("  from_smiles:", res)
            print("  union      :", res)


"""
from tools.chem_tools_combined import resolve_text_to_smiles, create_high_freq_table

DB_PATH = "compounds.db"
create_high_freq_table(DB_PATH)

text = "We tested aspirin and ibuprofen. Also we saw CCC(C)Br and CCO.CN <mol>."
res = resolve_text_to_smiles(text, db_path=DB_PATH, prefer_names=True)
print(res["from_names"])   # {'aspirin': '...', 'ibuprofen': '...'}  (依赖你的DB)
print(res["from_smiles"])  # ['CCC(C)Br', 'CCO', 'CN']  (RDKit canonical 后)
print(res["union_smiles"]) # 合并去重
"""

"""
from tools.chem_tools_combined import best_smiles_from_generated_mols

with torch.no_grad():
    gen_mols = diffusion.generate_mol_from_embedding(
        batch_size=len(emb1), embeddings=emb1, num_nodes_lig=None
    )

smi = best_smiles_from_generated_mols(gen_mols)  # 返回单个代表性 SMILES 或 None
"""
