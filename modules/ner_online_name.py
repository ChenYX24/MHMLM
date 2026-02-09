# ner_online.py
import re
import requests
import logging
from typing import Optional, List, Dict
import os

# 配置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# 延迟导入 chemdataextractor，避免网络问题
_Document = None
_HAS_CDE = False

def _get_document_class():
    """延迟导入 Document 类，避免网络连接问题"""
    global _Document, _HAS_CDE
    if _Document is None and not _HAS_CDE:
        try:
            # 设置环境变量禁用自动下载（如果可能）
            os.environ.setdefault('CDE_DISABLE_DOWNLOADS', '1')
            from chemdataextractor.doc import Document as _Document
            _HAS_CDE = True
        except Exception as e:
            logging.warning(f"Failed to import chemdataextractor: {e}. Name-to-SMILES mapping will be disabled.")
            _HAS_CDE = False
    return _Document

# ======= 工具函数 =======
def clean_text(text: str) -> str:
    """清理文本中的特定标记。"""
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'\$\{.*?\}\}', '', text)
    return text.strip()

def preprocess_cem(cem: str) -> str:
    """预处理化学实体名称以适应 PubChem 请求。"""
    cleaned = re.sub(r'\s*\([^)]*\)\s*', '', cem)
    cleaned = re.sub(r'[^a-zA-Z0-9\s-]', '_', cleaned)
    cleaned = cleaned.strip().replace(' ', '+')
    cleaned = cleaned.strip('_+')
    return cleaned

def is_likely_smiles(s: str) -> bool:
    """检查字符串是否可能是有效的SMILES。"""
    smiles_chars = set("BCNOPSFIKLHRecnops1234567890@-=#$()[]+\\/%")
    return all(c in smiles_chars for c in s) and ' ' not in s and len(s) > 0

def safe_document(text: str) -> Optional:
    """安全地解析文本为 ChemDataExtractor Document。"""
    try:
        Document = _get_document_class()
        if Document is None:
            return None
        if not text or not text.strip():
            return None
        return Document(text)
    except Exception as e:
        logging.warning(f"Failed to parse text with ChemDataExtractor. Error: {e}")
        return None

def get_smiles_from_pubchem(name: str, proxy: Optional[str] = None) -> Optional[str]:
    """
    从 PubChem API 获取 SMILES，支持化学名和 SMILES 两种输入。
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    proxies = {'http': proxy, 'https': proxy} if proxy else None

    # --- Case 1: 如果输入本身就是 SMILES ---
    if is_likely_smiles(name):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{name}/property/CanonicalSMILES/TXT"
    else:
        # --- Case 2: 输入是化学名 ---
        preprocessed_name = preprocess_cem(name).lower()
        preprocessed_name = preprocessed_name.replace("sulphuric", "sulfuric")  # 修正英式拼写
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{preprocessed_name}/property/CanonicalSMILES/TXT"

    try:
        response = requests.get(url, headers=headers, proxies=proxies, timeout=8)
        response.raise_for_status()
        text = response.text.strip()
        # print(f"[DEBUG] PubChem 返回内容: {text}")  # 调试用
        if text and not text.startswith("Page not found") and is_likely_smiles(text):
            return text
        else:
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"PubChem API request failed for '{name}': {e}")
        return None

def extract_and_convert_online(text: str, proxy: Optional[str] = None) -> Dict[str, str]:
    """
    在线提取文本中的化学实体并将其转换为 SMILES。
    增强版：在 CDE 失败时，使用正则和 SMILES 检测作为 fallback。
    """
    results = {}
    cleaned_text = clean_text(text)

    # Step 1: 尝试用 ChemDataExtractor
    doc = safe_document(cleaned_text)
    cems: List[str] = []
    if doc:
        try:
            cems = list(set(
                cem.text.strip()
                for cem in getattr(doc, "cems", []) or []
                if cem.text and cem.text.strip()
            ))
        except Exception as e:
            logging.warning(f"Error extracting CEMS from document: {e}")

    # Step 2: fallback 正则
    if not cems:
        logging.info("CDE未识别实体，尝试正则匹配常见化学名")
        regex_candidates = re.findall(
            r'\b[A-Za-z][a-z]{1,}(?:ic acid|ate|ene|one|ol|ide|ium|ane|yne)?\b',
            cleaned_text
        )
        cems = list(set(regex_candidates))

    # Step 3: fallback 检测 SMILES
    if not cems and is_likely_smiles(cleaned_text):
        cems = [cleaned_text]

    # Step 4: 查询 PubChem
    for cem_name in cems:
        smiles = get_smiles_from_pubchem(cem_name, proxy)
        if smiles:
            results[cem_name] = smiles

    # print(f"[DEBUG] 提取到候选实体: {cems}")
    return results

# ======= 示例：LLM 服务如何调用此模块 =======
def handle_mol_token(llm_context: str, proxy: Optional[str] = None) -> str:
    """
    一个模拟的函数，展示 LLM 服务如何调用实体识别工具。
    增强版：支持直接识别 SMILES 作为 fallback。
    """
    # 1. 尝试从上下文中识别化学实体并转换为SMILES
    smiles_map = extract_and_convert_online(llm_context, proxy)

    # 2. 从 llm_context 中提取最可能需要转换的实体（例如，最后一个实体）
    last_cem = ""
    last_idx = -1
    for cem_name in smiles_map:
        idx = llm_context.rfind(cem_name)
        if idx > last_idx:
            last_idx = idx
            last_cem = cem_name

    # 3. 如果识别到了化学实体并成功转成 SMILES
    if last_cem and last_cem in smiles_map:
        smiles = smiles_map[last_cem]
        print(f"✅ 成功将 '{last_cem}' 转换为 SMILES: '{smiles}'")
        return smiles  # 暂时返回 SMILES 字符串作为演示

    # 4. fallback: 检查上下文本身是不是 SMILES
    if is_likely_smiles(llm_context.strip()):
        print(f"⚡ 直接检测到输入是 SMILES: '{llm_context.strip()}'")
        return llm_context.strip()

    # 5. 否则走 Diffusion 管道占位符
    print("❌ 未能识别或转换化学实体，走 Diffusion 管道占位符。")
    return "<mol_not_found>"

# ======= 测试 =======
# if __name__ == "__main__":
#     test_texts = [
#         "A novel synthesis of Benzene was reported.",
#         "The reaction uses Sulphuric acid.",
#         "A mixture of ethyl acetate and isopropanol was used.",
#         "The molecule C1=CC=CC=C1 has a unique structure."
#     ]
    
#     proxy_url = "http://127.0.0.1:7899"  # 如果需要代理，请设置此变量

#     for text in test_texts:
#         print("-" * 20)
#         print(f"输入文本: '{text}'")
        
#         # 模拟LLM调用
#         result_smiles_or_token = handle_mol_token(text, proxy=proxy_url)
#         print(f"处理结果: {result_smiles_or_token}")
