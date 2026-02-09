#!/usr/bin/env python3
import os
import json
import time
import argparse
import warnings
import requests

# 忽略 HTTPS 验证警告
from requests.packages.urllib3.exceptions import InsecureRequestWarning
warnings.simplefilter('ignore', InsecureRequestWarning)

# 复用原来的 ADMETLab session / CSRF 逻辑
session = requests.Session()
get_url = "https://admetlab3.scbdd.com/server/evaluationCal"
session.get(get_url, verify=False)
csrf_token = session.cookies.get("csrftoken")
post_url = get_url
headers_admet = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": get_url,
}

# 新增：ChEMBL REST API 配置，用于根据 molecule_id 拿 SMILES
CHEMBL_BASE    = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_HEADERS = {
    "Accept": "application/json",
    "X-CHEMBL-DB-RELEASE": "CHEMBL23"
}

def fetch_canonical_smiles(mol_id: str) -> str:
    """
    从 ChEMBL REST API 拿 canonical_smiles。
    """
    url = f"{CHEMBL_BASE}/molecule/{mol_id}.json"
    resp = requests.get(url, headers=CHEMBL_HEADERS, verify=False)
    resp.raise_for_status()
    data = resp.json()
    # REST 返回字段可能在 "molecules" 或者顶层
    mols = data.get("molecules") or []
    if mols:
        return mols[0].get("canonical_smiles")
    # 兜底：有些接口直接在 data 顶层
    return data.get("canonical_smiles") or data.get("molecule_structures", {}).get("canonical_smiles", "")

def search_admet(smiles: str) -> dict:
    """
    原有的 ADMETLab 查询函数，保持不变。
    """
    data = {
        "csrfmiddlewaretoken": csrf_token,
        "smiles": smiles,
        "method": "1"
    }
    resp = session.post(post_url, data=data, headers=headers_admet, verify=False)
    text = resp.text
    result = {}
    # 解析 HTML 表格...
    segs = text.split('<td width="60%">')
    for seg in segs[1:]:
        key = seg.split('</td>')[0].strip()
        val = seg.split('<td width="20%">')[1].split('</td>')[0].strip()
        if '<span>' in val:
            val = val.split('<span>')[-1].split('</span>')[0]
        result[key] = val
    return result

def main():
    parser = argparse.ArgumentParser(
        description="从 ChEMBL JSONL 读分子 ID，批量爬取 ADMETLab 指标并保存 JSONL"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="输入文件（JSONL），每行包含一个字典，必须有 'molecule_id' 字段"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="输出 JSONL 文件，每行一个合并了 ADMET 和原始字段的字典"
    )
    parser.add_argument(
        "-s", "--sleep", type=float, default=1.0,
        help="每次请求后的延迟（秒）"
    )
    args = parser.parse_args()

    # 确保输出目录存在
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:

        for idx, line in enumerate(fin):
            rec = json.loads(line)
            mol_id = rec.get("molecule_id")
            if not mol_id:
                print(f"[WARN] line {idx} 缺少 molecule_id，跳过")
                continue

            # 1) 拿 SMILES
            try:
                smiles = fetch_canonical_smiles(mol_id)
            except Exception as e:
                print(f"[ERROR] 无法获取 {mol_id} 的 SMILES：{e}")
                continue

            # 2) 计算 ADMET
            try:
                admet = search_admet(smiles)
            except Exception as e:
                print(f"[ERROR] ADMET 查询失败 {mol_id} ({smiles})：{e}")
                continue

            # 3) 合并并输出
            rec["SMILES"] = smiles
            rec.update(admet)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{idx}] {mol_id} → OK")
            time.sleep(args.sleep)

if __name__ == "__main__":
    main()
