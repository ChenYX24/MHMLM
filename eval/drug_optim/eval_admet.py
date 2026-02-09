import json
from pathlib import Path
from eval.drug_optim.scoring.admet_reasoning_richness import RewardEngine, _is_valid_smiles

CORE_FEATURES = {
    "Caco-2 Permeability", "F50%", "CYP3A4 inhibitor", "CYP2D6 inhibitor",
    "Pgp substrate", "hERG Blockers", "DILI", "Human Hepatotoxicity",
    "AMES Toxicity", "Genotoxicity", "Toxicity: Drug-induced Neurotoxicity",
    "QED", "SAScore", "GASA", "Lipinski Rule", "HLM Stability"
}


infer_dir = Path("/data1/chenyuxuan/MHMLM/eval/drug_optim/eval_output/diffusion_sft/005-LDMol/infer")

generated_file = infer_dir / "generated_molecules_t2m.txt"
score_summary_path = infer_dir / "score_summary.json"
# generated_jsonl_file = infer_dir / "tmp_generated_molecules_t2m.jsonl"
# generated_pred_only_file = infer_dir / "tmp_generated_molecules_t2m_pred_only.txt"

w_main = 1.0
w_bonus = 1.0
engine = RewardEngine(w_main=w_main, w_bonus=w_bonus)
reasoning = ""


total_norm_main = 0.0
total_norm_bonus = 0.0
total_f1 = 0.0
total_norm_total = 0.0
count = 0


paired = 0
skipped_invalid_after = 0

# stats
bad_feat_stats = {}
pred_feat_stats = {}
reasoning_length_stats = []
f1_score_distribution = []
feature_coverage_stats = {}
has_reasoning_count = 0  


with (
    generated_file.open("r", encoding="utf-8") as f_generated,
    score_summary_path.open("w", encoding="utf-8") as f_score_summary,
):
    header = f_generated.readline()
    assert header == "orig_smiles\tdescription\tpred_smiles\n"
    for idx, line in enumerate(f_generated):
        paired += 1
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        assert len(parts) == 3
        orig_smiles, description, opt_smiles = parts

        if not _is_valid_smiles(opt_smiles):
            skipped_invalid_after += 1
            print(f"[SKIP] sample {idx}: invalid after SMILES -> {opt_smiles[:80]}")
            continue

        # 评分
        norm_total, detail = engine.compute_sample(orig_smiles, opt_smiles, reasoning, idx=idx)
        print(f"Norm total: {norm_total}")
        print(f"Detail: {detail}")

        # 统计信息收集
        reasoning_length_stats.append(len(reasoning))
        f1_score_distribution.append(detail["bonus_f1"])
        
        # 统计是否有reasoning
        if detail["has_reasoning"]:
            has_reasoning_count += 1
        
        # 统计坏特征
        for feat in detail["bad_feats"]:
            bad_feat_stats[feat] = bad_feat_stats.get(feat, 0) + 1
        
        # 统计预测特征
        for feat in detail["pred_feats"]:
            pred_feat_stats[feat] = pred_feat_stats.get(feat, 0) + 1
        
        # 统计特征覆盖率（使用核心特征集）
        core_eval_feats = [f for f in detail["eval_feats"] if f in CORE_FEATURES]
        coverage = len(core_eval_feats) / len(CORE_FEATURES)
        feature_coverage_stats[idx] = coverage

        # 控制台打印（可按需精简）
        print(f"\n=== Sample {idx} ===")
        print("orig_smiles:", orig_smiles)
        print("opt_smiles :", opt_smiles)
        print("reasoning  :", reasoning[:200] + "..." if len(reasoning) > 200 else reasoning)
        print("detail     :", detail)

        # 汇总
        total_norm_main  += detail["norm_main"]
        total_norm_bonus += detail["norm_bonus"]
        total_f1         += detail["bonus_f1"]
        total_norm_total += norm_total
        count += 1



    # 输出平均和统计信息
    if count > 0:
        averages = {
            "avg_main_reward": round(total_norm_main / count, 4),
            "avg_bonus": round(total_norm_bonus / count, 4),
            "avg_bonus_f1": round(total_f1 / count, 4),
            "avg_total_reward": round(total_norm_total / count, 4),
            "w_main": w_main,
            "w_bonus": w_bonus,
            "paired": paired,
            "used_pairs": count,
            "skipped_invalid_after": skipped_invalid_after,
            "reasoning_loaded": 0,  
            "avg_reasoning_length": round(sum(reasoning_length_stats) / len(reasoning_length_stats), 2),
            "f1_score_stats": {
                "min": round(min(f1_score_distribution), 4),
                "max": round(max(f1_score_distribution), 4),
                "median": round(sorted(f1_score_distribution)[len(f1_score_distribution)//2], 4),
                "std": round((sum((x - sum(f1_score_distribution)/len(f1_score_distribution))**2 for x in f1_score_distribution) / len(f1_score_distribution))**0.5, 4)
            },
            "top_bad_features": dict(sorted(bad_feat_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_pred_features": dict(sorted(pred_feat_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
            "avg_feature_coverage": round(sum(feature_coverage_stats.values()) / len(feature_coverage_stats), 4),
            "has_reasoning_count": has_reasoning_count,
            "has_reasoning_ratio": round(has_reasoning_count / count, 4) if count > 0 else 0.0
        }
    print(f"Saving score summary to {score_summary_path}")
    f_score_summary.write(json.dumps(averages, ensure_ascii=False) + "\n")