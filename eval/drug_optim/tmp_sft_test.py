import json
from pathlib import Path
from sft_tester import MolAwareGenerator2
import re

config = {
    "ckpt_dir": "/data1/chenyuxuan/checkpoint/qwen3_8b_cpt_sft/epoch2/LLM_nofreeze/checkpoint-4200",
    "device": "cuda:4",
    "device_map": None,  # 单卡模式
    "dtype": "bf16",
    "debug": True,
    "token_classifier_path": "/data1/lvchangwei/LLM/Lora/llama_mlp_token_classifier.pt",
}
gen = MolAwareGenerator2()
gen.load(config)

input_file = Path("/data1/chenyuxuan/MHMLM/eval_results/data/ldmol/drug_optim/raw/converted_reasoning_test5_fixed_updated.jsonl")
output_dir = Path("/data1/chenyuxuan/MHMLM/eval/drug_optim/eval_output/llm_cpt_sft_gvp")
output_file = output_dir / "output.txt"
output_dir.mkdir(parents=True, exist_ok=True)

# pattern = r"```smiles\s*(.*?)\s*```"
# match = re.search(pattern, content, re.DOTALL)
#         if match:
#             smiles = match.group(1).strip()
#             f_out.write(smiles + "\n")

with (open(input_file, 'r', encoding='utf-8') as f_in,
     open(output_file, 'w', encoding='utf-8') as f_out):
     
     for i, line in enumerate(f_in):
        obj = json.loads(line)
        prompt = obj["input"]
        content = gen.generate(
            prompt,

            add_dialog_wrapper=True,
            skip_special_tokens=True,
            task_type="drug_optim",

            realtime_mol=True,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.05,
        )
        f_out.write(content + '\n')
        f_out.flush()

print("\n==== Generation Finish ======")

# # prompt = "Describe this molecule: CCCCCCC(O)C/C=C\\CCCCCCCC(=O)[O-]\nPlease only output the answer."
# prompt = "Original SMILES: O=C(Nc1cccc(O)c1)c1cccnc1\n\nADMET Profile:\nMolecular Weight (MW): 214.07\nnRing: 2.0\nMaxRing: 6.0\nnHet: 4.0\nfChar: 0.0\nnRig: 13.0\nFlexibility: 0.231\nTPSA: 62.22\nlogS: -2.762\nlogP: 1.604\nlogD7.4: 1.887\npka (Acid): 8.018\npka (Base): 5.378\nQED: 0.804\nSAscore: Easy\nGASA: Easy\nFsp<sup>3</sup>: 0.0\nMCE-18: 10.0\nNPscore: -1.279\nLipinski Rule: Accepted\nPfizer Rule: Accepted\nGSK Rule: Accepted\nGoldenTriangle: Accepted\nPAINS: 0\nAlarm_NMR Rule: 2\nBMS Rule: 0\nChelating Rule: 0\nColloidal aggregators: 0.017\nReactive compounds: 0.021\nPromiscuous compounds: 0.086\nCaco-2 Permeability: -4.901\nMDCK Permeability: 0.0\nPAMPA: +\nPgp inhibitor: --\nPgp substrate: ---\nHIA: ---\nF20%: ---\nF30%: --\nF50%: -\nPPB: 80.7%\nVDss: 0.291\nBBB: ---\nFu: 17.1%\nOATP1B1 inhibitor: ++\nOATP1B3 inhibitor: +++\nBCRP inhibitor: --\nMRP1 inhibitor: +\nBSEP inhibitor: --\nCYP1A2 inhibitor: +++\nCYP1A2 substrate: --\nCYP2C19 inhibitor: ---\nCYP2C19 substrate: ---\nCYP2C9 inhibitor: +\nCYP2C9 substrate: ---\nCYP2D6 inhibitor: ---\nCYP2D6 substrate: ---\nCYP3A4 inhibitor: +++\nCYP3A4 substrate: ---\nCYP2B6 inhibitor: --\nCYP2B6 substrate: ---\nCYP2C8 inhibitor: +++\nHLM Stability: --\nCL<sub>plasma</sub>: 3.391\nT1/2: 0.906\nAquatic Toxicity Rule: 0\nGenotoxic Carcinogenicity Mutagenicity Rule: 1\nNonGenotoxic Carcinogenicity Rule: 0\nSkin Sensitization Rule: 5\nAcute Toxicity Rule: 0\nhERG Blockers: 0.385\nhERG Blockers (10um): 0.367\nDILI: 0.884\nAMES Toxicity: 0.462\nRat Oral Acute Toxicity: 0.246\nFDAMDD: 0.566\nSkin Sensitization: 0.958\nCarcinogenicity: 0.215\nEye Corrosion: 0.001\nEye Irritation: 0.988\nRespiratory: 0.476\nHuman Hepatotoxicity: 0.525\nDrug-induced Nephrotoxicity: 0.498\nDrug-induced Neurotoxicity: 0.448\nOtotoxicity: 0.105\nHematotoxicity: 0.215\nGenotoxicity: 0.927\nRPMI-8226 Immunitoxicity: 0.109\nA549 Cytotoxicity: 0.037\nHek293 Cytotoxicity: 0.465\nNR-AhR: ++\nNR-AR: ---\nNR-AR-LBD: ---\nNR-Aromatase: ---\nNR-ER: --\nNR-ER-LBD: ---\nNR-PPAR-gamma: ---\nSR-ARE: -\nSR-ATAD5: ---\nSR-HSE: ---\nSR-MMP: ++\nSR-p53: ---",

# print("\n=== Generated Text ===")
# print(text)
