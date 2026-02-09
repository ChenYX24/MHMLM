from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols

import numpy as np
import pandas as pd
import json
import argparse
import os

def molfinger_evaluate(targets, preds, morgan_r=2, verbose=False):
    outputs = []
    bad_mols = 0

    for i in range(len(targets)):
        try:
            gt_smi = targets[i]
            ot_smi = preds[i]
            gt_m = Chem.MolFromSmiles(gt_smi)
            ot_m = Chem.MolFromSmiles(ot_smi)

            if ot_m is None:
                raise ValueError('Bad SMILES: {ot_smi}')
            outputs.append((gt_m, ot_m))
        except:
            bad_mols += 1

    validity_score = len(outputs) / (len(outputs) + bad_mols)
    if verbose:
        print('validity:', validity_score)

    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    enum_list = outputs

    for i, (gt_m, ot_m) in enumerate(enum_list):

        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m, morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)
    if verbose:
        print('Average MACCS Similarity:', maccs_sims_score)
        print('Average RDK Similarity:', rdk_sims_score)
        print('Average Morgan Similarity:', morgan_sims_score)

    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score

def main(args):
    input_df = pd.read_csv(args.input_file, sep="\t")
    target_smiles = input_df[args.target_key].tolist()
    pred_smiles = input_df[args.pred_key].tolist()
    validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = molfinger_evaluate(target_smiles, pred_smiles)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "fingerprint_summary.json"), "w") as f:
        json.dump({
            "validity": validity_score,
            "maccs_smilarity": maccs_sims_score,
            "rdk_smilarity": rdk_sims_score,
            "morgan_smilarity": morgan_sims_score
        }, f)
    print(f"Fingerprint summary saved to {os.path.join(args.output_dir, 'fingerprint_summary.json')}")
    print(f"Validity: {validity_score}")
    print(f"MACCS Similarity: {maccs_sims_score}")
    print(f"RDK Similarity: {rdk_sims_score}")
    print(f"Morgan Similarity: {morgan_sims_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--target_key", type=str, required=True, default="target_smiles")
    parser.add_argument("--pred_key", type=str, required=True, default="pred_smiles")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)