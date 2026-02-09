import os
import argparse
import json

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from modules.ldmol_component import LDMolTrainer
from eval.drug_optim.scoring.admet_reasoning_richness import main_from_extracted

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Avoid tokenizer_parallelism dead lock!

def main(args):

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"world_size: {world_size}, rank: {rank}, local_rank: {local_rank}")        

    config = OmegaConf.load(args.config)
    results_dir = args.results_dir
    trainer = LDMolTrainer(config, rank=rank, world_size=world_size, local_rank=local_rank, results_dir=results_dir)

    data_paths = [
        "/data1/chenyuxuan/MHMLM/eval_results/data/ldmol/drug_optim/processed/test_t2m.txt"
    ]
    trainer.generate_smi_t2m(
        ckpt=args.ckpt,
        data_paths=data_paths,
        using_cfg=True,
        cfg_scale=2.5
    )
    dist.barrier()
    dist.destroy_process_group()
    
    infer_dir = trainer.experiment_dir / "infer"
    logger = trainer.logger

    generated_file = infer_dir / "generated_molecules_t2m.txt"
    score_summary_path = infer_dir / "score_summary.json"
    generated_jsonl_file = infer_dir / "tmp_generated_molecules_t2m.jsonl"
    generated_pred_only_file = infer_dir / "tmp_generated_molecules_t2m_pred_only.txt"
    with (
        generated_file.open("r", encoding="utf-8") as f_generated,
        generated_jsonl_file.open("w", encoding="utf-8") as f_jsonl,
        generated_pred_only_file.open("w", encoding="utf-8") as f_pred_only,
    ):
        header = f_generated.readline()
        assert header == "orig_smiles\tdescription\tpred_smiles\n"
        for line in f_generated:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            assert len(parts) == 3
            orig_smiles, description, pred_smiles = parts
            f_jsonl.write(json.dumps({
                "input": f"Original SMILES: {orig_smiles}\n\nADMET Profile:\n{description}"
            }) + "\n")
            f_pred_only.write(pred_smiles + "\n")
        logger.info(f"Running ADMET scoring...")
        main_from_extracted(
            orig_jsonl=str(generated_jsonl_file),
            after_smi_path=str(generated_pred_only_file),
            out_path=str(score_summary_path),
            w_main=1.0,
            w_bonus=1.0,
        )
    
    # Clean up temporary files
    generated_jsonl_file.unlink(missing_ok=True)
    generated_pred_only_file.unlink(missing_ok=True)

    logger.info(f"ADMET scoring completed!")
    logger.info(f"Score summary: {score_summary_path}")



if __name__ == "__main__":
    """
    Train: 
    # cd 至项目根目录
    cd /data1/chenyuxuan/MHMLM/

    # 使用.venv
    source .venv/bin/activate
    
    # 推理
    CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 -m eval.drug_optim.eval_diffusion --config=modules/ldmol_component/assets/ldmol-drug_optim.yaml --results_dir=/data1/chenyuxuan/MHMLM/eval/drug_optim/eval_output/diffusion_sft --ckpt=/data1/chenyuxuan/MHMLM/eval_results/results/ldmol-qwen_cpt_sft-drug_optim/000-LDMol/checkpoints/0007000.pt
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    main(args)
