import os
import argparse

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from modules.ldmol_component import LDMolTrainer

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
    trainer = LDMolTrainer(config, rank=rank, world_size=world_size, local_rank=local_rank)

    trainer.train()

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    """
    Train: 
    # cd 至项目根目录
    cd /data1/chenyuxuan/MHMLM/

    # 使用.venv
    source .venv/bin/activate
    
    # 训练
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m scripts.train_ldmol --config modules/ldmol_component/assets/ldmol-train.yaml


    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m scripts.train_ldmol --config modules/ldmol_component/assets/ldmol-train.yaml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args)
