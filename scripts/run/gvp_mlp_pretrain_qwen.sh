#!/bin/bash
# 训练 GNN MLP 适配器（Qwen 30k 数据）

# 激活conda环境
source /home/chenyuxuan/miniconda3/etc/profile.d/conda.sh
conda activate llam3.2

# 切换到项目目录
cd /data1/chenyuxuan/MHMLM

# 运行预训练
python scripts/train/gvp_mlp_pretrain.py \
  --data /data1/chenyuxuan/MSMLM/data/traindata/chatmol/chatmol_gnn_qwen_30k.pkl \
  --outdir /data1/chenyuxuan/MSMLM/model/gnn_mlp_qwen \
  --gnn-class modules.gnn:GVPEncoder \
  --gnn-ckpt /data1/lvchangwei/GNN/Project/GVP/checkpoints_256_wo/gvp_weights_best.pt \
  --gnn-batch-size 128 \
  --hidden-dim 1536 \
  --num-layers 2 \
  --epochs 100 \
  --batch-size 256 \
  --lr 1e-5 \
  --alpha 0.5 \
  --weight-decay 0.0 \
  --scheduler plateau \
  --grad-clip 1.0 \
  --target-normalize none \
  --val-ratio 0.05 \
  --seed 42 \
  --use-cache \
  --tensorboard \
  --bf16 \
  --amp 

