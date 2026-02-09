#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# LDMol 训练测试脚本
#
# 用法:
#   # 单卡训练
#   bash test_trainer.sh
#
#   # 多卡训练
#   GPUS=0,1,2,3 NPROC=4 bash test_trainer.sh
#
#   # 自定义参数
#   DATA_PATH=./data/my_train.txt EPOCHS=50 bash test_trainer.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PWD}"

# ----- 可配置参数 -----
: "${GPUS:=0}"                              # GPU 设备号，多卡用逗号分隔
: "${NPROC:=1}"                             # GPU 数量
: "${MASTER_PORT:=29500}"                   # DDP master port

: "${DATA_PATH:=./data/chatmol/train.txt}"  # 训练数据路径
: "${TEXT_ENCODER_PATH:=/data1/chenyuxuan/base_model/qwen3_8b}"  # Text Encoder 路径
: "${VAE_CKPT_PATH:=/data1/chenyuxuan/checkpoint/diffusion_pretrained/official/checkpoint_autoencoder.ckpt}"  # VAE 权重
: "${LDMOL_CKPT_PATH:=}"                    # DiT 预训练权重（可选）

: "${EPOCHS:=100}"                          # 训练轮数
: "${GLOBAL_BATCH_SIZE:=64}"                # 全局 batch size
: "${DESCRIPTION_LENGTH:=256}"              # 文本最大长度
: "${RESULTS_DIR:=./training_output}"       # 输出目录

: "${LOG_EVERY:=100}"                       # 日志频率
: "${CKPT_EVERY:=5000}"                     # 保存频率
: "${SEED:=0}"                              # 随机种子

# ----- 执行训练 -----
echo "=============================================="
echo "LDMol Training"
echo "=============================================="
echo "GPUs: ${GPUS} (${NPROC} processes)"
echo "Data: ${DATA_PATH}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${GLOBAL_BATCH_SIZE}"
echo "Output: ${RESULTS_DIR}"
echo "=============================================="

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    -m ldmol_component.LDMolTrainer \
    --data_path "${DATA_PATH}" \
    --text_encoder_path "${TEXT_ENCODER_PATH}" \
    --vae_ckpt_path "${VAE_CKPT_PATH}" \
    ${LDMOL_CKPT_PATH:+--ldmol_ckpt_path "${LDMOL_CKPT_PATH}"} \
    --epochs "${EPOCHS}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --description_length "${DESCRIPTION_LENGTH}" \
    --results_dir "${RESULTS_DIR}" \
    --log_every "${LOG_EVERY}" \
    --ckpt_every "${CKPT_EVERY}" \
    --global_seed "${SEED}"
