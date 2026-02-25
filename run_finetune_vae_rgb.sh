#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=False

VAE_PATH="checkpoints/Wan-AI/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth"
DATA_ROOT="/root/autodl-tmp/Matrix-3D/data/dataset_train_round1"
OUTPUT_DIR="./results/vae_rgb_finetuned"

LOG_DIR="log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/finetune_vae_rgb_${TIMESTAMP}.log"
echo "Logging to $LOG_FILE"

python code/VideoX-Fun/scripts/wan2.1_fun/code/finetune_vae_rgb.py \
    --vae_path $VAE_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --height 128 \
    --width 256 \
    --batch_size 1 \
    --max_frames 41 \
    --lr 1e-5 \
    --max_steps 5000 \
    --warmup_steps 200 \
    --lambda_l1 1.0 \
    --lambda_edge 0.5 \
    --lambda_lpips 0.1 \
    --lambda_kl 1e-6 \
    --lambda_prox 1e-4 \
    --kl_warmup_start 500 \
    --kl_warmup_end 1000 \
    --ema_decay 0.9999 \
    --encoder_lr_ratio 0.1 \
    --spike_threshold 5.0 \
    --lpips_warmup_start 200 \
    --lpips_warmup_end 500 \
    --gradient_checkpointing \
    --mixed_precision bf16 \
    --log_steps 10 \
    --save_steps 500 \
    2>&1 | tee "$LOG_FILE"
