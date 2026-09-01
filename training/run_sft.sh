#!/bin/bash
# =============================================================================
# SFT Training — Qwen3-VL-4B LoRA on 4×RTX 3090
# =============================================================================
# Environment: pv_train (docs/INSTALL.md #4), not the pv_bench evaluation env.
# Usage: bash training/run_sft.sh
#
# Data:   17124 samples, ~3 epochs → ~51k steps / effective_batch
# Memory: ~20 GiB/card with batch=2 + ZeRO-2
# Time:   ~3-5 hours
# =============================================================================

set -e

# ---- Paths ----
MODEL_PATH="./models/Qwen3-VL-4B-Instruct"
DATASET="./data/pv_dataset/train_sft/sft_data_v2.jsonl"
OUTPUT_DIR="./outputs/training/sft_v2"

# ---- Training ----
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
  --model "$MODEL_PATH" \
  --dataset "$DATASET" \
  --train_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --freeze_vit true \
  --output_dir "$OUTPUT_DIR" \
  --max_length 2048 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --bf16 true \
  --gradient_checkpointing true \
  --save_strategy epoch \
  --save_total_limit 3 \
  --save_only_model true \
  --report_to tensorboard \
  --logging_steps 10

echo ""
echo "=== SFT training complete ==="
echo "Output:      $OUTPUT_DIR"
echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR --port 6006"
