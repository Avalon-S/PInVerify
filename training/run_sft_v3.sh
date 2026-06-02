#!/bin/bash
# =============================================================================
# SFT v3 Training — Qwen3-VL-4B LoRA on 4×RTX 3090
# =============================================================================
# v3 change: neg_same CoT uses concrete attribute comparison
# (single controlled variable vs v2)
#
# Usage: bash training/run_sft_v3.sh
# =============================================================================

set -e

# ---- Paths ----
MODEL_PATH="./models/Qwen3-VL-4B-Instruct"
DATASET="./data/pv_dataset/train_sft/sft_data_v3.jsonl"
OUTPUT_DIR="./outputs/training/sft"

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
echo "=== SFT v3 training complete ==="
echo "Output:      $OUTPUT_DIR"
echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR --port 6006"
