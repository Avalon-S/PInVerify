#!/bin/bash
# =============================================================================
# DPO Training — Qwen3-VL-4B LoRA on 4×RTX 3090
# =============================================================================
# DPO on top of SFT v3 adapter.
# Uses preference pairs (chosen/rejected) — no online generation needed.
# Training speed similar to SFT (~1.5x due to reference model forward pass).
#
# Usage: bash training/run_dpo.sh
# =============================================================================

set -e

# ---- Paths ----
MODEL_PATH="./models/Qwen3-VL-4B-Instruct"
DATASET="./data/pv_dataset/train_rl/dpo_data_v3.jsonl"
SFT_ADAPTER="./outputs/training/sft/best_adapter"
OUTPUT_DIR="./data/dpo_v3_output"

if [ ! -d "$SFT_ADAPTER" ]; then
    echo "ERROR: SFT v3 adapter not found at $SFT_ADAPTER"
    echo "Run SFT v3 first: bash training/run_sft_v3.sh"
    echo "Then copy/symlink best checkpoint to $SFT_ADAPTER"
    exit 1
fi

# ---- Training ----
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
  --rlhf_type dpo \
  --model "$MODEL_PATH" \
  --adapters "$SFT_ADAPTER" \
  --dataset "$DATASET" \
  --train_type lora \
  --lora_rank 16 \
  --freeze_vit true \
  --max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-7 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --num_train_epochs 1 \
  --bf16 true \
  --gradient_checkpointing true \
  --save_steps 200 \
  --save_total_limit 5 \
  --save_only_model true \
  --report_to tensorboard \
  --logging_steps 10 \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "=== DPO v3 training complete ==="
echo "Output:      $OUTPUT_DIR"
echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR --port 6009"
