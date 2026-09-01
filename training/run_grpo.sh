#!/bin/bash
# =============================================================================
# GRPO Training — Qwen3-VL-4B LoRA on 4×RTX 3090
# =============================================================================
# Environment: pv_train (docs/INSTALL.md #4), not the pv_bench evaluation env.
# Two modes (set MODE variable):
#   scratch  — GRPO from base model (no SFT)
#   sft      — GRPO on top of SFT adapter (recommended)
#
# Usage:
#   MODE=scratch bash training/run_grpo.sh
#   MODE=sft     bash training/run_grpo.sh    # default
# =============================================================================

set -e

MODE="${MODE:-sft}"

# ---- Paths ----
MODEL_PATH="./models/Qwen3-VL-4B-Instruct"
DATASET="./data/pv_dataset/train_rl/rl_data_v2.jsonl"
SFT_ADAPTER="./outputs/training/sft_v2/best_adapter"
REWARD_PLUGIN="training/reward.py"

# ---- Mode-specific config ----
if [ "$MODE" = "sft" ]; then
    echo "=== GRPO mode: SFT + GRPO ==="
    ADAPTER_FLAG="--adapters $SFT_ADAPTER"
    OUTPUT_DIR="./outputs/training/grpo_v2_from_sft"

    if [ ! -d "$SFT_ADAPTER" ]; then
        echo "ERROR: SFT adapter not found at $SFT_ADAPTER"
        echo "Run SFT first: bash training/run_sft.sh"
        echo "Then copy/symlink best checkpoint to $SFT_ADAPTER"
        exit 1
    fi
else
    echo "=== GRPO mode: from scratch ==="
    ADAPTER_FLAG=""
    OUTPUT_DIR="./outputs/training/grpo_v2_from_scratch"
fi

# ---- Training ----
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
  --rlhf_type grpo \
  --model "$MODEL_PATH" \
  $ADAPTER_FLAG \
  --dataset "$DATASET" \
  --external_plugins "$REWARD_PLUGIN" \
  --reward_funcs pv_reward pv_format \
  --train_type lora \
  --lora_rank 16 \
  --freeze_vit true \
  --num_generations 4 \
  --max_completion_length 1024 \
  --temperature 1.0 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --num_train_epochs 1 \
  --bf16 true \
  --gradient_checkpointing true \
  --save_total_limit 3 \
  --save_only_model true \
  --report_to tensorboard \
  --logging_steps 10 \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "=== GRPO training complete (mode=$MODE) ==="
echo "Output:      $OUTPUT_DIR"
echo "TensorBoard: tensorboard --logdir $OUTPUT_DIR --port 6007"
