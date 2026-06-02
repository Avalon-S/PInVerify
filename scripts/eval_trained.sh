#!/bin/bash
# =============================================================================
# Evaluate Trained End-to-End Models (SFT / SFT+GRPO / SFT+GSPO)
# =============================================================================
# The trained model makes ALL decisions: navigation + verification + action.
# Uses configs/agent/trained_e2e.yaml (NOT the training-free agent configs).
#
# Prerequisites:
#   ADAPTER=<path> bash scripts/start_multigpu_servers_lora.sh <num_gpus>
#
# Usage:
#   # Basic: evaluate with a tag name
#   bash scripts/eval_trained.sh sft
#   bash scripts/eval_trained.sh grpo_v2
#   bash scripts/eval_trained.sh gspo_v2
#
#   # Custom split size (default: all)
#   bash scripts/eval_trained.sh sft 50
#
#   # Custom GPU config
#   NUM_GPUS=5 GPU_IDS=0,1,2,3,4 BASE_PORTS=12182,12282,12382,12482,12582 \
#     bash scripts/eval_trained.sh sft
#
#   # Specific bbox mode only (default: both gt and dino)
#   BBOX_MODES=gt bash scripts/eval_trained.sh sft
#   BBOX_MODES=dino bash scripts/eval_trained.sh grpo_v2
#
# Arguments:
#   $1 = model tag (sft / grpo_v1 / grpo_v2 / gspo_v2 / base) — for output dir
#   $2 = split size (default: all)
# =============================================================================

set -e
export PYTHONPATH=$PYTHONPATH:.

TAG="${1:?Usage: bash scripts/eval_trained.sh <tag> [split]}"
LIMIT="${2:-all}"

# ===== GPU Configuration =====
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
BASE_PORTS="${BASE_PORTS:-12182,12282,12382,12482}"

# ===== Bbox modes to evaluate =====
BBOX_MODES="${BBOX_MODES:-gt dino}"

INDEX_FILE="pv_index_${LIMIT}.jsonl"
OUT_BASE="./outputs/trained_${TAG}_${LIMIT}"
CONFIG="configs/agent/trained_e2e.yaml"

echo "========================================"
echo "Trained E2E Model Evaluation"
echo "  Tag:        ${TAG}"
echo "  Config:     ${CONFIG}"
echo "  Split:      ${INDEX_FILE}"
echo "  Bbox modes: ${BBOX_MODES}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Output:     ${OUT_BASE}"
echo "========================================"

if [ ! -f "$CONFIG" ]; then
    echo "[ERROR] Config not found: $CONFIG"
    exit 1
fi

for mode in $BBOX_MODES; do
    out_dir="${OUT_BASE}/trained_${TAG}_${mode}_${LIMIT}"

    echo ""
    echo "=============================="
    echo "[trained_e2e / ${mode}] Start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================="

    python scripts/evaluate_multigpu_dynamic.py \
        --config "$CONFIG" \
        --num_gpus $NUM_GPUS \
        --gpu_ids "$GPU_IDS" \
        --base_ports "$BASE_PORTS" \
        --dataset_index "$INDEX_FILE" \
        --output_dir "$out_dir" \
        dataset.index_file="$INDEX_FILE" \
        method.bbox_mode=$mode \
        output.save_viz=true

    echo "[trained_e2e / ${mode}] Completed: $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "========================================"
echo "Evaluation complete: ${TAG}"
echo "Results: ${OUT_BASE}"
echo "========================================"
