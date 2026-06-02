#!/bin/bash
# Run ALL agents with Dynamic Work Stealing
# Usage: bash run_all_multigpu_500_dynamic.sh [split]
#   split: 50 | 100 | 500 (default: 500)
# Advantages over static chunking:
#   - Better load balancing (fast GPUs don't wait for slow ones)
#   - Automatic failover (failed episodes can be retried)
#   - Real-time progress monitoring

set -e
export PYTHONPATH=$PYTHONPATH:.

# ===== GPU Configuration =====
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
BASE_PORTS="${BASE_PORTS:-12182,12282,12382,12482}"

# ===== Parse split argument =====
LIMIT="${1:-500}"
INDEX_FILE="pv_index_${LIMIT}.jsonl"
OUT_BASE="./outputs/multigpu_dynamic_${LIMIT}"

# All agents
AGENTS=(
    # === Single-view baselines (no navigation, not affected by NBV fix) ===
    # "single_view_direct"
    # "single_view_attr"
    # "single_view_merged"

    # === Multi-view without LLM (no navigation, not affected by NBV fix) ===
    # "multi_view_direct_random"  # next
    # "multi_view_attr_random"

    # === Multi-view with Geometric NBV (FPS-based, no LLM navigation) ===
    # "multi_view_direct_fps"
    # "multi_view_attr_fps"

    # === Multi-view with LLM (NBV-based, affected by navigation fix) ===
    # "multi_view_direct_llm"  # next
    # "multi_view_attr_llm"
    # "multi_view_attr_viewhint"

    # === Oracle NBV (upper bound, uses all viewpoints) ===
    # "multi_view_direct_oracle"
    # "multi_view_attr_oracle"

    # === Adaptive stopping (attr_majority fusion, max_steps=6) ===
    # "multi_view_attr_adaptive_random"
    # "multi_view_attr_adaptive_fps"
    # "multi_view_attr_adaptive_llm"  # next

    # "multi_view_direct_adaptive_random"
    # "multi_view_direct_adaptive_fps"
    # "multi_view_direct_adaptive_llm"  # next

    # === Visibility-weighted adaptive stopping (vis_weighted fusion, max_steps=6) ===
    "multi_view_attr_adaptive_vis_random"
    "multi_view_direct_adaptive_vis_random"
)

echo "========================================"
echo "Dynamic Work Stealing Multi-GPU Evaluation"
echo "Running ${#AGENTS[@]} agents x 2 modes (DINO + GT)"
echo "Split: ${INDEX_FILE} (${LIMIT} samples)"
echo "GPUs: ${NUM_GPUS}"
echo "Strategy: Task queue with dynamic load balancing"
echo "========================================"

# for mode in gt dino; do
for mode in dino; do
    echo ""
    echo "========== ${mode^^} Mode =========="

    for agent in "${AGENTS[@]}"; do
        echo ""
        echo "=============================="
        echo "[${agent}_${mode}_${LIMIT}] Start: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=============================="

        python scripts/evaluate_multigpu_dynamic.py \
            --config "configs/agent/${agent}.yaml" \
            --num_gpus $NUM_GPUS \
            --gpu_ids "$GPU_IDS" \
            --base_ports "$BASE_PORTS" \
            --dataset_index "$INDEX_FILE" \
            --output_dir "${OUT_BASE}/${agent}_${mode}_${LIMIT}" \
            dataset.index_file="$INDEX_FILE" \
            method.bbox_mode=$mode \
            output.save_viz=false

        echo "[${agent}_${mode}_${LIMIT}] Completed: $(date '+%Y-%m-%d %H:%M:%S')"
    done
done

echo ""
echo "========================================"
echo "All ${#AGENTS[@]} agents x 2 modes completed!"
echo "Results saved to: ${OUT_BASE}"
echo ""
echo "Key advantages of dynamic scheduling:"
echo "  - No GPU idle time (faster GPUs process more episodes)"
echo "  - Better fault tolerance (failed episodes isolated)"
echo "  - Real-time progress tracking"
echo "========================================"
