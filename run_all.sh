#!/bin/bash
# =============================================================================
# PInVerify — Training-free evaluation runner (paper reproduction)
# =============================================================================
# Reproduces the training-free tables of the FMEA @ CVPR 2026 paper:
#   Table 4 (main training-free table)  -> AGENT_SET=main   (default)
#   Table 5 (cross-model, embedding-only rows) -> AGENT_SET=embedding
# Trained agents (Table 6) use a different entry point: scripts/eval_trained.sh
#
# All paper numbers use adaptive stopping over T=6 sectors on the 3,000-episode
# test split with Grounding DINO detection. GT-box ablations rerun the same
# configs with BBOX_MODES=gt.
#
# Prerequisites:
#   # MLLM agents (AGENT_SET=main)
#   bash scripts/start_multigpu_servers.sh 4
#   # Embedding baselines (AGENT_SET=embedding)
#   bash scripts/start_multigpu_servers.sh 4      # GDINO is still needed
#   # SenseNova-SI backbone
#   bash scripts/start_multigpu_servers_sensenova.sh 4
#
# Usage:
#   bash run_all.sh                  # main table, 3,000 episodes, DINO
#   bash run_all.sh 50               # 50-episode smoke split
#   BBOX_MODES="dino gt" bash run_all.sh
#   AGENT_SET=embedding bash run_all.sh
#   OUT_BASE=./outputs/qwen3_vl_4b SAVE_VIZ=true bash run_all.sh   # for the figures
#   NUM_GPUS=8 GPU_IDS=0,1,2,3,4,5,6,7 \
#     BASE_PORTS=12182,12282,12382,12482,12582,12682,12782,12882 bash run_all.sh
#
# Arguments:
#   $1 = split size: 50 | 100 | 500 | 1000 | 3000 | all   (default: all = 3,000)
# =============================================================================

set -e
export PYTHONPATH=$PYTHONPATH:.

# ===== GPU configuration =====
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
BASE_PORTS="${BASE_PORTS:-12182,12282,12382,12482}"

# ===== Split =====
LIMIT="${1:-all}"
if [ "$LIMIT" = "3000" ]; then LIMIT="all"; fi     # the full test split is pv_index_all.jsonl
INDEX_FILE="pv_index_${LIMIT}.jsonl"

# ===== Detection mode(s) =====
# The paper's main table uses DINO; "gt" reproduces the GT-box ablations.
BBOX_MODES="${BBOX_MODES:-dino}"

# ===== Agent set =====
AGENT_SET="${AGENT_SET:-main}"

# ===== Output =====
# The figure scripts in scripts/ read fixed paths under ./outputs, so set
# OUT_BASE to match them when reproducing the paper figures, e.g.
#   OUT_BASE=./outputs/qwen3_vl_4b SAVE_VIZ=true bash run_all.sh
# SAVE_VIZ writes per-episode episode.json, which plot_nbv_polar_dino.py and
# plot_case_study.py need. It costs disk but does not change any metric.
SAVE_VIZ="${SAVE_VIZ:-false}"

# Table 4: 3 single-view + 6 multi-view configs, all with adaptive stopping.
MAIN_AGENTS=(
    # --- Single-view (1 step, no navigation) ---
    "single_view_attr"                      # Table 4, "Single-View / Attr"
    "single_view_direct"                    # Table 4, "Single-View / Direct"
    "single_view_merged"                    # Table 4, "Single-View / Merged"
    # --- Multi-view, attribute decomposition (adaptive stopping, T=6) ---
    "multi_view_attr_adaptive_random"       # Table 4, "MV attr + Random"
    "multi_view_attr_adaptive_fps"          # Table 4, "MV attr + Angular FPS"
    "multi_view_attr_adaptive_llm"          # Table 4, "MV attr + LLM-NBV"  (TF-Best)
    # --- Multi-view, direct query (adaptive stopping, T=6) ---
    "multi_view_direct_adaptive_random"     # Table 4, "MV direct + Random"
    "multi_view_direct_adaptive_fps"        # Table 4, "MV direct + Angular FPS"
    "multi_view_direct_adaptive_llm"        # Table 4, "MV direct + LLM-NBV"
)

# Table 5 (CLIP / SigLIP2 rows): the paper reports the SV-Merged row for each;
# the remaining configs are the appendix sweep (Table 18).
EMBEDDING_AGENTS=(
    "clip_single_view"
    "clip_single_view_merged"                       # Table 5, "CLIP / SV-Merged"
    "clip_multi_view_adaptive_random"
    "clip_multi_view_adaptive_random_merged"
    "clip_multi_view_adaptive_fps"
    "clip_multi_view_adaptive_fps_merged"
    "siglip2_single_view"
    "siglip2_single_view_merged"                    # Table 5, "SigLIP2 / SV-Merged"
    "siglip2_multi_view_adaptive_random"
    "siglip2_multi_view_adaptive_random_merged"
    "siglip2_multi_view_adaptive_fps"
    "siglip2_multi_view_adaptive_fps_merged"
)

case "$AGENT_SET" in
    main)      AGENTS=("${MAIN_AGENTS[@]}") ;;
    embedding) AGENTS=("${EMBEDDING_AGENTS[@]}") ;;
    all)       AGENTS=("${MAIN_AGENTS[@]}" "${EMBEDDING_AGENTS[@]}") ;;
    *) echo "[ERROR] Unknown AGENT_SET: $AGENT_SET (expected main | embedding | all)"; exit 1 ;;
esac

OUT_BASE="${OUT_BASE:-./outputs/${AGENT_SET}_${LIMIT}}"

echo "========================================"
echo "PInVerify training-free evaluation"
echo "  Agent set:  ${AGENT_SET} (${#AGENTS[@]} configs)"
echo "  Split:      ${INDEX_FILE}"
echo "  Bbox modes: ${BBOX_MODES}"
echo "  GPUs:       ${NUM_GPUS} (${GPU_IDS})"
echo "  Output:     ${OUT_BASE}"
echo "  save_viz:   ${SAVE_VIZ}"
echo "========================================"

for agent in "${AGENTS[@]}"; do
    if [ ! -f "configs/agent/${agent}.yaml" ]; then
        echo "[ERROR] Missing config: configs/agent/${agent}.yaml"
        exit 1
    fi
done

for mode in $BBOX_MODES; do
    echo ""
    echo "========== ${mode} mode =========="

    for agent in "${AGENTS[@]}"; do
        out_dir="${OUT_BASE}/${agent}_${mode}_${LIMIT}"

        echo ""
        echo "=============================="
        echo "[${agent} / ${mode}] Start: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=============================="

        python scripts/evaluate_multigpu_dynamic.py \
            --config "configs/agent/${agent}.yaml" \
            --num_gpus $NUM_GPUS \
            --gpu_ids "$GPU_IDS" \
            --base_ports "$BASE_PORTS" \
            --dataset_index "$INDEX_FILE" \
            --output_dir "$out_dir" \
            dataset.index_file="$INDEX_FILE" \
            method.bbox_mode=$mode \
            output.save_viz=$SAVE_VIZ

        echo "[${agent} / ${mode}] Completed: $(date '+%Y-%m-%d %H:%M:%S')"
    done
done

echo ""
echo "========================================"
echo "Done: ${#AGENTS[@]} configs x $(echo $BBOX_MODES | wc -w) mode(s)"
echo "Results: ${OUT_BASE}"
echo ""
echo "Aggregate with:"
echo "  python scripts/summarize_all_agents.py --root ${OUT_BASE}"
echo "  python scripts/compare_metrics.py --root ${OUT_BASE}"
echo "========================================"
