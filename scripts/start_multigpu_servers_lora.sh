#!/bin/bash
# =============================================================================
# Multi-GPU Server Launcher with LoRA Adapter Support
# =============================================================================
# Each GPU runs Qwen (base + optional LoRA) + GDINO servers in tmux
#
# Usage:
#   # Base model (same as start_multigpu_servers.sh)
#   bash scripts/start_multigpu_servers_lora.sh 4
#
#   # SFT model
#   ADAPTER=./data/sft_output/v3-20260302-092616/checkpoint-536 \
#     bash scripts/start_multigpu_servers_lora.sh 4
#
#   # SFT+GRPO model
#   ADAPTER=./data/grpo_sft_output/v0-xxx/checkpoint-500 \
#     bash scripts/start_multigpu_servers_lora.sh 4
#
#   # Custom model path (e.g., 8B)
#   MODEL=./models/Qwen3-VL-8B-Instruct \
#     bash scripts/start_multigpu_servers_lora.sh 7
#
# Port scheme: GPU i -> Qwen = BASE_PORT + i*100, GDINO = BASE_PORT + i*100 + 1
# =============================================================================

NUM_GPUS="${1:-${NUM_GPUS:-4}}"
GPU_START="${2:-${GPU_START:-0}}"
BASE_PORT="${BASE_PORT:-12182}"
SESSION_NAME="${SESSION_NAME:-pver_multigpu}"
CONDA_ENV="${CONDA_ENV:-pv_bench}"

# Optional: LoRA adapter and model override
ADAPTER="${ADAPTER:-}"
MODEL="${MODEL:-}"

# Check tmux
if ! command -v tmux &> /dev/null; then
    echo "tmux not found, please install it."
    exit 1
fi

# Kill existing session
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

# Build Qwen launch command
QWEN_EXTRA_ARGS=""
if [ -n "$ADAPTER" ]; then
    QWEN_EXTRA_ARGS="$QWEN_EXTRA_ARGS --adapter $ADAPTER"
fi
if [ -n "$MODEL" ]; then
    QWEN_EXTRA_ARGS="$QWEN_EXTRA_ARGS --model $MODEL"
fi

echo "========================================="
echo "Starting ${NUM_GPUS}x Qwen + ${NUM_GPUS}x GDINO"
if [ -n "$MODEL" ]; then
    echo "Model:   $MODEL"
fi
if [ -n "$ADAPTER" ]; then
    echo "Adapter: $ADAPTER"
else
    echo "Adapter: (none, base model)"
fi
echo ""
echo "Port Assignment:"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id=$((GPU_START + i))
    qwen_port=$((BASE_PORT + i * 100))
    gdino_port=$((qwen_port + 1))
    echo "  GPU $gpu_id: Qwen=$qwen_port, GDINO=$gdino_port"
done
echo "========================================="

# Create tmux session
tmux new-session -d -s "$SESSION_NAME" -n "init"

win_idx=0

for i in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id=$((GPU_START + i))
    qwen_port=$((BASE_PORT + i * 100))
    gdino_port=$((qwen_port + 1))

    # Qwen server
    if [ $win_idx -eq 0 ]; then
        tmux rename-window -t "$SESSION_NAME:0" "GPU${gpu_id}_Qwen"
    else
        tmux new-window -t "$SESSION_NAME" -n "GPU${gpu_id}_Qwen"
    fi
    tmux send-keys -t "$SESSION_NAME:$win_idx" "cd ~/pv_benchmark/servers" C-m
    tmux send-keys -t "$SESSION_NAME:$win_idx" "conda activate $CONDA_ENV" C-m
    tmux send-keys -t "$SESSION_NAME:$win_idx" "CUDA_VISIBLE_DEVICES=$gpu_id python run_qwen_batched.py --port $qwen_port $QWEN_EXTRA_ARGS" C-m
    win_idx=$((win_idx + 1))

    # GDINO server
    tmux new-window -t "$SESSION_NAME" -n "GPU${gpu_id}_GDINO"
    tmux send-keys -t "$SESSION_NAME:$win_idx" "cd ~/pv_benchmark/servers/GroundingDINO" C-m
    tmux send-keys -t "$SESSION_NAME:$win_idx" "conda activate $CONDA_ENV" C-m
    tmux send-keys -t "$SESSION_NAME:$win_idx" "CUDA_VISIBLE_DEVICES=$gpu_id python run_groundingdino_server.py --port $gdino_port" C-m
    win_idx=$((win_idx + 1))
done

# Generate matching evaluate command hint
gpu_ids=""
base_ports=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id=$((GPU_START + i))
    qwen_port=$((BASE_PORT + i * 100))
    [ -n "$gpu_ids" ] && gpu_ids="$gpu_ids,"
    gpu_ids="$gpu_ids$gpu_id"
    [ -n "$base_ports" ] && base_ports="$base_ports,"
    base_ports="$base_ports$qwen_port"
done

echo ""
echo "========================================="
echo "All $((NUM_GPUS * 2)) services started in tmux!"
echo ""
echo "Commands:"
echo "  Attach:  tmux attach -t $SESSION_NAME"
echo "  Stop:    tmux kill-session -t $SESSION_NAME"
echo ""
echo "Evaluate:"
echo "  NUM_GPUS=$NUM_GPUS GPU_IDS=$gpu_ids BASE_PORTS=$base_ports \\"
echo "    bash scripts/eval_trained.sh <agent> <tag>"
echo ""
echo "Navigate windows: Ctrl+b then number 0-$((win_idx - 1))"
echo "========================================="
