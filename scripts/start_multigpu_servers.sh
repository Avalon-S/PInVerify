#!/bin/bash
# =============================================================================
# Multi-GPU Server Launcher — Configurable GPU count
# =============================================================================
# Each GPU runs Qwen (text+vl) + GDINO servers in tmux
#
# Usage:
#   bash scripts/start_multigpu_servers.sh              # default: 4 GPUs (0-3)
#   bash scripts/start_multigpu_servers.sh 6             # 6 GPUs (0-5)
#   bash scripts/start_multigpu_servers.sh 6 2           # 6 GPUs starting from GPU 2 (2-7)
#   NUM_GPUS=8 bash scripts/start_multigpu_servers.sh    # via env var
#
# Port scheme: GPU i → Qwen = 12182 + i*100, GDINO = 12183 + i*100
# =============================================================================

NUM_GPUS="${1:-${NUM_GPUS:-4}}"
GPU_START="${2:-${GPU_START:-0}}"
BASE_PORT="${BASE_PORT:-12182}"
SESSION_NAME="${SESSION_NAME:-pver_multigpu}"
CONDA_ENV="${CONDA_ENV:-pv_bench}"

# Check tmux
if ! command -v tmux &> /dev/null; then
    echo "tmux not found, please install it."
    exit 1
fi

# Kill existing session
tmux kill-session -t "$SESSION_NAME" 2>/dev/null

echo "========================================="
echo "Starting ${NUM_GPUS}×Qwen + ${NUM_GPUS}×GDINO"
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
    tmux send-keys -t "$SESSION_NAME:$win_idx" "CUDA_VISIBLE_DEVICES=$gpu_id python run_qwen_batched.py --port $qwen_port" C-m
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
echo "  python scripts/evaluate_multigpu.py --config configs/agent/xxx.yaml \\"
echo "    --num_gpus $NUM_GPUS --gpu_ids $gpu_ids --base_ports $base_ports"
echo ""
echo "Navigate windows: Ctrl+b then number 0-$((win_idx - 1))"
echo "========================================="
