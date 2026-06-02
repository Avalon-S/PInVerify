#!/bin/bash
# scripts/manage_servers_multigpu.sh
# Launch multiple pairs of servers on different GPUs.
# Usage: bash scripts/manage_servers_multigpu.sh start [GPUS]
#        bash scripts/manage_servers_multigpu.sh stop

SESSION_NAME="pver_multigpu"

# Default to 4 GPUs if not specified
NUM_GPUS=${2:-4}

# Port base settings
QWEN_PORT_BASE=12182
GDINO_PORT_BASE=12183
PORT_OFFSET=100

if ! command -v tmux &> /dev/null; then
    echo "tmux could not be found, please install it."
    exit 1
fi

start_servers() {
    tmux kill-session -t $SESSION_NAME 2>/dev/null
    
    echo "Starting servers on ${NUM_GPUS} GPUs..."
    echo "Port mapping:"
    for i in $(seq 0 $((NUM_GPUS-1))); do
        qwen_port=$((QWEN_PORT_BASE + i * PORT_OFFSET))
        gdino_port=$((GDINO_PORT_BASE + i * PORT_OFFSET))
        echo "  GPU $i: Qwen=$qwen_port, GDINO=$gdino_port"
    done
    
    # Create session with first window
    tmux new-session -d -s $SESSION_NAME -n "GPU0"
    
    for i in $(seq 0 $((NUM_GPUS-1))); do
        qwen_port=$((QWEN_PORT_BASE + i * PORT_OFFSET))
        gdino_port=$((GDINO_PORT_BASE + i * PORT_OFFSET))
        
        if [ $i -gt 0 ]; then
            tmux new-window -t $SESSION_NAME -n "GPU$i"
        fi
        
        # Top pane: Qwen server
        tmux send-keys -t $SESSION_NAME:GPU$i "cd ~/pv_benchmark/servers" C-m
        tmux send-keys -t $SESSION_NAME:GPU$i "conda activate pv_bench" C-m
        tmux send-keys -t $SESSION_NAME:GPU$i "CUDA_VISIBLE_DEVICES=$i python run_qwen3_server.py --port $qwen_port" C-m
        
        # Bottom pane: GDINO server
        tmux split-window -v -t $SESSION_NAME:GPU$i
        tmux send-keys -t $SESSION_NAME:GPU$i.1 "cd ~/pv_benchmark/servers/GroundingDINO" C-m
        tmux send-keys -t $SESSION_NAME:GPU$i.1 "conda activate pv_bench" C-m
        tmux send-keys -t $SESSION_NAME:GPU$i.1 "CUDA_VISIBLE_DEVICES=$i python run_groundingdino_server.py --port $gdino_port" C-m
    done
    
    echo ""
    echo "All servers started in tmux session '$SESSION_NAME'"
    echo "Attach with: tmux attach -t $SESSION_NAME"
    echo "Switch windows with: Ctrl+B, then number (0-$((NUM_GPUS-1)))"
    
    # Attach by default
    tmux attach-session -t $SESSION_NAME
}

stop_servers() {
    tmux kill-session -t $SESSION_NAME 2>/dev/null
    echo "Session '$SESSION_NAME' killed."
}

case "$1" in
    start)
        start_servers
        ;;
    stop)
        stop_servers
        ;;
    *)
        echo "Usage: $0 {start|stop} [NUM_GPUS]"
        echo ""
        echo "Examples:"
        echo "  $0 start 4    # Start servers on 4 GPUs"
        echo "  $0 stop       # Stop all servers"
        exit 1
esac
