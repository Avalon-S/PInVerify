#!/bin/bash
# scripts/manage_servers.sh
# Supports both the Qwen and SenseNova-SI VLM servers

SESSION_NAME="pver_v2"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "tmux could not be found, please install it."
    exit 1
fi

start_qwen() {
    # Start the Qwen + GDINO servers
    tmux kill-session -t $SESSION_NAME 2>/dev/null
    
    echo "Starting Qwen + GDINO servers..."
    
    tmux new-session -d -s $SESSION_NAME -n "DualServer"
    
    # --- Top Pane: Qwen (pv_bench env) ---
    tmux send-keys -t $SESSION_NAME:0.0 "cd ~/pv_benchmark/servers" C-m
    tmux send-keys -t $SESSION_NAME:0.0 "conda activate pv_bench" C-m
    tmux send-keys -t $SESSION_NAME:0.0 "python run_qwen_batched.py" C-m
    
    # --- Bottom Pane: GDINO (pv_bench env) ---
    tmux split-window -v -t $SESSION_NAME:0
    tmux send-keys -t $SESSION_NAME:0.1 "cd ~/pv_benchmark/servers/GroundingDINO" C-m
    tmux send-keys -t $SESSION_NAME:0.1 "conda activate pv_bench" C-m
    tmux send-keys -t $SESSION_NAME:0.1 "python run_groundingdino_server.py" C-m
    
    echo "Layout: Top=Qwen(port 12182), Bottom=GDINO(port 12183)"
    tmux attach-session -t $SESSION_NAME
}

start_sensenova() {
    # Start the SenseNova-SI + GDINO servers
    tmux kill-session -t $SESSION_NAME 2>/dev/null
    
    echo "Starting SenseNova-SI + GDINO servers..."
    
    tmux new-session -d -s $SESSION_NAME -n "DualServer"
    
    # --- Top Pane: SenseNova-SI (sensenova env) ---
    # The script lives in servers/, but PYTHONPATH must point at the SenseNova-SI checkout
    tmux send-keys -t $SESSION_NAME:0.0 "cd ~/pv_benchmark/servers" C-m
    tmux send-keys -t $SESSION_NAME:0.0 "conda activate sensenova" C-m
    tmux send-keys -t $SESSION_NAME:0.0 "export PYTHONPATH=${SENSENOVA_PATH:-./SenseNova-SI}:\$PYTHONPATH" C-m
    tmux send-keys -t $SESSION_NAME:0.0 "python run_sensenova_si_server.py" C-m
    
    # --- Bottom Pane: GDINO (pv_bench env) ---
    tmux split-window -v -t $SESSION_NAME:0
    tmux send-keys -t $SESSION_NAME:0.1 "cd ~/pv_benchmark/servers/GroundingDINO" C-m
    tmux send-keys -t $SESSION_NAME:0.1 "conda activate pv_bench" C-m
    tmux send-keys -t $SESSION_NAME:0.1 "python run_groundingdino_server.py" C-m
    
    echo "Layout: Top=SenseNova-SI(port 12182), Bottom=GDINO(port 12183)"
    tmux attach-session -t $SESSION_NAME
}

stop_servers() {
    tmux kill-session -t $SESSION_NAME 2>/dev/null
    echo "Session '$SESSION_NAME' killed."
}

case "$1" in
    qwen)
        start_qwen
        ;;
    sensenova)
        start_sensenova
        ;;
    stop)
        stop_servers
        ;;
    *)
        echo "Usage: $0 {qwen|sensenova|stop}"
        echo ""
        echo "Commands:"
        echo "  qwen      - Start Qwen + GDINO servers (both use pv_bench env)"
        echo "  sensenova - Start SenseNova-SI + GDINO servers"
        echo "  stop      - Stop all servers"
        exit 1
esac
