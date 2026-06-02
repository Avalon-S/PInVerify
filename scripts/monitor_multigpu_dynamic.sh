#!/bin/bash
# tmux Monitoring Dashboard for Dynamic Multi-GPU Evaluation
# Shows real-time progress with work-stealing load balancing

SESSION_NAME="multigpu_dynamic_monitor"
NUM_GPUS=4

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Attaching..."
    tmux attach-session -t "$SESSION_NAME"
    exit 0
fi

# Get log directory
if [ -z "$1" ]; then
    LOG_BASE_DIR="logs/multigpu_dynamic/latest"
    if [ ! -L "$LOG_BASE_DIR" ]; then
        echo "Error: No running dynamic evaluation found."
        echo "Expected symlink: $LOG_BASE_DIR"
        echo ""
        echo "Usage:"
        echo "  $0 [log_directory]"
        echo ""
        echo "Example:"
        echo "  $0 logs/multigpu_dynamic/20250203_143022"
        exit 1
    fi
else
    LOG_BASE_DIR="$1"
fi

echo "Creating tmux monitoring session: $SESSION_NAME"
echo "Log directory: $LOG_BASE_DIR"
echo "Layout: $NUM_GPUS GPU workers"

# Create new session
tmux new-session -d -s "$SESSION_NAME" -n "Dynamic"

# Layout: 2x2 grid
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# Setup monitoring for each worker
setup_pane() {
    local PANE_ID=$1
    local WORKER_ID=$2
    local LOG_FILE="$LOG_BASE_DIR/worker_${WORKER_ID}.log"

    tmux select-pane -t "$PANE_ID"
    tmux send-keys "clear" C-m
    tmux send-keys "echo '════════════════════════════════════════'" C-m
    tmux send-keys "echo '  Worker $WORKER_ID (Dynamic Queue)'" C-m
    tmux send-keys "echo '════════════════════════════════════════'" C-m
    tmux send-keys "echo 'Waiting for log file: $LOG_FILE'" C-m
    tmux send-keys "echo ''" C-m

    # Wait for log file and tail it
    tmux send-keys "while [ ! -f '$LOG_FILE' ]; do sleep 1; done" C-m
    tmux send-keys "echo 'Log file detected! Monitoring...'" C-m
    tmux send-keys "echo ''" C-m
    tmux send-keys "tail -f '$LOG_FILE' 2>/dev/null || echo 'Waiting for log...'" C-m
}

# Setup all 4 panes
for i in $(seq 0 3); do
    setup_pane "$i" "$i"
done

# Enable mouse
tmux set-option -g mouse on

# Pane borders
tmux set-option -g pane-border-style fg=white
tmux set-option -g pane-active-border-style fg=green

# Status bar
tmux set-option -g status-style bg=black,fg=white
tmux set-option -g status-left "#[fg=green]Dynamic Multi-GPU #[fg=white]| "
tmux set-option -g status-right "#[fg=yellow]%Y-%m-%d %H:%M:%S"
tmux set-option -g status-interval 1

# Synchronize toggle
tmux bind-key S set-window-option synchronize-panes

echo ""
echo "════════════════════════════════════════"
echo "Tmux session created: $SESSION_NAME"
echo "════════════════════════════════════════"
echo ""
echo "Key features:"
echo "  • Real-time progress from all 4 workers"
echo "  • Each worker pulls episodes from shared queue"
echo "  • Fast workers process more episodes automatically"
echo ""
echo "Keyboard shortcuts:"
echo "  Ctrl+B then Arrow: Navigate panes"
echo "  Ctrl+B then S:     Synchronize panes"
echo "  Ctrl+B then D:     Detach session"
echo "  Ctrl+B then [:     Scroll mode (q to exit)"
echo ""
echo "To re-attach: tmux attach -t $SESSION_NAME"
echo "To kill: tmux kill-session -t $SESSION_NAME"
echo "════════════════════════════════════════"

# Attach
tmux attach-session -t "$SESSION_NAME"
