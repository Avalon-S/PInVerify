#!/bin/bash
# Check if multi-GPU API servers are running correctly

echo "========================================="
echo "Checking Multi-GPU API Server Status"
echo "========================================="
echo ""

# Expected ports
PORTS=(12182 12282 12382 12482)
GDINO_PORTS=(12183 12283 12383 12483)

check_endpoint() {
    local URL=$1
    local NAME=$2

    if curl -s --connect-timeout 2 "$URL" > /dev/null 2>&1; then
        echo "  ✓ $NAME - OK"
        return 0
    else
        echo "  ✗ $NAME - NOT RESPONDING"
        return 1
    fi
}

all_ok=true

for i in {0..3}; do
    PORT=${PORTS[$i]}
    GDINO_PORT=${GDINO_PORTS[$i]}

    echo "GPU $i:"

    # Check Qwen server
    if ! check_endpoint "http://127.0.0.1:$PORT/qwen-text" "Qwen (port $PORT)"; then
        all_ok=false
    fi

    # Check GDINO server
    if ! check_endpoint "http://127.0.0.1:$GDINO_PORT/groundingdino" "GDINO (port $GDINO_PORT)"; then
        all_ok=false
    fi

    echo ""
done

echo "========================================="

if $all_ok; then
    echo "✓ All servers are running!"
    echo ""
    echo "You can now run:"
    echo "  bash run_all_multigpu_500_dynamic.sh"
else
    echo "✗ Some servers are not responding!"
    echo ""
    echo "To start servers, run:"
    echo "  bash scripts/start_multigpu_servers.sh"
    echo ""
    echo "To check tmux session:"
    echo "  tmux list-sessions"
    echo "  tmux attach -t pver_multigpu"
    echo ""
    echo "To check if old servers are blocking ports:"
    echo "  lsof -i :12182"
    echo "  pkill -f 'run_qwen_batched.py'"
    echo "  pkill -f 'run_groundingdino_server.py'"
fi

echo "========================================="
