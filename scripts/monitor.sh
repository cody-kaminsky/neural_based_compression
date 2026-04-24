#!/usr/bin/env bash
# Start TensorBoard and tail the latest training log.
set -euo pipefail

tensorboard --logdir runs/ --bind_all --port 6006 &
TB_PID=$!
echo "TensorBoard running on port 6006 (pid $TB_PID)"

LATEST_LOG=$(ls -t logs/ 2>/dev/null | head -1)
if [ -z "$LATEST_LOG" ]; then
    echo "No logs found in logs/ — waiting for training to start..."
    while [ -z "$LATEST_LOG" ]; do
        sleep 5
        LATEST_LOG=$(ls -t logs/ 2>/dev/null | head -1)
    done
fi

echo "Tailing logs/$LATEST_LOG"
tail -f "logs/$LATEST_LOG"
