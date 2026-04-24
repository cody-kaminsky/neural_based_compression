#!/usr/bin/env bash
# monitor.sh — start TensorBoard and tail all training logs
tensorboard --logdir /workspace/runs --bind_all --port 6006 &
echo "TensorBoard running on port 6006"
tail -f /workspace/logs/train_lmbda_0.001.log /workspace/logs/train_lmbda_0.005.log /workspace/logs/train_lmbda_0.01.log
