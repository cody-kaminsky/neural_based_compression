#!/usr/bin/env bash
# Start TensorBoard and tail all 3 training logs simultaneously.
set -euo pipefail

tensorboard --logdir runs/ --bind_all --port 6006 &
echo "TensorBoard running on port 6006 (pid $!)"

tail -f logs/train_lmbda_0.001.log logs/train_lmbda_0.005.log logs/train_lmbda_0.01.log
