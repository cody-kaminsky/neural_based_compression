#!/usr/bin/env bash
# cloud_setup.sh — RunPod/Vast.ai setup for neural_based_compression training
# Requires a pod with 3x RTX 4090 GPUs.
# On RunPod: select 3x RTX 4090 under GPU count when creating the pod.
# Usage: bash scripts/cloud_setup.sh
set -euo pipefail

WORKSPACE=/workspace
REPO_DIR=$WORKSPACE/neural_based_compression

echo "=== Installing Python dependencies ==="
pip install -q compressai brevitas pytorch-msssim tensorboard onnx onnxruntime pillow tqdm pandas scipy

echo "=== Downloading COCO 2017 train images ==="
mkdir -p $WORKSPACE/dataset/raw
cd $WORKSPACE/dataset/raw
wget -q --show-progress -O coco_train2017.zip "http://images.cocodataset.org/zips/train2017.zip"
unzip -q coco_train2017.zip -d .
rm coco_train2017.zip
echo "COCO 2017 downloaded."

echo "=== Organizing dataset ==="
# COCO images are flat (no sequence structure), create synthetic sequences of 100 images each
cd $WORKSPACE
python3 - <<'EOF'
import os, shutil
from pathlib import Path

src = Path("/workspace/dataset/raw/train2017")
images = sorted(src.glob("*.jpg"))
seq_size = 100
train_dir = Path("/workspace/dataset/train")
val_dir = Path("/workspace/dataset/val")

# Split: last 10% of sequences to val
sequences = [images[i:i+seq_size] for i in range(0, len(images), seq_size)]
n_val = max(1, len(sequences) // 10)
train_seqs = sequences[:-n_val]
val_seqs = sequences[-n_val:]

for split_seqs, split_dir in [(train_seqs, train_dir), (val_seqs, val_dir)]:
    for i, seq in enumerate(split_seqs):
        seq_dir = split_dir / f"seq_{i:04d}"
        seq_dir.mkdir(parents=True, exist_ok=True)
        for img in seq:
            dst = seq_dir / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

print(f"Train: {len(train_seqs)} sequences, Val: {len(val_seqs)} sequences")
EOF
echo "Dataset organized."

echo "=== Starting training (3 models in parallel, one per GPU) ==="
mkdir -p $WORKSPACE/checkpoints/lmbda_0.001 $WORKSPACE/checkpoints/lmbda_0.005 $WORKSPACE/checkpoints/lmbda_0.01
mkdir -p $WORKSPACE/logs

CUDA_VISIBLE_DEVICES=0 python -m train.train \
    --lmbda 0.001 --epochs 200 --batch-size 16 --workers 4 \
    --amp --amp-dtype bfloat16 \
    --data-dir $WORKSPACE/dataset \
    --checkpoint-dir $WORKSPACE/checkpoints/lmbda_0.001 \
    2>&1 | tee $WORKSPACE/logs/train_lmbda_0.001.log &

CUDA_VISIBLE_DEVICES=1 python -m train.train \
    --lmbda 0.005 --epochs 200 --batch-size 16 --workers 4 \
    --amp --amp-dtype bfloat16 \
    --data-dir $WORKSPACE/dataset \
    --checkpoint-dir $WORKSPACE/checkpoints/lmbda_0.005 \
    2>&1 | tee $WORKSPACE/logs/train_lmbda_0.005.log &

CUDA_VISIBLE_DEVICES=2 python -m train.train \
    --lmbda 0.01 --epochs 200 --batch-size 16 --workers 4 \
    --amp --amp-dtype bfloat16 \
    --data-dir $WORKSPACE/dataset \
    --checkpoint-dir $WORKSPACE/checkpoints/lmbda_0.01 \
    2>&1 | tee $WORKSPACE/logs/train_lmbda_0.01.log &

echo "All 3 training processes launched. PIDs: $(jobs -p)"
echo "Monitor with: bash scripts/monitor.sh"
wait
echo "=== All training complete ==="
echo "Download checkpoints with:"
echo "  scp -r <pod-user>@<pod-ip>:$WORKSPACE/checkpoints/ ./checkpoints/"
