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

echo "=== Downloading VisDrone-VID dataset ==="
mkdir -p $WORKSPACE/dataset/raw
cd $WORKSPACE/dataset/raw
wget -q --show-progress -O visdrone_train.zip "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-VID-train.zip"
wget -q --show-progress -O visdrone_val.zip "https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-VID-val.zip"
unzip -q visdrone_train.zip -d .
unzip -q visdrone_val.zip -d .
rm visdrone_train.zip visdrone_val.zip
echo "Dataset downloaded and extracted."

echo "=== Extracting frames at 1fps ==="
cd $WORKSPACE
mkdir -p dataset/frames
for f in $(find dataset/raw -name "*.mp4" -o -name "*.avi" -o -name "*.MOV" 2>/dev/null); do
    seq=$(basename $(dirname "$f"))_$(basename "${f%.*}")
    mkdir -p "dataset/frames/$seq"
    ffmpeg -i "$f" -vf fps=1 -q:v 1 "dataset/frames/$seq/%06d.png" -y -loglevel error
done
echo "Frame extraction complete."

echo "=== Splitting dataset into train/val ==="
cd $REPO_DIR
python scripts/split_dataset.py $WORKSPACE/dataset/frames $WORKSPACE/dataset/train $WORKSPACE/dataset/val

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
