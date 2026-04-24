#!/usr/bin/env bash
# Cloud instance setup for neural_based_compression training.
# Tested on Ubuntu with CUDA pre-installed (RunPod / Vast.ai).
# Requires a pod with 3 GPUs. On RunPod, select '3x RTX 4090' under GPU count when creating the pod.
set -euo pipefail

REPO_URL="https://github.com/cody-kaminsky/neural_based_compression"
DATASET_SOURCE="${DATASET_SOURCE:-visdrone}"

# ---------------------------------------------------------------------------
# 1. Python dependencies
# ---------------------------------------------------------------------------
echo "==> Installing Python dependencies"
pip install compressai brevitas pytorch-msssim tensorboard onnx onnxruntime pillow tqdm pandas scipy

# ---------------------------------------------------------------------------
# 2. Clone / update repo
# ---------------------------------------------------------------------------
echo "==> Fetching repo"
if [ ! -d "neural_based_compression" ]; then
    git clone "$REPO_URL"
fi
cd neural_based_compression
git pull origin master

# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------
mkdir -p dataset/raw logs

case "$DATASET_SOURCE" in
    visdrone)
        echo "==> Downloading VisDrone-VID"
        VISDRONE_URL="https://github.com/VisDrone/VisDrone-Dataset/releases/download/v1.0/VisDrone2019-VID-train.zip"
        wget -q --show-progress -O dataset/raw/visdrone_train.zip "$VISDRONE_URL"
        unzip -q dataset/raw/visdrone_train.zip -d dataset/raw/
        rm dataset/raw/visdrone_train.zip
        ;;
    uavid)
        echo "==> UAVid requires manual download (registration required)."
        echo "    1. Register at https://uavid.nl"
        echo "    2. Download the video dataset and place .mp4 files under dataset/raw/"
        echo "    3. Re-run this script with DATASET_SOURCE=uavid after placing files."
        exit 0
        ;;
    *)
        echo "ERROR: Unknown DATASET_SOURCE='$DATASET_SOURCE'. Use 'visdrone' or 'uavid'." >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# 4. Extract frames at 1 fps
# ---------------------------------------------------------------------------
echo "==> Extracting frames at 1 fps"
shopt -s globstar nullglob
for f in dataset/raw/**/*.mp4; do
    seq=$(basename "$(dirname "$f")")
    mkdir -p "dataset/frames/$seq"
    ffmpeg -i "$f" -vf fps=1 -q:v 1 "dataset/frames/$seq/%06d.png" -y -loglevel error
done

echo "==> Splitting sequences into train/val (90/10)"
python scripts/split_dataset.py dataset/frames/ dataset/train/ dataset/val/

# ---------------------------------------------------------------------------
# 5. Training — parallel lambda sweep (one process per GPU)
# ---------------------------------------------------------------------------
mkdir -p logs checkpoints/lmbda_0.001 checkpoints/lmbda_0.005 checkpoints/lmbda_0.01

echo "==> Launching 3 training jobs in parallel (GPU 0, 1, 2)"

CUDA_VISIBLE_DEVICES=0 python -m train.train --lmbda 0.001 --epochs 200 --batch-size 16 --workers 4 --amp --amp-dtype bfloat16 --data-dir dataset/ --checkpoint-dir checkpoints/lmbda_0.001/ 2>&1 | tee logs/train_lmbda_0.001.log &

CUDA_VISIBLE_DEVICES=1 python -m train.train --lmbda 0.005 --epochs 200 --batch-size 16 --workers 4 --amp --amp-dtype bfloat16 --data-dir dataset/ --checkpoint-dir checkpoints/lmbda_0.005/ 2>&1 | tee logs/train_lmbda_0.005.log &

CUDA_VISIBLE_DEVICES=2 python -m train.train --lmbda 0.01  --epochs 200 --batch-size 16 --workers 4 --amp --amp-dtype bfloat16 --data-dir dataset/ --checkpoint-dir checkpoints/lmbda_0.01/  2>&1 | tee logs/train_lmbda_0.01.log &

wait
echo "All 3 models done."

# ---------------------------------------------------------------------------
# 6. Done
# ---------------------------------------------------------------------------
echo ""
echo "Training complete. Download your checkpoints:"
echo "  scp -r <pod-user>@<pod-ip>:neural_based_compression/checkpoints/ ./checkpoints/"
