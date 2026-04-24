"""Split sequence directories 90/10 into train and val by creating symlinks.

Usage:
    python scripts/split_dataset.py <frames_dir> <train_dir> <val_dir>

Sequences are sorted and the last 10% go to val.
"""

import os
import sys


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(f"Usage: {sys.argv[0]} <frames_dir> <train_dir> <val_dir>")

    frames_dir, train_dir, val_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    sequences = sorted(
        d for d in os.listdir(frames_dir)
        if os.path.isdir(os.path.join(frames_dir, d))
    )

    if not sequences:
        sys.exit(f"No subdirectories found in {frames_dir}")

    split = max(1, int(len(sequences) * 0.9))
    train_seqs = sequences[:split]
    val_seqs = sequences[split:]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    frames_abs = os.path.abspath(frames_dir)

    for seq in train_seqs:
        src = os.path.join(frames_abs, seq)
        dst = os.path.join(train_dir, seq)
        if not os.path.exists(dst):
            os.symlink(src, dst)

    for seq in val_seqs:
        src = os.path.join(frames_abs, seq)
        dst = os.path.join(val_dir, seq)
        if not os.path.exists(dst):
            os.symlink(src, dst)

    print(f"Split {len(sequences)} sequences: {len(train_seqs)} train, {len(val_seqs)} val")


if __name__ == "__main__":
    main()
