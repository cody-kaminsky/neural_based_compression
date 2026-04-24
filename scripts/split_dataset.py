#!/usr/bin/env python3
"""Split frame sequences into train/val sets (90/10 by sequence)."""
import os, sys, shutil
from pathlib import Path

def split(frames_dir, train_dir, val_dir, val_fraction=0.1):
    frames_dir = Path(frames_dir)
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    sequences = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    n_val = max(1, int(len(sequences) * val_fraction))
    val_seqs = sequences[-n_val:]
    train_seqs = sequences[:-n_val]
    for seqs, dst in [(train_seqs, train_dir), (val_seqs, val_dir)]:
        dst.mkdir(parents=True, exist_ok=True)
        for seq in seqs:
            link = dst / seq.name
            if not link.exists():
                os.symlink(seq.resolve(), link)
    print(f"Split: {len(train_seqs)} train sequences, {len(val_seqs)} val sequences")

if __name__ == "__main__":
    split(sys.argv[1], sys.argv[2], sys.argv[3])
