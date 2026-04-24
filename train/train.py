"""Training loop for neural compression model.

Usage:
    python train/train.py \\
        --data /path/to/images \\
        --lmbda 0.05 \\
        --epochs 200 \\
        --batch-size 8 \\
        --lr 1e-4 \\
        --workers 4 \\
        --output-dir checkpoints/ \\
        --resume checkpoints/last.pth
"""

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from pytorch_msssim import ms_ssim
except ImportError as e:
    raise ImportError("pytorch_msssim is required: pip install pytorch-msssim") from e

try:
    from train.model import NeuralEncoderModel
except ImportError as e:
    raise ImportError(
        "train/model.py not found. NeuralEncoderModel must be provided by the "
        "model implementation task before training can begin."
    ) from e

from train.dataset import AerialVideoDataset
from train.dataset_utils import yuv_to_rgb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train neural compression model")
    p.add_argument("--data", required=True, help="Root directory of training images")
    p.add_argument("--lmbda", type=float, default=0.05, help="Rate-distortion tradeoff λ")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output-dir", default="checkpoints/")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--patch-size", type=int, default=256)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def rate_from_likelihoods(likelihoods: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
    """Bits per pixel from a likelihoods tensor."""
    return (-torch.log2(likelihoods)).sum() / (b * h * w)


def compute_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    y_likelihoods: torch.Tensor,
    z_likelihoods: torch.Tensor,
    lmbda: float,
) -> tuple[torch.Tensor, dict]:
    b, _, h, w = x.shape

    # Convert YUV back to RGB for MS-SSIM (ms_ssim expects [0,1] RGB)
    x_rgb = yuv_to_rgb(x)
    x_hat_rgb = yuv_to_rgb(x_hat.clamp(0, 1))

    _ms = min(x_rgb.shape[-2:])
    if _ms > 96:
        msssim_val = ms_ssim(x_rgb, x_hat_rgb, data_range=1.0, size_average=True, win_size=7)
    elif _ms > 32:
        msssim_val = ms_ssim(x_rgb, x_hat_rgb, data_range=1.0, size_average=True, win_size=3)
    else:
        msssim_val = torch.tensor(0.9, device=x_rgb.device)
    D = 1.0 - msssim_val

    R = rate_from_likelihoods(y_likelihoods, b, h, w) + \
        rate_from_likelihoods(z_likelihoods, b, h, w)

    loss = lmbda * D + R

    # PSNR for logging
    mse = torch.mean((x_rgb - x_hat_rgb) ** 2)
    psnr = -10.0 * torch.log10(mse + 1e-10)

    return loss, {"loss": loss.item(), "bpp": R.item(), "psnr": psnr.item()}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: Adam) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    lmbda: float,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> float:
    model.train()
    total_loss = 0.0
    global_step = epoch * len(loader)

    for step, x in enumerate(loader):
        x = x.to(device)
        optimizer.zero_grad()

        out = model(x)
        loss, metrics = compute_loss(
            x, out["x_hat"], out["y_likelihoods"], out["z_likelihoods"], lmbda
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += metrics["loss"]

        if step % 50 == 0:
            print(
                f"Epoch {epoch+1} [{step}/{len(loader)}]  "
                f"loss={metrics['loss']:.4f}  bpp={metrics['bpp']:.4f}  "
                f"psnr={metrics['psnr']:.2f} dB"
            )
        writer.add_scalar("train/loss", metrics["loss"], global_step + step)

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    lmbda: float,
    device: torch.device,
) -> dict:
    model.eval()
    totals = {"loss": 0.0, "bpp": 0.0, "psnr": 0.0}

    for x in loader:
        x = x.to(device)
        out = model(x)
        _, metrics = compute_loss(
            x, out["x_hat"], out["y_likelihoods"], out["z_likelihoods"], lmbda
        )
        for k in totals:
            totals[k] += metrics[k]

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = AerialVideoDataset(args.data, patch_size=args.patch_size, split="train")
    val_ds = AerialVideoDataset(args.data, patch_size=args.patch_size, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = NeuralEncoderModel().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))

    best_val_loss = math.inf

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, args.lmbda, device, epoch, writer
        )
        val_metrics = validate(model, val_loader, args.lmbda, device)
        scheduler.step()

        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/psnr", val_metrics["psnr"], epoch)
        writer.add_scalar("val/bpp", val_metrics["bpp"], epoch)

        print(
            f"[Epoch {epoch+1}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_psnr={val_metrics['psnr']:.2f} dB  val_bpp={val_metrics['bpp']:.4f}"
        )

        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_metrics["loss"],
        }

        save_checkpoint(checkpoint, os.path.join(args.output_dir, "last.pth"))
        if is_best:
            save_checkpoint(checkpoint, os.path.join(args.output_dir, "best.pth"))
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                checkpoint,
                os.path.join(args.output_dir, f"epoch_{epoch+1:04d}.pth"),
            )

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
