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
from torch.amp import GradScaler, autocast
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
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed-precision training (AMP is on by default on CUDA)")
    p.add_argument("--mse-weight", type=float, default=0.16,
                   help="Weight on MSE term in hybrid distortion (1-MSSSIM gets 1-mse_weight)")
    p.add_argument("--mse-scale", type=float, default=10.0,
                   help="Scale factor on MSE to bring it into 1-MSSSIM's numerical range")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output-dir", default="checkpoints/")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Epochs to ramp rate-term scale from 0 to 1")
    p.add_argument("--subset-fraction", type=float, default=1.0,
                   help="Fraction of dataset to use (e.g. 0.1 for 10%) for fast iteration")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Rate-term warmup schedule
# ---------------------------------------------------------------------------

def rate_scale_for_epoch(epoch: int, warmup_epochs: int) -> float:
    """Linear ramp of rate-term scale from 0 (epoch 0) to 1 (epoch warmup_epochs).

    Epoch 0 is pure autoencoder training (rate term zeroed) so synthesis learns
    to use the latent before the rate gradient starts pushing y→0. λ warmup was
    not enough because R competes with λ*D from step 0 regardless of λ.
    """
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return 1.0
    return epoch / warmup_epochs


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
    rate_scale: float = 1.0,
    mse_weight: float = 0.16,
    mse_scale: float = 10.0,
) -> tuple[torch.Tensor, dict]:
    """Hybrid distortion: D = (1-α)·(1-MSSSIM) + α·(mse_scale·MSE).

    The mse_scale brings MSE into the same numerical range as 1-MS-SSIM (~0.04
    at convergence) so existing λ values keep their meaning, and so neither
    term dominates the gradient. Tune mse_scale if PSNR/MS-SSIM trade-off
    needs shifting.

    Pure MS-SSIM training leaves PSNR on the floor because the loss never sees
    per-pixel error. Adding an MSE term recovers ~2 dB PSNR at negligible
    MS-SSIM cost (Liu et al. recipe).
    """
    b, _, h, w = x.shape

    # Convert YUV back to RGB for MS-SSIM (ms_ssim expects [0,1] RGB).
    # Cast to FP32 explicitly so MS-SSIM's window convolutions and the MSE term
    # are stable when this is called inside an autocast (FP16) region.
    x_rgb = yuv_to_rgb(x).float()
    x_hat_rgb = yuv_to_rgb(x_hat.clamp(0, 1)).float()

    _ms = min(x_rgb.shape[-2:])
    if _ms > 96:
        msssim_val = ms_ssim(x_rgb, x_hat_rgb, data_range=1.0, size_average=True, win_size=7)
    elif _ms > 32:
        msssim_val = ms_ssim(x_rgb, x_hat_rgb, data_range=1.0, size_average=True, win_size=3)
    else:
        msssim_val = torch.tensor(0.9, device=x_rgb.device)

    mse_val = torch.mean((x_rgb - x_hat_rgb) ** 2)

    D_msssim = 1.0 - msssim_val
    D_mse = mse_scale * mse_val
    D = (1.0 - mse_weight) * D_msssim + mse_weight * D_mse

    R = rate_from_likelihoods(y_likelihoods, b, h, w) + \
        rate_from_likelihoods(z_likelihoods, b, h, w)

    loss = lmbda * D + rate_scale * R

    # PSNR for logging
    psnr = -10.0 * torch.log10(mse_val + 1e-10)

    return loss, {
        "loss": loss.item(),
        "bpp": R.item(),
        "psnr": psnr.item(),
        "msssim": msssim_val.item(),
        "d_msssim": D_msssim.item(),
        "d_mse": D_mse.item(),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Adam,
    aux_optimizer: Adam,
    scaler: GradScaler | None = None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "aux_optimizer" in ckpt:
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0)


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam,
    aux_optimizer: Adam,
    scaler: GradScaler,
    lmbda: float,
    rate_scale: float,
    mse_weight: float,
    mse_scale: float,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    global_step = epoch * len(loader)

    for step, x in enumerate(loader):
        x = x.to(device, non_blocking=True)

        # Main optimizer step (AMP-wrapped forward + loss).
        # Aux step intentionally stays in FP32: EntropyBottleneck.loss() is a
        # |logits - target| sum that's small in magnitude and known-flaky in
        # FP16. The aux step is cheap, so the FP32 path costs ~nothing.
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(x)
            loss, metrics = compute_loss(
                x, out["x_hat"], out["y_likelihoods"], out["z_likelihoods"],
                lmbda, rate_scale, mse_weight, mse_scale,
            )

        scaler.scale(loss).backward()
        # Unscale before clipping so the threshold (1.0) means real gradient norm,
        # not scaled gradient norm. scaler.step() then skips if any inf/nan.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Aux optimizer step: trains entropy bottleneck quantile parameters
        aux_optimizer.zero_grad(set_to_none=True)
        aux_loss = model.entropy_bottleneck.loss()
        aux_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in aux_optimizer.param_groups for p in g["params"]], 1.0
        )
        aux_optimizer.step()

        total_loss += metrics["loss"]

        if step % 50 == 0:
            print(
                f"Epoch {epoch+1} [{step}/{len(loader)}]  "
                f"loss={metrics['loss']:.4f}  bpp={metrics['bpp']:.4f}  "
                f"psnr={metrics['psnr']:.2f} dB  msssim={metrics['msssim']:.4f}  "
                f"aux={aux_loss.item():.4f}"
            )
        writer.add_scalar("train/loss", metrics["loss"], global_step + step)
        writer.add_scalar("train/bpp", metrics["bpp"], global_step + step)
        writer.add_scalar("train/psnr", metrics["psnr"], global_step + step)
        writer.add_scalar("train/msssim", metrics["msssim"], global_step + step)
        writer.add_scalar("train/d_msssim", metrics["d_msssim"], global_step + step)
        writer.add_scalar("train/d_mse", metrics["d_mse"], global_step + step)
        writer.add_scalar("train/aux_loss", aux_loss.item(), global_step + step)

    return total_loss / len(loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    lmbda: float,
    mse_weight: float,
    mse_scale: float,
    device: torch.device,
) -> dict:
    """Run validation in FP32 regardless of AMP setting — eval metrics should
    be deterministic across AMP/FP32 runs so val curves stay comparable."""
    model.eval()
    totals = {"loss": 0.0, "bpp": 0.0, "psnr": 0.0, "msssim": 0.0}

    for x in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        _, metrics = compute_loss(
            x, out["x_hat"], out["y_likelihoods"], out["z_likelihoods"],
            lmbda, 1.0, mse_weight, mse_scale,
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

    train_ds = AerialVideoDataset(
        args.data, patch_size=args.patch_size, split="train",
        subset_fraction=args.subset_fraction,
    )
    val_ds = AerialVideoDataset(
        args.data, patch_size=args.patch_size, split="val",
        subset_fraction=args.subset_fraction,
    )
    print(f"Train: {len(train_ds)} images  Val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = NeuralEncoderModel().to(device)

    # Separate main params (all except entropy bottleneck quantiles) from aux params.
    # EntropyBottleneck trains its quantile parameters via a dedicated aux loss.
    main_params = [p for n, p in model.named_parameters() if not n.endswith(".quantiles")]
    aux_params = [p for n, p in model.named_parameters() if n.endswith(".quantiles")]
    optimizer = Adam(main_params, lr=args.lr)
    aux_optimizer = Adam(aux_params, lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP only on CUDA. CPU autocast supports bf16, but we don't bother:
    # if you're training on CPU you've got bigger problems than precision.
    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = GradScaler(device.type, enabled=use_amp)
    print(f"Mixed precision (AMP, fp16): {'ON' if use_amp else 'OFF'}")
    print(f"Hybrid loss: (1-{args.mse_weight})·(1-MSSSIM) + {args.mse_weight}·{args.mse_scale}·MSE")

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer, aux_optimizer, scaler)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))

    best_val_loss = math.inf

    for epoch in range(start_epoch, args.epochs):
        rate_scale = rate_scale_for_epoch(epoch, args.warmup_epochs)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, aux_optimizer, scaler,
            args.lmbda, rate_scale, args.mse_weight, args.mse_scale,
            device, epoch, writer, use_amp,
        )
        # Update entropy bottleneck CDFs so val likelihoods are meaningful
        model.entropy_bottleneck.update(force=True)
        val_metrics = validate(
            model, val_loader, args.lmbda, args.mse_weight, args.mse_scale, device,
        )
        scheduler.step()

        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/psnr", val_metrics["psnr"], epoch)
        writer.add_scalar("val/bpp", val_metrics["bpp"], epoch)
        writer.add_scalar("val/msssim", val_metrics["msssim"], epoch)
        writer.add_scalar("train/rate_scale", rate_scale, epoch)

        print(
            f"[Epoch {epoch+1}/{args.epochs}] rate_scale={rate_scale:.3f}  "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"val_psnr={val_metrics['psnr']:.2f} dB  val_msssim={val_metrics['msssim']:.4f}  "
            f"val_bpp={val_metrics['bpp']:.4f}"
        )

        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "aux_optimizer": aux_optimizer.state_dict(),
            "scaler": scaler.state_dict(),
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
