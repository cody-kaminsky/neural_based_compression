"""BD-rate evaluation script.

Encodes images at multiple quality levels using both a neural model and
x264 (via ffmpeg), then computes Bjøntegaard Delta Rate (BD-rate) to
measure coding efficiency relative to x264.

Usage:
    python eval/bdrate.py \\
        --model checkpoints/best.pth \\
        --images /path/to/kodak \\
        --output eval/results/
"""

from __future__ import annotations

import argparse
import io
import math
import os
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.interpolate import PchipInterpolator

# Optional model dependency
try:
    from train.model import NeuralEncoderModel
    _MODEL_AVAILABLE = True
except ImportError:
    _MODEL_AVAILABLE = False

from train.dataset_utils import rgb_to_yuv, yuv_to_rgb, scan_image_files


# ---------------------------------------------------------------------------
# BD-rate computation
# ---------------------------------------------------------------------------

def bjontegaard_delta_rate(
    rates1: list[float],
    psnrs1: list[float],
    rates2: list[float],
    psnrs2: list[float],
) -> float:
    """Compute BD-rate (%) of curve 2 relative to curve 1.

    Negative value means curve 2 is more efficient (fewer bits for same PSNR).

    Uses piecewise cubic (PCHIP) interpolation over the overlapping PSNR range,
    integrates in log-rate domain.

    Args:
        rates1, psnrs1: anchor codec rate-distortion points (bpp, dB)
        rates2, psnrs2: test codec rate-distortion points (bpp, dB)
    Returns:
        BD-rate percentage.
    """
    log_rates1 = np.log(np.array(rates1, dtype=float))
    log_rates2 = np.log(np.array(rates2, dtype=float))
    psnrs1 = np.array(psnrs1, dtype=float)
    psnrs2 = np.array(psnrs2, dtype=float)

    # Overlapping PSNR range
    min_psnr = max(psnrs1.min(), psnrs2.min())
    max_psnr = min(psnrs1.max(), psnrs2.max())

    if min_psnr >= max_psnr:
        raise ValueError(
            f"No overlapping PSNR range: [{psnrs1.min():.1f},{psnrs1.max():.1f}] "
            f"vs [{psnrs2.min():.1f},{psnrs2.max():.1f}]"
        )

    # Sort by PSNR for interpolation
    order1 = np.argsort(psnrs1)
    order2 = np.argsort(psnrs2)

    interp1 = PchipInterpolator(psnrs1[order1], log_rates1[order1])
    interp2 = PchipInterpolator(psnrs2[order2], log_rates2[order2])

    grid = np.linspace(min_psnr, max_psnr, 100)
    avg_diff = np.trapz(interp2(grid) - interp1(grid), grid) / (max_psnr - min_psnr)

    return (np.exp(avg_diff) - 1.0) * 100.0


# ---------------------------------------------------------------------------
# PSNR helper
# ---------------------------------------------------------------------------

def psnr_db(orig: torch.Tensor, recon: torch.Tensor) -> float:
    mse = torch.mean((orig.float() - recon.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# x264 encode/decode via ffmpeg
# ---------------------------------------------------------------------------

X264_CRFS = [18, 23, 28, 35]


def encode_decode_x264(img_path: str, crf: int, tmp_dir: str) -> tuple[float, float]:
    """Encode with x264 at given CRF, return (bpp, psnr_db)."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    encoded_path = os.path.join(tmp_dir, f"enc_crf{crf}.mkv")
    decoded_path = os.path.join(tmp_dir, f"dec_crf{crf}.png")

    # Encode
    cmd_enc = [
        "ffmpeg", "-y", "-i", img_path,
        "-vcodec", "libx264", "-crf", str(crf),
        "-preset", "medium", "-frames:v", "1",
        encoded_path,
    ]
    result = subprocess.run(cmd_enc, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg encode failed (CRF {crf}):\n{result.stderr.decode()}"
        )

    # Decode
    cmd_dec = [
        "ffmpeg", "-y", "-i", encoded_path,
        "-frames:v", "1", decoded_path,
    ]
    result = subprocess.run(cmd_dec, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed (CRF {crf}):\n{result.stderr.decode()}"
        )

    file_bytes = os.path.getsize(encoded_path)
    bpp = (file_bytes * 8) / (w * h)

    orig_t = TF.to_tensor(img)
    recon_t = TF.to_tensor(Image.open(decoded_path).convert("RGB"))
    psnr = psnr_db(orig_t, recon_t)

    return bpp, psnr


# ---------------------------------------------------------------------------
# Neural model encode/decode
# ---------------------------------------------------------------------------

NEURAL_QUALITIES = [1, 2, 3, 4]  # model quality levels (passed as lambda indices)


def encode_decode_neural(
    img_path: str,
    model: torch.nn.Module,
    quality: int,
    device: torch.device,
) -> tuple[float, float]:
    """Encode with neural model, return (bpp, psnr_db)."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    # Pad to multiple of 64 (typical for hyperprior models)
    pad_w = (64 - w % 64) % 64
    pad_h = (64 - h % 64) % 64

    x = TF.to_tensor(img).unsqueeze(0).to(device)
    if pad_w or pad_h:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

    x_yuv = rgb_to_yuv(x)

    with torch.no_grad():
        out = model.compress(x_yuv)
        x_hat_yuv = model.decompress(out["strings"], out["shape"])["x_hat"]

    # Crop back
    x_hat_yuv = x_hat_yuv[:, :, :h, :w]

    x_rgb = yuv_to_rgb(x_yuv[:, :, :h, :w])
    x_hat_rgb = yuv_to_rgb(x_hat_yuv.clamp(0, 1))

    # Count compressed bytes
    total_bits = sum(len(s) * 8 for strings in out["strings"] for s in strings)
    bpp = total_bits / (w * h)

    psnr = psnr_db(x_rgb, x_hat_rgb)
    return bpp, psnr


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BD-rate evaluation")
    p.add_argument("--model", required=True, help="Path to model checkpoint")
    p.add_argument("--images", required=True, help="Directory of test images")
    p.add_argument("--output", default="eval/results/", help="Output directory")
    p.add_argument("--skip-neural", action="store_true",
                   help="Skip neural model (only run x264 baseline)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load neural model
    model = None
    if not args.skip_neural:
        if not _MODEL_AVAILABLE:
            print(
                "WARNING: train/model.py not found — NeuralEncoderModel unavailable. "
                "Use --skip-neural to run x264-only evaluation.",
                file=sys.stderr,
            )
            sys.exit(1)
        ckpt = torch.load(args.model, map_location=device)
        model = NeuralEncoderModel()
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        print(f"Loaded model from {args.model}")

    image_files = scan_image_files(args.images)
    if not image_files:
        print(f"No images found in {args.images!r}", file=sys.stderr)
        sys.exit(1)
    print(f"Evaluating on {len(image_files)} images")

    # Aggregate per-quality results
    x264_results: dict[int, list[tuple[float, float]]] = {c: [] for c in X264_CRFS}
    neural_results: dict[int, list[tuple[float, float]]] = {q: [] for q in NEURAL_QUALITIES}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for img_idx, img_path in enumerate(image_files):
            print(f"  [{img_idx+1}/{len(image_files)}] {os.path.basename(img_path)}")

            # x264
            for crf in X264_CRFS:
                try:
                    bpp, psnr = encode_decode_x264(img_path, crf, tmp_dir)
                    x264_results[crf].append((bpp, psnr))
                except RuntimeError as e:
                    print(f"    x264 CRF={crf} failed: {e}", file=sys.stderr)

            # Neural
            if model is not None:
                for quality in NEURAL_QUALITIES:
                    try:
                        bpp, psnr = encode_decode_neural(img_path, model, quality, device)
                        neural_results[quality].append((bpp, psnr))
                    except Exception as e:
                        print(f"    Neural quality={quality} failed: {e}", file=sys.stderr)

    # Average across images
    def avg_rd(results: dict) -> tuple[list[float], list[float]]:
        rates, psnrs = [], []
        for key in sorted(results):
            pts = results[key]
            if pts:
                rates.append(float(np.mean([p[0] for p in pts])))
                psnrs.append(float(np.mean([p[1] for p in pts])))
        return rates, psnrs

    x264_rates, x264_psnrs = avg_rd(x264_results)
    neural_rates, neural_psnrs = avg_rd(neural_results)

    # BD-rate
    bdrate_val: float | None = None
    if model is not None and len(neural_rates) >= 2 and len(x264_rates) >= 2:
        try:
            bdrate_val = bjontegaard_delta_rate(
                x264_rates, x264_psnrs, neural_rates, neural_psnrs
            )
            print(f"\nBD-rate vs x264: {bdrate_val:+.2f}%")
            print("  Negative means the neural model is more efficient.")
        except ValueError as e:
            print(f"BD-rate computation failed: {e}", file=sys.stderr)

    # Save BD-rate to text
    if bdrate_val is not None:
        with open(os.path.join(args.output, "bdrate.txt"), "w") as f:
            f.write(f"BD-rate vs x264: {bdrate_val:+.2f}%\n")
            f.write(f"Images evaluated: {len(image_files)}\n")
            f.write("\nx264 RD points (bpp, PSNR dB):\n")
            for r, p in zip(x264_rates, x264_psnrs):
                f.write(f"  {r:.4f}  {p:.2f}\n")
            if neural_rates:
                f.write("\nNeural RD points (bpp, PSNR dB):\n")
                for r, p in zip(neural_rates, neural_psnrs):
                    f.write(f"  {r:.4f}  {p:.2f}\n")

    # Plot RD curves
    fig, ax = plt.subplots(figsize=(8, 5))
    if x264_rates:
        ax.plot(x264_rates, x264_psnrs, "b-o", label="x264")
    if neural_rates:
        ax.plot(neural_rates, neural_psnrs, "r-s", label="Neural")
    ax.set_xlabel("Rate (bpp)")
    ax.set_ylabel("PSNR (dB)")
    title = "Rate-Distortion Curve"
    if bdrate_val is not None:
        title += f"\nBD-rate vs x264: {bdrate_val:+.2f}%"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(args.output, "rd_curve.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"RD curve saved to {plot_path}")


if __name__ == "__main__":
    main()
