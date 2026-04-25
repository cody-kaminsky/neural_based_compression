"""Visualize what a checkpoint's reconstruction looks like on a single image.

Saves a side-by-side PNG (original | reconstruction) and prints PSNR / bpp.

Usage:
    python eval/inspect_reconstruction.py \
        --ckpt checkpoints/run-l16/last.pth \
        --image /path/to/input.jpg \
        --out reconstruction.png
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

from train.model import NeuralEncoderModel
from train.dataset_utils import rgb_to_yuv, yuv_to_rgb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pth")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--out", default="reconstruction.png", help="Output side-by-side PNG")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    model = NeuralEncoderModel().to(device)
    # strict=False: checkpoint contains entropy_bottleneck CDF buffers
    # (_offset/_quantized_cdf/_cdf_length) populated by the training script's
    # update(force=True) call. The fresh model has them at size [0]. We don't
    # need them — forward() uses the soft (noise-based) likelihood path.
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Load image, crop to multiple of 8 (matches val transform)
    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    new_w, new_h = (w // 8) * 8, (h // 8) * 8
    left, top = (w - new_w) // 2, (h - new_h) // 2
    img = TF.crop(img, top, left, new_h, new_w)
    rgb = TF.to_tensor(img).unsqueeze(0).to(device)  # [1,3,H,W] in [0,1]
    yuv = rgb_to_yuv(rgb)

    # Forward (model handles its own pad/crop to multiple of 64)
    with torch.no_grad():
        out = model(yuv)
    yuv_hat = out["x_hat"].clamp(0, 1)
    rgb_hat = yuv_to_rgb(yuv_hat)

    # Metrics
    b, _, H, W = rgb.shape
    bpp_y = (-torch.log2(out["y_likelihoods"])).sum() / (b * H * W)
    bpp_z = (-torch.log2(out["z_likelihoods"])).sum() / (b * H * W)
    bpp = (bpp_y + bpp_z).item()
    mse = torch.mean((rgb - rgb_hat) ** 2).item()
    psnr = -10.0 * torch.log10(torch.tensor(mse + 1e-10)).item()

    print(f"Image: {args.image}  ({new_w}x{new_h})")
    print(f"Checkpoint: {args.ckpt}  (epoch {ckpt.get('epoch', '?')})")
    print(f"PSNR: {psnr:.2f} dB   bpp: {bpp:.4f}   (y={bpp_y.item():.4f}, z={bpp_z.item():.4f})")

    # Save side-by-side
    side_by_side = torch.cat([rgb[0].cpu(), rgb_hat[0].cpu()], dim=2)  # along width
    out_img = TF.to_pil_image(side_by_side.clamp(0, 1))
    out_img.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
