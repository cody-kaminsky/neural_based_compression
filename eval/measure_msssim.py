"""Measure averaged PSNR / MS-SSIM / bpp for a checkpoint over a directory of images.

Usage:
    python -m eval.measure_msssim \\
        --ckpt checkpoints/run-l16/last.pth \\
        --images /path/to/kodak
"""

import argparse
import math
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pytorch_msssim import ms_ssim

from train.model import NeuralEncoderModel
from train.dataset_utils import rgb_to_yuv, yuv_to_rgb, scan_image_files


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", required=True, help="Directory of test images")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N images (deterministic, by filename order)")
    args = p.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = NeuralEncoderModel().to(device)
    state = {k: v for k, v in ckpt["model"].items()
             if not k.startswith("entropy_bottleneck._")}
    model.load_state_dict(state, strict=False)
    model.eval()

    files = scan_image_files(args.images)
    if not files:
        raise SystemExit(f"No images found in {args.images!r}")
    if args.limit is not None:
        files = files[: args.limit]
    print(f"Evaluating on {len(files)} images")

    psnrs, msssims, bpps = [], [], []
    for idx, path in enumerate(files):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        new_w, new_h = (w // 8) * 8, (h // 8) * 8
        left, top = (w - new_w) // 2, (h - new_h) // 2
        img = TF.crop(img, top, left, new_h, new_w)
        rgb = TF.to_tensor(img).unsqueeze(0).to(device)
        yuv = rgb_to_yuv(rgb)

        with torch.no_grad():
            out = model(yuv)
        rgb_hat = yuv_to_rgb(out["x_hat"].clamp(0, 1))

        b, _, H, W = rgb.shape
        bpp_y = (-torch.log2(out["y_likelihoods"])).sum() / (b * H * W)
        bpp_z = (-torch.log2(out["z_likelihoods"])).sum() / (b * H * W)
        bpp = (bpp_y + bpp_z).item()

        mse = torch.mean((rgb - rgb_hat) ** 2).item()
        psnr = -10.0 * math.log10(mse + 1e-10)

        # ms_ssim needs window 7 by default and at least ~96 px on the smaller side
        win_size = 7 if min(H, W) > 96 else 3
        msssim = ms_ssim(rgb, rgb_hat, data_range=1.0, size_average=True, win_size=win_size).item()

        psnrs.append(psnr)
        msssims.append(msssim)
        bpps.append(bpp)

        print(f"  [{idx+1}/{len(files)}] {path.split('/')[-1]:30s}  "
              f"bpp={bpp:.3f}  PSNR={psnr:.2f}  MS-SSIM={msssim:.4f}")

    n = len(files)
    print()
    print(f"Average over {n} images:")
    print(f"  bpp     : {sum(bpps)/n:.4f}")
    print(f"  PSNR    : {sum(psnrs)/n:.2f} dB")
    print(f"  MS-SSIM : {sum(msssims)/n:.4f}")
    print(f"  MS-SSIM (dB scale, -10 log10(1-x)): {-10.0 * math.log10(1.0 - sum(msssims)/n + 1e-10):.2f}")


if __name__ == "__main__":
    main()
