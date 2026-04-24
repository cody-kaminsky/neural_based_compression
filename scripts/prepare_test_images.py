"""
Prepare 5 golden reference test images for the neural_based_compression project.

Outputs (in reference_vectors/test_images/):
  - 5 × PNG, exactly 1280×720, Lanczos-resampled
  - 5 × YUV  4:2:0 binary, exactly 1,382,400 bytes each
  - manifest.json

Images:
  1. kodim05  – Kodak image #05 (woman with toys, rich colour)
  2. kodim15  – Kodak image #15 (doors, mid-texture)
  3. kodim23  – Kodak image #23 (lighthouse scene)
  4. aerial_low_texture  – synthetic agricultural-style flat field (numpy stand-in)
  5. aerial_high_texture – synthetic dense-urban-style structured pattern (numpy stand-in)

Run:
    python scripts/prepare_test_images.py
"""

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reference_vectors" / "test_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_W, TARGET_H = 1280, 720
YUV_BYTES = TARGET_W * TARGET_H + (TARGET_W // 2) * (TARGET_H // 2) * 2  # 1 382 400

KODAK_BASE = "http://r0k.us/graphics/kodak/kodak"
KODAK_IMAGES = [
    {
        "stem": "kodim05",
        "source_url": f"{KODAK_BASE}/kodim05.png",
        "content_description": "Kodak image 05 – woman with stuffed toys, rich saturated colours",
    },
    {
        "stem": "kodim15",
        "source_url": f"{KODAK_BASE}/kodim15.png",
        "content_description": "Kodak image 15 – painted doors and windows, moderate texture",
    },
    {
        "stem": "kodim23",
        "source_url": f"{KODAK_BASE}/kodim23.png",
        "content_description": "Kodak image 23 – lighthouse and sky, smooth gradients with fine detail",
    },
]


def download_image(url: str, timeout: int = 30) -> Image.Image:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def make_synthetic_low_texture() -> Image.Image:
    """Flat agricultural field: smooth green-brown gradient + subtle Gaussian noise."""
    rng = np.random.default_rng(seed=42)
    # Base gradient: left dark-green → right sandy-brown
    x = np.linspace(0, 1, TARGET_W, dtype=np.float32)
    y = np.linspace(0, 1, TARGET_H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    r = (0.35 + 0.25 * xx + 0.05 * yy) * 255
    g = (0.40 + 0.15 * xx - 0.05 * yy) * 255
    b = (0.15 + 0.10 * xx + 0.05 * yy) * 255

    noise = rng.normal(0, 4, (TARGET_H, TARGET_W, 3)).astype(np.float32)
    img_arr = np.stack([r, g, b], axis=-1) + noise
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(img_arr, "RGB")


def make_synthetic_high_texture() -> Image.Image:
    """Dense urban grid: regular block pattern + per-pixel variation to mimic rooftops."""
    rng = np.random.default_rng(seed=99)
    arr = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)

    block_h, block_w = 20, 20
    palette = [
        (80, 80, 90),    # asphalt
        (160, 140, 120), # concrete
        (60, 90, 60),    # vegetation patch
        (200, 180, 140), # light rooftop
        (50, 50, 60),    # dark rooftop
    ]
    for row in range(0, TARGET_H, block_h):
        for col in range(0, TARGET_W, block_w):
            colour = palette[(row // block_h + col // block_w) % len(palette)]
            arr[row:row + block_h, col:col + block_w] = colour

    noise = rng.integers(-18, 19, (TARGET_H, TARGET_W, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def rgb_to_yuv420(rgb: Image.Image) -> bytes:
    """Convert an RGB PIL image to YUV 4:2:0 planar binary (Y | U | V)."""
    assert rgb.size == (TARGET_W, TARGET_H), f"Expected {TARGET_W}×{TARGET_H}, got {rgb.size}"

    # BT.601 full-range coefficients
    arr = np.array(rgb, dtype=np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    Y = np.clip(0.299 * R + 0.587 * G + 0.114 * B, 0, 255).astype(np.uint8)
    U = np.clip(-0.16874 * R - 0.33126 * G + 0.5 * B + 128, 0, 255).astype(np.uint8)
    V = np.clip(0.5 * R - 0.41869 * G - 0.08131 * B + 128, 0, 255).astype(np.uint8)

    # 4:2:0 chroma subsampling: simple 2×2 box average
    U420 = ((U[0::2, 0::2].astype(np.uint16)
              + U[0::2, 1::2]
              + U[1::2, 0::2]
              + U[1::2, 1::2]) // 4).astype(np.uint8)
    V420 = ((V[0::2, 0::2].astype(np.uint16)
              + V[0::2, 1::2]
              + V[1::2, 0::2]
              + V[1::2, 1::2]) // 4).astype(np.uint8)

    return Y.tobytes() + U420.tobytes() + V420.tobytes()


def process_image(img: Image.Image, stem: str) -> None:
    resized = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)

    png_path = OUT_DIR / f"{stem}.png"
    resized.save(png_path, format="PNG", compress_level=6)
    print(f"  PNG  -> {png_path.relative_to(REPO_ROOT)}  ({png_path.stat().st_size:,} bytes)")

    yuv_path = OUT_DIR / f"{stem}.yuv"
    yuv_data = rgb_to_yuv420(resized)
    yuv_path.write_bytes(yuv_data)
    assert len(yuv_data) == YUV_BYTES, f"YUV size mismatch: {len(yuv_data)} != {YUV_BYTES}"
    print(f"  YUV  -> {yuv_path.relative_to(REPO_ROOT)}  ({yuv_path.stat().st_size:,} bytes)")


def main() -> None:
    manifest = []

    # --- Kodak images ---
    for entry in KODAK_IMAGES:
        stem = entry["stem"]
        print(f"\n[{stem}] Downloading from {entry['source_url']} …")
        try:
            img = download_image(entry["source_url"])
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        process_image(img, stem)
        manifest.append(
            {
                "filename": f"{stem}.png",
                "source_url": entry["source_url"],
                "content_description": entry["content_description"],
                "width": TARGET_W,
                "height": TARGET_H,
            }
        )

    # --- Aerial: low-texture (synthetic stand-in) ---
    print("\n[aerial_low_texture] Generating synthetic flat-field image …")
    low = make_synthetic_low_texture()
    process_image(low, "aerial_low_texture")
    manifest.append(
        {
            "filename": "aerial_low_texture.png",
            "source_url": "synthetic",
            "content_description": (
                "Synthetic low-texture aerial stand-in: smooth agricultural-field "
                "gradient (green-brown) with Gaussian noise σ=4. "
                "Generated by scripts/prepare_test_images.py (numpy seed=42). "
                "Replace with a real UC Merced 'agricultural' tile if desired."
            ),
            "width": TARGET_W,
            "height": TARGET_H,
        }
    )

    # --- Aerial: high-texture (synthetic stand-in) ---
    print("\n[aerial_high_texture] Generating synthetic urban-grid image …")
    high = make_synthetic_high_texture()
    process_image(high, "aerial_high_texture")
    manifest.append(
        {
            "filename": "aerial_high_texture.png",
            "source_url": "synthetic",
            "content_description": (
                "Synthetic high-texture aerial stand-in: regular urban block grid "
                "(5-colour palette, 20×20 px blocks) with per-pixel noise ±18. "
                "Generated by scripts/prepare_test_images.py (numpy seed=99). "
                "Replace with a real UC Merced 'buildings' or 'denseresidential' tile if desired."
            ),
            "width": TARGET_W,
            "height": TARGET_H,
        }
    )

    # --- manifest ---
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[manifest] -> {manifest_path.relative_to(REPO_ROOT)}")

    print(f"\nDone. {len(manifest)} images prepared in {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
