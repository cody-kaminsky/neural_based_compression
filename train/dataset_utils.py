"""Utility functions for image format conversion and file scanning."""

import os
import torch

# BT.601 RGB -> YCbCr matrix
_RGB_TO_YUV = torch.tensor([
    [ 0.299,     0.587,     0.114   ],
    [-0.168736, -0.331264,  0.5     ],
    [ 0.5,      -0.418688, -0.081312],
], dtype=torch.float32)

# Inverse (YCbCr -> RGB)
_YUV_TO_RGB = torch.linalg.inv(_RGB_TO_YUV)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def rgb_to_yuv(img: torch.Tensor) -> torch.Tensor:
    """BT.601 RGB -> YCbCr, all channels in [0, 1].

    Args:
        img: (..., 3, H, W) float32 tensor in [0, 1], channels in RGB order.
    Returns:
        (..., 3, H, W) float32 tensor (Y, Cb, Cr) each normalised to [0, 1].
    """
    mat = _RGB_TO_YUV.to(img.device)
    # Reshape to (..., H*W, 3) for matmul then reshape back
    *leading, c, h, w = img.shape
    pixels = img.reshape(*leading, c, h * w).transpose(-2, -1)  # (..., H*W, 3)
    yuv_pixels = pixels @ mat.T  # (..., H*W, 3)
    yuv = yuv_pixels.transpose(-2, -1).reshape(*leading, c, h, w)

    # Y is already in [0,1]; Cb and Cr are in [-0.5, 0.5] — shift to [0, 1]
    yuv[..., 1:, :, :] += 0.5
    return yuv.clamp(0.0, 1.0)


def yuv_to_rgb(yuv: torch.Tensor) -> torch.Tensor:
    """BT.601 YCbCr -> RGB.

    Args:
        yuv: (..., 3, H, W) float32 tensor in [0, 1] (Y, Cb, Cr).
    Returns:
        (..., 3, H, W) float32 tensor in [0, 1] (R, G, B).
    """
    mat = _YUV_TO_RGB.to(yuv.device)
    yuv = yuv.clone()
    yuv[..., 1:, :, :] -= 0.5

    *leading, c, h, w = yuv.shape
    pixels = yuv.reshape(*leading, c, h * w).transpose(-2, -1)
    rgb_pixels = pixels @ mat.T
    rgb = rgb_pixels.transpose(-2, -1).reshape(*leading, c, h, w)
    return rgb.clamp(0.0, 1.0)


def scan_image_files(root: str) -> list[str]:
    """Return sorted list of all .jpg/.jpeg/.png paths under root."""
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in _IMAGE_EXTS:
                paths.append(os.path.join(dirpath, fname))
    return sorted(paths)
