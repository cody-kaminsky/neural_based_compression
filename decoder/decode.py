"""Ground-station decoder for the neural video compression pipeline.

Parses a bitstream written per docs/bitstream_spec.md and reconstructs the
original frame via HyperSynthesis + SynthesisNet.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .ans import RANSDecoder
from .factorized import FactorizedEntropyCoder

MAGIC = 0x9E          # "NE" sentinel: 0b10011110 = 158
EOF_MARKER = 0xDEADBEEF
STRIP_ROWS = 64
Y_CHANNELS = 128
Z_CHANNELS = 64
SPATIAL_STRIDE = 8    # SynthesisNet total downsampling (3 × stride-2 blocks)
HYPER_STRIDE = 16     # HyperSynthesis adds one more stride-2 on top of SPATIAL_STRIDE


class CorruptStreamError(ValueError):
    """Raised when the bitstream fails any validation check."""


# ── Header parsers ────────────────────────────────────────────────────────────

def _parse_frame_header(data: bytes) -> Tuple[int, int, int, int]:
    """Return (width, height, model_id, num_strips) from the 7-byte frame header."""
    if len(data) < 7:
        raise CorruptStreamError("Bitstream too short for frame header")
    magic = data[0]
    if magic != MAGIC:
        raise CorruptStreamError(
            f"Invalid magic byte: {magic:#04x} (expected {MAGIC:#04x})"
        )
    width = int.from_bytes(data[1:3], "big")
    height = int.from_bytes(data[3:5], "big")
    model_id = data[5]
    if model_id > 2:
        raise CorruptStreamError(f"Reserved model_id: {model_id}")
    num_strips = data[6]
    return width, height, model_id, num_strips


def _parse_strip_header(
    data: bytes, offset: int
) -> Tuple[int, int, int, int]:
    """Return (strip_y, z_len, y_len, new_offset) from the 7-byte strip header."""
    strip_y = int.from_bytes(data[offset: offset + 2], "big")
    z_len = int.from_bytes(data[offset + 2: offset + 4], "big")
    y_len = int.from_bytes(data[offset + 4: offset + 7], "big")
    return strip_y, z_len, y_len, offset + 7


# ── Strip assembly ────────────────────────────────────────────────────────────

def _assemble_strips(
    strip_list: list,
    overlap_rows: int = 0,
    target_height: Optional[int] = None,
) -> np.ndarray:
    """Concatenate decoded strip tensors along the height axis.

    Args:
        strip_list:    list of (3, H_i, W) float32 arrays
        overlap_rows:  rows to discard from the top of each strip except strip 0
        target_height: if set, crop the assembled frame to exactly this many rows

    Returns:
        (3, total_H, W) float32 array
    """
    cropped = []
    for i, strip in enumerate(strip_list):
        if i > 0 and overlap_rows > 0:
            strip = strip[:, overlap_rows:, :]
        cropped.append(strip)
    frame = np.concatenate(cropped, axis=1)
    if target_height is not None:
        frame = frame[:, :target_height, :]
    return frame


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models(model_dir: str):
    """Return (backend, synthesis_model, hyper_model).

    Tries ONNX Runtime first (looking for synthesis.onnx / hyper_synthesis.onnx
    in model_dir); falls back to PyTorch SynthesisNet / HyperSynthesis with
    randomly-initialised weights.
    """
    if model_dir:
        try:
            import onnxruntime as ort
            synth_path = os.path.join(model_dir, "synthesis.onnx")
            hyper_path = os.path.join(model_dir, "hyper_synthesis.onnx")
            if os.path.isfile(synth_path) and os.path.isfile(hyper_path):
                return "onnx", ort.InferenceSession(synth_path), ort.InferenceSession(hyper_path)
        except ImportError:
            pass

    # PyTorch fallback — add project root so train.modules is importable
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from train.modules.synthesis import SynthesisNet  # type: ignore
    from train.modules.hyper import HyperSynthesis    # type: ignore
    import torch

    synth = SynthesisNet()
    hyper = HyperSynthesis()
    synth.eval()
    hyper.eval()
    return "torch", synth, hyper


def _run_hyper(models, z_hat_np: np.ndarray) -> np.ndarray:
    """z_hat_np: (1, Z_CHANNELS, Hz, Wz) float32  →  sigma: (1, Y_CHANNELS, Hy, Wy)."""
    backend, _, hyper = models
    if backend == "onnx":
        name = hyper.get_inputs()[0].name
        return hyper.run(None, {name: z_hat_np})[0]
    import torch
    with torch.no_grad():
        return hyper(torch.from_numpy(z_hat_np)).numpy()


def _run_synthesis(models, y_hat_np: np.ndarray) -> np.ndarray:
    """y_hat_np: (1, Y_CHANNELS, Hy, Wy) float32  →  pixels: (1, 3, H, W) in [0,1]."""
    backend, synth, _ = models
    if backend == "onnx":
        name = synth.get_inputs()[0].name
        return synth.run(None, {name: y_hat_np})[0]
    import torch
    with torch.no_grad():
        return synth(torch.from_numpy(y_hat_np)).numpy()


# ── CDF table (module-level cache) ───────────────────────────────────────────

_CDF_CACHE: Optional[Tuple] = None


def _get_cdf_tables():
    """Return (cdf_tbl, cumfreq_tbl, boundaries) — computed once per process."""
    global _CDF_CACHE
    if _CDF_CACHE is None:
        from .cdf_table import compute_laplace_cdf_table
        cdf_tbl, cumfreq_tbl = compute_laplace_cdf_table()
        sigmas = np.geomspace(0.01, 64.0, num=256)
        log_s = np.log(sigmas)
        boundaries = (log_s[:-1] + log_s[1:]) / 2
        _CDF_CACHE = (cdf_tbl, cumfreq_tbl, boundaries)
    return _CDF_CACHE


# ── rANS probability helper ───────────────────────────────────────────────────

def _make_ans_freq_fn(
    sigma_flat: np.ndarray,
    cdf_tbl: np.ndarray,
    cumfreq_tbl: np.ndarray,
    boundaries: np.ndarray,
) -> Callable:
    """Build a stateful freq_fn for RANSDecoder driven by per-symbol sigma values.

    RANSDecoder iterates i from (num_symbols - 1) down to 0 and calls
    freq_fn(i % 64, slot) exactly once per symbol.  A call counter maps
    each invocation back to the symbol index and its sigma.
    """
    num_symbols = len(sigma_flat)
    log_sigma = np.log(np.clip(sigma_flat.astype(np.float64), 0.01, 64.0))
    scale_idx = np.searchsorted(boundaries, log_sigma).clip(0, 255).astype(np.int32)

    # Slot → symbol lookup cached per scale index
    _cache: dict = {}

    def _lookup(si: int):
        if si not in _cache:
            probs = cdf_tbl[si]
            cfreqs = cumfreq_tbl[si]
            slot_sym = np.zeros(4096, dtype=np.uint8)
            for s in range(256):
                slot_sym[cfreqs[s]: cfreqs[s] + probs[s]] = s
            _cache[si] = (slot_sym, probs, cfreqs)
        return _cache[si]

    counter = [num_symbols - 1]

    def freq_fn(stream_idx: int, slot: int) -> Tuple[int, int, int]:
        i = counter[0]
        counter[0] -= 1
        si = int(scale_idx[i])
        slot_sym, probs, cfreqs = _lookup(si)
        sym = int(slot_sym[min(slot, 4095)])
        return sym, int(probs[sym]), int(cfreqs[sym])

    return freq_fn


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_png(frame: np.ndarray, path: str) -> None:
    """frame: (3, H, W) float32 in [0, 1]."""
    from PIL import Image
    arr = (frame * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    Image.fromarray(arr, "RGB").save(path)


def _save_yuv420(frame: np.ndarray, path: str) -> None:
    """frame: (3, H, W) float32 in [0, 1].  Writes planar YCbCr 4:2:0."""
    a = (frame * 255).clip(0, 255).astype(np.float32)
    R, G, B = a[0], a[1], a[2]
    Y  = (0.299 * R + 0.587 * G + 0.114 * B).clip(0, 255).astype(np.uint8)
    Cb = (128 - 0.168736 * R - 0.331264 * G + 0.5 * B).clip(0, 255).astype(np.uint8)
    Cr = (128 + 0.5 * R - 0.418688 * G - 0.081312 * B).clip(0, 255).astype(np.uint8)
    with open(path, "wb") as fh:
        fh.write(Y.tobytes())
        fh.write(Cb[::2, ::2].tobytes())
        fh.write(Cr[::2, ::2].tobytes())


# ── Main decode entry point ───────────────────────────────────────────────────

def decode_frame(
    bitstream_path: str,
    model_dir: str,
    output_path: str,
    output_format: str = "png",
    overlap_rows: int = 0,
    _y_freq_fn: Optional[Callable] = None,
) -> Dict:
    """Parse bitstream per docs/bitstream_spec.md and reconstruct the frame.

    For each strip:
      - Read z_bitstream → FactorizedEntropyCoder.decode() → z_hat
      - z_hat → HyperSynthesis → sigma scale map
      - Read y_bitstream → RANSDecoder.decode() → y_hat
      - y_hat → SynthesisNet → strip pixels
      - Discard overlap rows (top rows of each strip except strip 0)

    Assemble strips → full frame.  Save as PNG or raw YUV 4:2:0.

    Args:
        bitstream_path: path to .bin compressed frame
        model_dir:      directory with synthesis.onnx / hyper_synthesis.onnx
                        (or any path; PyTorch fallback is used when absent)
        output_path:    destination image file
        output_format:  "png" or "yuv420"
        overlap_rows:   rows to drop from the top of strips 1..N-1
        _y_freq_fn:     optional freq_fn override for rANS y decoding (testing only)

    Returns:
        dict with keys: width, height, model_id, num_strips, output_path
    """
    with open(bitstream_path, "rb") as fh:
        data = fh.read()

    width, height, model_id, num_strips = _parse_frame_header(data)
    offset = 7

    fact_coder = FactorizedEntropyCoder(FactorizedEntropyCoder.default_prob_table())
    cdf_tbl, cumfreq_tbl, boundaries = _get_cdf_tables()
    models = _load_models(model_dir)
    ans_dec = RANSDecoder()

    strip_pixels: list = []

    for strip_idx in range(num_strips):
        strip_y, z_len, y_len, offset = _parse_strip_header(data, offset)

        strip_h = min(STRIP_ROWS, height - strip_idx * STRIP_ROWS)
        y_h = strip_h // SPATIAL_STRIDE
        y_w = width // SPATIAL_STRIDE
        z_h = strip_h // HYPER_STRIDE
        z_w = width // HYPER_STRIDE
        num_z = Z_CHANNELS * z_h * z_w
        num_y = Y_CHANNELS * y_h * y_w

        # z: arithmetic decode → float tensor → HyperSynthesis → sigma
        z_data = data[offset: offset + z_len]
        offset += z_len
        z_int8 = fact_coder.decode(z_data, num_z)
        z_hat = z_int8.reshape(1, Z_CHANNELS, z_h, z_w).astype(np.float32)
        sigma = _run_hyper(models, z_hat)           # (1, Y_CHANNELS, y_h, y_w)
        sigma_flat = sigma.flatten()                # (num_y,)

        # y: rANS decode → float tensor → SynthesisNet → pixels
        y_data = data[offset: offset + y_len]
        offset += y_len

        freq_fn = (
            _y_freq_fn
            if _y_freq_fn is not None
            else _make_ans_freq_fn(sigma_flat, cdf_tbl, cumfreq_tbl, boundaries)
        )

        y_symbols = ans_dec.decode(y_data, num_y, freq_fn)
        y_hat = (
            np.array(y_symbols, dtype=np.uint8)
            .view(np.int8)
            .reshape(1, Y_CHANNELS, y_h, y_w)
            .astype(np.float32)
        )

        pixels = _run_synthesis(models, y_hat)      # (1, 3, strip_h, width)
        strip_pixels.append(pixels[0])              # (3, strip_h, width)

    eof_val = int.from_bytes(data[offset: offset + 4], "big")
    if eof_val != EOF_MARKER:
        raise CorruptStreamError(
            f"EOF marker mismatch: {eof_val:#010x} (expected {EOF_MARKER:#010x})"
        )

    frame = _assemble_strips(strip_pixels, overlap_rows=overlap_rows, target_height=height)

    fmt = output_format.lower()
    if fmt == "png":
        _save_png(frame, output_path)
    elif fmt in ("yuv420", "yuv"):
        _save_yuv420(frame, output_path)
    else:
        raise ValueError(f"Unknown output_format: {output_format!r}")

    return {
        "width": width,
        "height": height,
        "model_id": model_id,
        "num_strips": num_strips,
        "output_path": output_path,
    }
