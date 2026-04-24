"""Structural tests for decoder/decode.py."""
from __future__ import annotations

import os

import numpy as np
import pytest

from decoder.decode import (
    MAGIC,
    EOF_MARKER,
    CorruptStreamError,
    _parse_frame_header,
    _assemble_strips,
    decode_frame,
)


# ── Test 1: frame header parsing ─────────────────────────────────────────────

class TestBitstreamHeaderParse:
    def _make_header(self, width=320, height=240, model_id=1, num_strips=4):
        buf = bytearray(7)
        buf[0] = MAGIC
        buf[1:3] = width.to_bytes(2, "big")
        buf[3:5] = height.to_bytes(2, "big")
        buf[5] = model_id
        buf[6] = num_strips
        return bytes(buf)

    def test_bitstream_header_parse(self):
        header = self._make_header(width=320, height=240, model_id=1, num_strips=4)
        width, height, model_id, num_strips = _parse_frame_header(header)
        assert width == 320
        assert height == 240
        assert model_id == 1
        assert num_strips == 4

    def test_header_various_sizes(self):
        for w, h in [(128, 64), (1920, 1080), (1, 1)]:
            ns = max(1, (h + 63) // 64)
            hdr = self._make_header(width=w, height=h, model_id=0, num_strips=ns)
            pw, ph, mid, pns = _parse_frame_header(hdr)
            assert pw == w and ph == h and mid == 0 and pns == ns

    def test_bad_magic_raises(self):
        header = bytearray(self._make_header())
        header[0] = 0x00
        with pytest.raises(CorruptStreamError, match="magic"):
            _parse_frame_header(bytes(header))

    def test_reserved_model_id_raises(self):
        header = bytearray(self._make_header())
        header[5] = 3
        with pytest.raises(CorruptStreamError, match="model_id"):
            _parse_frame_header(bytes(header))

    def test_too_short_raises(self):
        with pytest.raises(CorruptStreamError):
            _parse_frame_header(b"\x9e\x00")


# ── Test 2: strip reassembly ──────────────────────────────────────────────────

class TestStripReassembly:
    def test_strip_reassembly(self):
        overlap = 4
        W, C = 128, 3
        # Three strips filled with known sentinel values (0.0, 1.0, 2.0)
        strips = [
            np.full((C, 64, W), float(i), dtype=np.float32)
            for i in range(3)
        ]
        assembled = _assemble_strips(strips, overlap_rows=overlap)

        # strip 0: all 64 rows  |  strips 1 & 2: 64 - overlap = 60 rows each
        expected_h = 64 + 60 + 60
        assert assembled.shape == (C, expected_h, W), (
            f"Expected shape ({C}, {expected_h}, {W}), got {assembled.shape}"
        )
        # Value checks: first 64 rows from strip 0
        np.testing.assert_array_equal(assembled[:, :64, :], 0.0)
        # Next 60 rows from strip 1 (rows 4..63, value=1.0)
        np.testing.assert_array_equal(assembled[:, 64:124, :], 1.0)
        # Last 60 rows from strip 2 (rows 4..63, value=2.0)
        np.testing.assert_array_equal(assembled[:, 124:, :], 2.0)

    def test_no_overlap_concatenates_fully(self):
        strips = [np.ones((3, 64, 64), dtype=np.float32) * i for i in range(4)]
        assembled = _assemble_strips(strips, overlap_rows=0)
        assert assembled.shape == (3, 256, 64)

    def test_target_height_crops(self):
        strips = [np.ones((3, 64, 64), dtype=np.float32)]
        assembled = _assemble_strips(strips, overlap_rows=0, target_height=48)
        assert assembled.shape == (3, 48, 64)


# ── Test 3: end-to-end structural roundtrip ───────────────────────────────────

def _build_freq_fn(probs: list, cumfreqs: list):
    """Stateless freq_fn using a fixed 256-entry probability table."""
    slot_sym = np.zeros(4096, dtype=np.uint8)
    for sym, (cf, p) in enumerate(zip(cumfreqs, probs)):
        if p > 0:
            slot_sym[cf: cf + p] = sym

    def freq_fn(stream_idx: int, slot: int):
        sym = int(slot_sym[min(slot, 4095)])
        return sym, probs[sym], cumfreqs[sym]

    return freq_fn


def _build_fake_bitstream(width: int, height: int) -> tuple:
    """Return (bitstream_bytes, y_freq_fn) for a single-strip frame."""
    from decoder.ans import RANSEncoder
    from decoder.factorized import FactorizedEntropyCoder

    strip_h = height          # single strip → strip_h == height
    y_h = strip_h // 8
    y_w = width // 8
    z_h = strip_h // 16
    z_w = width // 16
    num_z = 64 * z_h * z_w
    num_y = 128 * y_h * y_w

    # ── z: encode with default Laplace table, all-zero symbols ───────────────
    fact = FactorizedEntropyCoder(FactorizedEntropyCoder.default_prob_table())
    z_symbols = np.zeros(num_z, dtype=np.int8)
    z_data = fact.encode(z_symbols)

    # ── y: encode with default Laplace table, all-zero symbols ───────────────
    prob_arr = FactorizedEntropyCoder.default_prob_table()
    probs = [int(p) for p in prob_arr]
    cumfreqs = []
    c = 0
    for p in probs:
        cumfreqs.append(c)
        c += p

    y_symbols_enc = [0] * num_y
    enc = RANSEncoder()
    y_data = enc.encode(y_symbols_enc, probs, cumfreqs)

    # ── pack bitstream ────────────────────────────────────────────────────────
    z_len = len(z_data)
    y_len = len(y_data)

    buf = bytearray()
    buf.append(MAGIC)
    buf.extend(width.to_bytes(2, "big"))
    buf.extend(height.to_bytes(2, "big"))
    buf.append(1)    # model_id = medium
    buf.append(1)    # num_strips = 1
    # Strip header
    buf.extend((0).to_bytes(2, "big"))          # strip_y
    buf.extend(z_len.to_bytes(2, "big"))        # z_len (16-bit is enough here)
    buf.extend(y_len.to_bytes(3, "big"))        # y_len (24-bit)
    buf.extend(z_data)
    buf.extend(y_data)
    buf.extend(EOF_MARKER.to_bytes(4, "big"))

    freq_fn = _build_freq_fn(probs, cumfreqs)
    return bytes(buf), freq_fn


def test_roundtrip_structural(tmp_path):
    """Build a minimal 128×64 fake bitstream and verify decode_frame completes."""
    W, H = 128, 64
    bitstream, y_freq_fn = _build_fake_bitstream(W, H)

    bs_path = tmp_path / "frame.bin"
    bs_path.write_bytes(bitstream)
    out_path = str(tmp_path / "out.png")

    result = decode_frame(
        str(bs_path),
        str(tmp_path),        # model_dir — no ONNX files → PyTorch fallback
        out_path,
        output_format="png",
        _y_freq_fn=y_freq_fn,
    )

    assert os.path.exists(out_path), "Output PNG was not created"
    assert result["width"] == W
    assert result["height"] == H
    assert result["num_strips"] == 1
    assert result["output_path"] == out_path
