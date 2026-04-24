"""Tests for decoder/factorized.py — factorized entropy coder."""

import numpy as np
import pytest

from decoder.factorized import FactorizedEntropyCoder

RNG = np.random.default_rng(42)


def _laplace_prob_table(scale: float = 4.0) -> np.ndarray:
    """Build a Laplace-shaped prob table for testing."""
    return FactorizedEntropyCoder.default_prob_table()


def _make_coder(scale: float = 4.0) -> FactorizedEntropyCoder:
    return FactorizedEntropyCoder(_laplace_prob_table(scale))


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------

def test_roundtrip_500_random_int8():
    """Encode 500 random INT8 symbols, decode, verify exact recovery."""
    coder = _make_coder()
    symbols = RNG.integers(-128, 128, size=500, dtype=np.int8)
    compressed = coder.encode(symbols)
    recovered = coder.decode(compressed, len(symbols))
    np.testing.assert_array_equal(recovered, symbols)


def test_roundtrip_all_zeros():
    """All-zero input: should round-trip exactly."""
    coder = _make_coder()
    symbols = np.zeros(200, dtype=np.int8)
    compressed = coder.encode(symbols)
    recovered = coder.decode(compressed, len(symbols))
    np.testing.assert_array_equal(recovered, symbols)


def test_roundtrip_single_symbol():
    """Single symbol: length-1 array."""
    coder = _make_coder()
    for val in [-128, -1, 0, 1, 127]:
        symbols = np.array([val], dtype=np.int8)
        compressed = coder.encode(symbols)
        recovered = coder.decode(compressed, 1)
        np.testing.assert_array_equal(recovered, symbols, err_msg=f"Failed for val={val}")


def test_roundtrip_laplace_distributed():
    """Laplace-distributed input: round-trip 1000 symbols."""
    coder = _make_coder()
    # Draw from Laplace, clip and cast to int8
    samples = np.random.default_rng(7).laplace(loc=0, scale=4, size=1000)
    symbols = np.clip(np.round(samples), -128, 127).astype(np.int8)
    compressed = coder.encode(symbols)
    recovered = coder.decode(compressed, len(symbols))
    np.testing.assert_array_equal(recovered, symbols)


def test_roundtrip_all_same_nonzero():
    """All symbols equal to a non-zero value."""
    coder = _make_coder()
    for val in [-128, 127, 42, -1]:
        symbols = np.full(100, val, dtype=np.int8)
        compressed = coder.encode(symbols)
        recovered = coder.decode(compressed, len(symbols))
        np.testing.assert_array_equal(recovered, symbols)


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------

def test_compression_ratio_laplace():
    """
    For Laplace(scale=4) input, compressed bytes < raw bytes (8 bits/symbol).
    Uses a moderately large sample so the ratio is stable.
    """
    coder = _make_coder()
    rng = np.random.default_rng(99)
    samples = rng.laplace(loc=0, scale=4, size=2000)
    symbols = np.clip(np.round(samples), -128, 127).astype(np.int8)

    compressed = coder.encode(symbols)
    raw_bytes = len(symbols)  # 1 byte per symbol = 8 bits/symbol
    # Expect meaningful compression; allow some overhead for flush bytes
    assert len(compressed) < raw_bytes, (
        f"Expected compressed ({len(compressed)}) < raw ({raw_bytes})"
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_bad_shape_raises():
    with pytest.raises(AssertionError):
        FactorizedEntropyCoder(np.ones(128, dtype=np.uint16))


def test_bad_sum_raises():
    bad = np.ones(256, dtype=np.uint16)  # sums to 256, not 4096
    with pytest.raises(AssertionError):
        FactorizedEntropyCoder(bad)


# ---------------------------------------------------------------------------
# default_prob_table
# ---------------------------------------------------------------------------

def test_default_prob_table_shape_and_sum():
    pt = FactorizedEntropyCoder.default_prob_table()
    assert pt.shape == (256,)
    assert pt.dtype == np.uint16
    assert int(pt.astype(np.int64).sum()) == 4096


def test_default_prob_table_all_positive():
    pt = FactorizedEntropyCoder.default_prob_table()
    assert (pt > 0).all(), "Every symbol must have non-zero probability"


def test_default_prob_table_peak_at_zero():
    """Probability of z=0 (unsigned byte 0) should be the highest."""
    pt = FactorizedEntropyCoder.default_prob_table()
    assert pt[0] == pt.max(), f"Expected peak at index 0, got max at {pt.argmax()}"
