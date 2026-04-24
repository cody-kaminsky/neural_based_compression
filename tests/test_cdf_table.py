"""
Tests for decoder/cdf_table.py — Laplace CDF probability table generator.

Spot-check reference values (computed analytically):
  - table[0][0]   = 4096  (sigma=0.01, k=0:  nearly all mass at 0)
  - table[255][63]  = 11    (sigma=64.0, k=63:  tail of broad distribution)
  - table[255][127] = 3     (sigma=64.0, k=127: far tail of broad distribution)

Derivation for spot checks (sigma=64.0, k>=1):
  P(k|sigma) = 0.5 * exp(-(k-0.5)*sqrt(2)/sigma) * (1 - exp(-sqrt(2)/sigma))
  k=63:  P = 0.5 * exp(-62.5*sqrt(2)/64) * 0.02185 ≈ 0.002746 → round(11.24) = 11
  k=127: P = 0.5 * exp(-126.5*sqrt(2)/64) * 0.02185 ≈ 0.000668 → round(2.73)  = 3
"""

import math
import pathlib
import tempfile

import numpy as np
import pytest

from decoder.cdf_table import (
    _laplace_cdf,
    compute_laplace_cdf_table,
    export_to_mem_file,
    export_to_npy,
)

PROB_BITS = 12
TOTAL = 1 << PROB_BITS  # 4096


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cdf_table():
    return compute_laplace_cdf_table()


@pytest.fixture(scope="module")
def table(cdf_table):
    return cdf_table[0]


@pytest.fixture(scope="module")
def cumfreq(cdf_table):
    return cdf_table[1]


# ---------------------------------------------------------------------------
# Row-sum invariant: every row must sum to exactly 4096
# ---------------------------------------------------------------------------

def test_all_rows_sum_to_total(table):
    row_sums = table.astype(np.int64).sum(axis=1)
    bad = np.where(row_sums != TOTAL)[0]
    assert len(bad) == 0, (
        f"{len(bad)} rows do not sum to {TOTAL}: "
        f"indices={bad[:5]}, sums={row_sums[bad[:5]]}"
    )


def test_no_negative_probabilities(table):
    # uint16 can't be negative, but the cast from int64 might wrap — check raw
    assert table.min() >= 0


# ---------------------------------------------------------------------------
# Symmetry: verify P(-k | sigma) = P(k | sigma) for the Laplace CDF formula
# ---------------------------------------------------------------------------

def test_laplace_cdf_symmetry():
    """CDF(x) + CDF(-x) == 1 for all x != 0 (Laplace is symmetric about 0)."""
    rng = np.random.default_rng(42)
    xs = rng.uniform(0.01, 200.0, size=500)
    sigmas = rng.uniform(0.1, 32.0, size=500)
    for x, sigma in zip(xs, sigmas):
        cdf_pos = _laplace_cdf(x, sigma)
        cdf_neg = _laplace_cdf(-x, sigma)
        assert abs(cdf_pos + cdf_neg - 1.0) < 1e-12, (
            f"Symmetry violated at x={x}, sigma={sigma}: "
            f"CDF({x})={cdf_pos}, CDF({-x})={cdf_neg}"
        )


def test_laplace_cdf_probability_symmetry():
    """P(k | sigma) = P(-k | sigma) for integer k, verified via CDF differences."""
    sigma = 2.5
    for k in [1, 5, 10, 50]:
        p_pos = _laplace_cdf(k + 0.5, sigma) - _laplace_cdf(k - 0.5, sigma)
        p_neg = _laplace_cdf(-k + 0.5, sigma) - _laplace_cdf(-k - 0.5, sigma)
        assert abs(p_pos - p_neg) < 1e-14, (
            f"P({k})={p_pos} != P({-k})={p_neg} for sigma={sigma}"
        )


# ---------------------------------------------------------------------------
# Spot-checks: 3 specific (sigma_index, k) pairs vs manually computed values
# ---------------------------------------------------------------------------

def test_spot_check_small_sigma_k0(table):
    """sigma=0.01: virtually all probability mass is at k=0 → table[0][0] = 4096."""
    # P(0|0.01) = 1 - exp(-sqrt(2)*0.5/0.01) ≈ 1.0 exactly to float precision
    assert table[0, 0] == 4096, f"Expected 4096, got {table[0, 0]}"


def test_spot_check_large_sigma_k63(table):
    """sigma=64.0, k=63: analytically P ≈ 0.002746 → round(11.24) = 11."""
    # This bin is not the largest (k=0 is), so it is unaffected by the sum adjustment.
    assert table[255, 63] == 11, f"Expected 11, got {table[255, 63]}"


def test_spot_check_large_sigma_k127(table):
    """sigma=64.0, k=127: analytically P ≈ 0.000668 → round(2.73) = 3."""
    assert table[255, 127] == 3, f"Expected 3, got {table[255, 127]}"


# ---------------------------------------------------------------------------
# Cumulative frequency correctness
# ---------------------------------------------------------------------------

def test_cumfreq_is_exclusive_prefix_sum(table, cumfreq):
    expected = np.zeros_like(cumfreq)
    expected[:, 1:] = np.cumsum(table[:, :-1], axis=1)
    np.testing.assert_array_equal(
        cumfreq, expected, err_msg="cumfreq is not the exclusive prefix sum of table"
    )


def test_cumfreq_first_column_is_zero(cumfreq):
    assert (cumfreq[:, 0] == 0).all(), "cumfreq[:, 0] must be all zeros"


def test_cumfreq_last_value_less_than_total(table, cumfreq):
    last = cumfreq[:, -1].astype(np.int64) + table[:, -1].astype(np.int64)
    assert (last == TOTAL).all(), "Last cumfreq + last prob must equal total"


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

def test_table_shape_and_dtype(table):
    assert table.shape == (256, 256)
    assert table.dtype == np.uint16


def test_cumfreq_shape_and_dtype(cumfreq):
    assert cumfreq.shape == (256, 256)
    assert cumfreq.dtype == np.uint32


# ---------------------------------------------------------------------------
# .mem file: round-trip verification
# ---------------------------------------------------------------------------

def test_mem_file_line_count(table):
    with tempfile.NamedTemporaryFile(suffix=".mem", delete=False, mode="w") as f:
        mem_path = f.name
    export_to_mem_file(table, mem_path)
    with open(mem_path) as f:
        lines = f.readlines()
    assert len(lines) == 256 * 128, f"Expected 32768 lines, got {len(lines)}"


def test_mem_file_round_trip(table):
    """Values packed into the .mem file unpack back to the original table."""
    with tempfile.NamedTemporaryFile(suffix=".mem", delete=False, mode="w") as f:
        mem_path = f.name
    export_to_mem_file(table, mem_path)

    recovered = np.zeros((256, 256), dtype=np.uint16)
    with open(mem_path) as f:
        for row in range(256):
            for j in range(128):
                word = int(f.readline().strip(), 16)
                recovered[row, 2 * j] = (word >> 16) & 0xFFFF
                recovered[row, 2 * j + 1] = word & 0xFFFF

    np.testing.assert_array_equal(
        recovered, table, err_msg=".mem round-trip mismatch"
    )


def test_mem_file_hex_format(table):
    """Each line must be exactly 8 lowercase hex characters."""
    with tempfile.NamedTemporaryFile(suffix=".mem", delete=False, mode="w") as f:
        mem_path = f.name
    export_to_mem_file(table, mem_path)
    with open(mem_path) as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            assert len(stripped) == 8, f"Line {i}: expected 8 chars, got '{stripped}'"
            assert stripped == stripped.lower(), f"Line {i}: not lowercase hex"


# ---------------------------------------------------------------------------
# .npy export / import
# ---------------------------------------------------------------------------

def test_npy_round_trip(table, cumfreq):
    with tempfile.TemporaryDirectory() as tmpdir:
        tp = pathlib.Path(tmpdir) / "cdf_table.npy"
        cp = pathlib.Path(tmpdir) / "cumfreq_table.npy"
        export_to_npy(table, cumfreq, tp, cp)
        loaded_table = np.load(tp)
        loaded_cumfreq = np.load(cp)
    np.testing.assert_array_equal(loaded_table, table)
    np.testing.assert_array_equal(loaded_cumfreq, cumfreq)


# ---------------------------------------------------------------------------
# Monotonicity: for fixed k, probabilities should be roughly increasing in sigma
# (broader distribution → more mass spread to non-zero bins, so P(0) decreases).
# ---------------------------------------------------------------------------

def test_p0_decreases_with_sigma(table):
    """P(k=0) should decrease as sigma increases (distribution widens)."""
    p0 = table[:, 0]
    # Due to rounding there can be ties, but overall trend must hold:
    # compare first vs last 10 rows
    assert p0[:10].mean() > p0[-10:].mean(), (
        "Expected P(0) to decrease as sigma grows"
    )
