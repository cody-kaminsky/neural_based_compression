"""
Laplace CDF probability table generator for arithmetic coding.

Generates a (256, 256) table of 12-bit quantized probabilities where:
  - Rows index log-spaced scale values sigma in [0.01, 64.0]
  - Columns index symbol values k in [0, 255]
  - P(k | sigma) = CDF(k+0.5 | sigma) - CDF(k-0.5 | sigma)

The Laplace CDF used is:
  CDF(x | sigma) = 0.5 * (1 + sign(x) * (1 - exp(-|x| * sqrt(2) / sigma)))

By symmetry P(-k | sigma) = P(k | sigma), so this table serves both
positive and negative symbols.
"""

import numpy as np


def _laplace_cdf(x, sigma):
    """Laplace CDF with mean=0 and std sigma (scale = sigma/sqrt(2))."""
    sqrt2 = np.sqrt(2.0)
    s = np.sign(x)
    return 0.5 * (1.0 + s * (1.0 - np.exp(-np.abs(x) * sqrt2 / sigma)))


def compute_laplace_cdf_table(num_scales=256, num_symbols=256, prob_bits=12):
    """
    Generate a Laplace CDF probability table for arithmetic coding.

    Parameters
    ----------
    num_scales : int
        Number of sigma values (rows). Default 256.
    num_symbols : int
        Number of symbol bins (columns), covering k=0..num_symbols-1. Default 256.
    prob_bits : int
        Probability precision in bits (total = 2**prob_bits). Default 12.

    Returns
    -------
    table : np.ndarray, shape (num_scales, num_symbols), dtype uint16
        12-bit quantized probabilities. Each row sums to exactly 2**prob_bits.
    cumfreq : np.ndarray, shape (num_scales, num_symbols), dtype uint32
        Exclusive cumulative frequency per row (cumfreq[i, k] = sum(table[i, :k])).
    """
    total = 1 << prob_bits  # 4096

    # 256 log-spaced sigma values from 0.01 to 64.0
    sigmas = np.geomspace(0.01, 64.0, num=num_scales)

    table = np.zeros((num_scales, num_symbols), dtype=np.uint16)

    k = np.arange(num_symbols, dtype=np.float64)

    for i, sigma in enumerate(sigmas):
        # P(k | sigma) = CDF(k + 0.5) - CDF(k - 0.5)
        p = _laplace_cdf(k + 0.5, sigma) - _laplace_cdf(k - 0.5, sigma)

        p_int = np.round(p * total).astype(np.int64)

        # Absorb rounding error into the largest bin so the row sums to exactly 4096
        diff = total - int(p_int.sum())
        if diff != 0:
            p_int[int(np.argmax(p_int))] += diff

        table[i] = p_int.astype(np.uint16)

    # Exclusive prefix sum (cumulative frequencies)
    cumfreq = np.zeros((num_scales, num_symbols), dtype=np.uint32)
    cumfreq[:, 1:] = np.cumsum(table[:, :-1], axis=1)

    return table, cumfreq


def export_to_mem_file(table, output_path):
    """
    Export the CDF table to Xilinx BRAM .mem format.

    Two consecutive uint16 values are packed into one 32-bit word:
      word = (table[row, 2*j] << 16) | table[row, 2*j+1]

    Writes num_scales * (num_symbols // 2) lines, one 8-hex-digit word per line.
    For the default 256x256 table this produces 32768 lines.
    """
    num_scales, num_symbols = table.shape
    if num_symbols % 2 != 0:
        raise ValueError("num_symbols must be even for 32-bit packing")

    with open(output_path, "w") as f:
        for row in range(num_scales):
            for j in range(num_symbols // 2):
                hi = int(table[row, 2 * j])
                lo = int(table[row, 2 * j + 1])
                f.write(f"{(hi << 16) | lo:08x}\n")


def export_to_npy(table, cumfreq, table_path, cumfreq_path):
    """Save the probability table and cumulative frequency table as .npy files."""
    np.save(table_path, table)
    np.save(cumfreq_path, cumfreq)


if __name__ == "__main__":
    import pathlib

    out_dir = pathlib.Path(__file__).parent.parent / "reference_vectors"
    out_dir.mkdir(exist_ok=True)

    print("Computing Laplace CDF table (256 scales × 256 symbols, 12-bit) ...")
    table, cumfreq = compute_laplace_cdf_table()

    export_to_npy(
        table,
        cumfreq,
        out_dir / "cdf_table.npy",
        out_dir / "cumfreq_table.npy",
    )
    export_to_mem_file(table, out_dir / "cdf_table.mem")

    print(f"Saved cdf_table.npy       shape={table.shape}   dtype={table.dtype}")
    print(f"Saved cumfreq_table.npy   shape={cumfreq.shape} dtype={cumfreq.dtype}")
    print(f"Saved cdf_table.mem       ({256 * 128} lines)")
