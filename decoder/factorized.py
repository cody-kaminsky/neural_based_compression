"""
Factorized (non-parametric) entropy coder for the hyperprior z latent.

Uses arithmetic coding with a learned per-symbol probability table.
z symbols are INT8 in [-128, 127], stored as unsigned bytes 0..255.
The probability model is a 256-entry table of 12-bit integers summing to 4096.

Codec: Witten-Neal-Cleary bit-output arithmetic coder (carry-free, provably
correct). Bits are packed MSB-first into output bytes.
"""

from __future__ import annotations

import numpy as np

_PROB_BITS = 12
_TOTAL = 1 << _PROB_BITS  # 4096

# WNC arithmetic coder precision (32-bit interval)
_BITS = 32
_FULL = 1 << _BITS          # 2^32
_HALF = _FULL >> 1          # 2^31
_QRTR = _FULL >> 2          # 2^30


class FactorizedEntropyCoder:
    """
    Simple arithmetic coder for the z hyperprior latent.
    Uses a learned non-parametric (factorized) probability model.
    z symbols are INT8 values in [-128, 127].

    The probability model is a fixed lookup table of 256 probabilities
    (one per symbol value 0..255 for unsigned interpretation),
    stored as 12-bit integers summing to 4096.
    """

    def __init__(self, prob_table: np.ndarray):
        # prob_table: shape (256,) uint16, probabilities summing to 4096
        assert prob_table.shape == (256,), f"Expected shape (256,), got {prob_table.shape}"
        assert int(prob_table.astype(np.int64).sum()) == _TOTAL, (
            f"prob_table must sum to {_TOTAL}, got {prob_table.astype(np.int64).sum()}"
        )
        self._probs: list[int] = [int(p) for p in prob_table]
        self._cumfreq: list[int] = []
        c = 0
        for p in self._probs:
            self._cumfreq.append(c)
            c += p

        # Slot lookup: slot ∈ [0, 4096) → symbol (uint8) for O(1) decode
        self._slot_sym = np.zeros(_TOTAL, dtype=np.uint8)
        for sym, (cf, p) in enumerate(zip(self._cumfreq, self._probs)):
            self._slot_sym[cf:cf + p] = sym

    def encode(self, symbols: np.ndarray) -> bytes:
        """
        symbols: flat INT8 array
        Returns: compressed bytes (WNC arithmetic coding, bits packed MSB-first).
        """
        syms = symbols.astype(np.int8).view(np.uint8).tolist()
        probs = self._probs
        cumfreq = self._cumfreq

        low = 0
        high = _FULL - 1
        follow = 0          # pending follow (opposite-polarity) bits
        bits: list[int] = []

        def _emit(bit: int) -> None:
            nonlocal follow
            bits.append(bit)
            inv = bit ^ 1
            for _ in range(follow):
                bits.append(inv)
            follow = 0

        for s in syms:
            rng = high - low + 1
            step = rng // _TOTAL
            high = low + (cumfreq[s] + probs[s]) * step - 1
            low = low + cumfreq[s] * step

            while True:
                if high < _HALF:            # both in lower half → emit 0
                    _emit(0)
                    low <<= 1
                    high = (high << 1) | 1
                elif low >= _HALF:          # both in upper half → emit 1
                    _emit(1)
                    low = (low - _HALF) << 1
                    high = ((high - _HALF) << 1) | 1
                elif low >= _QRTR and high < _HALF + _QRTR:  # straddle centre
                    follow += 1
                    low = (low - _QRTR) << 1
                    high = ((high - _QRTR) << 1) | 1
                else:
                    break

        # Flush: resolve the pending interval
        follow += 1
        _emit(0 if low < _QRTR else 1)

        # Pack bits MSB-first into bytes
        out = bytearray()
        for i in range(0, len(bits), 8):
            chunk = bits[i:i + 8]
            byte = 0
            for j, b in enumerate(chunk):
                byte |= b << (7 - j)
            out.append(byte)
        return bytes(out)

    def decode(self, data: bytes, num_symbols: int) -> np.ndarray:
        """Returns: flat INT8 array matching original symbols."""
        probs = self._probs
        cumfreq = self._cumfreq
        slot_sym = self._slot_sym

        # Unpack bytes to bit stream
        bits: list[int] = []
        for byte in data:
            for j in range(7, -1, -1):
                bits.append((byte >> j) & 1)
        pos = 0

        def _read_bit() -> int:
            nonlocal pos
            if pos < len(bits):
                b = bits[pos]
                pos += 1
                return b
            return 0

        low = 0
        high = _FULL - 1
        value = 0
        for _ in range(_BITS):
            value = (value << 1) | _read_bit()

        result = np.zeros(num_symbols, dtype=np.uint8)
        for i in range(num_symbols):
            rng = high - low + 1
            step = rng // _TOTAL
            slot = min((value - low) // step, _TOTAL - 1)
            sym = int(slot_sym[slot])

            high = low + (cumfreq[sym] + probs[sym]) * step - 1
            low = low + cumfreq[sym] * step

            while True:
                if high < _HALF:
                    low <<= 1
                    high = (high << 1) | 1
                    value = (value << 1) | _read_bit()
                elif low >= _HALF:
                    low = (low - _HALF) << 1
                    high = ((high - _HALF) << 1) | 1
                    value = ((value - _HALF) << 1) | _read_bit()
                elif low >= _QRTR and high < _HALF + _QRTR:
                    low = (low - _QRTR) << 1
                    high = ((high - _QRTR) << 1) | 1
                    value = ((value - _QRTR) << 1) | _read_bit()
                else:
                    break

            result[i] = sym
        return result.view(np.int8)

    @staticmethod
    def default_prob_table() -> np.ndarray:
        """
        Returns a Laplace-shaped distribution centered at 0, scale=4,
        quantised to 12-bit probs summing to 4096.

        Unsigned byte b maps to signed z: b for b<128, b-256 for b>=128.
        """
        scale = 4.0

        def _laplace_cdf(v: float) -> float:
            if v >= 0:
                return 1.0 - 0.5 * np.exp(-v / scale)
            return 0.5 * np.exp(v / scale)

        raw = np.zeros(256)
        for b in range(256):
            z = float(b if b < 128 else b - 256)
            raw[b] = max(0.0, _laplace_cdf(z + 0.5) - _laplace_cdf(z - 0.5))

        # Give every bin a floor of 1; distribute remaining 3840 units
        # proportionally so the z=0 peak isn't deflated by rounding error.
        budget = _TOTAL - 256   # 3840 units for proportional distribution
        raw_sum = raw.sum()
        extras = np.round((raw / raw_sum) * budget).astype(np.int64)
        diff = budget - int(extras.sum())
        extras[int(np.argmax(extras))] += diff
        probs = (1 + extras).astype(np.int64)

        assert int(probs.sum()) == _TOTAL
        assert (probs > 0).all()
        return probs.astype(np.uint16)
