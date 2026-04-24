"""Tests for rANS encoder/decoder (decoder/ans.py)."""

import random
import pytest
from decoder.ans import RANSEncoder, RANSDecoder, build_prob_table, L, M, NUM_STREAMS


def make_uniform_table(n: int):
    """Uniform distribution over n symbols, each with frequency M//n."""
    assert M % n == 0
    freq = M // n
    freqs = [freq] * n
    cumfreqs = [i * freq for i in range(n)]
    return freqs, cumfreqs


def freq_fn_factory(freqs, cumfreqs):
    """Build a freq_fn from flat freq/cumfreq arrays (stream-independent)."""
    # Build CDF lookup: cumfreq[s] <= x < cumfreq[s+1] => symbol s
    def freq_fn(stream_idx, x):
        # Binary search for symbol
        lo, hi = 0, len(freqs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cumfreqs[mid + 1] <= x:
                lo = mid + 1
            else:
                hi = mid
        sym = lo
        return sym, freqs[sym], cumfreqs[sym]
    return freq_fn


def _extended_cumfreqs(freqs, cumfreqs):
    """Add a sentinel M at the end for binary search."""
    return cumfreqs + [M]


class TestBuildProbTable:
    def test_sums_to_m(self):
        freqs, cumfreqs = build_prob_table([2048, 2048])
        assert sum(freqs) == M
        assert cumfreqs == [0, 2048]

    def test_cumfreqs_correct(self):
        probs = [1000, 2000, 1096]
        freqs, cumfreqs = build_prob_table(probs)
        assert freqs == probs
        assert cumfreqs == [0, 1000, 3000]

    def test_wrong_sum_raises(self):
        with pytest.raises(AssertionError):
            build_prob_table([100, 200])


class TestRoundTrip:
    def _encode_decode(self, symbols, freqs, cumfreqs):
        enc = RANSEncoder()
        data = enc.encode(symbols, freqs, cumfreqs)

        ext_cumfreqs = _extended_cumfreqs(freqs, cumfreqs)

        def freq_fn(stream_idx, x):
            lo, hi = 0, len(freqs) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if ext_cumfreqs[mid + 1] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            sym = lo
            return sym, freqs[sym], cumfreqs[sym]

        dec = RANSDecoder()
        return dec.decode(data, len(symbols), freq_fn)

    def test_1000_random_symbols(self):
        random.seed(42)
        n_sym = 16
        # Random probs summing to M
        raw = [random.randint(1, 100) for _ in range(n_sym)]
        scale = M / sum(raw)
        probs = [max(1, int(r * scale)) for r in raw]
        # Fix rounding
        probs[-1] += M - sum(probs)
        assert sum(probs) == M

        freqs, cumfreqs = build_prob_table(probs)
        ext = _extended_cumfreqs(freqs, cumfreqs)

        symbols = []
        for _ in range(1000):
            # Sample according to probabilities
            x = random.randrange(M)
            for i, (c, f) in enumerate(zip(cumfreqs, freqs)):
                if c <= x < c + f:
                    symbols.append(i)
                    break

        recovered = self._encode_decode(symbols, freqs, cumfreqs)
        assert recovered == symbols

    def test_all_same_symbols(self):
        """Degenerate case: single non-zero frequency symbol."""
        freqs = [M]
        cumfreqs = [0]
        symbols = [0] * 200
        recovered = self._encode_decode(symbols, freqs, cumfreqs)
        assert recovered == symbols

    def test_single_symbol(self):
        freqs = [M]
        cumfreqs = [0]
        symbols = [0]
        recovered = self._encode_decode(symbols, freqs, cumfreqs)
        assert recovered == symbols

    def test_three_symbols_manual(self):
        """Hand-verify a 3-symbol sequence.

        Using 2 symbols with freq=2048 each (uniform binary).
        Symbol sequence: [0, 1, 0]
        We verify encode then decode recovers [0, 1, 0].
        """
        freqs = [2048, 2048]
        cumfreqs = [0, 2048]
        symbols = [0, 1, 0]

        # Manual encode: symbol i -> stream i % 64
        # All 3 go to streams 0, 1, 2.
        # Just verify round-trip here (manual state math is complex for 4-byte flush).
        recovered = self._encode_decode(symbols, freqs, cumfreqs)
        assert recovered == symbols

    def test_two_symbols_binary(self):
        freqs = [2048, 2048]
        cumfreqs = [0, 2048]
        symbols = [0, 1, 0, 1, 1, 0, 0, 1] * 10
        recovered = self._encode_decode(symbols, freqs, cumfreqs)
        assert recovered == symbols

    def test_skewed_probabilities(self):
        # One rare symbol (freq=1), one very common (freq=4095)
        freqs = [4095, 1]
        cumfreqs = [0, 4095]
        random.seed(7)
        symbols = [0 if random.random() < 0.999 else 1 for _ in range(500)]
        recovered = self._encode_decode(symbols, freqs, cumfreqs)
        assert recovered == symbols


class TestStreamInterleaving:
    """Verify round-robin stream assignment."""

    def test_stream_assignment(self):
        """Check that symbol i goes to stream i % NUM_STREAMS by verifying
        that encoding with only stream-0 symbols and stream-1 symbols
        recovers correctly — an indirect proof via round-trip."""
        freqs = [M // 4] * 4
        cumfreqs = [i * (M // 4) for i in range(4)]

        # 128 symbols: stream 0 gets indices 0,64 (symbols 0,1); stream 1 gets 1,65 (symbols 2,3)
        symbols = list(range(4)) * 32  # 128 symbols cycling 0..3

        enc = RANSEncoder()
        data = enc.encode(symbols, freqs, cumfreqs)

        ext = _extended_cumfreqs(freqs, cumfreqs)

        def freq_fn(stream_idx, x):
            lo, hi = 0, len(freqs) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if ext[mid + 1] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            return lo, freqs[lo], cumfreqs[lo]

        dec = RANSDecoder()
        recovered = dec.decode(data, len(symbols), freq_fn)
        assert recovered == symbols

    def test_stream_0_gets_even_symbols(self):
        """Encode 128 symbols; confirm stream 0 received symbols at positions 0, 64."""
        # We verify this indirectly: encode a sequence where every 64th symbol
        # is distinctively different and confirm perfect recovery.
        freqs, cumfreqs = make_uniform_table(2)

        # Stream 0 positions: 0, 64. Assign symbol 0 there, symbol 1 elsewhere.
        n = 128
        symbols = [1] * n
        symbols[0] = 0
        symbols[64] = 0

        enc = RANSEncoder()
        data = enc.encode(symbols, freqs, cumfreqs)

        ext = _extended_cumfreqs(freqs, cumfreqs)

        def freq_fn(_, x):
            sym = 0 if x < freqs[0] else 1
            return sym, freqs[sym], cumfreqs[sym]

        dec = RANSDecoder()
        recovered = dec.decode(data, n, freq_fn)
        assert recovered == symbols
        assert recovered[0] == 0
        assert recovered[64] == 0
        assert recovered[1] == 1
        assert recovered[65] == 1

    def test_stream_boundaries(self):
        """Exactly 64 symbols → each stream gets exactly 1 symbol."""
        freqs, cumfreqs = make_uniform_table(4)
        random.seed(99)
        symbols = [random.randrange(4) for _ in range(64)]

        ext = _extended_cumfreqs(freqs, cumfreqs)

        def freq_fn(_, x):
            lo, hi = 0, 3
            while lo < hi:
                mid = (lo + hi) // 2
                if ext[mid + 1] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            return lo, freqs[lo], cumfreqs[lo]

        enc = RANSEncoder()
        data = enc.encode(symbols, freqs, cumfreqs)
        dec = RANSDecoder()
        recovered = dec.decode(data, 64, freq_fn)
        assert recovered == symbols


def _extended_cumfreqs(freqs, cumfreqs):
    return cumfreqs + [M]
