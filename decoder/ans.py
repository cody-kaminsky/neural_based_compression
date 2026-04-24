"""
rANS (range Asymmetric Numeral Systems) encoder and decoder.

Matches the VHDL hardware spec:
  L = 2^23, b = 256, M = 2^12
  64 independent streams, round-robin interleaving, streams flushed 0..63.

Output format:
  [64 x 4-byte LE stream lengths] [stream 0 bytes] ... [stream 63 bytes]

Each stream's bytes are in decoder-read order (encoder emission reversed),
so the decoder reads them linearly front-to-back.
"""

from __future__ import annotations
from typing import Callable, List, Tuple

L = 1 << 23       # lower bound of normalised state range
B = 256            # output alphabet size (byte)
M = 1 << 12       # probability precision (4096)
NUM_STREAMS = 64
HEADER_BYTES = NUM_STREAMS * 4  # 256 bytes


def build_prob_table(probs: List[int]) -> Tuple[List[int], List[int]]:
    """Convert raw 12-bit probabilities to (freqs, cumfreqs).

    probs must be non-negative integers summing to M=4096.
    """
    assert sum(probs) == M, f"probs must sum to {M}, got {sum(probs)}"
    cumfreqs = []
    c = 0
    for p in probs:
        cumfreqs.append(c)
        c += p
    return list(probs), cumfreqs


class RANSEncoder:
    """Encode a sequence of symbols into a 64-stream interleaved rANS bitstream."""

    def encode(
        self,
        symbols: List[int],
        freqs: List[int],
        cumfreqs: List[int],
    ) -> bytes:
        """Encode symbols and return the interleaved byte output.

        Args:
            symbols:  sequence of symbol indices
            freqs:    frequency[symbol] for each symbol
            cumfreqs: cumulative frequency[symbol] for each symbol

        Returns:
            Bytes: 256-byte length header followed by 64 stream payloads.
        """
        states = [L] * NUM_STREAMS
        # Emission list per stream: bytes appended LSB-first during normalisation
        stream_emit: List[List[int]] = [[] for _ in range(NUM_STREAMS)]

        for i, sym in enumerate(symbols):
            s = i % NUM_STREAMS
            freq = freqs[sym]
            cumfreq = cumfreqs[sym]
            x = states[s]

            # Normalise: x must land in [freq*(L//M), freq*(L//M)*B) before update
            upper = freq * (L // M) * B
            while x >= upper:
                stream_emit[s].append(x & 0xFF)
                x >>= 8

            # rANS update
            x = (x // freq) * M + cumfreq + (x % freq)
            states[s] = x

        # Flush: emit 4 state bytes per stream (LSB first)
        # State is in [L, B*L) = [2^23, 2^31) — fits in 4 bytes.
        for s in range(NUM_STREAMS):
            x = states[s]
            for _ in range(4):
                stream_emit[s].append(x & 0xFF)
                x >>= 8

        # Reverse each stream so decoder can read linearly (flush bytes come first,
        # then renorm bytes in high-to-low order as the decoder expects).
        result = bytearray()
        header = bytearray()
        payloads = bytearray()
        for s in range(NUM_STREAMS):
            payload = bytes(reversed(stream_emit[s]))
            header.extend(len(payload).to_bytes(4, 'little'))
            payloads.extend(payload)

        return bytes(header) + bytes(payloads)


class RANSDecoder:
    """Decode a 64-stream interleaved rANS bitstream."""

    def decode(
        self,
        data: bytes,
        num_symbols: int,
        freq_fn: Callable[[int, int], Tuple[int, int, int]],
    ) -> List[int]:
        """Decode num_symbols symbols from data.

        Args:
            data:        byte output of RANSEncoder.encode
            num_symbols: number of symbols to decode
            freq_fn:     callable(stream_idx, slot) -> (symbol, freq, cumfreq)
                         where slot = state % M
        """
        # Parse stream lengths from header
        lengths = [
            int.from_bytes(data[s * 4:(s + 1) * 4], 'little')
            for s in range(NUM_STREAMS)
        ]

        # Slice per-stream payloads (already in decoder-read order; no re-reversal)
        stream_bufs: List[bytes] = []
        offset = HEADER_BYTES
        for s in range(NUM_STREAMS):
            stream_bufs.append(data[offset:offset + lengths[s]])
            offset += lengths[s]

        # Read initial state from first 4 bytes of each stream.
        # The encoder flushed LSB-first then reversed, so these 4 bytes are
        # [high, ..., low] — read with left-shift to reconstruct.
        states = [0] * NUM_STREAMS
        cursors = [4] * NUM_STREAMS
        for s in range(NUM_STREAMS):
            x = 0
            for b in range(4):
                x = (x << 8) | stream_bufs[s][b]
            states[s] = x

        # Decode symbols in reverse order (last encoded is decoded first)
        symbols = [0] * num_symbols
        for i in range(num_symbols - 1, -1, -1):
            s = i % NUM_STREAMS
            x = states[s]

            # rANS decode step
            slot = x % M
            sym, freq, cumfreq = freq_fn(s, slot)
            x = freq * (x // M) + slot - cumfreq
            symbols[i] = sym

            # Renorm: read bytes from stream using left-shift (MSB comes first
            # because encoder emitted LSB-first then reversed)
            while x < L:
                x = (x << 8) | stream_bufs[s][cursors[s]]
                cursors[s] += 1

            states[s] = x

        return symbols
