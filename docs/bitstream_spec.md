# Neural Compression Bitstream Specification

**Version**: 1.0  
**Status**: Normative  
**Scope**: Defines the byte-exact wire format exchanged between the VHDL hardware encoder and the Python software decoder.

---

## 1. Overview

A **frame bitstream** is a self-contained, byte-aligned sequence representing one compressed video frame. It contains:

1. A fixed-size **frame header** (7 bytes)
2. `num_strips` **per-strip packets**, each containing a hyperprior (`z`) bitstream and a spatial (`y`) bitstream
3. A 4-byte **EOF marker**

All multi-byte integers are **big-endian** unless stated otherwise. There is no alignment padding between fields; every field begins immediately after the last byte of the preceding field.

---

## 2. Frame Header (7 bytes)

| Offset | Size | Type    | Field         | Description                                      |
|--------|------|---------|---------------|--------------------------------------------------|
| 0      | 1    | uint8   | `magic`       | Magic byte = `0xNE` (0b1001_1110 = 158)         |
| 1      | 2    | uint16  | `frame_width` | Frame width in pixels, big-endian                |
| 3      | 2    | uint16  | `frame_height`| Frame height in pixels, big-endian               |
| 5      | 1    | uint8   | `model_id`    | Quality tier: 0 = low, 1 = medium, 2 = high      |
| 6      | 1    | uint8   | `num_strips`  | Number of horizontal strips = ⌈frame_height/64⌉ |

**Total header size**: 7 bytes.

### Field constraints

- `magic` **must** equal `0xNE`. The decoder **must** reject any stream where byte 0 ≠ `0xNE`.
- `frame_width` and `frame_height` are in the range [1, 65535].
- `model_id` values 3–255 are reserved; the decoder **must** reject them.
- `num_strips` **must** equal `⌈frame_height / 64⌉`. The last strip may be shorter than 64 rows.

---

## 3. Per-Strip Packet (repeated `num_strips` times)

Each strip packet covers a horizontal band of up to 64 rows. Strip packets are concatenated immediately after the frame header, in top-to-bottom order (strip 0 first).

### 3.1 Strip Header (7 bytes)

| Offset | Size | Type    | Field      | Description                                       |
|--------|------|---------|------------|---------------------------------------------------|
| 0      | 2    | uint16  | `strip_y`  | Row index of the top row of this strip, big-endian|
| 2      | 2    | uint16  | `z_len`    | Byte length of the following `z_bitstream`        |
| 4      | 3    | uint24  | `y_len`    | Byte length of the following `y_bitstream`        |

`y_len` is a **24-bit big-endian unsigned integer** (3 bytes). A full-quality 1080p strip can produce > 65 535 bytes; a 16-bit field would overflow.

### 3.2 Strip Payload

Immediately following the 7-byte strip header:

```
[z_bitstream: z_len bytes] [y_bitstream: y_len bytes]
```

#### 3.2.1 `z_bitstream` — Hyperprior Arithmetic Coding

The `z` latent is a 2-D tensor of INT8 values, flattened in row-major (C) order before encoding. Symbols are encoded with the **Witten-Neal-Cleary (WNC) bit-output arithmetic coder** implemented in `decoder/factorized.py`.

- Probability model: the 256-entry Laplace-shaped table returned by `FactorizedEntropyCoder.default_prob_table()` (12-bit integers summing to 4096).
- Symbol alphabet: unsigned bytes 0–255, where byte `b` represents signed INT8 value `b` (for b < 128) or `b − 256` (for b ≥ 128).
- Bits are packed **MSB-first** into output bytes; the last byte may be zero-padded.
- The coder produces a self-delimiting stream: the decoder requires `num_symbols` as a side-channel parameter (stored in the strip count / tensor shape, not in the bitstream itself).

#### 3.2.2 `y_bitstream` — Spatial rANS Coding

The `y` latent is encoded using the 64-stream interleaved rANS coder in `decoder/ans.py`.

**Internal layout of `y_bitstream`**:

| Offset within `y_bitstream` | Size        | Description                                      |
|-----------------------------|-------------|--------------------------------------------------|
| 0                           | 256 bytes   | Stream-length header: 64 × 4-byte **LE** uint32  |
| 256                         | variable    | Stream 0 payload (length from header)            |
| 256 + len(stream 0)         | variable    | Stream 1 payload                                 |
| …                           | …           | …                                                |
| sum of all lengths – last   | variable    | Stream 63 payload                                |

- Each 4-byte header entry is **little-endian** (matching the hardware's native 32-bit word order).
- Payloads are concatenated in stream order 0 → 63 with no gaps.
- Each stream payload is in **decoder-read order**: the encoder emits bytes MSB-first after reversing its internal emission order, so the decoder reads them front-to-back.

---

## 4. EOF Marker (4 bytes)

| Offset | Size | Value        | Description       |
|--------|------|--------------|-------------------|
| 0      | 4    | `0xDEADBEEF` | End-of-frame sentinel (big-endian) |

The decoder **must** verify the EOF marker. Any value other than `0xDEADBEEF` indicates stream corruption.

---

## 5. Complete Frame Layout

```
Offset 0:   [magic: 1B] [width: 2B] [height: 2B] [model_id: 1B] [num_strips: 1B]
Offset 7:   ┌── Strip 0 ─────────────────────────────────────────────────────────┐
            │  [strip_y: 2B] [z_len: 2B] [y_len: 3B]                            │
            │  [z_bitstream: z_len bytes]                                        │
            │  [y_bitstream: y_len bytes]                                        │
            └────────────────────────────────────────────────────────────────────┘
            ┌── Strip 1 ─────────────────────────────────────────────────────────┐
            │  …                                                                 │
            └────────────────────────────────────────────────────────────────────┘
            …
            ┌── Strip N-1 ───────────────────────────────────────────────────────┐
            │  …                                                                 │
            └────────────────────────────────────────────────────────────────────┘
EOF:        [0xDE] [0xAD] [0xBE] [0xEF]
```

---

## 6. Byte Ordering and Alignment

| Field class              | Byte order   | Notes                                    |
|--------------------------|--------------|------------------------------------------|
| `frame_width`, `frame_height`, `strip_y`, `z_len` | Big-endian | All header integer fields |
| `y_len` (3-byte uint24)  | Big-endian   | MSB at lower address                     |
| rANS stream-length table | Little-endian| 64 × 4-byte, matches VHDL 32-bit LE word |
| EOF marker               | Big-endian   | Read as a single 32-bit BE word          |
| `z_bitstream` payload    | N/A          | Opaque byte sequence (WNC bit-packed)    |
| `y_bitstream` payload    | N/A          | Opaque byte sequences (rANS streams)     |

There is **no padding or alignment** between any adjacent fields. Every byte is significant.

---

## 7. Worked Example — 2-Strip Frame

### Scenario

- Frame: 128 × 96 pixels, model_id = 1 (medium quality)
- `num_strips` = ⌈96 / 64⌉ = 2
- Strip 0: rows 0–63 (full strip)
- Strip 1: rows 64–95 (32 rows — partial strip)
- Hypothetical compressed sizes:
  - Strip 0: z_len = 40, y_len = 3 200
  - Strip 1: z_len = 22, y_len = 1 600

### Step-by-step byte layout

```
Byte  0:  0xNE                          ← magic
Byte  1:  0x00                          ← frame_width high byte (128 = 0x0080)
Byte  2:  0x80                          ← frame_width low byte
Byte  3:  0x00                          ← frame_height high byte (96 = 0x0060)
Byte  4:  0x60                          ← frame_height low byte
Byte  5:  0x01                          ← model_id = 1 (medium)
Byte  6:  0x02                          ← num_strips = 2

── Strip 0 header (7 bytes, offset 7) ──────────────────────────────────────
Byte  7:  0x00                          ← strip_y high byte (0 = 0x0000)
Byte  8:  0x00                          ← strip_y low byte
Byte  9:  0x00                          ← z_len high byte (40 = 0x0028)
Byte 10:  0x28                          ← z_len low byte
Byte 11:  0x00                          ← y_len byte 2 (MSB)  (3200 = 0x000C80)
Byte 12:  0x0C                          ← y_len byte 1
Byte 13:  0x80                          ← y_len byte 0 (LSB)

── Strip 0 payload (40 + 3200 = 3240 bytes, offset 14) ─────────────────────
Bytes 14–53:   z_bitstream[0]  (40 bytes, WNC arithmetic coded)
Bytes 54–57:   y_bitstream[0] stream-length table, streams 0–0   (first 4 of 256 B)
  …
Bytes 54–309:  y_bitstream[0] header (256 bytes: 64 × uint32 LE stream lengths)
Bytes 310–3253: y_bitstream[0] payloads streams 0–63 (3200 − 256 = 2944 bytes data)

── Strip 1 header (7 bytes, offset 3254) ───────────────────────────────────
Byte 3254: 0x00                         ← strip_y high byte (64 = 0x0040)
Byte 3255: 0x40                         ← strip_y low byte
Byte 3256: 0x00                         ← z_len high byte (22 = 0x0016)
Byte 3257: 0x16                         ← z_len low byte
Byte 3258: 0x00                         ← y_len byte 2 (MSB)  (1600 = 0x000640)
Byte 3259: 0x06                         ← y_len byte 1
Byte 3260: 0x40                         ← y_len byte 0 (LSB)

── Strip 1 payload (22 + 1600 = 1622 bytes, offset 3261) ───────────────────
Bytes 3261–3282: z_bitstream[1]  (22 bytes)
Bytes 3283–3538: y_bitstream[1] header (256 bytes)
Bytes 3539–4882: y_bitstream[1] payloads (1344 bytes)

── EOF marker (offset 4883) ────────────────────────────────────────────────
Byte 4883: 0xDE
Byte 4884: 0xAD
Byte 4885: 0xBE
Byte 4886: 0xEF

Total frame size: 4887 bytes
```

### Decoder walkthrough

1. Read bytes 0–6. Verify magic = `0xNE`. Parse width=128, height=96, model_id=1, num_strips=2.
2. **Strip 0**: Read bytes 7–13. Parse strip_y=0, z_len=40, y_len=3200.
   - Load 40 bytes → `FactorizedEntropyCoder.decode(data, num_z_symbols)`.
   - Load 3200 bytes → parse 256-byte LE header → reconstruct 64 stream slices → `RANSDecoder.decode(...)`.
3. **Strip 1**: Read next 7 bytes. Parse strip_y=64, z_len=22, y_len=1600.
   - Decode z (22 bytes) and y (1600 bytes) as above.
4. Read 4 bytes. Verify = `0xDEADBEEF`. If mismatch, raise `CorruptStreamError`.

---

## 8. Padding and Edge Cases

- **Partial strip**: the last strip covers `frame_height mod 64` rows (or 64 if divisible). The encoder processes partial strips normally; the decoder reconstructs only the rows present.
- **Minimum stream**: a 1×1 frame has num_strips = 1. The minimum valid frame is 7 + 7 + z_len + y_len + 4 bytes.
- **Maximum y_len**: a 1080p strip at high quality could produce up to ≈ 2^22 bytes. The 3-byte y_len field supports up to 16 777 215 bytes, which is sufficient.
- **Zero-length z or y**: not permitted. Every strip must have z_len ≥ 1 and y_len ≥ 257 (1 byte of arithmetic data + 256-byte rANS header).
