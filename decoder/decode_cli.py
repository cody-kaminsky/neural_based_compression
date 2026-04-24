"""Command-line entry point for the ground-station decoder.

Usage:
    python -m decoder.decode_cli <bitstream.bin> --model-dir <dir> --output <out.png>
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="decoder.decode_cli",
        description="Neural video compression ground-station decoder",
    )
    parser.add_argument("bitstream", help="Compressed bitstream (.bin)")
    parser.add_argument("--model-dir", required=True, metavar="DIR",
                        help="Directory containing synthesis.onnx / hyper_synthesis.onnx "
                             "(PyTorch fallback used when absent)")
    parser.add_argument("--output", required=True, metavar="FILE",
                        help="Output image path (.png or .yuv)")
    parser.add_argument("--format", choices=["png", "yuv420"], default="png",
                        help="Output format (default: png)")
    parser.add_argument("--overlap-rows", type=int, default=0, metavar="N",
                        help="Overlap rows to discard from each strip top (default: 0)")
    args = parser.parse_args()

    from .decode import decode_frame

    try:
        result = decode_frame(
            args.bitstream,
            args.model_dir,
            args.output,
            output_format=args.format,
            overlap_rows=args.overlap_rows,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Decoded {result['width']}x{result['height']} "
        f"({result['num_strips']} strip(s), model_id={result['model_id']})"
    )
    print(f"Output: {result['output_path']}")


if __name__ == "__main__":
    main()
