"""Tests for P0-T38 golden reference test images."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = REPO_ROOT / "reference_vectors" / "test_images"

TARGET_W, TARGET_H = 1280, 720
YUV_BYTES = TARGET_W * TARGET_H + (TARGET_W // 2) * (TARGET_H // 2) * 2  # 1 382 400

STEMS = ["kodim05", "kodim15", "kodim23", "aerial_low_texture", "aerial_high_texture"]


@pytest.fixture(scope="module")
def manifest():
    path = IMG_DIR / "manifest.json"
    assert path.exists(), f"manifest.json not found at {path}"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("stem", STEMS)
def test_png_dimensions(stem):
    path = IMG_DIR / f"{stem}.png"
    assert path.exists(), f"PNG not found: {path}"
    with Image.open(path) as img:
        assert img.size == (TARGET_W, TARGET_H), (
            f"{stem}.png: expected {TARGET_W}x{TARGET_H}, got {img.size}"
        )


@pytest.mark.parametrize("stem", STEMS)
def test_yuv_size(stem):
    path = IMG_DIR / f"{stem}.yuv"
    assert path.exists(), f"YUV not found: {path}"
    size = path.stat().st_size
    assert size == YUV_BYTES, (
        f"{stem}.yuv: expected {YUV_BYTES} bytes, got {size}"
    )


@pytest.mark.parametrize("stem", STEMS)
def test_y_plane_mean(stem):
    path = IMG_DIR / f"{stem}.yuv"
    assert path.exists(), f"YUV not found: {path}"
    y_size = TARGET_W * TARGET_H
    raw = path.read_bytes()
    y_plane = np.frombuffer(raw[:y_size], dtype=np.uint8)
    mean = float(y_plane.mean())
    assert 20.0 <= mean <= 235.0, (
        f"{stem}.yuv Y-plane mean {mean:.2f} outside [20, 235]"
    )


def test_manifest_entries(manifest):
    assert len(manifest) == len(STEMS), (
        f"manifest has {len(manifest)} entries, expected {len(STEMS)}"
    )
    required_fields = {"filename", "source_url", "content_description", "width", "height"}
    for entry in manifest:
        missing = required_fields - entry.keys()
        assert not missing, f"manifest entry missing fields: {missing}  entry={entry}"
        assert entry["width"] == TARGET_W
        assert entry["height"] == TARGET_H


def test_manifest_filenames_match_stems(manifest):
    manifest_stems = {Path(e["filename"]).stem for e in manifest}
    assert manifest_stems == set(STEMS), (
        f"manifest stems mismatch: {manifest_stems} != {set(STEMS)}"
    )
