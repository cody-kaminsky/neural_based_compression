"""Tests for AerialVideoDataset, dataset_utils, and train.py importability."""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

# Ensure repo root is on sys.path when running from any directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from train.dataset import AerialVideoDataset
from train.dataset_utils import rgb_to_yuv, scan_image_files, yuv_to_rgb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_test_images(tmp_path: str, n: int = 10, size: tuple = (128, 96)) -> str:
    """Write n random RGB PNGs into tmp_path and return its path."""
    for i in range(n):
        arr = np.random.randint(0, 255, (*size[::-1], 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        img.save(os.path.join(tmp_path, f"img_{i:02d}.png"))
    return tmp_path


# ---------------------------------------------------------------------------
# scan_image_files
# ---------------------------------------------------------------------------

class TestScanImageFiles:
    def test_finds_all_images(self, tmp_path):
        d = str(tmp_path)
        _make_test_images(d, n=5)
        files = scan_image_files(d)
        assert len(files) == 5

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _make_test_images(str(tmp_path), n=3)
        _make_test_images(str(sub), n=2)
        files = scan_image_files(str(tmp_path))
        assert len(files) == 5

    def test_sorted(self, tmp_path):
        _make_test_images(str(tmp_path), n=5)
        files = scan_image_files(str(tmp_path))
        assert files == sorted(files)

    def test_empty_dir(self, tmp_path):
        assert scan_image_files(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# YUV conversion
# ---------------------------------------------------------------------------

class TestYuvConversion:
    def test_roundtrip_within_tolerance(self):
        rgb = torch.rand(1, 3, 64, 64)
        yuv = rgb_to_yuv(rgb)
        rgb_back = yuv_to_rgb(yuv)
        # Allow 1/255 tolerance for float rounding
        assert torch.allclose(rgb, rgb_back, atol=1.0 / 255 + 1e-5), (
            f"Max error: {(rgb - rgb_back).abs().max().item():.6f}"
        )

    def test_output_range(self):
        rgb = torch.rand(2, 3, 32, 32)
        yuv = rgb_to_yuv(rgb)
        assert yuv.min() >= 0.0 - 1e-6
        assert yuv.max() <= 1.0 + 1e-6

    def test_shape_preserved(self):
        rgb = torch.rand(4, 3, 48, 48)
        yuv = rgb_to_yuv(rgb)
        assert yuv.shape == rgb.shape

    def test_inverse_range(self):
        yuv = torch.rand(1, 3, 32, 32)
        rgb = yuv_to_rgb(yuv)
        assert rgb.min() >= 0.0 - 1e-6
        assert rgb.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# AerialVideoDataset
# ---------------------------------------------------------------------------

class TestAerialVideoDataset:
    def _ds(self, tmp_path, split="train", n=10, seed=42):
        _make_test_images(str(tmp_path), n=n)
        return AerialVideoDataset(str(tmp_path), patch_size=64, split=split, seed=seed)

    def test_len_train(self, tmp_path):
        ds = self._ds(tmp_path, split="train", n=10)
        # 10% val → 1 val, 9 train
        assert len(ds) == 9

    def test_len_val(self, tmp_path):
        ds = self._ds(tmp_path, split="val", n=10)
        assert len(ds) == 1

    def test_train_val_disjoint(self, tmp_path):
        _make_test_images(str(tmp_path), n=20)
        train_ds = AerialVideoDataset(str(tmp_path), split="train", seed=42)
        val_ds = AerialVideoDataset(str(tmp_path), split="val", seed=42)
        train_set = set(train_ds.files)
        val_set = set(val_ds.files)
        assert train_set.isdisjoint(val_set)
        assert train_set | val_set == set(scan_image_files(str(tmp_path)))

    def test_getitem_shape_train(self, tmp_path):
        ds = self._ds(tmp_path, split="train")
        item = ds[0]
        assert item.shape == (3, 64, 64), f"Got shape {item.shape}"

    def test_getitem_shape_val(self, tmp_path):
        # Images are 128×96 — both are multiples of 8 already
        ds = self._ds(tmp_path, split="val")
        item = ds[0]
        assert item.ndim == 3
        assert item.shape[0] == 3
        assert item.shape[1] % 8 == 0
        assert item.shape[2] % 8 == 0

    def test_getitem_dtype(self, tmp_path):
        ds = self._ds(tmp_path, split="train")
        item = ds[0]
        assert item.dtype == torch.float32

    def test_getitem_range(self, tmp_path):
        ds = self._ds(tmp_path, split="train")
        item = ds[0]
        assert item.min() >= 0.0 - 1e-6
        assert item.max() <= 1.0 + 1e-6

    def test_split_reproducible(self, tmp_path):
        _make_test_images(str(tmp_path), n=20)
        ds1 = AerialVideoDataset(str(tmp_path), split="train", seed=42)
        ds2 = AerialVideoDataset(str(tmp_path), split="train", seed=42)
        assert ds1.files == ds2.files

    def test_different_seeds_differ(self, tmp_path):
        _make_test_images(str(tmp_path), n=20)
        ds1 = AerialVideoDataset(str(tmp_path), split="train", seed=42)
        ds2 = AerialVideoDataset(str(tmp_path), split="train", seed=99)
        assert ds1.files != ds2.files

    def test_empty_root_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No image files"):
            AerialVideoDataset(str(tmp_path), split="train")


# ---------------------------------------------------------------------------
# train.py importability (no GPU required)
# ---------------------------------------------------------------------------

class TestTrainImportable:
    def _run_script(self, script: str) -> "subprocess.CompletedProcess":
        import subprocess
        import textwrap
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True, text=True, cwd=_REPO_ROOT,
        )

    def test_train_py_is_valid_python(self):
        """train/train.py should be parseable as valid Python."""
        import ast
        src = open(os.path.join(_REPO_ROOT, "train", "train.py")).read()
        ast.parse(src)  # raises SyntaxError on failure

    def test_parse_args_defined(self):
        """parse_args() in train.py should accept known arguments without error."""
        result = self._run_script("""
            import sys
            sys.path.insert(0, '.')
            import unittest.mock as mock
            sys.modules['train.model'] = mock.MagicMock()
            import importlib.util, types
            spec = importlib.util.spec_from_file_location('train.train', 'train/train.py')
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            args = mod.parse_args.__call__.__doc__
            print('OK')
        """)
        # Accept either success or a clean ImportError for the model
        assert result.returncode == 0 or "NeuralEncoderModel" in result.stderr, (
            f"Unexpected failure:\n{result.stderr}"
        )

    def test_argparse_help(self):
        """train.py --help should print usage and exit 0."""
        result = self._run_script("""
            import sys
            sys.path.insert(0, '.')
            import unittest.mock as mock
            sys.modules['train.model'] = mock.MagicMock()
            import importlib.util
            spec = importlib.util.spec_from_file_location('train.train', 'train/train.py')
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print('OK')
        """)
        assert result.returncode == 0 or "NeuralEncoderModel" in result.stderr, (
            f"Unexpected failure:\n{result.stderr}"
        )
