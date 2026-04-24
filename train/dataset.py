"""Aerial video dataset loader for neural compression training."""

import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from train.dataset_utils import rgb_to_yuv, scan_image_files


class AerialVideoDataset(Dataset):
    """Loads images from a directory tree for compression training.

    Crops random 256x256 patches during training, returns centre-cropped
    images (multiple of 8) for validation. Returns YUV 4:2:0-style 3-channel
    float32 tensor in [0, 1] with channels (Y, Cb, Cr).

    Args:
        root: path to directory containing images (jpg/png, any depth)
        patch_size: random crop size used during training (default 256)
        split: 'train' or 'val'
        val_fraction: fraction held out for validation (default 0.1)
        seed: random seed for train/val split (default 42)
        augment: enable horizontal flip; defaults to True when split='train'
    """

    def __init__(
        self,
        root: str,
        patch_size: int = 256,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        augment: bool | None = None,
    ) -> None:
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.patch_size = patch_size
        self.split = split
        self.augment = (split == "train") if augment is None else augment

        all_files = scan_image_files(root)
        if not all_files:
            raise ValueError(f"No image files found under {root!r}")

        rng = random.Random(seed)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)

        n_val = max(1, int(len(all_files) * val_fraction))
        if split == "val":
            selected = sorted(indices[:n_val])
        else:
            selected = sorted(indices[n_val:])

        self.files = [all_files[i] for i in selected]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.files[idx]).convert("RGB")

        if self.split == "train":
            tensor = self._train_transform(img)
        else:
            tensor = self._val_transform(img)

        return rgb_to_yuv(tensor)

    # ------------------------------------------------------------------
    def _train_transform(self, img: Image.Image) -> torch.Tensor:
        p = self.patch_size
        w, h = img.size
        # Pad if smaller than patch
        if w < p or h < p:
            img = TF.pad(img, [max(0, p - w), max(0, p - h)], padding_mode="reflect")
            w, h = img.size

        left = random.randint(0, w - p)
        top = random.randint(0, h - p)
        img = TF.crop(img, top, left, p, p)

        if self.augment and random.random() < 0.5:
            img = TF.hflip(img)

        return TF.to_tensor(img)

    def _val_transform(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        # Centre-crop to nearest multiple of 8
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img = TF.crop(img, top, left, new_h, new_w)
        return TF.to_tensor(img)
