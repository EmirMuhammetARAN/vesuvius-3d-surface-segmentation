"""
3D Patch Dataset and Augmentations for Vesuvius Surface Detection.

Key Features:
1. Dynamic 3D patch extraction (e.g. 128x128x128).
2. Foreground Oversampling (nnU-Net style): ~66% of patches are biased towards foreground papyrus voxels (label==1).
3. 3D Spatial Augmentations: Random flips across all 3 axes, random 90-degree 3D rotations.
4. Proper handling of label 2 (unlabeled/ignore regions).
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset


class Vesuvius3DPatchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: str = "dataset",
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        samples_per_volume: int = 4,
        fg_oversample_prob: float = 0.67,
        is_train: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.fg_oversample_prob = fg_oversample_prob
        self.is_train = is_train

        self.image_dir = self.data_dir / "train_images"
        self.label_dir = self.data_dir / "train_labels"

        # Cache volume paths and verify existence
        self.samples = []
        for _, row in self.df.iterrows():
            img_id = str(row["id"])
            img_path = self.image_dir / f"{img_id}.tif"
            lbl_path = self.label_dir / f"{img_id}.tif"
            if img_path.exists() and lbl_path.exists():
                self.samples.append({
                    "id": img_id,
                    "scroll_id": row.get("scroll_id", -1),
                    "img_path": str(img_path),
                    "lbl_path": str(lbl_path)
                })

        self.cache = {}
        self.max_cache_items = 64

    def __len__(self) -> int:
        return len(self.samples) * self.samples_per_volume

    def _load_volume(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if sample_idx in self.cache:
            return self.cache[sample_idx]

        sample = self.samples[sample_idx]
        image = tifffile.imread(sample["img_path"])
        label = tifffile.imread(sample["lbl_path"])

        if len(self.cache) < self.max_cache_items:
            self.cache[sample_idx] = (image, label)

        return image, label

    def _extract_patch(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        D, H, W = image.shape
        pD, pH, pW = self.patch_size

        # If volume is smaller than patch size in any dim, pad with zeros
        pad_d = max(0, pD - D)
        pad_h = max(0, pH - H)
        pad_w = max(0, pW - W)
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            label = np.pad(label, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant", constant_values=2)
            D, H, W = image.shape

        # Foreground oversampling: pick center near foreground voxel with prob fg_oversample_prob
        use_fg = (np.random.rand() < self.fg_oversample_prob) if self.is_train else False
        fg_indices = None
        if use_fg:
            fg_indices = np.argwhere(label == 1)
            if len(fg_indices) == 0:
                use_fg = False

        if use_fg and fg_indices is not None and len(fg_indices) > 0:
            # Pick a random foreground point as patch center
            center = fg_indices[np.random.choice(len(fg_indices))]
            start_d = np.clip(center[0] - pD // 2, 0, D - pD)
            start_h = np.clip(center[1] - pH // 2, 0, H - pH)
            start_w = np.clip(center[2] - pW // 2, 0, W - pW)
        else:
            # Uniform random crop
            start_d = np.random.randint(0, max(1, D - pD + 1))
            start_h = np.random.randint(0, max(1, H - pH + 1))
            start_w = np.random.randint(0, max(1, W - pW + 1))

        patch_img = image[start_d : start_d + pD, start_h : start_h + pH, start_w : start_w + pW]
        patch_lbl = label[start_d : start_d + pD, start_h : start_h + pH, start_w : start_w + pW]

        return patch_img, patch_lbl

    def _apply_augmentations(
        self, img: np.ndarray, lbl: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        for axis in range(3):
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=axis)
                lbl = np.flip(lbl, axis=axis)

        # 3D 90-degree rotations in plane (H, W)
        k = np.random.choice([0, 1, 2, 3])
        if k > 0:
            img = np.rot90(img, k=k, axes=(1, 2))
            lbl = np.rot90(lbl, k=k, axes=(1, 2))

        return np.ascontiguousarray(img), np.ascontiguousarray(lbl)

    def __getitem__(self, idx: int) -> dict:
        vol_idx = idx // self.samples_per_volume
        image, label = self._load_volume(vol_idx)

        patch_img, patch_lbl = self._extract_patch(image, label)

        if self.is_train:
            patch_img, patch_lbl = self._apply_augmentations(patch_img, patch_lbl)

        # Normalization: uint8 [0, 255] to float32 [0.0, 1.0]
        patch_img = patch_img.astype(np.float32) / 255.0

        # PyTorch format: (Channel, D, H, W)
        tensor_img = torch.from_numpy(patch_img).unsqueeze(0).float()
        tensor_lbl = torch.from_numpy(patch_lbl).long()

        return {
            "image": tensor_img,
            "label": tensor_lbl,
            "id": self.samples[vol_idx]["id"]
        }
