"""RICE1 (Remote Sensing Image Cloud & Haze Removal) Dataset (500 pairs)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import torch

from src.datasets.base_paired import BasePairedDataset


class RICE1Dataset(BasePairedDataset):
    """RICE-I Remote Sensing Cloud & Haze Removal Dataset.

    Paper Reference:
        "A dataset for cloud removal in remote sensing images"
        (Lin et al., Remote Sensing Letters, 2019)

    Dataset Specifications:
        - Total Samples: 500 paired satellite images (Google Earth RGB, 512x512)
        - Folders: 'cloud' (hazy/cloudy input) and 'label' (clear reference)

    Supported Directory Layouts:
        Layout A:
            data/RICE_DATASET/RICE1/
            ├── cloud/  (500 images: 0.png - 499.png)
            └── label/  (500 images: 0.png - 499.png)
        Layout B:
            data/RICE1/
            ├── cloud/ (or hazy/)
            └── label/ (or clear/)
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = "data/RICE_DATASET/RICE1",
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        crop_size: Optional[Tuple[int, int]] = (256, 256),
        seed: int = 42,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize RICE-1 dataset.

        Args:
            root_dir: Root directory (e.g. data/RICE_DATASET/RICE1 or data/RICE1 or data/RICE_DATASET).
            split: 'train', 'val', 'test', or 'all'.
            split_ratios: (train_ratio, val_ratio, test_ratio).
            crop_size: (height, width) for synchronized paired cropping.
            seed: Seed for deterministic train/val/test splitting.
            transform: Optional custom transform.
        """
        resolved_root = Path(root_dir)

        # Handle nested data/RICE_DATASET without RICE1 in path
        if (resolved_root / "RICE1").is_dir() and not (resolved_root / "cloud").is_dir():
            resolved_root = resolved_root / "RICE1"

        super().__init__(
            root_dir=resolved_root,
            split=split,
            split_ratios=split_ratios,
            crop_size=crop_size,
            seed=seed,
            transform=transform,
        )

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        item = super().__getitem__(idx)
        item["dataset"] = "RICE1"
        return item
