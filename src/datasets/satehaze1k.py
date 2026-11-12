"""SateHaze1k / Haze1k Remote Sensing Image Dehazing Dataset (thin, moderate, thick subsets)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import ConcatDataset, Dataset

from src.datasets.base_paired import BasePairedDataset


def _find_subset_dir(root: Path, subset_name: str) -> Optional[Path]:
    """Resolve subset directory handling both 'Haze1k_thin' and 'thin' naming conventions."""
    candidates = [
        root / f"Haze1k_{subset_name}",
        root / subset_name,
        root / "Haze1k" / f"Haze1k_{subset_name}",
        root / "Haze1k" / subset_name,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


class _Haze1kSubsetDataset(Dataset):
    """Loads a single Haze1k subset (thin, moderate, or thick) with official train/val/test splits."""

    def __init__(
        self,
        subset_dir: Path,
        subset_name: str,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        crop_size: Optional[Tuple[int, int]] = (256, 256),
        seed: int = 42,
        transform: Optional[Callable] = None,
    ) -> None:
        self.subset_name = subset_name
        self.split = split.lower()
        if self.split == "valid":
            self.split = "val"

        # Check if subset has a 'dataset' subfolder (standard Haze1k official structure)
        ds_root = subset_dir / "dataset" if (subset_dir / "dataset").is_dir() else subset_dir

        # Check for official split folders: train, val (or valid), test
        train_dir = ds_root / "train"
        val_dir = ds_root / "valid" if (ds_root / "valid").is_dir() else ds_root / "val"
        test_dir = ds_root / "test"

        has_official_splits = train_dir.is_dir() and val_dir.is_dir() and test_dir.is_dir()

        if has_official_splits:
            if self.split == "train":
                self.sub_datasets = [
                    BasePairedDataset(train_dir, split="all", crop_size=crop_size, seed=seed, transform=transform)
                ]
            elif self.split in ("val", "valid"):
                self.sub_datasets = [
                    BasePairedDataset(val_dir, split="all", crop_size=crop_size, seed=seed, transform=transform)
                ]
            elif self.split == "test":
                self.sub_datasets = [
                    BasePairedDataset(test_dir, split="all", crop_size=crop_size, seed=seed, transform=transform)
                ]
            elif self.split == "all":
                # Combine train (320), val (35), and test (45) -> 400 pairs
                self.sub_datasets = [
                    BasePairedDataset(train_dir, split="all", crop_size=crop_size, seed=seed, transform=transform),
                    BasePairedDataset(val_dir, split="all", crop_size=crop_size, seed=seed, transform=transform),
                    BasePairedDataset(test_dir, split="all", crop_size=crop_size, seed=seed, transform=transform),
                ]
            else:
                raise ValueError(f"Unknown split: {self.split}")
        else:
            # Flat folder: use deterministic split via BasePairedDataset
            self.sub_datasets = [
                BasePairedDataset(
                    ds_root,
                    split=self.split,
                    split_ratios=split_ratios,
                    crop_size=crop_size,
                    seed=seed,
                    transform=transform,
                )
            ]

        self._dataset = ConcatDataset(self.sub_datasets) if len(self.sub_datasets) > 1 else self.sub_datasets[0]

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        item = self._dataset[idx]
        item["subset"] = self.subset_name
        return item


class SateHaze1kDataset(Dataset):
    """SateHaze1k / Haze1k Satellite Image Dehazing Benchmark Dataset.

    Paper Reference:
        "SateHaze1k: An Extensive Benchmark for Satellite Image Dehazing" / Haze1k
    
    Specifications:
        - 3 Subsets: 'thin' (400 pairs), 'moderate' (400 pairs), 'thick' (400 pairs)
        - Native Image Resolution: 512 x 512 pixels
        - Total Samples: 1,200 pairs
        - Official Splits per subset: train (320), val/valid (35), test (45)

    Supports both actual disk layouts:
        Layout A (nested):
            data/Haze1k/Haze1k/Haze1k_<thin|moderate|thick>/dataset/<train|val|valid|test>/<input|target>/
        Layout B (standard):
            data/SateHaze1k/<thin|moderate|thick>/<hazy|clear>/
    """

    SUBSETS = ("thin", "moderate", "thick")

    def __init__(
        self,
        root_dir: Union[str, Path] = "data/Haze1k",
        subset: str = "thin",
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        crop_size: Optional[Tuple[int, int]] = (256, 256),
        seed: int = 42,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize SateHaze1k / Haze1k dataset.

        Args:
            root_dir: Root directory of dataset (e.g. data/Haze1k or data/Haze1k/Haze1k or data/SateHaze1k).
            subset: One of 'thin', 'moderate', 'thick', or 'all' (combines all three).
            split: 'train', 'val', 'test', or 'all'.
            split_ratios: (train_ratio, val_ratio, test_ratio) for flat datasets.
            crop_size: (height, width) for synchronized paired cropping.
            seed: Seed for deterministic splitting.
            transform: Optional custom paired transformation.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.subset = subset.lower()
        self.split = split.lower()
        self.split_ratios = split_ratios
        self.crop_size = crop_size
        self.seed = seed

        valid_subsets = {"thin", "moderate", "thick", "all"}
        if self.subset not in valid_subsets:
            raise ValueError(f"Invalid subset '{self.subset}'. Must be one of {valid_subsets}")

        # Resolve root directory if nested
        resolved_root = self.root_dir
        if (self.root_dir / "Haze1k").is_dir() and not (self.root_dir / "Haze1k_thin").is_dir():
            resolved_root = self.root_dir / "Haze1k"

        # Check if root_dir itself directly contains images in hazy/clear or input/target
        is_direct = any((resolved_root / f).is_dir() for f in ["hazy", "input", "inputs"]) and any(
            (resolved_root / f).is_dir() for f in ["clear", "target", "targets", "clean"]
        )

        if is_direct:
            self._underlying: Dataset = BasePairedDataset(
                root_dir=resolved_root,
                split=self.split,
                split_ratios=self.split_ratios,
                crop_size=self.crop_size,
                seed=self.seed,
                transform=transform,
            )
            return

        subsets_to_load = list(self.SUBSETS) if self.subset == "all" else [self.subset]
        datasets = []

        for s_name in subsets_to_load:
            s_dir = _find_subset_dir(resolved_root, s_name)
            if not s_dir:
                raise FileNotFoundError(
                    f"Subset '{s_name}' directory not found under {resolved_root}. "
                    f"Expected folder like 'Haze1k_{s_name}' or '{s_name}'."
                )

            ds = _Haze1kSubsetDataset(
                subset_dir=s_dir,
                subset_name=s_name,
                split=self.split,
                split_ratios=self.split_ratios,
                crop_size=self.crop_size,
                seed=self.seed,
                transform=transform,
            )
            datasets.append(ds)

        if len(datasets) == 1:
            self._underlying = datasets[0]
        else:
            self._underlying = ConcatDataset(datasets)

    def __len__(self) -> int:
        return len(self._underlying)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        return self._underlying[idx]


# Alias for compatibility
Haze1kDataset = SateHaze1kDataset
