"""RRSHID (Realistic Real-World Remote Sensing Hazy Image Dataset) Dataset Module.

Dataset Specifications:
    - Real-world satellite & aerial hazy remote sensing scenes (3,053 pairs total).
    - Resolution: 256x256 RGB pixels.
    - Three haze density subsets:
        * Thin (TN): 763 pairs (train: 610, val: 77, test: 76)
        * Moderate (M): 1,526 pairs (train: 1220, val: 154, test: 152)
        * Thick (TK): 764 pairs (train: 611, val: 77, test: 76)
        * Total: 3,053 pairs across all subsets and splits.
    - Official dataset splits: 'train', 'val', 'test' (or 'all' for full corpus).

Directory Structure on Disk:
    data/RRSHID/
    ├── thin_fog/
    │   ├── train/ (gt/, hazy/ - 610 pairs)
    │   ├── val/   (GT/, hazy/ - 77 pairs)
    │   └── test/  (gt/, hazy/ - 76 pairs)
    ├── moderate_fog/
    │   ├── train/ (clear/, hazy/ - 1220 pairs)
    │   ├── val/   (GT/, hazy/ - 154 pairs)
    │   └── test/  (clear/, hazy/ - 152 pairs)
    └── thick_fog/
        ├── train/ (clear/, hazy/ - 611 pairs)
        ├── val/   (GT/, hazy/ - 77 pairs)
        └── test/  (clear/, hazy/ - 76 pairs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import ConcatDataset, Dataset

from src.datasets.base_paired import BasePairedDataset


# Map aliases to the exact disk folder names
SUBSET_NAME_MAP = {
    # Thin
    "thin": "thin_fog",
    "tn": "thin_fog",
    "thin_fog": "thin_fog",
    # Moderate
    "moderate": "moderate_fog",
    "m": "moderate_fog",
    "moderate_fog": "moderate_fog",
    # Thick
    "thick": "thick_fog",
    "tk": "thick_fog",
    "thick_fog": "thick_fog",
}


def _find_rrshid_root(root: Path) -> Path:
    """Find the root directory containing thin_fog, moderate_fog, thick_fog."""
    candidates = [
        root,
        root / "RRSHID",
        root / "data" / "RRSHID",
    ]
    for c in candidates:
        if (c / "thin_fog").is_dir() or (c / "moderate_fog").is_dir() or (c / "thick_fog").is_dir():
            return c
    return root


class _RRSHIDSingleSubsetDataset(Dataset):
    """Loads a single RRSHID subset (thin_fog, moderate_fog, or thick_fog)."""

    def __init__(
        self,
        subset_dir: Path,
        subset_name: str,
        split: str = "train",
        crop_size: Optional[Tuple[int, int]] = None,
        seed: int = 42,
        transform: Optional[Callable] = None,
    ) -> None:
        self.subset_name = subset_name
        self.split = split.lower()
        if self.split == "valid":
            self.split = "val"

        train_dir = subset_dir / "train"
        val_dir = subset_dir / "val"
        test_dir = subset_dir / "test"

        has_official_splits = train_dir.is_dir() and val_dir.is_dir() and test_dir.is_dir()
        if not has_official_splits:
            raise FileNotFoundError(
                f"Expected official split directories (train, val, test) inside {subset_dir}"
            )

        def _resolve_clear_dir(split_path: Path) -> Path:
            for cand in ["clear", "gt", "GT", "target", "label"]:
                if (split_path / cand).is_dir():
                    return split_path / cand
            raise FileNotFoundError(f"Could not find clear/gt directory in {split_path}")

        def _make_base_dataset(sp_dir: Path) -> BasePairedDataset:
            h_dir = sp_dir / "hazy"
            c_dir = _resolve_clear_dir(sp_dir)
            return BasePairedDataset(
                root_dir=sp_dir,
                split="all",
                crop_size=crop_size,
                seed=seed,
                transform=transform,
                hazy_dir=h_dir,
                clear_dir=c_dir,
            )

        if self.split == "train":
            self.sub_datasets = [_make_base_dataset(train_dir)]
        elif self.split in ("val", "valid"):
            self.sub_datasets = [_make_base_dataset(val_dir)]
        elif self.split == "test":
            self.sub_datasets = [_make_base_dataset(test_dir)]
        elif self.split == "all":
            self.sub_datasets = [
                _make_base_dataset(train_dir),
                _make_base_dataset(val_dir),
                _make_base_dataset(test_dir),
            ]
        else:
            raise ValueError(f"Unknown split '{split}'. Expected one of ['train', 'val', 'valid', 'test', 'all']")

        self._dataset = ConcatDataset(self.sub_datasets) if len(self.sub_datasets) > 1 else self.sub_datasets[0]

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        item = self._dataset[idx]
        item["subset"] = self.subset_name
        return item


class RRSHIDDataset(Dataset):
    """RRSHID (Realistic Real-World Remote Sensing Hazy Image Dataset) paired dataset.

    Paper Reference:
        "Remote sensing image dehazing: A systematic review of progress, challenges, and prospects"
        (ISPRS Journal of Photogrammetry and Remote Sensing, Vol. 235, 2026)

    Subsets:
        - 'thin' / 'TN' / 'thin_fog': Thin real haze (763 pairs)
        - 'moderate' / 'M' / 'moderate_fog': Moderate real haze (1,526 pairs)
        - 'thick' / 'TK' / 'thick_fog': Thick real haze (764 pairs)
        - 'all' / 'combined': Combined across all 3 subsets (3,053 pairs)

    Splits:
        - 'train': Official training split
        - 'val' / 'valid': Official validation split
        - 'test': Official test split
        - 'all': All splits combined
    """

    def __init__(
        self,
        root_dir: Union[str, Path] = "data/RRSHID",
        subset: str = "all",
        split: str = "train",
        crop_size: Optional[Tuple[int, int]] = None,
        seed: int = 42,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initialize RRSHID dataset.

        Args:
            root_dir: Path to RRSHID root folder.
            subset: One of 'thin' ('TN'), 'moderate' ('M'), 'thick' ('TK'), or 'all'/'combined'.
            split: One of 'train', 'val', 'valid', 'test', or 'all'.
            crop_size: (height, width) for synchronized paired cropping. None for native (256x256).
            seed: Seed for deterministic operations.
            transform: Custom paired transform (optional).
        """
        super().__init__()
        self.root_dir = _find_rrshid_root(Path(root_dir))
        self.subset_arg = subset.lower()
        self.split = split.lower()
        if self.split == "valid":
            self.split = "val"
        self.crop_size = crop_size
        self.seed = seed
        self.transform = transform

        if self.subset_arg in ("all", "combined"):
            sub_names = ["thin_fog", "moderate_fog", "thick_fog"]
        elif self.subset_arg in SUBSET_NAME_MAP:
            sub_names = [SUBSET_NAME_MAP[self.subset_arg]]
        else:
            valid_keys = list(SUBSET_NAME_MAP.keys()) + ["all", "combined"]
            raise ValueError(f"Invalid subset '{subset}'. Must be one of {valid_keys}")

        self.sub_datasets: List[_RRSHIDSingleSubsetDataset] = []
        for s_folder in sub_names:
            s_dir = self.root_dir / s_folder
            if not s_dir.is_dir():
                raise FileNotFoundError(f"RRSHID subset folder not found at: {s_dir}")
            # Identify standard human-friendly subset tag (thin, moderate, thick)
            friendly_name = s_folder.replace("_fog", "")
            self.sub_datasets.append(
                _RRSHIDSingleSubsetDataset(
                    subset_dir=s_dir,
                    subset_name=friendly_name,
                    split=self.split,
                    crop_size=crop_size,
                    seed=seed,
                    transform=transform,
                )
            )

        self._dataset = ConcatDataset(self.sub_datasets) if len(self.sub_datasets) > 1 else self.sub_datasets[0]

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        return self._dataset[idx]

