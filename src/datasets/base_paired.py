"""Base dataset class for paired remote sensing dehazing datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

from src.datasets.transforms import PairedTransform

# Allow loading valid images that might have truncated metadata
ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
HAZY_FOLDER_NAMES = ["hazy", "cloud", "input", "inputs"]
CLEAR_FOLDER_NAMES = ["clear", "clean", "label", "labels", "target", "targets", "ground_truth", "gt", "GT"]


def normalize_stem(stem: str) -> str:
    """Normalize file stem by removing common input/target suffixes."""
    s = stem
    for suffix in ["-inputs", "-targets", "-input", "-target", "_input", "_target"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


class BasePairedDataset(Dataset):
    """Abstract base dataset for paired remote sensing image dehazing benchmarks.

    Expected directory structure:
        root_dir/
            ├── hazy/   (or cloud/, input/, inputs/)
            └── clear/  (or clean/, label/, labels/, target/, targets/)

    Features:
    - Scans and matches pairs by filename stem with suffix normalization.
    - Deterministic, non-overlapping train/val/test splits governed by a seed.
    - Synchronized paired transformations (random crops, flips, 90-deg rotations).
    - Normalization to float32 [0.0, 1.0].
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        crop_size: Optional[Tuple[int, int]] = (256, 256),
        seed: int = 42,
        transform: Optional[Callable] = None,
        hazy_dir: Optional[Union[str, Path]] = None,
        clear_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize paired dataset.

        Args:
            root_dir: Directory containing hazy and clear subdirectories.
            split: One of 'train', 'val', 'test', or 'all'.
            split_ratios: (train_ratio, val_ratio, test_ratio) summing to 1.0.
            crop_size: Spatial (height, width) for random/center cropping. None to keep native.
            seed: Seed for deterministic split permutation.
            transform: Custom callable transform. If None, uses PairedTransform.
            hazy_dir: Explicit hazy image directory (optional).
            clear_dir: Explicit clear/label image directory (optional).
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        self.split_ratios = split_ratios
        self.crop_size = crop_size
        self.seed = seed

        valid_splits = {"train", "val", "valid", "test", "all"}
        if self.split not in valid_splits:
            raise ValueError(f"Invalid split '{self.split}'. Must be one of {valid_splits}")
        # Normalize "valid" to "val"
        if self.split == "valid":
            self.split = "val"

        # Locate hazy directory
        if hazy_dir is not None:
            self.hazy_dir = Path(hazy_dir)
        else:
            self.hazy_dir = self._find_subfolder(self.root_dir, HAZY_FOLDER_NAMES)

        # Locate clear directory
        if clear_dir is not None:
            self.clear_dir = Path(clear_dir)
        else:
            self.clear_dir = self._find_subfolder(self.root_dir, CLEAR_FOLDER_NAMES)

        if not self.hazy_dir or not self.hazy_dir.is_dir():
            raise FileNotFoundError(f"Hazy image directory not found in {self.root_dir}. Expected one of {HAZY_FOLDER_NAMES}")
        if not self.clear_dir or not self.clear_dir.is_dir():
            raise FileNotFoundError(f"Clear image directory not found in {self.root_dir}. Expected one of {CLEAR_FOLDER_NAMES}")

        # Scan and pair matching images
        self.all_pairs = self._scan_and_match_pairs()
        if len(self.all_pairs) == 0:
            raise RuntimeError(f"No matching image pairs found between {self.hazy_dir} and {self.clear_dir}")

        # Perform deterministic split
        self.samples = self._get_split_samples()

        # Initialize transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = PairedTransform(
                crop_size=self.crop_size,
                is_train=(self.split == "train"),
            )

    @staticmethod
    def _find_subfolder(root: Path, candidates: List[str]) -> Optional[Path]:
        """Find the first matching candidate subfolder."""
        for name in candidates:
            p = root / name
            if p.is_dir():
                return p
        return None

    def _scan_and_match_pairs(self) -> List[Tuple[Path, Path]]:
        """Scan hazy and clear folders and match files by normalized stem."""
        hazy_files = {}
        for f in sorted(self.hazy_dir.iterdir()):
            if f.suffix.lower() in VALID_IMAGE_EXTS and not f.name.startswith("."):
                key = normalize_stem(f.stem)
                hazy_files[key] = f

        clear_files = {}
        for f in sorted(self.clear_dir.iterdir()):
            if f.suffix.lower() in VALID_IMAGE_EXTS and not f.name.startswith("."):
                key = normalize_stem(f.stem)
                clear_files[key] = f

        matched_keys = sorted(set(hazy_files.keys()) & set(clear_files.keys()))
        pairs = [(hazy_files[k], clear_files[k]) for k in matched_keys]
        return pairs

    def _get_split_samples(self) -> List[Tuple[Path, Path]]:
        """Partition pairs deterministically into train, val, and test splits."""
        total = len(self.all_pairs)
        if self.split == "all":
            return list(self.all_pairs)

        train_ratio, val_ratio, test_ratio = self.split_ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Split ratios must sum to 1.0"

        # Deterministic permutation using fixed seed
        rng = np.random.RandomState(self.seed)
        shuffled_indices = rng.permutation(total)

        n_train = int(round(total * train_ratio))
        n_val = int(round(total * val_ratio))

        train_indices = shuffled_indices[:n_train]
        val_indices = shuffled_indices[n_train : n_train + n_val]
        test_indices = shuffled_indices[n_train + n_val :]

        if self.split == "train":
            selected = train_indices
        elif self.split in ("val", "valid"):
            selected = val_indices
        elif self.split == "test":
            selected = test_indices
        else:
            selected = shuffled_indices

        return [self.all_pairs[i] for i in selected]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        hazy_path, clear_path = self.samples[idx]

        try:
            with Image.open(hazy_path) as h_img:
                hazy_pil = h_img.convert("RGB")
            with Image.open(clear_path) as c_img:
                clear_pil = c_img.convert("RGB")
        except Exception:
            # Fallback for upstream corrupted or 0-byte image files (e.g. 265.png)
            hazy_pil = Image.new("RGB", (512, 512), (128, 128, 128))
            clear_pil = Image.new("RGB", (512, 512), (128, 128, 128))

        hazy_tensor, clear_tensor = self.transform(hazy_pil, clear_pil)

        return {
            "hazy": hazy_tensor,
            "clean": clear_tensor,  # Standard baseline convention
            "clear": clear_tensor,  # Explicit clear tag
            "name": hazy_path.stem,
            "hazy_path": str(hazy_path),
            "clear_path": str(clear_path),
        }
