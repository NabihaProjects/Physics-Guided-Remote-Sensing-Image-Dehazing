"""Remote sensing dehazing dataset classes and data loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from src.physics.atmospheric_model import generate_nonuniform_transmission, synthesize_haze
from src.utils.config import ExperimentConfig


def create_synthetic_rs_sample(
    height: int = 256,
    width: int = 256,
    seed: int = 42,
) -> torch.Tensor:
    """Generate a synthetic clean remote sensing image patch with multi-terrain features.

    Simulates typical remote sensing land-cover patterns:
    - Green vegetation areas
    - Water bodies (deep blue/cyan)
    - Soil/desert (warm brown/ochre)
    - Built-up urban geometric patterns

    Args:
        height: Image height.
        width: Image width.
        seed: Random seed for procedural generation.

    Returns:
        Tensor of shape (3, height, width) with float values in [0, 1].
    """
    rng = np.random.RandomState(seed)

    # Base low-frequency terrain gradient
    y, x = np.mgrid[0:height, 0:width]
    base = np.sin(x / 40.0) * np.cos(y / 40.0)

    # 3-channel remote sensing spectral bands (RGB)
    r = 0.25 + 0.35 * (base > 0.0) + 0.15 * rng.rand(height, width)
    g = 0.45 + 0.30 * (base <= 0.0) + 0.10 * rng.rand(height, width)
    b = 0.20 + 0.25 * (base > 0.4) + 0.08 * rng.rand(height, width)

    # Add geometric field / building textures
    grid_mask = ((x // 32) % 2 == 0) & ((y // 32) % 2 == 0)
    r[grid_mask] += 0.15
    g[grid_mask] += 0.05
    b[grid_mask] += 0.10

    img = np.stack([r, g, b], axis=0).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return torch.from_numpy(img)


class RemoteSensingDehazeDataset(Dataset):
    """Dataset for remote sensing image dehazing supporting paired folders or dynamic synthesis."""

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        split: str = "train",
        image_size: int = 256,
        is_synthetic: bool = False,
        num_synthetic_samples: int = 100,
        beta_range: Tuple[float, float] = (0.04, 0.12),
        atmospheric_light_range: Tuple[float, float] = (0.75, 0.95),
        seed: int = 42,
        augment: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            root_dir: Path to directory containing clean/hazy subdirectories.
            split: Split name ("train", "val", "test").
            image_size: Spatial resolution to crop/resize to.
            is_synthetic: If True, dynamically synthesizes haze over clean images.
            num_synthetic_samples: Number of samples when using procedural synthesis.
            beta_range: Atmospheric scattering coefficient range for synthesis.
            atmospheric_light_range: Atmospheric light range [A_min, A_max].
            seed: Base random seed.
            augment: Whether to apply remote-sensing specific isometric augmentations.
        """
        super().__init__()
        self.root_dir = Path(root_dir) if root_dir else None
        self.split = split
        self.image_size = image_size
        self.is_synthetic = is_synthetic
        self.num_synthetic_samples = num_synthetic_samples
        self.beta_range = beta_range
        self.atmospheric_light_range = atmospheric_light_range
        self.seed = seed
        self.augment = augment and (split == "train")

        self.samples: List[Tuple[Optional[Path], Optional[Path]]] = []

        if not self.is_synthetic and self.root_dir and self.root_dir.exists():
            hazy_dir = self.root_dir / split / "hazy"
            clean_dir = self.root_dir / split / "clean"
            if hazy_dir.exists() and clean_dir.exists():
                hazy_files = sorted(list(hazy_dir.glob("*.png")) + list(hazy_dir.glob("*.jpg")))
                for hf in hazy_files:
                    cf = clean_dir / hf.name
                    if cf.exists():
                        self.samples.append((hf, cf))

        # Fallback to procedural synthesis if no files found
        if len(self.samples) == 0:
            self.is_synthetic = True

    def __len__(self) -> int:
        if self.is_synthetic:
            return self.num_synthetic_samples
        return len(self.samples)

    def _apply_augmentations(
        self,
        hazy: torch.Tensor,
        clean: torch.Tensor,
        transmission: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply isometric transformations (horizontal/vertical flip, 90-degree rotations)."""
        if torch.rand(1).item() > 0.5:
            hazy = TF.hflip(hazy)
            clean = TF.hflip(clean)
            if transmission is not None:
                transmission = TF.hflip(transmission)

        if torch.rand(1).item() > 0.5:
            hazy = TF.vflip(hazy)
            clean = TF.vflip(clean)
            if transmission is not None:
                transmission = TF.vflip(transmission)

        rot_choice = torch.randint(0, 4, (1,)).item()
        if rot_choice > 0:
            angle = float(rot_choice * 90)
            hazy = TF.rotate(hazy, angle)
            clean = TF.rotate(clean, angle)
            if transmission is not None:
                transmission = TF.rotate(transmission, angle)

        return hazy, clean, transmission

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        if self.is_synthetic:
            # Procedural sample generation
            sample_seed = self.seed + idx
            clean = create_synthetic_rs_sample(
                height=self.image_size,
                width=self.image_size,
                seed=sample_seed,
            )

            gen = torch.Generator().manual_seed(sample_seed)
            t_min = float(torch.empty(1).uniform_(0.2, 0.4, generator=gen).item())
            t_max = float(torch.empty(1).uniform_(0.8, 0.98, generator=gen).item())

            trans = generate_nonuniform_transmission(
                height=self.image_size,
                width=self.image_size,
                t_min=t_min,
                t_max=t_max,
                generator=gen,
            ).squeeze(0)  # Shape (1, H, W)

            a_val = float(
                torch.empty(1).uniform_(
                    self.atmospheric_light_range[0],
                    self.atmospheric_light_range[1],
                    generator=gen,
                ).item()
            )

            hazy = synthesize_haze(
                clean_image=clean.unsqueeze(0),
                transmission=trans.unsqueeze(0),
                atmospheric_light=a_val,
            ).squeeze(0)  # Shape (3, H, W)

            if self.augment:
                hazy, clean, trans = self._apply_augmentations(hazy, clean, trans)

            return {
                "hazy": hazy,
                "clean": clean,
                "transmission": trans,
                "atmospheric_light": torch.tensor([a_val, a_val, a_val], dtype=torch.float32),
                "name": f"syn_rs_{idx:04d}",
            }

        # Paired dataset mode
        hazy_path, clean_path = self.samples[idx]
        assert hazy_path is not None and clean_path is not None

        with Image.open(hazy_path) as h_img:
            hazy_tensor = TF.to_tensor(h_img.convert("RGB"))
        with Image.open(clean_path) as c_img:
            clean_tensor = TF.to_tensor(c_img.convert("RGB"))

        # Resize / Center crop
        hazy_tensor = TF.resize(hazy_tensor, [self.image_size, self.image_size], antialias=True)
        clean_tensor = TF.resize(clean_tensor, [self.image_size, self.image_size], antialias=True)

        trans_tensor = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)

        if self.augment:
            hazy_tensor, clean_tensor, _ = self._apply_augmentations(hazy_tensor, clean_tensor, None)

        return {
            "hazy": hazy_tensor,
            "clean": clean_tensor,
            "transmission": trans_tensor,
            "atmospheric_light": torch.tensor([0.85, 0.85, 0.85], dtype=torch.float32),
            "name": hazy_path.stem,
        }


def seed_worker(worker_id: int) -> None:
    """Worker initialization function for deterministic multi-process data loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def create_dataloaders(
    config: ExperimentConfig,
    train_samples: int = 120,
    val_samples: int = 30,
) -> Tuple[DataLoader, DataLoader]:
    """Create reproducible PyTorch training and validation DataLoaders.

    Args:
        config: Experiment configuration object.
        train_samples: Sample count for training when in synthetic mode.
        val_samples: Sample count for validation when in synthetic mode.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    g = torch.Generator()
    g.manual_seed(config.project.seed)

    train_ds = RemoteSensingDehazeDataset(
        root_dir=config.data.data_dir,
        split="train",
        image_size=config.data.image_size,
        num_synthetic_samples=train_samples,
        seed=config.project.seed,
        augment=True,
    )

    val_ds = RemoteSensingDehazeDataset(
        root_dir=config.data.data_dir,
        split="val",
        image_size=config.data.image_size,
        num_synthetic_samples=val_samples,
        seed=config.project.seed + 1000,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        worker_init_fn=seed_worker,
        drop_last=False,
        pin_memory=False,
    )

    return train_loader, val_loader
