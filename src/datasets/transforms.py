"""Synchronized paired image transformations for remote sensing dehazing datasets."""

from __future__ import annotations

import random
from typing import Optional, Tuple
from PIL import Image
import torch
import torchvision.transforms.functional as TF


class PairedTransform:
    """Applies spatially synchronized transformations to paired hazy and clear images.

    Ensures that identical spatial crop windows, flips, and rotations are applied
    simultaneously to both hazy and clear remote sensing scenes.
    """

    def __init__(
        self,
        crop_size: Optional[Tuple[int, int]] = (256, 256),
        is_train: bool = True,
        flip_prob: float = 0.5,
        rot90_prob: float = 0.5,
    ) -> None:
        """Initialize paired transform.

        Args:
            crop_size: (height, width) for random cropping during training or center cropping
                       during evaluation. If None, preserves original dimensions.
            is_train: If True, applies random cropping, flips, and 90-degree rotations.
            flip_prob: Probability of horizontal / vertical flips.
            rot90_prob: Probability of applying 90/180/270-degree rotation.
        """
        self.crop_size = crop_size
        self.is_train = is_train
        self.flip_prob = flip_prob
        self.rot90_prob = rot90_prob

    def __call__(
        self,
        hazy: Image.Image,
        clear: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform a pair of PIL Images into normalized PyTorch tensors in [0, 1].

        Args:
            hazy: Hazy remote sensing PIL Image (RGB).
            clear: Clear ground-truth remote sensing PIL Image (RGB).

        Returns:
            Tuple of (hazy_tensor, clear_tensor), each shape (3, H, W) in float32 [0.0, 1.0].
        """
        assert hazy.size == clear.size, f"Size mismatch: hazy {hazy.size} vs clear {clear.size}"

        w, h = hazy.size

        # 1. Paired Cropping
        if self.crop_size is not None:
            th, tw = self.crop_size
            if w >= tw and h >= th:
                if self.is_train:
                    # Synchronized random crop coordinates
                    i = random.randint(0, h - th)
                    j = random.randint(0, w - tw)
                    hazy = TF.crop(hazy, i, j, th, tw)
                    clear = TF.crop(clear, i, j, th, tw)
                else:
                    # Synchronized center crop
                    hazy = TF.center_crop(hazy, (th, tw))
                    clear = TF.center_crop(clear, (th, tw))

        # 2. Synchronized Isometric Augmentations (Train mode only)
        if self.is_train:
            # Horizontal flip
            if random.random() < self.flip_prob:
                hazy = TF.hflip(hazy)
                clear = TF.hflip(clear)

            # Vertical flip
            if random.random() < self.flip_prob:
                hazy = TF.vflip(hazy)
                clear = TF.vflip(clear)

            # 90-degree isometric rotations
            if random.random() < self.rot90_prob:
                angle = random.choice([90, 180, 270])
                hazy = TF.rotate(hazy, angle)
                clear = TF.rotate(clear, angle)

        # 3. Convert PIL Image to PyTorch Tensor [0.0, 1.0] float32
        hazy_t = TF.to_tensor(hazy)
        clear_t = TF.to_tensor(clear)

        return hazy_t, clear_t

