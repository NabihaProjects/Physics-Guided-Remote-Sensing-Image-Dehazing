"""Unit tests for Step 2: Physical Simulation and Remote Sensing Dataset Pipeline."""

from __future__ import annotations

import pytest
import torch

from src.data.dataset import (
    RemoteSensingDehazeDataset,
    create_dataloaders,
    create_synthetic_rs_sample,
)
from src.physics.atmospheric_model import (
    AtmosphericScatteringModel,
    generate_nonuniform_transmission,
    invert_atmospheric_scattering,
    synthesize_haze,
)
from src.utils.config import ExperimentConfig


def test_synthetic_rs_sample():
    """Verify synthetic clean remote sensing image generation properties."""
    sample = create_synthetic_rs_sample(height=128, width=128, seed=42)
    assert sample.shape == (3, 128, 128)
    assert sample.dtype == torch.float32
    assert sample.min() >= 0.0
    assert sample.max() <= 1.0


def test_transmission_generation():
    """Verify non-uniform transmission map generation bounds and reproducibility."""
    gen1 = torch.Generator().manual_seed(99)
    t1 = generate_nonuniform_transmission(64, 64, t_min=0.25, t_max=0.90, generator=gen1)

    assert t1.shape == (1, 1, 64, 64)
    assert t1.min() >= 0.249
    assert t1.max() <= 0.901

    # Check reproducibility with identical seed
    gen2 = torch.Generator().manual_seed(99)
    t2 = generate_nonuniform_transmission(64, 64, t_min=0.25, t_max=0.90, generator=gen2)
    assert torch.allclose(t1, t2)


def test_asm_forward_and_inverse_consistency():
    """Verify that inverting the forward Koschmieder model recovers the original clean image."""
    clean = torch.tensor([[[[0.2, 0.6], [0.8, 0.4]]]], dtype=torch.float32).repeat(1, 3, 1, 1)
    trans = torch.tensor([[[[0.5, 0.7], [0.6, 0.8]]]], dtype=torch.float32)
    A = 0.9

    hazy = synthesize_haze(clean_image=clean, transmission=trans, atmospheric_light=A)
    assert hazy.shape == clean.shape
    assert hazy.min() >= 0.0
    assert hazy.max() <= 1.0

    # Invert using exact known parameters
    recovered = invert_atmospheric_scattering(
        hazy_image=hazy,
        transmission=trans,
        atmospheric_light=A,
        t0=0.05,
    )

    # Recovery should match clean within floating point tolerance
    assert torch.allclose(recovered, clean, atol=1e-5), "Physical inversion failed to reconstruct clean image"


def test_atmospheric_scattering_module():
    """Verify the nn.Module wrapper AtmosphericScatteringModel."""
    model = AtmosphericScatteringModel(t0=0.1)
    hazy = torch.rand(2, 3, 32, 32)
    trans = torch.rand(2, 1, 32, 32).clamp(0.2, 0.9)
    A = torch.tensor([[0.8, 0.8, 0.8], [0.85, 0.85, 0.85]])

    out = model(hazy, trans, A)
    assert out.shape == (2, 3, 32, 32)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_dataset_and_dataloaders():
    """Verify DataLoader batch creation, dictionary keys, and tensor validity."""
    config = ExperimentConfig()
    config.data.image_size = 64
    config.data.batch_size = 4
    config.hardware.num_workers = 0  # Single thread for quick test

    train_loader, val_loader = create_dataloaders(config, train_samples=8, val_samples=4)

    assert len(train_loader) == 2
    assert len(val_loader) == 1

    batch = next(iter(train_loader))
    assert "hazy" in batch
    assert "clean" in batch
    assert "transmission" in batch
    assert "atmospheric_light" in batch

    assert batch["hazy"].shape == (4, 3, 64, 64)
    assert batch["clean"].shape == (4, 3, 64, 64)
    assert batch["transmission"].shape == (4, 1, 64, 64)
    assert batch["atmospheric_light"].shape == (4, 3)

    assert batch["hazy"].min() >= 0.0 and batch["hazy"].max() <= 1.0
    assert batch["clean"].min() >= 0.0 and batch["clean"].max() <= 1.0

