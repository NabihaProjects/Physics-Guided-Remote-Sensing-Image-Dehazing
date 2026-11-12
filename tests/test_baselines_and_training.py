"""Unit tests for baseline models (DCP and AOD-Net) and the training pipeline."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.aodnet import AODNet
from src.models.dcp import DCP, RSDCP, guided_filter
from src.train.train import create_model, evaluate_val_split, get_training_device, train_model


def test_dcp_baseline_specifications():
    """Verify DCP baseline conforms to classical He et al. 2010 specs: patch=15, omega=0.95, t0=0.1."""
    dcp = DCP(patch_size=15, omega=0.95, t0=0.10)

    # Must be non-learning (0 parameters)
    params = list(dcp.parameters())
    assert len(params) == 0

    assert dcp.patch_size == 15
    assert dcp.omega == 0.95
    assert dcp.t0 == 0.10

    # Test 4D input
    hazy_4d = torch.rand(2, 3, 64, 64)
    dehazed_4d = dcp(hazy_4d)
    assert dehazed_4d.shape == (2, 3, 64, 64)
    assert dehazed_4d.min() >= 0.0
    assert dehazed_4d.max() <= 1.0

    # Test 3D input
    hazy_3d = torch.rand(3, 64, 64)
    dehazed_3d = dcp(hazy_3d)
    assert dehazed_3d.shape == (3, 64, 64)
    assert dehazed_3d.min() >= 0.0
    assert dehazed_3d.max() <= 1.0


def test_aodnet_baseline_under_10k_parameters():
    """Verify AOD-Net has fewer than 10k parameters and follows J = K*I - K + b."""
    model = AODNet(in_channels=3, b=1.0)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Strictly under 10k parameters
    assert total_params < 10000
    assert trainable_params == 1761

    # Forward pass
    hazy = torch.rand(4, 3, 128, 128)
    dehazed = model(hazy)
    assert dehazed.shape == (4, 3, 128, 128)
    assert dehazed.min() >= 0.0
    assert dehazed.max() <= 1.0

    # Gradient flow with L1 loss
    target = torch.rand(4, 3, 128, 128)
    loss = nn.L1Loss()(dehazed, target)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"


def test_common_interface():
    """Verify both baselines share the common interface forward(hazy) -> dehazed in [0, 1]."""
    models = [
        DCP(patch_size=15, omega=0.95, t0=0.10),
        AODNet(in_channels=3, b=1.0),
    ]

    for model in models:
        # Batch mode
        x_batch = torch.rand(2, 3, 32, 32)
        out_batch = model(x_batch)
        assert out_batch.shape == x_batch.shape
        assert out_batch.min() >= 0.0
        assert out_batch.max() <= 1.0

        # Single image mode
        x_single = torch.rand(3, 32, 32)
        out_single = model(x_single)
        assert out_single.shape == x_single.shape
        assert out_single.min() >= 0.0
        assert out_single.max() <= 1.0


class DummyDictDataset(torch.utils.data.Dataset):
    """Synthetic dataset yielding dictionary format matching SateHaze1kDataset."""
    def __init__(self, count: int = 16, size: tuple = (128, 128)):
        torch.manual_seed(42)
        self.hazy = torch.rand(count, 3, size[0], size[1], dtype=torch.float32)
        self.clean = (self.hazy * 0.8 + 0.1).clamp(0.0, 1.0)

    def __len__(self):
        return len(self.hazy)

    def __getitem__(self, idx):
        return {
            "hazy": self.hazy[idx],
            "clean": self.clean[idx],
            "name": f"sample_{idx:03d}",
        }


def test_train_model_and_checkpoint_resume(tmp_path: Path):
    """Verify training loop, checkpoint saving, and resume functionality."""
    train_ds = DummyDictDataset(count=16, size=(64, 64))
    val_ds = DummyDictDataset(count=8, size=(64, 64))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    model = AODNet(in_channels=3)
    out_dir = tmp_path / "exp_test"

    # Train for 2 epochs
    res1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=out_dir,
        epochs=2,
        learning_rate=1e-3,
        device=torch.device("cpu"),
        log_tensorboard=True,
    )

    assert res1["best_val_psnr"] > 0.0
    assert len(res1["epoch_durations"]) == 2
    assert (out_dir / "checkpoints" / "epoch_001.pth").exists()
    assert (out_dir / "checkpoints" / "epoch_002.pth").exists()
    assert (out_dir / "checkpoints" / "latest_checkpoint.pth").exists()

    # Test resume for 1 more epoch (total 3 epochs)
    res2 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=out_dir,
        epochs=3,
        learning_rate=1e-3,
        device=torch.device("cpu"),
        resume_path=out_dir / "checkpoints" / "latest_checkpoint.pth",
        log_tensorboard=False,
    )

    assert len(res2["epoch_durations"]) == 1  # Executed epoch 3
    assert (out_dir / "checkpoints" / "epoch_003.pth").exists()

