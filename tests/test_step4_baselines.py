"""Unit tests for Step 4: Benchmark Baselines (RS-DCP and AOD-Net)."""

from __future__ import annotations

import tempfile
import pytest
import torch

from src.losses.compound_loss import CompoundDehazeLoss
from src.models.aodnet import AODNet
from src.models.dcp import RSDCP
from src.utils.config import ExperimentConfig


def test_rs_dcp_baseline_forward():
    """Verify Remote Sensing Dark Channel Prior forward pass."""
    model = RSDCP(patch_size=7, filter_radius=5)
    dummy_hazy = torch.rand(2, 3, 48, 48)

    dehazed = model(dummy_hazy)
    assert dehazed.shape == (2, 3, 48, 48)
    assert dehazed.min() >= 0.0
    assert dehazed.max() <= 1.0


def test_aod_net_forward_and_backward():
    """Verify AOD-Net forward pass, backward gradient pass, and parameter updates."""
    model = AODNet(in_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss()

    hazy = torch.rand(2, 3, 32, 32)
    clean = torch.rand(2, 3, 32, 32)

    # Initial forward pass
    pred = model(hazy)
    assert pred.shape == (2, 3, 32, 32)
    assert pred.min() >= 0.0
    assert pred.max() <= 1.0

    loss1 = criterion(pred, clean)
    optimizer.zero_grad()
    loss1.backward()

    # Check gradients exist
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grads, "AOD-Net parameters did not receive gradients"

    optimizer.step()

    # Second pass should reflect update
    pred2 = model(hazy)
    loss2 = criterion(pred2, clean)
    assert loss2.item() < loss1.item() or torch.isclose(loss2, loss1, atol=1e-2)


def test_compound_loss_components():
    """Verify CompoundDehazeLoss computation of L1, SSIM, SAM, and TV."""
    loss_fn = CompoundDehazeLoss()

    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)
    trans = torch.rand(2, 1, 32, 32, requires_grad=True)

    loss, details = loss_fn(pred, target, predicted_transmission=trans)
    assert isinstance(loss, torch.Tensor)
    assert "loss_l1" in details
    assert "loss_ssim" in details
    assert "loss_spectral" in details
    assert "loss_tv" in details
    assert loss.item() > 0.0

    # Test backward pass through loss
    loss.backward()
    assert pred.grad is not None
    assert trans.grad is not None

