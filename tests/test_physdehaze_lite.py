"""Unit tests for PhysDehaze-Lite architecture, loss components, and one-batch overfit."""

from __future__ import annotations

import yaml
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.losses.physdehaze_loss import PhysicsConsistencyLoss, PhysDehazeLoss
from src.models.physdehaze_lite import PhysDehazeLite


def test_parameter_count():
    """Verify PhysDehaze-Lite parameter budget is strictly under 1,000,000 parameters."""
    model = PhysDehazeLite(in_channels=3, base_channels=32)
    total_params, trainable_params = model.count_parameters()

    print(f"\n[PhysDehaze-Lite Parameter Summary]")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Budget Limit:         1,000,000")

    assert total_params < 1_000_000, f"Expected < 1M parameters, got {total_params:,}"
    assert trainable_params == total_params
    assert total_params == 323_703


def test_shape_and_tensors():
    """Verify PhysDehaze-Lite produces exact tensor dimensions for 4D, 3D, and non-divisible inputs."""
    model = PhysDehazeLite(in_channels=3, base_channels=32)
    model.eval()

    with torch.no_grad():
        # Standard 4D batch (B=2, C=3, H=128, W=128)
        x_batch = torch.rand(2, 3, 128, 128)
        pred_dict = model(x_batch, return_dict=True)

        assert isinstance(pred_dict, dict)
        assert "output" in pred_dict
        assert "transmission" in pred_dict
        assert "airlight" in pred_dict
        assert "j_phys" in pred_dict

        # Shape verifications
        assert pred_dict["output"].shape == (2, 3, 128, 128)
        assert pred_dict["transmission"].shape == (2, 1, 128, 128)
        assert pred_dict["airlight"].shape == (2, 3, 1, 1)
        assert pred_dict["j_phys"].shape == (2, 3, 128, 128)

        # Value range verifications
        assert pred_dict["output"].min() >= 0.0 and pred_dict["output"].max() <= 1.0
        assert pred_dict["transmission"].min() >= 0.0 and pred_dict["transmission"].max() <= 1.0
        assert pred_dict["airlight"].min() >= 0.0 and pred_dict["airlight"].max() <= 1.0
        assert pred_dict["j_phys"].min() >= 0.0 and pred_dict["j_phys"].max() <= 1.0

        # Simple tensor output (return_dict=False)
        simple_out = model(x_batch, return_dict=False)
        assert isinstance(simple_out, torch.Tensor)
        assert simple_out.shape == (2, 3, 128, 128)

        # Single 3D image (C=3, H=128, W=128)
        x_3d = torch.rand(3, 128, 128)
        out_3d = model(x_3d)
        assert out_3d.shape == (3, 128, 128)

        # Arbitrary non-divisible spatial resolution (e.g. 100x100)
        x_odd = torch.rand(1, 3, 100, 100)
        out_odd = model(x_odd)
        assert out_odd.shape == (1, 3, 100, 100)


def test_losses_switchability_and_yaml_config():
    """Verify all 4 losses (L1, SSIM, Physics, SAM) are individually switchable and parse YAML configs."""
    yaml_str = """
    losses:
      l1:
        enabled: true
        weight: 1.0
      ssim:
        enabled: false
        weight: 0.2
      physics:
        enabled: true
        weight: 0.5
      sam:
        enabled: false
        weight: 0.1
    """
    config_dict = yaml.safe_load(yaml_str)
    loss_module = PhysDehazeLoss(config=config_dict)

    assert loss_module.use_l1 is True
    assert loss_module.weight_l1 == 1.0
    assert loss_module.use_ssim is False
    assert loss_module.use_physics is True
    assert loss_module.weight_physics == 0.5
    assert loss_module.use_sam is False

    hazy = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)
    model = PhysDehazeLite(in_channels=3, base_channels=16)
    pred_dict = model(hazy, return_dict=True)

    loss_val, loss_breakdown = loss_module(pred_dict, target, hazy=hazy)
    assert "loss_l1" in loss_breakdown
    assert "loss_physics" in loss_breakdown
    assert "loss_ssim" not in loss_breakdown
    assert "loss_sam" not in loss_breakdown
    assert loss_val.item() > 0.0

    # Test SAM loss alone
    sam_only = PhysDehazeLoss(use_l1=False, use_ssim=False, use_physics=False, use_sam=True, weight_sam=1.0)
    loss_sam, breakdown_sam = sam_only(pred_dict, target)
    assert "loss_sam" in breakdown_sam
    assert loss_sam.item() > 0.0


def test_one_batch_overfit():
    """Verify that PhysDehaze-Lite can overfit a single batch in 100 steps (loss drops clearly)."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    model = PhysDehazeLite(in_channels=3, base_channels=16).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    criterion = PhysDehazeLoss(
        use_l1=True, weight_l1=1.0,
        use_ssim=True, weight_ssim=0.2,
        use_physics=True, weight_physics=0.5,
        use_sam=True, weight_sam=0.1,
    )

    # Physically grounded batch: clean ground truth J and synthetic hazy input I
    clean = torch.rand(2, 3, 64, 64, device=device)
    t_gt = 0.60
    A_gt = 0.85
    hazy = (clean * t_gt + A_gt * (1.0 - t_gt)).clamp(0.0, 1.0)

    initial_loss = 0.0
    final_loss = 0.0

    for step in range(100):
        optimizer.zero_grad()
        pred = model(hazy, return_dict=True)
        loss, _ = criterion(pred, clean, hazy=hazy)
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == 99:
            final_loss = loss.item()

    drop_percent = ((initial_loss - final_loss) / initial_loss) * 100.0

    print(f"\n[One-Batch Overfit Test (100 Steps)]")
    print(f"  Step 0 Loss:   {initial_loss:.4f}")
    print(f"  Step 100 Loss: {final_loss:.4f}")
    print(f"  Relative Drop: {drop_percent:.1f}%")

    # Loss must drop clearly by at least 70% in 100 steps
    assert final_loss < initial_loss * 0.30, (
        f"Loss did not drop sufficiently: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )
