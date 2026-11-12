"""Unit tests for Step 5: Novel Physics-Guided Architecture (RS-PGNet)."""

from __future__ import annotations

import pytest
import torch

from src.models.rs_pgnet import (
    GatedSpatialBlock,
    PhysicsEstimationBranch,
    RSPGNet,
    SpectralAttention,
)
from src.metrics.profiler import count_parameters


def test_spectral_attention():
    """Verify Spectral Attention module preserves shape and operates per-channel."""
    sam = SpectralAttention(channels=16)
    x = torch.rand(2, 16, 24, 24)
    out = sam(x)
    assert out.shape == x.shape
    assert out.min() >= 0.0


def test_gated_spatial_block():
    """Verify Gated Spatial Convolutional Block execution and residual connection."""
    block = GatedSpatialBlock(channels=16)
    x = torch.rand(2, 16, 24, 24)
    out = block(x)
    assert out.shape == x.shape


def test_physics_estimation_branch():
    """Verify physical transmission and atmospheric light estimation bounds."""
    branch = PhysicsEstimationBranch(in_channels=3, base_channels=16)
    x = torch.rand(2, 3, 32, 32)
    t, a = branch(x)

    assert t.shape == (2, 1, 32, 32)
    assert a.shape == (2, 3, 1, 1)

    # Physics bounds check
    assert t.min() >= 0.05 and t.max() <= 1.0
    assert a.min() >= 0.40 and a.max() <= 1.0


def test_rs_pgnet_forward_and_parameter_budget():
    """Verify end-to-end RS-PGNet forward pass and parameter budget (< 1.8M params)."""
    model = RSPGNet(in_channels=3, base_channels=32, num_refine_blocks=3)

    total_params, trainable_params, size_mb = count_parameters(model)
    print(f"\n[RS-PGNet] Total params: {total_params:,} ({size_mb:.2f} MB)")

    # Hard constraint: must be well below 1.8M params for Colab T4 / CPU feasibility
    assert total_params < 1_800_000, f"Model parameter count {total_params} exceeds 1.8M budget"
    assert trainable_params == total_params

    dummy_hazy = torch.rand(2, 3, 64, 64)
    res = model(dummy_hazy, return_dict=True)

    assert isinstance(res, dict)
    for key in ["output", "physical", "transmission", "atmospheric_light", "residual"]:
        assert key in res

    assert res["output"].shape == (2, 3, 64, 64)
    assert res["output"].min() >= 0.0 and res["output"].max() <= 1.0
    assert res["transmission"].shape == (2, 1, 64, 64)
    assert res["atmospheric_light"].shape == (2, 3, 1, 1)


def test_rs_pgnet_gradient_flow():
    """Verify backward gradient flow into both physics and refinement branches."""
    model = RSPGNet(in_channels=3, base_channels=16, num_refine_blocks=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dummy_hazy = torch.rand(2, 3, 32, 32)
    dummy_clean = torch.rand(2, 3, 32, 32)

    res = model(dummy_hazy, return_dict=True)
    loss = torch.nn.functional.l1_loss(res["output"], dummy_clean)

    optimizer.zero_grad()
    loss.backward()

    # Verify gradients reach physics branch
    physics_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.physics_branch.parameters()
    )
    assert physics_has_grad, "Physics branch did not receive backpropagation gradients"

    # Verify gradients reach refinement branch
    refine_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.refine_branch.parameters()
    )
    assert refine_has_grad, "Refinement branch did not receive backpropagation gradients"

