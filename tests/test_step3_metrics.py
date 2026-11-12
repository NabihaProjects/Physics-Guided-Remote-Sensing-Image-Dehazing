"""Unit tests for Step 3: Metrics & Evaluation Engine."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.metrics.metrics import (
    calculate_ciede2000,
    calculate_psnr,
    calculate_ssim,
    MetricTracker,
)
from src.metrics.profiler import count_parameters, profile_model


def test_psnr_metrics():
    """Verify PSNR calculation behavior for identical and degraded pairs."""
    img1 = torch.full((3, 64, 64), 0.5)
    img2 = torch.full((3, 64, 64), 0.5)

    # Identical images must return capped 100.0 dB
    psnr_identical = calculate_psnr(img1, img2)
    assert psnr_identical == 100.0

    # Degraded image
    img_degraded = img1 + 0.1
    psnr_degraded = calculate_psnr(img_degraded, img1)
    # MSE = 0.01 -> 10 * log10(1 / 0.01) = 20 dB
    assert pytest.approx(psnr_degraded, rel=1e-2) == 20.0


def test_ssim_metrics():
    """Verify SSIM calculation behavior."""
    img1 = torch.rand(3, 64, 64)
    img2 = img1.clone()

    ssim_identical = calculate_ssim(img1, img2)
    assert pytest.approx(ssim_identical, abs=1e-4) == 1.0

    # Perturbed image
    img_noise = (img1 + 0.2 * torch.randn(3, 64, 64)).clamp(0.0, 1.0)
    ssim_noisy = calculate_ssim(img_noise, img1)
    assert ssim_noisy < 1.0
    assert ssim_noisy > -1.0


def test_ciede2000_spectral_metric():
    """Verify CIEDE2000 color distortion calculation."""
    img1 = torch.rand(3, 32, 32)
    img2 = img1.clone()

    delta_e_identical = calculate_ciede2000(img1, img2)
    assert pytest.approx(delta_e_identical, abs=1e-4) == 0.0

    # Shift red channel
    img_shifted = img1.clone()
    img_shifted[0] = (img_shifted[0] + 0.3).clamp(0.0, 1.0)
    delta_e_shifted = calculate_ciede2000(img_shifted, img1)
    assert delta_e_shifted > 0.0


def test_metric_tracker():
    """Verify MetricTracker accumulation and reporting."""
    tracker = MetricTracker()
    pred_batch = torch.rand(4, 3, 32, 32)
    target_batch = pred_batch.clone()

    tracker.update(pred_batch, target_batch, extra_metrics={"loss": 0.042})
    summary = tracker.summary()

    assert "psnr" in summary
    assert "ssim" in summary
    assert "ciede2000" in summary
    assert "loss" in summary

    assert summary["psnr"]["mean"] == 100.0
    assert pytest.approx(summary["ssim"]["mean"], abs=1e-3) == 1.0
    assert pytest.approx(summary["ciede2000"]["mean"], abs=1e-3) == 0.0
    assert summary["loss"]["mean"] == 0.042


def test_model_profiler():
    """Verify model parameter counting and latency profiling."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 3, kernel_size=3, padding=1),
    )

    total, trainable, size_mb = count_parameters(model)
    # Layer 1: 3 * 16 * 9 + 16 = 448
    # Layer 2: 16 * 3 * 9 + 3 = 435
    # Total = 883
    assert total == 883
    assert trainable == 883
    assert size_mb > 0.0

    profile = profile_model(model, input_shape=(1, 3, 32, 32), num_runs=5)
    assert "total_parameters" in profile
    assert "mean_latency_ms" in profile
    assert "fps" in profile
    assert profile["mean_latency_ms"] > 0.0

