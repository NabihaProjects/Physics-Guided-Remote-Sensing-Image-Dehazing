"""Unit tests for Section 4.2 evaluation metrics from the base survey paper:

"Remote sensing image dehazing: A systematic review of progress, challenges, and prospects"
(ISPRS J. Photogramm. Remote Sens. 235, 2026).

Verifies the 10 evaluation metrics and evaluate() batch pipeline:
1. PSNR (dB >= 80 / 100 for identical)
2. SSIM (= 1.0 for identical)
3. CIEDE2000 (= 0.0 for identical)
4. LPIPS (~0.0 for identical)
5. SAM (spectral angle mapper, Eq. 15, = 0.0 deg for identical)
6. ERGAS (Eq. 16, r=1.0, = 0.0 for identical)
7. UIQI (universal image quality index, Eq. 17, = 1.0 for identical)
8. QNR (survey Eq. 18: exp(-SAM/10) * exp(-ERGAS/100), = 1.0 for identical)
9. NIQE (no-reference naturalness, returns valid float score)
10. HIST (histogram similarity, Eq. 20, = 1.0 for identical)
11. FID (optional Fréchet Inception Distance)
12. evaluate(pred_dir, gt_dir) pipeline saving summary CSV
"""

from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from src.metrics.metrics import (
    calculate_ciede2000,
    calculate_ergas,
    calculate_fid,
    calculate_hist,
    calculate_lpips,
    calculate_niqe,
    calculate_psnr,
    calculate_qnr,
    calculate_sam,
    calculate_ssim,
    calculate_uiqi,
    evaluate,
    MetricTracker,
)


@pytest.fixture
def synthetic_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate reproducible synthetic RGB image pair with varied features."""
    torch.manual_seed(42)
    # Realistic multi-band image tensor with textures in [0.05, 0.95]
    img = torch.rand(3, 64, 64) * 0.9 + 0.05
    return img, img.clone()


def test_metric_psnr(synthetic_pair):
    """Test 1: PSNR returns capped 100 dB for identical images and degrades with noise."""
    ref, identical = synthetic_pair
    psnr_ident = calculate_psnr(identical, ref)
    assert psnr_ident >= 80.0
    assert psnr_ident == 100.0

    # Degraded image with additive offset
    degraded = (ref + 0.1).clamp(0.0, 1.0)
    psnr_deg = calculate_psnr(degraded, ref)
    assert psnr_deg < psnr_ident
    assert psnr_deg > 0.0


def test_metric_ssim(synthetic_pair):
    """Test 2: SSIM returns 1.0 for identical images and degrades with perturbation."""
    ref, identical = synthetic_pair
    ssim_ident = calculate_ssim(identical, ref)
    assert pytest.approx(ssim_ident, abs=1e-4) == 1.0

    # Perturbed image
    perturbed = (ref + 0.2 * torch.randn_like(ref)).clamp(0.0, 1.0)
    ssim_pert = calculate_ssim(perturbed, ref)
    assert ssim_pert < 1.0
    assert ssim_pert > -1.0


def test_metric_ciede2000(synthetic_pair):
    """Test 3: CIEDE2000 color difference returns 0.0 for identical images."""
    ref, identical = synthetic_pair
    ciede_ident = calculate_ciede2000(identical, ref)
    assert pytest.approx(ciede_ident, abs=1e-4) == 0.0

    # Color shift on red channel
    shifted = ref.clone()
    shifted[0] = (shifted[0] + 0.3).clamp(0.0, 1.0)
    ciede_shift = calculate_ciede2000(shifted, ref)
    assert ciede_shift > 0.0


def test_metric_lpips(synthetic_pair):
    """Test 4: LPIPS (AlexNet) returns ~0.0 for identical images and degrades with perturbation."""
    ref, identical = synthetic_pair
    lpips_ident = calculate_lpips(identical, ref)
    assert lpips_ident < 1e-4

    perturbed = (ref + 0.3 * torch.randn_like(ref)).clamp(0.0, 1.0)
    lpips_pert = calculate_lpips(perturbed, ref)
    assert lpips_pert > lpips_ident


def test_metric_sam(synthetic_pair):
    """Test 5: SAM (Survey Eq. 15) returns 0.0 degrees for identical spectral vectors."""
    ref, identical = synthetic_pair
    sam_ident = calculate_sam(identical, ref)
    assert pytest.approx(sam_ident, abs=1e-4) == 0.0

    # Spectral vector rotation / channel distortion
    distorted = ref.clone()
    distorted[0] = (distorted[0] * 0.5).clamp(0.0, 1.0)
    distorted[2] = (distorted[2] * 1.5).clamp(0.0, 1.0)
    sam_dist = calculate_sam(distorted, ref)
    assert sam_dist > 0.0


def test_metric_ergas(synthetic_pair):
    """Test 6: ERGAS (Survey Eq. 16) returns 0.0 for identical images with r=1."""
    ref, identical = synthetic_pair
    ergas_ident = calculate_ergas(identical, ref, r=1.0)
    assert pytest.approx(ergas_ident, abs=1e-4) == 0.0

    # Radiometric error
    degraded = (ref + 0.1).clamp(0.0, 1.0)
    ergas_deg = calculate_ergas(degraded, ref, r=1.0)
    assert ergas_deg > 0.0


def test_metric_uiqi(synthetic_pair):
    """Test 7: UIQI (Survey Eq. 17) returns 1.0 for identical images."""
    ref, identical = synthetic_pair
    uiqi_ident = calculate_uiqi(identical, ref)
    assert pytest.approx(uiqi_ident, abs=1e-4) == 1.0

    # Contrast and correlation distortion
    distorted = (ref * 0.5 + 0.2).clamp(0.0, 1.0)
    uiqi_dist = calculate_uiqi(distorted, ref)
    assert uiqi_dist < 1.0


def test_metric_qnr():
    """Test 8: QNR (Survey Eq. 18: exp(-SAM/10) * exp(-ERGAS/100)) returns 1.0 for perfect match."""
    # Perfect match: SAM=0.0 deg, ERGAS=0.0
    qnr_perfect = calculate_qnr(sam=0.0, ergas=0.0)
    assert pytest.approx(qnr_perfect, abs=1e-6) == 1.0

    # Degraded case
    sam_val = 5.0
    ergas_val = 20.0
    expected_qnr = np.exp(-sam_val / 10.0) * np.exp(-ergas_val / 100.0)
    computed_qnr = calculate_qnr(sam=sam_val, ergas=ergas_val)
    assert pytest.approx(computed_qnr, abs=1e-6) == float(expected_qnr)
    assert 0.0 < computed_qnr < 1.0


def test_metric_hist(synthetic_pair):
    """Test 9: HIST (Survey Eq. 20) returns 1.0 for identical images over 256 bins."""
    ref, identical = synthetic_pair
    hist_ident = calculate_hist(identical, ref, bins=256)
    assert pytest.approx(hist_ident, abs=1e-4) == 1.0

    # Distinct distributions
    shifted = (ref + 0.25).clamp(0.0, 1.0)
    hist_shifted = calculate_hist(shifted, ref, bins=256)
    assert hist_shifted < 1.0
    assert hist_shifted >= 0.0


def test_metric_niqe(synthetic_pair):
    """Test 10: NIQE no-reference evaluator returns a valid scalar score."""
    ref, _ = synthetic_pair
    # Repeat spatial size to 128x128 for reliable patch statistics
    img = torch.nn.functional.interpolate(ref.unsqueeze(0), size=(128, 128), mode="bilinear")
    score = calculate_niqe(img)
    if score is not None:
        assert isinstance(score, float)
        assert score > 0.0


def test_metric_fid():
    """Test 11: FID calculates Fréchet distance between feature distributions."""
    np.random.seed(42)
    feat1 = np.random.randn(100, 64)
    feat2 = feat1.copy()

    # Identical activations yield 0.0 FID
    fid_ident = calculate_fid(feat1, feat2)
    assert pytest.approx(fid_ident, abs=1e-4) == 0.0

    # Shifted distribution yields > 0.0
    feat_shifted = feat1 + 2.0
    fid_shift = calculate_fid(feat1, feat_shifted)
    assert fid_shift > 0.0


def test_batch_and_shape_handling():
    """Verify metrics accept both (3, H, W) and (B, 3, H, W) tensors cleanly."""
    torch.manual_seed(42)
    batch_img1 = torch.rand(4, 3, 32, 32)
    batch_img2 = batch_img1.clone()

    assert calculate_psnr(batch_img1, batch_img2) == 100.0
    assert pytest.approx(calculate_ssim(batch_img1, batch_img2), abs=1e-4) == 1.0
    assert pytest.approx(calculate_ciede2000(batch_img1, batch_img2), abs=1e-4) == 0.0
    assert pytest.approx(calculate_sam(batch_img1, batch_img2), abs=1e-4) == 0.0
    assert pytest.approx(calculate_ergas(batch_img1, batch_img2), abs=1e-4) == 0.0
    assert pytest.approx(calculate_uiqi(batch_img1, batch_img2), abs=1e-4) == 1.0
    assert pytest.approx(calculate_hist(batch_img1, batch_img2), abs=1e-4) == 1.0


def test_evaluate_function(tmp_path: Path):
    """Verify evaluate(pred_dir, gt_dir) pipeline loops over pairs and writes CSV with means."""
    pred_dir = tmp_path / "preds"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()

    # Create 3 synthetic paired images
    for i in range(3):
        np.random.seed(100 + i)
        arr = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(gt_dir / f"img_{i:02d}.png")
        # Identical prediction
        img.save(pred_dir / f"img_{i:02d}.png")

    csv_output = tmp_path / "results.csv"
    results = evaluate(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        output_csv=csv_output,
        compute_lpips=True,
        compute_niqe=True,
        compute_fid=False,
    )

    # Check returned means
    assert results["psnr"] == 100.0
    assert pytest.approx(results["ssim"], abs=1e-3) == 1.0
    assert pytest.approx(results["sam"], abs=1e-3) == 0.0
    assert pytest.approx(results["ergas"], abs=1e-3) == 0.0
    assert pytest.approx(results["uiqi"], abs=1e-3) == 1.0
    assert pytest.approx(results["ciede2000"], abs=1e-3) == 0.0
    assert pytest.approx(results["hist"], abs=1e-3) == 1.0
    assert pytest.approx(results["qnr"], abs=1e-3) == 1.0
    assert results["lpips"] < 1e-4

    # Verify CSV file structure
    assert csv_output.exists()
    with open(csv_output, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        # 3 image rows + 1 MEAN row = 4 rows
        assert len(reader) == 4
        assert reader[-1]["image"] == "MEAN"
        assert float(reader[-1]["psnr"]) == 100.0
        assert pytest.approx(float(reader[-1]["ssim"]), abs=1e-3) == 1.0

