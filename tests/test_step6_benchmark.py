"""Integration test for Step 6: Benchmark Runner and Reporting."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from scripts.benchmark import run_benchmark
from src.utils.config import ExperimentConfig


def test_end_to_end_benchmark_run():
    """Verify end-to-end execution of benchmark suite across all methods."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = ExperimentConfig()
        cfg.data.image_size = 48
        cfg.model.base_channels = 16

        results = run_benchmark(
            config=cfg,
            num_test_samples=4,
            output_dir=tmp_dir,
            checkpoint_dir=None,
        )

        assert len(results) == 4, f"Expected 4 evaluated methods, got {len(results)}"

        methods = [r["method"] for r in results]
        assert "Hazy Input (Unprocessed)" in methods
        assert "RS-DCP (He et al. Adapted)" in methods
        assert "AOD-Net (ICCV 2017)" in methods
        assert "RS-PGNet (Proposed Novel)" in methods

        # Check file outputs
        out_path = Path(tmp_dir)
        assert (out_path / "benchmark_results.csv").is_file()
        assert (out_path / "benchmark_results.json").is_file()
        assert (out_path / "benchmark_summary.md").is_file()
        assert (out_path / "benchmark_comparison.png").is_file()

        # Check values
        for r in results:
            assert r["psnr"] > 0.0
            assert 0.0 <= r["ssim"] <= 1.0
            assert r["ciede2000"] >= 0.0

