"""Unit tests for the complete repo structure, TensorBoard logger, and Evaluator."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import torch

from src.datasets.dataset import RemoteSensingDehazeDataset
from src.eval.evaluator import Evaluator
from src.models.rs_pgnet import RSPGNet
from src.utils.config import ExperimentConfig
from src.utils.experiment_logger import HAS_TENSORBOARD, ExperimentLogger


def test_experiment_logger_tensorboard():
    """Verify ExperimentLogger logs scalars and image tensors to TensorBoard events."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        exp_logger = ExperimentLogger(
            log_dir=tmp_dir,
            experiment_name="test_tb_run",
            use_tensorboard=True,
            use_wandb=False,
        )

        assert HAS_TENSORBOARD, "TensorBoard is not available in environment"
        assert exp_logger.tb_writer is not None

        exp_logger.log_scalar("train/loss", 0.1234, step=1)
        exp_logger.log_metrics({"psnr": 22.5, "ssim": 0.85}, step=1, prefix="val")

        dummy_img = torch.rand(3, 32, 32)
        exp_logger.log_image("val/sample", dummy_img, step=1)
        exp_logger.close()

        # Verify TensorBoard event file exists
        tb_dir = Path(tmp_dir) / "tensorboard"
        assert tb_dir.is_dir()
        event_files = list(tb_dir.glob("events.out.tfevents.*"))
        assert len(event_files) >= 1, "TensorBoard events file was not generated"


def test_evaluator_engine():
    """Verify Evaluator runs across test dataset and outputs metrics JSON."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = ExperimentConfig()
        cfg.data.image_size = 32
        model = RSPGNet(in_channels=3, base_channels=16, num_refine_blocks=1)

        test_ds = RemoteSensingDehazeDataset(
            root_dir=None,
            split="test",
            image_size=32,
            num_synthetic_samples=2,
            seed=100,
            augment=False,
        )

        evaluator = Evaluator(model=model, device=torch.device("cpu"), output_dir=tmp_dir)
        metrics = evaluator.evaluate_dataset(test_ds, save_images=True)

        assert "psnr" in metrics
        assert "ssim" in metrics
        assert "ciede2000" in metrics
        assert (Path(tmp_dir) / "eval_metrics.json").is_file()
        assert (Path(tmp_dir) / "dehazed_images").is_dir()
        dehazed_files = list((Path(tmp_dir) / "dehazed_images").glob("*.png"))
        assert len(dehazed_files) == 2


def test_scripts_exist_and_executable():
    """Verify Makefile, training script, evaluation script, and demo script exist."""
    assert Path("Makefile").is_file()
    assert Path("scripts/train.py").is_file()
    assert Path("scripts/eval.py").is_file()
    assert Path("scripts/benchmark.py").is_file()
    assert Path("scripts/demo.py").is_file()
    assert Path("app/streamlit_app.py").is_file()
    assert Path("run_train.sh").is_file()
    assert Path("run_eval.sh").is_file()
    assert Path("run_demo.sh").is_file()

