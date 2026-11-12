"""Unit tests for Step 1: Foundation and Reproducibility Infrastructure."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.config import ExperimentConfig, load_config
from src.utils.logger import setup_experiment_dir, setup_logger
from src.utils.seed import get_device, set_seed


def test_seed_determinism_python_and_numpy():
    """Verify that set_seed ensures exact reproducibility in Python random and NumPy."""
    set_seed(12345)
    r1 = [random.random() for _ in range(5)]
    n1 = np.random.rand(5).tolist()

    # Re-seed and check identical generation
    set_seed(12345)
    r2 = [random.random() for _ in range(5)]
    n2 = np.random.rand(5).tolist()

    assert r1 == r2, "Python random generator is not deterministic across seed resets"
    assert n1 == n2, "NumPy random generator is not deterministic across seed resets"


def test_seed_determinism_pytorch():
    """Verify that set_seed ensures exact reproducibility in PyTorch tensor generation."""
    set_seed(777)
    t1 = torch.randn(4, 4)

    set_seed(777)
    t2 = torch.randn(4, 4)

    assert torch.equal(t1, t2), "PyTorch random tensor generation is not deterministic"


def test_get_device():
    """Verify device selection logic returns a valid torch.device."""
    device_auto = get_device("auto")
    device_cpu = get_device("cpu")

    assert isinstance(device_auto, torch.device)
    assert isinstance(device_cpu, torch.device)
    assert device_cpu.type == "cpu"


def test_config_loading_and_types():
    """Verify loading default YAML config into typed ExperimentConfig dataclass."""
    config_path = Path("configs/default.yaml")
    assert config_path.exists(), "configs/default.yaml does not exist"

    cfg = load_config(config_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.project.seed == 42
    assert cfg.logging.use_tensorboard is True
    assert cfg.model.in_channels == 3
    assert cfg.model.out_channels == 3
    assert isinstance(cfg.data.haze_simulation.beta_range, list)
    assert len(cfg.data.haze_simulation.beta_range) == 2


def test_config_overrides():
    """Verify that runtime overrides correctly update config parameters."""
    config_path = Path("configs/default.yaml")
    overrides = {
        "project.seed": 999,
        "training.epochs": 50,
        "model.base_channels": 64,
    }
    cfg = load_config(config_path, overrides=overrides)
    assert cfg.project.seed == 999
    assert cfg.training.epochs == 50
    assert cfg.model.base_channels == 64


def test_experiment_directory_and_logging():
    """Verify experiment directory structure creation and logger initialization."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg = load_config("configs/default.yaml")
        cfg.project.output_dir = tmp_dir
        cfg.project.experiment_name = "test_run"

        exp_dir, logger = setup_experiment_dir(cfg)

        assert exp_dir.exists()
        assert (exp_dir / "checkpoints").is_dir()
        assert (exp_dir / "visualizations").is_dir()
        assert (exp_dir / "metrics").is_dir()
        assert (exp_dir / "logs").is_dir()
        assert (exp_dir / "config.yaml").is_file()

        logger.info("Step 1 foundation test message logged successfully.")

