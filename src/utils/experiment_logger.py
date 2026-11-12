"""Unified experiment tracking logger supporting TensorBoard and Weights & Biases."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class ExperimentLogger:
    """Manages experiment metric logging across TensorBoard, Weights & Biases, and disk."""

    def __init__(
        self,
        log_dir: Path | str,
        experiment_name: str = "rs_dehazing",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize unified logger.

        Args:
            log_dir: Root directory for logs and TensorBoard events.
            experiment_name: Identifier for the current experiment run.
            use_tensorboard: Whether to log metrics and images to TensorBoard.
            use_wandb: Whether to initialize and stream metrics to Weights & Biases.
            config: Full configuration dictionary to log.
            logger: Standard python logger for console output.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.logger = logger or logging.getLogger("ExperimentLogger")

        # TensorBoard initialization
        self.tb_writer: Optional[Any] = None
        if use_tensorboard and HAS_TENSORBOARD:
            tb_path = self.log_dir / "tensorboard"
            tb_path.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_path))
            self.logger.info("TensorBoard logging initialized at: %s", tb_path)

        # Weights & Biases initialization
        self.wandb_run = None
        if use_wandb:
            if HAS_WANDB:
                self.wandb_run = wandb.init(
                    project="rs-dehazing-benchmark",
                    name=experiment_name,
                    config=config,
                    dir=str(self.log_dir),
                )
                self.logger.info("Weights & Biases logging initialized for run: %s", experiment_name)
            else:
                self.logger.warning("wandb requested but not installed. Skipping W&B tracking.")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric."""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, global_step=step)

        if self.wandb_run is not None:
            wandb.log({tag: value}, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """Log a collection of metrics with an optional prefix (e.g. 'train/' or 'val/')."""
        for k, v in metrics.items():
            tag = f"{prefix}/{k}" if prefix else k
            self.log_scalar(tag, float(v), step=step)

    def log_image(self, tag: str, image_tensor: torch.Tensor, step: int) -> None:
        """Log an image tensor of shape (C, H, W) in [0, 1]."""
        clamped = torch.clamp(image_tensor, 0.0, 1.0)
        if self.tb_writer is not None:
            self.tb_writer.add_image(tag, clamped, global_step=step)

        if self.wandb_run is not None:
            np_img = clamped.permute(1, 2, 0).cpu().numpy()
            wandb.log({tag: wandb.Image(np_img)}, step=step)

    def close(self) -> None:
        """Flush and close all logging handles."""
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()

        if self.wandb_run is not None:
            wandb.finish()

