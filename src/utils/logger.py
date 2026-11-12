"""Logging and experiment directory utilities."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

from src.utils.config import ExperimentConfig


def setup_logger(name: str = "rs_dehazing", log_file: Path | str | None = None) -> logging.Logger:
    """Configure a unified logger with console and optional file handlers.

    Args:
        name: Name of the logger.
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_experiment_dir(config: ExperimentConfig) -> Tuple[Path, logging.Logger]:
    """Create directory structure for an experiment run and persist the configuration.

    Directory structure:
        experiments/{experiment_name}_{timestamp}/
            ├── checkpoints/
            ├── visualizations/
            ├── metrics/
            ├── logs/
            │   └── run.log
            └── config.yaml

    Args:
        config: Validated ExperimentConfig instance.

    Returns:
        Tuple of (experiment_dir Path, configured Logger).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config.project.output_dir) / f"{config.project.experiment_name}_{timestamp}"

    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (exp_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Persist config snapshot
    config.save_yaml(exp_dir / "config.yaml")

    logger = setup_logger(
        name=f"exp_{config.project.experiment_name}",
        log_file=exp_dir / "logs" / "run.log",
    )
    logger.info("Initialized experiment directory at: %s", exp_dir)
    return exp_dir, logger

