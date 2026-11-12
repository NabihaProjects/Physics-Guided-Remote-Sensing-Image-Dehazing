"""Reproducibility module for deterministic execution across random number generators."""

from __future__ import annotations

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set seeds across Python random, NumPy, and PyTorch for full reproducibility.

    Args:
        seed: The integer seed to set. Default is 42.
        deterministic: If True, sets cuDNN to deterministic mode.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms where available in PyTorch
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass


def get_device(preferred: str = "auto") -> torch.device:
    """Select the appropriate PyTorch device based on user preference and hardware.

    Args:
        preferred: One of "auto", "cuda", "mps", "cpu".

    Returns:
        torch.device instance representing the chosen device.
    """
    preferred = preferred.lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")

    # Auto-detection
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

