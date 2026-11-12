"""Hardware efficiency and latency profiler for remote sensing dehazing models."""

from __future__ import annotations

import time
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """Count total and trainable parameters of a PyTorch model.

    Args:
        model: PyTorch nn.Module.

    Returns:
        Tuple of (total_params, trainable_params, size_in_mb).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total * 4.0) / (1024.0 * 1024.0)  # Assuming float32
    return total, trainable, size_mb


def measure_inference_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    device: torch.device | None = None,
    num_warmup: int = 10,
    num_runs: int = 50,
) -> Dict[str, float]:
    """Measure mean and standard deviation of model inference latency.

    Args:
        model: PyTorch nn.Module to benchmark.
        input_shape: Input tensor shape (B, C, H, W).
        device: Device to run benchmark on.
        num_warmup: Number of untimed warmup iterations.
        num_runs: Number of timed iterations.

    Returns:
        Dictionary with 'mean_ms', 'std_ms', and 'fps'.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times: list[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            times.append((t_end - t_start) * 1000.0)  # ms

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "mean_latency_ms": round(mean_ms, 3),
        "std_latency_ms": round(std_ms, 3),
        "fps": round(fps, 1),
    }


def estimate_conv_macs(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
) -> float:
    """Estimate Multiply-Accumulate Operations (MACs) for 2D Conv layers in the model.

    Args:
        model: PyTorch nn.Module.
        input_shape: Input tensor shape (1, C, H, W).

    Returns:
        Estimated GMACs (Giga MACs).
    """
    hooks = []
    total_macs = [0]

    def conv_hook(self: nn.Conv2d, inp: Tuple[torch.Tensor], out: torch.Tensor) -> None:
        batch_size = out.size(0)
        out_channels, out_h, out_w = out.size(1), out.size(2), out.size(3)
        in_channels = self.in_channels
        k_h, k_w = self.kernel_size
        groups = self.groups
        macs_per_output_pixel = (in_channels // groups) * k_h * k_w
        layer_macs = batch_size * out_channels * out_h * out_w * macs_per_output_pixel
        total_macs[0] += layer_macs

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))

    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    dummy = torch.zeros(*input_shape, device=device)

    model.eval()
    with torch.no_grad():
        try:
            _ = model(dummy)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    gmacs = total_macs[0] / 1e9
    return round(float(gmacs), 4)


def profile_model(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    device: torch.device | None = None,
    num_runs: int = 30,
) -> Dict[str, Union[int, float]]:
    """Comprehensive hardware efficiency profiler.

    Args:
        model: Neural network model.
        input_shape: Spatial dimension of input tensor.
        device: Device to run timing profile on.
        num_runs: Number of timing runs.

    Returns:
        Summary dict containing parameters, size, latency, FPS, and GMACs.
    """
    total, trainable, size_mb = count_parameters(model)
    timing = measure_inference_latency(
        model=model,
        input_shape=input_shape,
        device=device,
        num_warmup=5,
        num_runs=num_runs,
    )
    gmacs = estimate_conv_macs(model, input_shape=input_shape)

    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "model_size_mb": round(size_mb, 3),
        "gmacs": gmacs,
        **timing,
    }

