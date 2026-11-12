"""Atmospheric Scattering Model (Koschmieder) and haze simulation for remote sensing."""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def generate_nonuniform_transmission(
    height: int,
    width: int,
    t_min: float = 0.2,
    t_max: float = 0.95,
    octaves: int = 4,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate a realistic, smoothly varying non-uniform transmission map for RS imagery.

    Simulates spatial atmospheric thickness variations (e.g. cirrus / thin cloud / haze gradients)
    using multi-scale sinusoidal harmonic synthesis.

    Args:
        height: Spatial height of the image.
        width: Spatial width of the image.
        t_min: Minimum transmission value (thick haze regions).
        t_max: Maximum transmission value (clear regions).
        octaves: Number of spatial frequency harmonics.
        generator: PyTorch pseudo-random generator for reproducibility.
        device: Device on which to create the tensor.

    Returns:
        Tensor of shape (1, 1, height, width) with values bounded in [t_min, t_max].
    """
    device = device or torch.device("cpu")
    y = torch.linspace(-np.pi, np.pi, height, device=device).view(-1, 1)
    x = torch.linspace(-np.pi, np.pi, width, device=device).view(1, -1)

    grid = torch.zeros((height, width), device=device)

    for i in range(1, octaves + 1):
        freq = float(2 ** (i - 1))
        # Random phase shifts and frequency perturbations
        phase_x = float(torch.empty(1, device=device).uniform_(-np.pi, np.pi, generator=generator).item())
        phase_y = float(torch.empty(1, device=device).uniform_(-np.pi, np.pi, generator=generator).item())
        weight = 1.0 / (2.0 ** i)

        harmonic = torch.sin(freq * x + phase_x) * torch.cos(freq * y + phase_y)
        grid = grid + weight * harmonic

    # Normalize grid to [0, 1]
    g_min = grid.min()
    g_max = grid.max()
    denom = g_max - g_min if (g_max - g_min) > 1e-6 else torch.tensor(1.0, device=device)
    norm_grid = (grid - g_min) / denom

    # Scale to [t_min, t_max]
    t_map = t_min + norm_grid * (t_max - t_min)
    return t_map.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W)


def synthesize_haze(
    clean_image: torch.Tensor,
    transmission: torch.Tensor,
    atmospheric_light: Union[float, torch.Tensor] = 0.85,
) -> torch.Tensor:
    """Apply Koschmieder's Atmospheric Scattering Model: I(x) = J(x)*t(x) + A*(1 - t(x)).

    Args:
        clean_image: Haze-free scene radiance J, shape (B, C, H, W) in range [0, 1].
        transmission: Transmission map t, shape (B, 1, H, W) or (B, C, H, W) in range [0, 1].
        atmospheric_light: Global or spatially varying atmospheric light A.

    Returns:
        Hazy image tensor I of shape (B, C, H, W) clipped to [0, 1].
    """
    if isinstance(atmospheric_light, float):
        A = torch.tensor(atmospheric_light, dtype=clean_image.dtype, device=clean_image.device)
        A = A.view(1, 1, 1, 1)
    elif atmospheric_light.dim() == 1:
        # Per-channel atmospheric light (e.g., shape (C,))
        A = atmospheric_light.view(1, -1, 1, 1)
    else:
        A = atmospheric_light

    hazy = clean_image * transmission + A * (1.0 - transmission)
    return torch.clamp(hazy, 0.0, 1.0)


def invert_atmospheric_scattering(
    hazy_image: torch.Tensor,
    transmission: torch.Tensor,
    atmospheric_light: Union[float, torch.Tensor] = 0.85,
    t0: float = 0.1,
) -> torch.Tensor:
    """Invert Koschmieder model: J(x) = (I(x) - A) / max(t(x), t0) + A.

    Args:
        hazy_image: Hazy image I, shape (B, C, H, W) in range [0, 1].
        transmission: Estimated transmission map t, shape (B, 1, H, W) or (B, C, H, W).
        atmospheric_light: Estimated atmospheric light A.
        t0: Lower bound threshold on transmission to prevent division by zero or noise amplification.

    Returns:
        Recovered haze-free image J of shape (B, C, H, W) clipped to [0, 1].
    """
    if isinstance(atmospheric_light, float):
        A = torch.tensor(atmospheric_light, dtype=hazy_image.dtype, device=hazy_image.device)
        A = A.view(1, 1, 1, 1)
    elif atmospheric_light.dim() == 1:
        A = atmospheric_light.view(1, -1, 1, 1)
    elif atmospheric_light.dim() == 2:
        # Shape (B, C)
        A = atmospheric_light.view(atmospheric_light.size(0), atmospheric_light.size(1), 1, 1)
    else:
        A = atmospheric_light

    t_clamped = torch.clamp(transmission, min=t0, max=1.0)
    recovered = (hazy_image - A) / t_clamped + A
    return torch.clamp(recovered, 0.0, 1.0)


class AtmosphericScatteringModel(nn.Module):
    """PyTorch Module encapsulating the physical atmospheric scattering inversion."""

    def __init__(self, t0: float = 0.1) -> None:
        """Initialize ASM module.

        Args:
            t0: Minimum transmission clamping threshold.
        """
        super().__init__()
        self.t0 = t0

    def forward(
        self,
        hazy: torch.Tensor,
        transmission: torch.Tensor,
        atmospheric_light: torch.Tensor,
    ) -> torch.Tensor:
        """Invert ASM during forward pass of neural network.

        Args:
            hazy: Input hazy tensor (B, 3, H, W).
            transmission: Predicted transmission map (B, 1, H, W).
            atmospheric_light: Predicted atmospheric light (B, 3, 1, 1) or (B, 3).

        Returns:
            Physically dehazed tensor J_phy (B, 3, H, W).
        """
        return invert_atmospheric_scattering(
            hazy_image=hazy,
            transmission=transmission,
            atmospheric_light=atmospheric_light,
            t0=self.t0,
        )

