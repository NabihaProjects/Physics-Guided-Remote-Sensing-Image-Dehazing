"""Classical Dark Channel Prior (DCP) dehazing baseline with Guided Filter refinement.

Reference:
    He, K., Sun, J., & Tang, X. (2010). "Single Image Haze Removal Using Dark Channel Prior."
    IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 33(12), 2341-2353.
"""

from __future__ import annotations

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def _box_filter(x: torch.Tensor, r: int) -> torch.Tensor:
    """Fast 2D box filter implemented via average pooling."""
    kernel_size = 2 * r + 1
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=r)


def guided_filter(
    p: torch.Tensor,
    guidance: torch.Tensor,
    r: int = 15,
    eps: float = 1e-3,
) -> torch.Tensor:
    """PyTorch tensor-native guided image filter for edge-preserving transmission smoothing.

    Args:
        p: Filtering input tensor of shape (B, 1, H, W).
        guidance: Guidance image tensor of shape (B, 1, H, W) or (B, 3, H, W).
        r: Filter radius.
        eps: Regularization parameter.

    Returns:
        Filtered tensor of shape (B, 1, H, W).
    """
    if guidance.size(1) == 3:
        # Convert RGB guidance to grayscale luminance: Y = 0.299 R + 0.587 G + 0.114 B
        guidance = 0.299 * guidance[:, 0:1] + 0.587 * guidance[:, 1:2] + 0.114 * guidance[:, 2:3]

    mean_I = _box_filter(guidance, r)
    mean_p = _box_filter(p, r)
    mean_Ip = _box_filter(guidance * p, r)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = _box_filter(guidance * guidance, r)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = _box_filter(a, r)
    mean_b = _box_filter(b, r)

    q = mean_a * guidance + mean_b
    return q


class DCP(nn.Module):
    """Classical Dark Channel Prior dehazing baseline (He et al. 2010, non-learning).

    Attributes & Algorithmic Steps:
    1. Dark Channel Computation: Spatial and spectral min-pooling over RGB channels with patch size.
    2. Atmospheric Light (A) Estimation: Top 0.1% brightest dark channel pixels.
    3. Transmission Map Estimation: t_raw = 1 - omega * dark(I / A).
    4. Edge-Preserving Refinement: Guided image filter with radius r and regularization eps.
    5. Atmospheric Scattering Model Inversion: J = (I - A) / max(t, t0) + A, clamped to [0, 1].

    Parameters:
        patch_size: Window size for local dark channel min-pooling (default 15).
        omega: Haze retention factor (default 0.95).
        t0: Minimum transmission clamping threshold (default 0.1).
        filter_radius: Radius for guided image filtering (default 15).
        filter_eps: Guided filter regularization parameter (default 1e-3).
        a_max: Atmospheric light upper bound to avoid selecting reflective terrain targets.
    """

    def __init__(
        self,
        patch_size: int = 15,
        omega: float = 0.95,
        t0: float = 0.10,
        filter_radius: int = 15,
        filter_eps: float = 1e-3,
        a_max: float = 0.95,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.omega = omega
        self.t0 = t0
        self.filter_radius = filter_radius
        self.filter_eps = filter_eps
        self.a_max = a_max

    def _get_dark_channel(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the dark channel: min_c(x), followed by spatial min-pooling."""
        min_channel, _ = torch.min(x, dim=1, keepdim=True)
        pad = self.patch_size // 2
        # Min pooling via negative max pooling
        dark = -F.max_pool2d(-min_channel, kernel_size=self.patch_size, stride=1, padding=pad)
        return dark

    def _estimate_atmospheric_light(
        self,
        x: torch.Tensor,
        dark_channel: torch.Tensor,
        top_percent: float = 0.001,
    ) -> torch.Tensor:
        """Estimate atmospheric light vector A of shape (B, 3, 1, 1)."""
        B, C, H, W = x.shape
        num_pixels = H * W
        num_candidates = max(int(num_pixels * top_percent), 1)

        flat_dark = dark_channel.view(B, -1)
        flat_x = x.view(B, C, -1)

        # Indices of brightest pixels in dark channel
        _, indices = torch.topk(flat_dark, k=num_candidates, dim=-1)

        A = torch.zeros((B, C, 1, 1), device=x.device, dtype=x.dtype)
        for b in range(B):
            idx = indices[b]
            selected = flat_x[b, :, idx]  # (C, num_candidates)
            intensities = selected.mean(dim=0)
            max_idx = torch.argmax(intensities)
            chosen_a = selected[:, max_idx]
            A[b, :, 0, 0] = torch.clamp(chosen_a, max=self.a_max)

        return A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dehaze input image batch or single image.

        Common interface:
            forward(hazy: torch.Tensor) -> dehazed: torch.Tensor, output in [0, 1].

        Args:
            x: Input hazy image tensor with shape (B, 3, H, W) or (3, H, W) in [0, 1].

        Returns:
            Dehazed image tensor with shape matching input in [0, 1].
        """
        is_3d = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            is_3d = True
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")

        dark = self._get_dark_channel(x)
        A = self._estimate_atmospheric_light(x, dark)

        # Normalized image x / A
        norm_x = x / torch.clamp(A, min=1e-3)
        raw_transmission = 1.0 - self.omega * self._get_dark_channel(norm_x)

        # Refine transmission using guided filter
        refined_trans = guided_filter(
            p=raw_transmission,
            guidance=x,
            r=self.filter_radius,
            eps=self.filter_eps,
        )
        refined_trans = torch.clamp(refined_trans, min=self.t0, max=1.0)

        # Invert Koschmieder model: J = (I - A) / t + A
        dehazed = (x - A) / refined_trans + A
        dehazed = torch.clamp(dehazed, 0.0, 1.0)

        if is_3d:
            dehazed = dehazed.squeeze(0)
        return dehazed


# Alias for backwards compatibility
RSDCP = DCP
