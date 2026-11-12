"""Compound loss combining L1, SSIM, Spectral Angle Mapper (SAM), and Total Variation."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import LossWeightsConfig


class SpectralAngleLoss(nn.Module):
    """Spectral Angle Mapper (SAM) loss for remote sensing spectral fidelity.

    Measures the angular deviation of spectral vectors at each pixel:
        SAM(x, y) = arccos( <x, y> / (||x|| * ||y|| + eps) )
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean SAM loss across batch.

        Args:
            pred: Predicted tensor (B, C, H, W).
            target: Ground truth tensor (B, C, H, W).

        Returns:
            Scalar tensor representing mean spectral angle in radians.
        """
        dot_product = torch.sum(pred * target, dim=1)  # (B, H, W)
        norm_pred = torch.norm(pred, p=2, dim=1)
        norm_target = torch.norm(target, p=2, dim=1)

        denominator = norm_pred * norm_target + self.eps
        cos_similarity = torch.clamp(dot_product / denominator, min=-1.0 + self.eps, max=1.0 - self.eps)

        angle = torch.acos(cos_similarity)
        return torch.mean(angle)


class TotalVariationLoss(nn.Module):
    """Total Variation (TV) loss to regularize transmission map smoothness."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L1 TV loss.

        Args:
            x: Transmission tensor of shape (B, 1, H, W).

        Returns:
            Scalar TV loss.
        """
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return torch.mean(diff_h) + torch.mean(diff_w)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss for PyTorch optimization."""

    def __init__(self, window_size: int = 11, channels: int = 3) -> None:
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("kernel", self._create_gaussian_kernel(window_size, channels))

    def _create_gaussian_kernel(self, window_size: int, channels: int) -> torch.Tensor:
        """Generate 2D Gaussian convolution kernel."""
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(1) * g1d.unsqueeze(0)
        kernel = g2d.expand(channels, 1, window_size, window_size).contiguous()
        return kernel

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute 1.0 - SSIM(img1, img2)."""
        c1 = 0.01**2
        c2 = 0.03**2

        kernel = self.kernel.to(device=img1.device, dtype=img1.dtype)
        pad = self.window_size // 2

        mu1 = F.conv2d(img1, kernel, padding=pad, groups=self.channels)
        mu2 = F.conv2d(img2, kernel, padding=pad, groups=self.channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=self.channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return torch.clamp(1.0 - ssim_map.mean(), min=0.0)


class CompoundDehazeLoss(nn.Module):
    """Unified composite loss function for remote sensing dehazing."""

    def __init__(self, weights: Optional[LossWeightsConfig] = None) -> None:
        super().__init__()
        weights = weights or LossWeightsConfig()
        self.w_l1 = weights.l1
        self.w_ssim = weights.ssim
        self.w_spectral = weights.spectral
        self.w_tv = weights.tv_transmission

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.sam_loss = SpectralAngleLoss()
        self.tv_loss = TotalVariationLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        predicted_transmission: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate total weighted compound loss.

        Args:
            pred: Predicted dehazed image (B, 3, H, W).
            target: Ground truth clean image (B, 3, H, W).
            predicted_transmission: Optional predicted transmission map (B, 1, H, W).

        Returns:
            Tuple of (total_loss tensor, dictionary of individual loss values).
        """
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        spectral = self.sam_loss(pred, target)

        total = self.w_l1 * l1 + self.w_ssim * ssim + self.w_spectral * spectral
        loss_dict = {
            "loss_l1": float(l1.item()),
            "loss_ssim": float(ssim.item()),
            "loss_spectral": float(spectral.item()),
        }

        if predicted_transmission is not None and self.w_tv > 0:
            tv = self.tv_loss(predicted_transmission)
            total = total + self.w_tv * tv
            loss_dict["loss_tv"] = float(tv.item())

        loss_dict["loss_total"] = float(total.item())
        return total, loss_dict

