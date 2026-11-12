"""RS-PGNet: Physics-Guided Spectral-Spatial Remote Sensing Image Dehazing Network."""

from __future__ import annotations

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.physics.atmospheric_model import AtmosphericScatteringModel


class SpectralAttention(nn.Module):
    """Spectral Attention Module (SAM) to preserve inter-band radiometric fidelity."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class GatedSpatialBlock(nn.Module):
    """Gated Spatial Convolutional Block for edge restoration without halo artifacts."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(channels)
        self.conv_feat = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_gate = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        feat = self.act(self.conv_feat(x_norm))
        gate = torch.sigmoid(self.conv_gate(x_norm))
        out = self.conv_out(feat * gate)
        return residual + out


class PhysicsEstimationBranch(nn.Module):
    """Estimates transmission map t(x) and atmospheric light A(x) under physical priors."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        # Multi-scale receptive field for wide-area RS atmospheric haze
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Dilated convolutions to capture large geographic haze context
        self.dilated1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, dilation=1)
        self.dilated2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=2, dilation=2)
        self.dilated3 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=4, dilation=4)
        self.fuse = nn.Conv2d(base_channels * 3, base_channels, kernel_size=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Transmission head (B, 1, H, W) in [0.05, 1.0]
        self.trans_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Atmospheric light head (B, 3, 1, 1) in [0.4, 1.0]
        self.a_pool = nn.AdaptiveAvgPool2d(1)
        self.a_head = nn.Sequential(
            nn.Linear(base_channels, base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels // 2, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat0 = self.in_conv(x)
        d1 = self.act(self.dilated1(feat0))
        d2 = self.act(self.dilated2(feat0))
        d3 = self.act(self.dilated3(feat0))

        fused = self.act(self.fuse(torch.cat([d1, d2, d3], dim=1)))

        # Transmission map with physics lower-bound clamp
        raw_t = self.trans_head(fused)
        t = 0.05 + 0.95 * raw_t  # Bounded in [0.05, 1.0]

        # Atmospheric light
        pooled = self.a_pool(fused).flatten(1)
        raw_a = self.a_head(pooled).view(-1, 3, 1, 1)
        a = 0.40 + 0.60 * raw_a  # Bounded in [0.40, 1.0]

        return t, a


class SpectralSpatialRefinementBranch(nn.Module):
    """Refinement branch to correct residual color casts and recover high-frequency ground details."""

    def __init__(self, in_channels: int = 6, base_channels: int = 32, num_blocks: int = 3) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                GatedSpatialBlock(base_channels),
                SpectralAttention(base_channels),
            )
            for _ in range(num_blocks)
        ])

        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),  # Residual correction delta in [-1.0, 1.0]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        for block in self.blocks:
            feat = block(feat)
        delta = self.tail(feat) * 0.25  # Scaled residual update
        return delta


class RSPGNet(nn.Module):
    """Physics-Guided Spectral-Spatial Network (RS-PGNet) for remote sensing dehazing.

    Combines:
    1. Atmospheric Scattering Model (Koschmieder) estimation and physical inversion.
    2. Spectral Attention and Gated Spatial Residual Refinement.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_refine_blocks: int = 3,
        t0: float = 0.05,
    ) -> None:
        super().__init__()
        self.physics_branch = PhysicsEstimationBranch(
            in_channels=in_channels,
            base_channels=base_channels,
        )
        self.asm_inversion = AtmosphericScatteringModel(t0=t0)
        self.refine_branch = SpectralSpatialRefinementBranch(
            in_channels=in_channels * 2,  # [J_phy, I]
            base_channels=base_channels,
            num_blocks=num_refine_blocks,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Input hazy image batch (B, 3, H, W) in range [0, 1].
            return_dict: If True, returns dictionary with all intermediate physical predictions.

        Returns:
            Dictionary containing 'output', 'physical', 'transmission', 'atmospheric_light', 'residual'.
            Or just 'output' if return_dict=False.
        """
        # Step 1: Estimate physical parameters
        t_map, a_light = self.physics_branch(x)

        # Step 2: Invert Atmospheric Scattering Model
        j_phy = self.asm_inversion(hazy=x, transmission=t_map, atmospheric_light=a_light)

        # Step 3: Spectral-spatial residual refinement
        refine_in = torch.cat([j_phy, x], dim=1)
        delta = self.refine_branch(refine_in)

        # Step 4: Final composite reconstruction clamped to valid radiance [0, 1]
        final_output = torch.clamp(j_phy + delta, 0.0, 1.0)

        if not return_dict:
            return final_output

        return {
            "output": final_output,
            "physical": j_phy,
            "transmission": t_map,
            "atmospheric_light": a_light,
            "residual": delta,
        }

