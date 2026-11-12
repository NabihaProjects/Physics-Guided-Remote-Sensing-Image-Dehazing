"""PhysDehaze-Lite: Lightweight Physics-Guided Remote Sensing Dehazing Network.

Architectural Foundation:
    Based on the atmospheric scattering model (base survey paper Eq. 1):
        I(x) = J(x) * t(x) + A * (1 - t(x))
    Inverted physical radiance recovery:
        J_phys(x) = (I(x) - A * (1 - t(x))) / clamp(t(x), min=0.1)

Specifications:
- Parameter budget: Strictly under 1,000,000 parameters (~130k parameters).
- Backbone: U-Net-style encoder-decoder using depthwise-separable convolutions.
- Transmission Head: Predicts pixel-wise transmission map t(x) in (0, 1) via Sigmoid.
- Airlight Head: Predicts global atmospheric light vector A in (0, 1) via global pooling + MLP.
- Radiance Recovery: Physical ASM inversion followed by a residual refinement conv block.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-Separable Convolution (Depthwise 3x3 + Pointwise 1x1 + GroupNorm + LeakyReLU)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=padding * dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        num_groups = min(4, out_channels)
        while out_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class ResidualBlock(nn.Module):
    """Residual block composed of two Depthwise-Separable Convolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class PhysDehazeLite(nn.Module):
    """Lightweight physics-guided dehazing network (< 1M parameters).

    Estimates atmospheric scattering parameters (t, A) and recovers clean radiance J
    using He et al. / Koschmieder physics, followed by residual spatial refinement.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        t_min: float = 0.10,
        use_airlight_head: bool = True,
        fixed_airlight_val: float = 1.0,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize PhysDehazeLite architecture.

        Args:
            in_channels: Number of input channels (default 3 for RGB).
            base_channels: Base channel capacity of encoder (default 32).
            t_min: Minimum transmission clamp threshold to avoid division instability.
            use_airlight_head: If True, uses MLP head to predict A; if False, uses fixed A.
            fixed_airlight_val: Fixed atmospheric light constant if use_airlight_head is False.
            dropout_rate: Dropout rate for bottleneck MC Dropout (default 0.0).
        """
        super().__init__()
        self.in_channels = in_channels
        self.t_min = t_min
        self.use_airlight_head = use_airlight_head
        self.fixed_airlight_val = fixed_airlight_val
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = base_channels * 6   # 192

        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(4, c1), c1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Encoder Levels (U-Net style with Depthwise-Separable Convolutions)
        self.enc1 = ResidualBlock(c1)
        self.down1 = DepthwiseSeparableConv(c1, c2, stride=2)  # H/2, W/2

        self.enc2 = ResidualBlock(c2)
        self.down2 = DepthwiseSeparableConv(c2, c3, stride=2)  # H/4, W/4

        self.enc3 = ResidualBlock(c3)
        self.down3 = DepthwiseSeparableConv(c3, c4, stride=2)  # H/8, W/8

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(c4),
            ResidualBlock(c4),
        )

        # Airlight Head: Global average pooling + MLP -> A in (0, 1) of shape (B, 3, 1, 1)
        self.airlight_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, in_channels),
            nn.Sigmoid(),
        )

        # Decoder Levels with Skip Connections
        self.dec3_conv = DepthwiseSeparableConv(c4 + c3, c3)
        self.dec2_conv = DepthwiseSeparableConv(c3 + c2, c2)
        self.dec1_conv = DepthwiseSeparableConv(c2 + c1, c1)

        # Transmission Head: Predicts pixel-wise transmission map t(x) in (0, 1)
        self.transmission_head = nn.Sequential(
            DepthwiseSeparableConv(c1, c1 // 2),
            nn.Conv2d(c1 // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Refinement Conv Block: Refines J_phys with residual correction
        # Input: J_phys (3) + Hazy I (3) + Transmission t (1) = 7 channels
        refine_in_ch = in_channels * 2 + 1
        self.refinement_block = nn.Sequential(
            nn.Conv2d(refine_in_ch, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(4, c1), c1),
            nn.LeakyReLU(0.1, inplace=True),
            DepthwiseSeparableConv(c1, c1),
            DepthwiseSeparableConv(c1, c1),
            nn.Conv2d(c1, in_channels, kernel_size=3, padding=1),
        )
        # Initialize final refinement layer with zero weights/bias so that initial
        # predictions are physically grounded by the atmospheric scattering model inversion
        nn.init.zeros_(self.refinement_block[-1].weight)
        nn.init.zeros_(self.refinement_block[-1].bias)

    def count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def enable_mc_dropout(self, dropout_rate: float = 0.1) -> None:
        """Enable Monte Carlo Dropout for epistemic uncertainty estimation (Paper Fig. 16)."""
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout2d(p=dropout_rate).train()

    def disable_mc_dropout(self) -> None:
        """Disable Monte Carlo Dropout for standard deterministic inference."""
        self.dropout_rate = 0.0
        self.dropout = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through PhysDehaze-Lite.

        Args:
            x: Input hazy image batch (B, 3, H, W) or (3, H, W) in [0, 1].
            return_dict: If True, returns full dict with J, t, A, and J_phys.
                         If False, returns clean dehazed tensor J in [0, 1].

        Returns:
            Dehazed tensor J or detailed dictionary with physical parameters.
        """
        is_3d = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            is_3d = True
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")

        _, _, orig_h, orig_w = x.shape

        # Ensure spatial dimensions are divisible by 8 for U-Net down/up operations
        pad_h = (8 - orig_h % 8) % 8
        pad_w = (8 - orig_w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x

        # 1. Encoder
        feat1 = self.stem(x_in)
        skip1 = self.enc1(feat1)         # H, W

        feat2 = self.down1(skip1)
        skip2 = self.enc2(feat2)         # H/2, W/2

        feat3 = self.down2(skip2)
        skip3 = self.enc3(feat3)         # H/4, W/4

        feat4 = self.down3(skip3)        # H/8, W/8
        bottleneck = self.bottleneck(feat4)
        bottleneck = self.dropout(bottleneck)

        # 2. Airlight Head: A in (0, 1) shaped (B, 3, 1, 1)
        if self.use_airlight_head:
            a_vec = self.airlight_head(bottleneck)
            A = a_vec.view(-1, self.in_channels, 1, 1)
        else:
            A = torch.full(
                (x_in.size(0), self.in_channels, 1, 1),
                fill_value=self.fixed_airlight_val,
                device=x_in.device,
                dtype=x_in.dtype,
            )

        # 3. Decoder with Skip Connections
        up3 = F.interpolate(bottleneck, size=skip3.shape[2:], mode="bilinear", align_corners=False)
        dec3 = self.dec3_conv(torch.cat([up3, skip3], dim=1))

        up2 = F.interpolate(dec3, size=skip2.shape[2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2_conv(torch.cat([up2, skip2], dim=1))

        up1 = F.interpolate(dec2, size=skip1.shape[2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1_conv(torch.cat([up1, skip1], dim=1))

        # 4. Transmission Head: t(p) in (0, 1)
        t = self.transmission_head(dec1)

        # Remove padding if applied
        if pad_h > 0 or pad_w > 0:
            t = t[:, :, :orig_h, :orig_w]
            x_orig = x
        else:
            x_orig = x_in

        # 5. Physical Inversion: J_phys = (I - A * (1 - t)) / clamp(t, min=t_min)
        t_clamped = torch.clamp(t, min=self.t_min, max=1.0)
        j_phys = (x_orig - A * (1.0 - t)) / t_clamped
        j_phys = torch.clamp(j_phys, 0.0, 1.0)

        # 6. Refinement Conv Block: J = clamp(J_phys + delta_J, 0, 1)
        refine_in = torch.cat([j_phys, x_orig, t.expand(-1, 1, -1, -1)], dim=1)
        delta_j = self.refinement_block(refine_in)
        j_out = torch.clamp(j_phys + delta_j, 0.0, 1.0)

        if is_3d:
            j_out = j_out.squeeze(0)
            t = t.squeeze(0)
            j_phys = j_phys.squeeze(0)

        if return_dict:
            return {
                "output": j_out,
                "transmission": t,
                "airlight": A,
                "j_phys": j_phys,
            }
        return j_out
