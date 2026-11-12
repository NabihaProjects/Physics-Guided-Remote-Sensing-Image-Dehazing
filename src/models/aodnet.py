"""All-in-One Dehazing Network (AOD-Net) baseline.

Reference:
    Li, B., Peng, X., Wang, Z., Xu, J., & Feng, D. (2017). "AOD-Net: All-in-One Dehazing Network."
    IEEE International Conference on Computer Vision (ICCV), 4770-4778.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AODNet(nn.Module):
    """All-in-One Dehazing Network (AOD-Net, Li et al., ICCV 2017).

    A compact convolutional neural network (< 10,000 parameters, strictly 1,761 parameters)
    that directly formulates the atmospheric scattering model:
        J(x) = K(x) * I(x) - K(x) + b
    where K(x) integrates transmission map t(x) and atmospheric light A into a single
    learnable spatial transformation, eliminating error accumulation from separate estimation.

    Parameters:
        in_channels: Number of input spectral channels (default 3 for RGB).
        b: Constant bias term in the transformed equation (default 1.0).
    """

    def __init__(self, in_channels: int = 3, b: float = 1.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.b = b

        # Multi-scale receptive field convolutions (Total: 1,761 parameters)
        self.conv1 = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate K(x) and compute restored radiance J(x).

        Common interface:
            forward(hazy: torch.Tensor) -> dehazed: torch.Tensor, output in [0, 1].

        Args:
            x: Input hazy image batch (B, 3, H, W) or (3, H, W) in [0, 1].

        Returns:
            Dehazed image batch (B, 3, H, W) or (3, H, W) clamped to [0, 1].
        """
        is_3d = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            is_3d = True
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {x.shape}")

        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))

        cat1 = torch.cat([x1, x2], dim=1)
        x3 = self.relu(self.conv3(cat1))

        cat2 = torch.cat([x2, x3], dim=1)
        x4 = self.relu(self.conv4(cat2))

        cat3 = torch.cat([x1, x2, x3, x4], dim=1)
        k = self.relu(self.conv5(cat3))

        # Atmospheric scattering model inversion: J = K * I - K + b
        output = k * x - k + self.b
        output = torch.clamp(output, 0.0, 1.0)

        if is_3d:
            output = output.squeeze(0)
        return output
