"""Physics-guided dehazing loss module for PhysDehaze-Lite.

Implements all 4 individually switchable loss terms:
1. L1 Reconstruction Loss: || J - J_gt ||_1
2. SSIM Structural Loss: 1.0 - SSIM(J, J_gt)
3. Physics Consistency Loss: || I - (J * t + A * (1 - t)) ||_1
4. Spectral Angle Mapper (SAM) Loss: SAM(J, J_gt)

All loss components can be independently enabled or disabled and weighted
via configuration dictionaries or YAML files.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.compound_loss import SSIMLoss, SpectralAngleLoss


class PhysicsConsistencyLoss(nn.Module):
    """Enforces the atmospheric scattering model constraint: I = J * t + A * (1 - t).

    Re-synthesizes the hazy image from the predicted radiance J, transmission map t,
    and atmospheric airlight vector A, penalizing deviation from the observed hazy input I:
        L_phys = || I - (J * t + A * (1 - t)) ||_1
    """

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(
        self,
        hazy_input: torch.Tensor,
        dehazed_pred: torch.Tensor,
        transmission: torch.Tensor,
        airlight: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics consistency loss.

        Args:
            hazy_input: Observed hazy image I of shape (B, 3, H, W).
            dehazed_pred: Recovered radiance J of shape (B, 3, H, W).
            transmission: Predicted transmission map t of shape (B, 1, H, W) or (B, 3, H, W).
            airlight: Predicted atmospheric light A of shape (B, 3, 1, 1).

        Returns:
            Scalar tensor with L1 reconstruction error between I and synthetic hazy image.
        """
        # Ensure airlight is 4D for proper broadcasting
        if airlight.dim() == 2:
            airlight = airlight.unsqueeze(-1).unsqueeze(-1)

        # Forward atmospheric scattering equation (Eq. 1)
        hazy_synthesized = dehazed_pred * transmission + airlight * (1.0 - transmission)
        return self.l1(hazy_synthesized, hazy_input)


class PhysDehazeLoss(nn.Module):
    """Composite loss for PhysDehaze-Lite with individually switchable components.

    Configurable via YAML dictionary:
        losses:
          l1:
            enabled: true
            weight: 1.0
          ssim:
            enabled: true
            weight: 0.2
          physics:
            enabled: true
            weight: 0.5
          sam:
            enabled: true
            weight: 0.1
    """

    def __init__(
        self,
        use_l1: bool = True,
        weight_l1: float = 1.0,
        use_ssim: bool = True,
        weight_ssim: float = 0.2,
        use_physics: bool = True,
        weight_physics: float = 0.5,
        use_sam: bool = True,
        weight_sam: float = 0.1,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # If config dictionary provided, parse loss settings from it
        if config is not None:
            losses_cfg = config.get("losses", config)
            # L1
            l1_cfg = losses_cfg.get("l1", {})
            self.use_l1 = l1_cfg.get("enabled", use_l1)
            self.weight_l1 = float(l1_cfg.get("weight", weight_l1))
            # SSIM
            ssim_cfg = losses_cfg.get("ssim", {})
            self.use_ssim = ssim_cfg.get("enabled", use_ssim)
            self.weight_ssim = float(ssim_cfg.get("weight", weight_ssim))
            # Physics
            phys_cfg = losses_cfg.get("physics", {})
            self.use_physics = phys_cfg.get("enabled", use_physics)
            self.weight_physics = float(phys_cfg.get("weight", weight_physics))
            # SAM
            sam_cfg = losses_cfg.get("sam", {})
            self.use_sam = sam_cfg.get("enabled", use_sam)
            self.weight_sam = float(sam_cfg.get("weight", weight_sam))
        else:
            self.use_l1 = use_l1
            self.weight_l1 = weight_l1
            self.use_ssim = use_ssim
            self.weight_ssim = weight_ssim
            self.use_physics = use_physics
            self.weight_physics = weight_physics
            self.use_sam = use_sam
            self.weight_sam = weight_sam

        # Sub-loss modules
        self.l1_criterion = nn.L1Loss()
        self.ssim_criterion = SSIMLoss(window_size=11, channels=3)
        self.physics_criterion = PhysicsConsistencyLoss()
        self.sam_criterion = SpectralAngleLoss()

    def forward(
        self,
        pred: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target: torch.Tensor,
        hazy: Optional[torch.Tensor] = None,
        transmission: Optional[torch.Tensor] = None,
        airlight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the composite loss over predicted and target batches.

        Args:
            pred: Predicted dehazed tensor J or dictionary containing 'output',
                  'transmission', and 'airlight'.
            target: Ground truth clean image tensor J_gt.
            hazy: Input hazy image I (required if use_physics is True).
            transmission: Predicted transmission map t (optional if pred is dict).
            airlight: Predicted airlight vector A (optional if pred is dict).

        Returns:
            Tuple of (total_loss tensor, dictionary of individual loss metrics).
        """
        if isinstance(pred, dict):
            j_pred = pred["output"]
            t_pred = pred.get("transmission", transmission)
            a_pred = pred.get("airlight", airlight)
        else:
            j_pred = pred
            t_pred = transmission
            a_pred = airlight

        device = j_pred.device
        total_loss = torch.tensor(0.0, device=device, dtype=j_pred.dtype)
        loss_dict: Dict[str, float] = {}

        # 1. L1 Reconstruction Loss
        if self.use_l1 and self.weight_l1 > 0:
            l1_val = self.l1_criterion(j_pred, target)
            total_loss = total_loss + self.weight_l1 * l1_val
            loss_dict["loss_l1"] = float(l1_val.item())

        # 2. SSIM Structural Loss
        if self.use_ssim and self.weight_ssim > 0:
            ssim_val = self.ssim_criterion(j_pred, target)
            total_loss = total_loss + self.weight_ssim * ssim_val
            loss_dict["loss_ssim"] = float(ssim_val.item())

        # 3. Physics Consistency Loss: || I - (J*t + A*(1-t)) ||_1
        if self.use_physics and self.weight_physics > 0:
            if hazy is not None and t_pred is not None and a_pred is not None:
                phys_val = self.physics_criterion(hazy, j_pred, t_pred, a_pred)
                total_loss = total_loss + self.weight_physics * phys_val
                loss_dict["loss_physics"] = float(phys_val.item())

        # 4. Spectral Loss using SAM between J and ground truth
        if self.use_sam and self.weight_sam > 0:
            sam_val = self.sam_criterion(j_pred, target)
            total_loss = total_loss + self.weight_sam * sam_val
            loss_dict["loss_sam"] = float(sam_val.item())

        loss_dict["loss_total"] = float(total_loss.item())
        return total_loss, loss_dict

