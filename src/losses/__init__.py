"""Physics-informed and remote sensing loss functions."""

from src.losses.compound_loss import (
    CompoundDehazeLoss,
    SpectralAngleLoss,
    SSIMLoss,
    TotalVariationLoss,
)
from src.losses.physdehaze_loss import (
    PhysicsConsistencyLoss,
    PhysDehazeLoss,
)

__all__ = [
    "CompoundDehazeLoss",
    "SpectralAngleLoss",
    "TotalVariationLoss",
    "SSIMLoss",
    "PhysicsConsistencyLoss",
    "PhysDehazeLoss",
]
