"""Physics-based modeling and atmospheric scattering operations."""

from src.physics.atmospheric_model import (
    AtmosphericScatteringModel,
    generate_nonuniform_transmission,
    synthesize_haze,
    invert_atmospheric_scattering,
)

__all__ = [
    "AtmosphericScatteringModel",
    "generate_nonuniform_transmission",
    "synthesize_haze",
    "invert_atmospheric_scattering",
]

