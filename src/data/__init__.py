"""Data module alias pointing to src.datasets for consistency."""

from src.datasets.dataset import (
    RemoteSensingDehazeDataset,
    create_dataloaders,
    create_synthetic_rs_sample,
)

__all__ = [
    "RemoteSensingDehazeDataset",
    "create_dataloaders",
    "create_synthetic_rs_sample",
]
