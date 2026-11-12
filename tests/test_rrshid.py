"""Unit tests for RRSHID (Real-World Remote Sensing Hazy Image Dataset)."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from src.datasets.rrshid import RRSHIDDataset, SUBSET_NAME_MAP
from scripts.verify_dataset import verify_dataset


DATA_DIR = Path("data/RRSHID")


@pytest.mark.skipif(not DATA_DIR.exists(), reason="data/RRSHID directory not available")
def test_rrshid_counts_per_subset_and_split():
    """Verify exact sample counts for thin, moderate, thick across train, val, test."""
    expected_counts = {
        "thin": {"train": 610, "val": 77, "test": 76, "all": 763},
        "moderate": {"train": 1220, "val": 154, "test": 152, "all": 1526},
        "thick": {"train": 611, "val": 77, "test": 76, "all": 764},
    }

    for sub, splits in expected_counts.items():
        for sp, expected in splits.items():
            ds = RRSHIDDataset(root_dir=DATA_DIR, subset=sub, split=sp)
            assert len(ds) == expected, f"Expected {expected} pairs for {sub}/{sp}, got {len(ds)}"


@pytest.mark.skipif(not DATA_DIR.exists(), reason="data/RRSHID directory not available")
def test_rrshid_aliases_and_subsets():
    """Verify alias mapping for thin (TN), moderate (M), thick (TK)."""
    alias_tests = [
        ("thin", "TN"),
        ("moderate", "M"),
        ("thick", "TK"),
    ]
    for standard, alias in alias_tests:
        ds_std = RRSHIDDataset(root_dir=DATA_DIR, subset=standard, split="test")
        ds_alias = RRSHIDDataset(root_dir=DATA_DIR, subset=alias, split="test")
        assert len(ds_std) == len(ds_alias)


@pytest.mark.skipif(not DATA_DIR.exists(), reason="data/RRSHID directory not available")
def test_rrshid_combined_dataset():
    """Verify combined subset loading (3,053 pairs total across all splits)."""
    ds_all = RRSHIDDataset(root_dir=DATA_DIR, subset="all", split="all")
    assert len(ds_all) == 3053

    ds_train = RRSHIDDataset(root_dir=DATA_DIR, subset="all", split="train")
    assert len(ds_train) == 610 + 1220 + 611  # 2441

    ds_val = RRSHIDDataset(root_dir=DATA_DIR, subset="all", split="val")
    assert len(ds_val) == 77 + 154 + 77  # 308

    ds_test = RRSHIDDataset(root_dir=DATA_DIR, subset="all", split="test")
    assert len(ds_test) == 76 + 152 + 76  # 304


@pytest.mark.skipif(not DATA_DIR.exists(), reason="data/RRSHID directory not available")
def test_rrshid_item_shapes_and_values():
    """Verify tensor shape (3, 256, 256), dtype, and range [0, 1]."""
    ds = RRSHIDDataset(root_dir=DATA_DIR, subset="thin", split="test")
    item = ds[0]

    assert isinstance(item, dict)
    assert "hazy" in item and "clear" in item and "subset" in item
    assert item["subset"] == "thin"

    hazy = item["hazy"]
    clear = item["clear"]

    assert isinstance(hazy, torch.Tensor)
    assert isinstance(clear, torch.Tensor)
    assert hazy.shape == (3, 256, 256)
    assert clear.shape == (3, 256, 256)
    assert hazy.dtype == torch.float32
    assert clear.dtype == torch.float32
    assert 0.0 <= hazy.min() <= hazy.max() <= 1.0
    assert 0.0 <= clear.min() <= clear.max() <= 1.0


@pytest.mark.skipif(not DATA_DIR.exists(), reason="data/RRSHID directory not available")
def test_rrshid_verification_script():
    """Verify scripts/verify_dataset.py passes on RRSHID with 0 mismatches."""
    report = verify_dataset(DATA_DIR, dataset_type="rrshid", expected_dim=(256, 256))
    assert report["overall_status"] == "PASS"
    assert "RRSHID" in report["datasets"]
    assert report["datasets"]["RRSHID"]["total_pairs"] == 3053

