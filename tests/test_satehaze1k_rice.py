"""Unit tests for SateHaze1k and RICE1 datasets with fake tiny test images."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator, Tuple
import numpy as np
from PIL import Image
import pytest
import torch

from scripts.verify_dataset import verify_dataset
from src.datasets.rice import RICE1Dataset
from src.datasets.satehaze1k import SateHaze1kDataset
from src.datasets.transforms import PairedTransform


def create_fake_image_pair(
    hazy_path: Path,
    clear_path: Path,
    size: Tuple[int, int] = (64, 64),
    seed: int = 42,
) -> None:
    """Generate a tiny synthetic hazy and clear image pair with marked spatial patterns."""
    rng = np.random.RandomState(seed)
    w, h = size

    # Clear image: base RGB pattern with distinct colored square at corner
    clear_arr = (rng.rand(h, w, 3) * 150).astype(np.uint8)
    clear_arr[:10, :10, :] = [255, 0, 0]  # Red marker top-left

    # Hazy image: hazy appearance (higher mean, lower contrast)
    hazy_arr = (clear_arr * 0.6 + 80).astype(np.uint8)
    hazy_arr[:10, :10, :] = [200, 100, 100]  # Washed out red marker

    hazy_path.parent.mkdir(parents=True, exist_ok=True)
    clear_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(clear_arr).save(clear_path)
    Image.fromarray(hazy_arr).save(hazy_path)


@pytest.fixture
def fake_satehaze1k_dir() -> Generator[Path, None, None]:
    """Create a temporary fake SateHaze1k dataset directory tree."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / "SateHaze1k"
        subsets = ["thin", "moderate", "thick"]
        pairs_per_subset = 10

        for s_idx, subset in enumerate(subsets):
            for i in range(pairs_per_subset):
                hazy_file = root / subset / "hazy" / f"{subset}_{i:03d}.png"
                clear_file = root / subset / "clear" / f"{subset}_{i:03d}.png"
                create_fake_image_pair(hazy_file, clear_file, size=(64, 64), seed=s_idx * 100 + i)

        yield root


@pytest.fixture
def fake_rice1_dir() -> Generator[Path, None, None]:
    """Create a temporary fake RICE1 dataset directory tree."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir) / "RICE1"
        num_pairs = 20

        for i in range(num_pairs):
            hazy_file = root / "hazy" / f"rice_{i:03d}.png"
            clear_file = root / "clear" / f"rice_{i:03d}.png"
            create_fake_image_pair(hazy_file, clear_file, size=(64, 64), seed=500 + i)

        yield root


def test_paired_transform_synchronization():
    """Verify paired transform crops, flips, and rotates both images synchronously."""
    transform = PairedTransform(crop_size=(32, 32), is_train=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        h_path = Path(tmp_dir) / "h.png"
        c_path = Path(tmp_dir) / "c.png"
        create_fake_image_pair(h_path, c_path, size=(64, 64))

        with Image.open(h_path) as h_img, Image.open(c_path) as c_img:
            h_tensor, c_tensor = transform(h_img.convert("RGB"), c_img.convert("RGB"))

        assert h_tensor.shape == (3, 32, 32)
        assert c_tensor.shape == (3, 32, 32)
        assert h_tensor.dtype == torch.float32
        assert c_tensor.dtype == torch.float32
        assert 0.0 <= h_tensor.min() <= h_tensor.max() <= 1.0
        assert 0.0 <= c_tensor.min() <= c_tensor.max() <= 1.0


def test_satehaze1k_subsets_and_loading(fake_satehaze1k_dir: Path):
    """Verify SateHaze1k dataset loads thin, moderate, thick, and combined subsets."""
    # Test single subset
    ds_thin = SateHaze1kDataset(
        root_dir=fake_satehaze1k_dir,
        subset="thin",
        split="all",
        crop_size=(32, 32),
    )
    assert len(ds_thin) == 10
    sample = ds_thin[0]
    assert sample["subset"] == "thin"
    assert sample["hazy"].shape == (3, 32, 32)
    assert sample["clear"].shape == (3, 32, 32)
    assert sample["clean"].shape == (3, 32, 32)

    # Test combined all subsets
    ds_all = SateHaze1kDataset(
        root_dir=fake_satehaze1k_dir,
        subset="all",
        split="all",
        crop_size=(32, 32),
    )
    assert len(ds_all) == 30


def test_rice1_loading(fake_rice1_dir: Path):
    """Verify RICE1 dataset loads correctly with hazy and clear pairs."""
    ds = RICE1Dataset(
        root_dir=fake_rice1_dir,
        split="all",
        crop_size=(32, 32),
    )
    assert len(ds) == 20
    sample = ds[0]
    assert sample["dataset"] == "RICE1"
    assert "hazy" in sample
    assert "clear" in sample
    assert sample["hazy"].shape == (3, 32, 32)
    assert sample["clear"].shape == (3, 32, 32)


def test_deterministic_splits_no_overlap(fake_rice1_dir: Path):
    """Verify train, val, and test splits are strictly disjoint and reproducible."""
    seed = 1234
    ratios = (0.6, 0.2, 0.2)  # For 20 items: 12 train, 4 val, 4 test

    ds_train = RICE1Dataset(root_dir=fake_rice1_dir, split="train", split_ratios=ratios, seed=seed)
    ds_val = RICE1Dataset(root_dir=fake_rice1_dir, split="val", split_ratios=ratios, seed=seed)
    ds_test = RICE1Dataset(root_dir=fake_rice1_dir, split="test", split_ratios=ratios, seed=seed)

    assert len(ds_train) == 12
    assert len(ds_val) == 4
    assert len(ds_test) == 4

    train_names = {s["name"] for s in ds_train}
    val_names = {s["name"] for s in ds_val}
    test_names = {s["name"] for s in ds_test}

    # Strict non-overlap assertion
    assert len(train_names & val_names) == 0, "Train and Val splits share samples"
    assert len(train_names & test_names) == 0, "Train and Test splits share samples"
    assert len(val_names & test_names) == 0, "Val and Test splits share samples"

    # Union covers full dataset
    all_names = train_names | val_names | test_names
    assert len(all_names) == 20

    # Test reproducibility: identical seed yields identical train set
    ds_train_repeat = RICE1Dataset(root_dir=fake_rice1_dir, split="train", split_ratios=ratios, seed=seed)
    train_names_repeat = {s["name"] for s in ds_train_repeat}
    assert train_names == train_names_repeat


def test_verify_dataset_script_pass(fake_satehaze1k_dir: Path, fake_rice1_dir: Path):
    """Verify pair verification tool passes on well-formed datasets."""
    report_sat = verify_dataset(fake_satehaze1k_dir, dataset_type="satehaze1k", expected_dim=(64, 64))
    assert report_sat["overall_status"] == "PASS"
    haze_info = report_sat["datasets"].get("Haze1k", report_sat.get("subsets", {}))
    total_sat = haze_info.get("total_pairs", haze_info.get("total_pairs_across_subsets", 0))
    assert total_sat == 30

    report_rice = verify_dataset(fake_rice1_dir, dataset_type="rice1", expected_dim=(64, 64))
    assert report_rice["overall_status"] == "PASS"
    rice_info = report_rice["datasets"].get("RICE1", report_rice.get("subsets", {}).get("main", {}))
    assert rice_info["total_pairs"] == 20


def test_verify_dataset_script_detects_orphans_and_corruption(fake_rice1_dir: Path):
    """Verify pair verification tool flags orphaned and corrupted files."""
    # Create an orphaned hazy file
    orphan_hazy = fake_rice1_dir / "hazy" / "orphan_999.png"
    Image.new("RGB", (64, 64), color="red").save(orphan_hazy)

    # Create a corrupted file in clear
    corrupt_clear = fake_rice1_dir / "clear" / "rice_000.png"
    with open(corrupt_clear, "wb") as f:
        f.write(b"not an image file")

    report = verify_dataset(fake_rice1_dir, dataset_type="rice1")
    assert report["overall_status"] == "FAIL"

    rice_info = report["datasets"].get("RICE1", {})
    details = rice_info.get("details", rice_info)
    assert details["orphaned_hazy_count"] >= 1
    assert "orphan_999" in details["orphaned_hazy_samples"]
    assert details["corrupted_pairs_count"] >= 1

