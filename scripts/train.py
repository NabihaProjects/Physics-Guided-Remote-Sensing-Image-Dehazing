"""CLI training script for remote sensing dehazing benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Ensure PYTORCH_ENABLE_MPS_FALLBACK=1
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from src.datasets.satehaze1k import SateHaze1kDataset
from src.datasets.rrshid import RRSHIDDataset
from src.train.train import create_model, get_training_device, train_model
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a remote sensing dehazing model on Haze1k + RRSHID.")
    parser.add_argument("--config", type=str, default="configs/physdehaze_lite.yaml", help="Path to config YAML.")
    parser.add_argument("--model", type=str, default="physdehaze_lite", choices=["physdehaze_lite", "aodnet", "dcp", "rs_pgnet"], help="Model name.")
    parser.add_argument("--data-dir", type=str, default="data/Haze1k", help="Dataset root directory.")
    parser.add_argument("--rrshid-dir", type=str, default="data/RRSHID", help="RRSHID dataset root directory.")
    parser.add_argument("--rrshid-samples", type=int, default=1000, help="Number of sampled RRSHID training pairs (default: 1000).")
    parser.add_argument("--output-dir", type=str, default="experiments/physdehaze_lite", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--num-workers", type=int, default=0, choices=[0, 2], help="DataLoader workers.")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-epoch smoke test on 20 images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_training_device()
    print(f"Executing training on: {device} (PYTORCH_ENABLE_MPS_FALLBACK=1)")

    crop_size = (128, 128)

    # 1. Haze1k Train (thin + moderate + thick combined = 960 pairs)
    haze1k_train = SateHaze1kDataset(
        root_dir=args.data_dir,
        subset="all",
        split="train",
        crop_size=crop_size,
        seed=args.seed,
    )

    # 2. RRSHID Train (randomly sample 1,000 pairs with seed 42)
    rrshid_path = Path(args.rrshid_dir)
    if rrshid_path.exists() and args.rrshid_samples > 0:
        rrshid_train_full = RRSHIDDataset(
            root_dir=rrshid_path,
            subset="all",
            split="train",
            crop_size=crop_size,
            seed=args.seed,
        )
        rng = np.random.RandomState(args.seed)
        sample_count = min(args.rrshid_samples, len(rrshid_train_full))
        rrshid_indices = rng.permutation(len(rrshid_train_full))[:sample_count].tolist()
        rrshid_train_sampled = Subset(rrshid_train_full, rrshid_indices)
        train_dataset: Dataset = ConcatDataset([haze1k_train, rrshid_train_sampled])
        print(f"Loaded training data: Haze1k ({len(haze1k_train)} pairs) + RRSHID ({len(rrshid_train_sampled)} pairs) = {len(train_dataset)} total pairs.")
    else:
        train_dataset = haze1k_train
        print(f"Loaded training data: Haze1k ({len(haze1k_train)} pairs).")

    # 3. Validation: Haze1k val/valid splits ONLY (105 pairs). RICE1 is NEVER used for training!
    val_dataset: Dataset = SateHaze1kDataset(
        root_dir=args.data_dir,
        subset="all",
        split="val",
        crop_size=crop_size,
        seed=args.seed,
    )
    print(f"Loaded validation data: Haze1k val splits ({len(val_dataset)} pairs). RICE1 is excluded from training.")

    epochs = args.epochs
    if args.smoke_test:
        epochs = 2
        train_dataset = Subset(train_dataset, range(min(20, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(10, len(val_dataset))))
        print(f"\n[Smoke Test Activated]: 2 epochs on 20 training images (val on 10 images).\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = create_model(args.model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model.upper()} ({param_count:,} trainable parameters)")

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        epochs=epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        resume_path=args.resume,
    )

    print("\nTraining completed.")
    print(f"  Best Val PSNR: {results['best_val_psnr']:.2f} dB")
    print(f"  Avg Seconds / Epoch: {results['avg_seconds_per_epoch']:.2f} s")


if __name__ == "__main__":
    main()
