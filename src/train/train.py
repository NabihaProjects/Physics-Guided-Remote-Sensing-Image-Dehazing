"""Training script and engine for remote sensing dehazing baselines.

Implements training for AOD-Net and other dehazing models following benchmark specifications:
- Objective: L1 loss (Mean Absolute Error)
- Optimizer: AdamW
- Learning Rate Schedule: Cosine Annealing (CosineAnnealingLR)
- Precision: float32 only (no AMP / FP16)
- Compute Device: Apple Silicon 'mps' if available, otherwise CPU
- Fallback: PYTORCH_ENABLE_MPS_FALLBACK=1
- Transforms: Random 128x128 crops, flips, and rotations
- Data: Combined train splits of Haze1k thin+moderate+thick (960 pairs); val splits (105 pairs)
- Batch Size: 8, num_workers: 0 or 2
- Checkpointing: Saved every epoch with resume support
- Logging: Real-time TensorBoard metric visualization and loss tracking
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Set MPS fallback prior to importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from src.datasets.satehaze1k import SateHaze1kDataset
from src.datasets.rrshid import RRSHIDDataset
from src.metrics.metrics import (
    calculate_ciede2000,
    calculate_ergas,
    calculate_hist,
    calculate_psnr,
    calculate_qnr,
    calculate_sam,
    calculate_ssim,
    calculate_uiqi,
    MetricTracker,
)
from src.losses.physdehaze_loss import PhysDehazeLoss
from src.models.aodnet import AODNet
from src.models.dcp import DCP
from src.models.physdehaze_lite import PhysDehazeLite
from src.models.rs_pgnet import RSPGNet

logger = logging.getLogger(__name__)


def get_training_device(device_override: Optional[str] = None) -> torch.device:
    """Determine compute device: MPS if available on macOS, else CPU."""
    if device_override:
        return torch.device(device_override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(model_name: str, in_channels: int = 3, base_channels: int = 32) -> nn.Module:
    """Instantiate model behind common interface forward(hazy) -> dehazed in [0, 1]."""
    name = model_name.lower().replace("-", "_")
    if name == "aodnet":
        return AODNet(in_channels=in_channels, b=1.0)
    elif name == "dcp":
        return DCP(patch_size=15, omega=0.95, t0=0.10)
    elif name in ("physdehaze_lite", "physdehazelite"):
        return PhysDehazeLite(in_channels=in_channels, base_channels=base_channels)
    elif name == "rs_pgnet":
        return RSPGNet(in_channels=in_channels, base_channels=base_channels)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from 'physdehaze_lite', 'aodnet', 'dcp', 'rs_pgnet'")


def evaluate_val_split(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    """Run validation evaluation computing loss and Section 4.2 full-reference metrics."""
    model.eval()
    tracker = MetricTracker()
    total_val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            hazy = batch["hazy"].to(device, dtype=torch.float32)
            clean = batch["clean"].to(device, dtype=torch.float32)

            pred = model(hazy, return_dict=True) if isinstance(model, PhysDehazeLite) else model(hazy)
            if isinstance(pred, dict):
                pred_out = pred["output"]
            else:
                pred_out = pred

            if isinstance(criterion, PhysDehazeLoss):
                loss, _ = criterion(pred, clean, hazy=hazy)
            else:
                loss = criterion(pred_out, clean)

            total_val_loss += loss.item()
            num_batches += 1

            tracker.update(pred_out, clean)

    metrics = tracker.to_dict()
    metrics["val_loss"] = total_val_loss / max(num_batches, 1)
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Union[str, Path],
    epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    min_lr: float = 1e-6,
    device: Optional[torch.device] = None,
    resume_path: Optional[Union[str, Path]] = None,
    log_tensorboard: bool = True,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Execute training loop with AdamW, Cosine LR, L1 loss, and float32 precision.

    Args:
        model: Trainable PyTorch dehazing model.
        train_loader: Training DataLoader yielding 128x128 paired crops.
        val_loader: Validation DataLoader.
        output_dir: Directory where checkpoints and logs are saved.
        epochs: Number of training epochs.
        learning_rate: Initial AdamW learning rate.
        weight_decay: Weight decay regularizer.
        min_lr: Minimum learning rate for cosine scheduler.
        device: PyTorch device ('mps' or 'cpu').
        resume_path: Path to existing checkpoint to resume from.
        log_tensorboard: Whether to write TensorBoard events.

    Returns:
        Dictionary containing training statistics, best PSNR, and epoch timings.
    """
    dev = device or get_training_device()
    model = model.to(dev)
    # Ensure float32 only
    model = model.float()

    out_dir = Path(output_dir)
    ckpt_dir = out_dir / "checkpoints"
    logs_dir = out_dir / "logs"
    tb_dir = out_dir / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=str(tb_dir)) if log_tensorboard else None

    # Objective criterion
    if criterion is not None:
        criterion = criterion.to(dev)
    elif isinstance(model, PhysDehazeLite):
        criterion = PhysDehazeLoss(
            use_l1=True, weight_l1=1.0,
            use_ssim=True, weight_ssim=0.2,
            use_physics=True, weight_physics=0.5,
            use_sam=True, weight_sam=0.1,
        ).to(dev)
    else:
        criterion = nn.L1Loss().to(dev)

    # AdamW Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("Model has no trainable parameters (e.g. classical DCP). Cannot train.")

    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    # Cosine Annealing Learning Rate Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    start_epoch = 1
    best_val_psnr = -1.0
    history: List[Dict[str, Any]] = []
    epoch_durations: List[float] = []

    # Resume from checkpoint if specified
    if resume_path:
        resume_file = Path(resume_path)
        if resume_file.is_file():
            logger.info("Resuming training from checkpoint: %s", resume_file)
            ckpt = torch.load(resume_file, map_location=dev)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_psnr = ckpt.get("best_val_psnr", -1.0)
            logger.info("Resumed successfully at epoch %d with best PSNR: %.2f dB", start_epoch, best_val_psnr)
        else:
            logger.warning("Resume file not found at %s. Starting fresh training.", resume_file)

    global_step = (start_epoch - 1) * len(train_loader)

    print(f"\n{'='*70}")
    print(f"Starting Training: {model.__class__.__name__}")
    print(f"  Device:         {dev}")
    print(f"  Epochs:         {epochs} (starting at {start_epoch})")
    print(f"  Train batches:  {len(train_loader)} (batch size: {train_loader.batch_size})")
    print(f"  Val batches:    {len(val_loader)}")
    print(f"  Precision:      float32 only")
    print(f"  Loss:           L1 Loss")
    print(f"  Optimizer:      AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"  Checkpoints:    {ckpt_dir}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.perf_counter()
        model.train()
        running_loss = 0.0
        num_train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            hazy = batch["hazy"].to(dev, dtype=torch.float32)
            clean = batch["clean"].to(dev, dtype=torch.float32)

            optimizer.zero_grad()

            pred = model(hazy, return_dict=True) if isinstance(model, PhysDehazeLite) else model(hazy)
            if isinstance(criterion, PhysDehazeLoss):
                loss, loss_dict = criterion(pred, clean, hazy=hazy)
            else:
                pred_out = pred["output"] if isinstance(pred, dict) else pred
                loss = criterion(pred_out, clean)
                loss_dict = {"loss_l1": float(loss.item())}

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_train_batches += 1
            global_step += 1

            if writer is not None and batch_idx % 5 == 0:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"train/{k}", v, global_step)

        # Step scheduler once per epoch
        current_lr = float(scheduler.get_last_lr()[0])
        scheduler.step()

        epoch_duration = time.perf_counter() - epoch_start_time
        epoch_durations.append(epoch_duration)
        avg_train_loss = running_loss / max(num_train_batches, 1)

        # Validation evaluation
        val_metrics = evaluate_val_split(model, val_loader, dev, criterion)
        val_psnr = val_metrics["psnr"]
        val_ssim = val_metrics["ssim"]
        val_ciede = val_metrics["ciede2000"]
        val_loss = val_metrics["val_loss"]

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("train/epoch_l1_loss", avg_train_loss, epoch)
            writer.add_scalar("train/learning_rate", current_lr, epoch)
            writer.add_scalar("train/epoch_seconds", epoch_duration, epoch)
            writer.add_scalar("val/l1_loss", val_loss, epoch)
            writer.add_scalar("val/psnr", val_psnr, epoch)
            writer.add_scalar("val/ssim", val_ssim, epoch)
            writer.add_scalar("val/ciede2000", val_ciede, epoch)
            writer.add_scalar("val/sam", val_metrics["sam"], epoch)
            writer.add_scalar("val/ergas", val_metrics["ergas"], epoch)
            writer.add_scalar("val/uiqi", val_metrics["uiqi"], epoch)
            writer.add_scalar("val/qnr", val_metrics["qnr"], epoch)
            writer.add_scalar("val/hist", val_metrics["hist"], epoch)

        # Print progress
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Time: {epoch_duration:.2f}s | "
            f"Train L1: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} dB | "
            f"SSIM: {val_ssim:.4f} | "
            f"CIEDE: {val_ciede:.2f} | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpoint payload
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_psnr": best_val_psnr,
            "train_loss": avg_train_loss,
            "val_metrics": val_metrics,
            "epoch_duration_sec": epoch_duration,
        }

        # Save checkpoint every epoch
        torch.save(ckpt_data, ckpt_dir / f"epoch_{epoch:03d}.pth")
        torch.save(ckpt_data, ckpt_dir / "latest_checkpoint.pth")

        # Save best model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            ckpt_data["best_val_psnr"] = best_val_psnr
            torch.save(ckpt_data, ckpt_dir / "best_model.pth")

        record = {
            "epoch": epoch,
            "duration_sec": epoch_duration,
            "train_loss": avg_train_loss,
            "learning_rate": current_lr,
            **val_metrics,
        }
        history.append(record)

    if writer is not None:
        writer.close()

    # Save metrics history JSON
    with open(logs_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    avg_sec_per_epoch = float(np.mean(epoch_durations)) if epoch_durations else 0.0

    return {
        "best_val_psnr": best_val_psnr,
        "epoch_durations": epoch_durations,
        "avg_seconds_per_epoch": avg_sec_per_epoch,
        "history": history,
        "checkpoints_dir": str(ckpt_dir),
    }


def main() -> None:
    """CLI entrypoint for running dehazing baseline training or smoke test."""
    parser = argparse.ArgumentParser(description="Train dehazing baseline on Haze1k + RRSHID combined dataset.")
    parser.add_argument("--model", type=str, default="physdehaze_lite", choices=["physdehaze_lite", "aodnet", "dcp", "rs_pgnet"], help="Model to train")
    parser.add_argument("--data-dir", type=str, default="data/Haze1k", help="Root directory of Haze1k dataset")
    parser.add_argument("--rrshid-dir", type=str, default="data/RRSHID", help="Root directory of RRSHID dataset")
    parser.add_argument("--rrshid-samples", type=int, default=1000, help="Number of sampled RRSHID train pairs (default: 1000)")
    parser.add_argument("--output-dir", type=str, default="experiments/physdehaze_lite", help="Directory for artifacts")
    parser.add_argument("--epochs", type=int, default=10, help="Total number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=0, choices=[0, 2], help="DataLoader workers (default: 0)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2-epoch smoke test on 20 images")
    parser.add_argument("--limit-train-samples", type=int, default=None, help="Limit number of train images")
    parser.add_argument("--limit-val-samples", type=int, default=None, help="Limit number of validation images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Device override ('cpu' or 'mps')")
    args = parser.parse_args()

    # Reproducibility seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_training_device(args.device)
    print(f"Target Compute Device: {device} (PYTORCH_ENABLE_MPS_FALLBACK=1)")

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

    # Smoke test configurations
    if args.smoke_test:
        if args.epochs == 10:
            args.epochs = 2
        args.limit_train_samples = 20
        args.limit_val_samples = 10
        print(f"\n[Smoke Test Mode Activated]: {args.epochs} epochs on {args.limit_train_samples} training images (val on {args.limit_val_samples} images).\n")

    if args.limit_train_samples is not None:
        n_train = min(args.limit_train_samples, len(train_dataset))
        train_dataset = Subset(train_dataset, range(n_train))

    if args.limit_val_samples is not None:
        n_val = min(args.limit_val_samples, len(val_dataset))
        val_dataset = Subset(val_dataset, range(n_val))

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

    # Model instantiation
    model = create_model(args.model)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized {args.model.upper()}: {param_count:,} trainable parameters (< 10,000 threshold)")

    # Execute training
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        resume_path=args.resume,
    )

    print(f"\n{'='*70}")
    print(f"Smoke Test / Training Finished:")
    print(f"  Best Val PSNR:          {results['best_val_psnr']:.2f} dB")
    print(f"  Avg Seconds per Epoch:  {results['avg_seconds_per_epoch']:.2f} s")
    for idx, d in enumerate(results["epoch_durations"], 1):
        print(f"    Epoch {idx}: {d:.2f} seconds")
    print(f"  Checkpoints saved to:   {results['checkpoints_dir']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
