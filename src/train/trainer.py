"""Unified trainer and evaluation pipeline for remote sensing dehazing models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.losses.compound_loss import CompoundDehazeLoss
from src.metrics.metrics import MetricTracker
from src.utils.config import ExperimentConfig
from src.utils.logger import setup_logger
from src.utils.experiment_logger import ExperimentLogger


class DehazeTrainer:
    """Standardized trainer and evaluator for remote sensing dehazing experiments."""

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_dir: Path | str,
        device: torch.device | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: PyTorch model to train.
            config: Validated ExperimentConfig instance.
            train_loader: DataLoader for training set.
            val_loader: DataLoader for validation/test set.
            experiment_dir: Output path for checkpoints, metrics, and logs.
            device: PyTorch device.
        """
        self.config = config
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.exp_dir = Path(experiment_dir)

        self.logger = setup_logger("trainer", self.exp_dir / "logs" / "train.log")
        self.exp_logger = ExperimentLogger(
            log_dir=self.exp_dir,
            experiment_name=self.config.project.experiment_name,
            use_tensorboard=self.config.logging.use_tensorboard,
            use_wandb=self.config.logging.use_wandb,
            config=self.config.to_dict(),
            logger=self.logger,
        )

        self.criterion = CompoundDehazeLoss(config.loss_weights).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.min_lr,
        )

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=(self.device.type == "cuda" and config.hardware.mixed_precision),
            )
        else:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=(self.device.type == "cuda" and config.hardware.mixed_precision)
            )
        self.best_psnr = -1.0
        self.history: list[dict] = []
        self.global_step = 0

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            hazy = batch["hazy"].to(self.device)
            clean = batch["clean"].to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda" and self.config.hardware.mixed_precision)):
                pred = self.model(hazy)
                pred_t = None
                if isinstance(pred, dict):
                    pred_t = pred.get("transmission")
                    pred = pred["output"]

                loss, loss_dict = self.criterion(pred, clean, predicted_transmission=pred_t)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log step metrics to TensorBoard / W&B
            if self.global_step % 5 == 0:
                self.exp_logger.log_metrics(loss_dict, step=self.global_step, prefix="train_step")

        self.scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        current_lr = float(self.scheduler.get_last_lr()[0])

        self.exp_logger.log_scalar("train/loss_epoch", avg_loss, step=epoch)
        self.exp_logger.log_scalar("train/learning_rate", current_lr, step=epoch)

        return {"train_loss": avg_loss, "lr": current_lr}

    @torch.no_grad()
    def evaluate(self, save_visuals: bool = False, epoch: int = 0) -> Dict[str, float]:
        """Run full evaluation on validation set."""
        self.model.eval()
        tracker = MetricTracker()
        sample_to_save: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

        for batch in self.val_loader:
            hazy = batch["hazy"].to(self.device)
            clean = batch["clean"].to(self.device)

            pred = self.model(hazy)
            if isinstance(pred, dict):
                pred = pred["output"]

            tracker.update(pred, clean)

            if sample_to_save is None and save_visuals:
                sample_to_save = (hazy[0].cpu(), pred[0].cpu(), clean[0].cpu())

        summary = tracker.to_dict()

        # Log validation metrics
        self.exp_logger.log_metrics(summary, step=epoch, prefix="val")

        if sample_to_save is not None:
            self._save_comparison_image(
                hazy=sample_to_save[0],
                pred=sample_to_save[1],
                clean=sample_to_save[2],
                epoch=epoch,
            )
            # Log triplet visual to TensorBoard / W&B
            triplet = torch.cat([sample_to_save[0], sample_to_save[1], sample_to_save[2]], dim=2)
            self.exp_logger.log_image("val/triplet_hazy_pred_gt", triplet, step=epoch)

        return summary

    def _save_comparison_image(
        self,
        hazy: torch.Tensor,
        pred: torch.Tensor,
        clean: torch.Tensor,
        epoch: int,
    ) -> None:
        """Save triplet comparison image: Hazy | Dehazed | Clean."""
        h_np = hazy.permute(1, 2, 0).clamp(0, 1).numpy()
        p_np = pred.permute(1, 2, 0).clamp(0, 1).numpy()
        c_np = clean.permute(1, 2, 0).clamp(0, 1).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        axes[0].imshow(h_np)
        axes[0].set_title("Hazy Input")
        axes[0].axis("off")

        axes[1].imshow(p_np)
        axes[1].set_title("Dehazed (Model)")
        axes[1].axis("off")

        axes[2].imshow(c_np)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = self.exp_dir / "visualizations" / f"epoch_{epoch:03d}.png"
        fig.savefig(save_path, dpi=120)
        plt.close(fig)

    def train(self) -> Dict[str, float]:
        """Execute full training procedure according to config."""
        self.logger.info("Starting training for %d epochs...", self.config.training.epochs)

        try:
            for epoch in range(1, self.config.training.epochs + 1):
                train_metrics = self.train_epoch(epoch)

                if epoch % self.config.training.eval_interval == 0:
                    val_metrics = self.evaluate(save_visuals=True, epoch=epoch)
                    self.logger.info(
                        "Epoch %d/%d - Loss: %.4f | Val PSNR: %.2f dB | SSIM: %.4f | CIEDE2000: %.2f",
                        epoch,
                        self.config.training.epochs,
                        train_metrics["train_loss"],
                        val_metrics["psnr"],
                        val_metrics["ssim"],
                        val_metrics["ciede2000"],
                    )

                    epoch_record = {
                        "epoch": epoch,
                        **train_metrics,
                        **val_metrics,
                    }
                    self.history.append(epoch_record)

                    # Save best checkpoint
                    if val_metrics["psnr"] > self.best_psnr:
                        self.best_psnr = val_metrics["psnr"]
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "best_psnr": self.best_psnr,
                                "config": self.config.to_dict(),
                            },
                            self.exp_dir / "checkpoints" / "best_model.pth",
                        )
        finally:
            self.exp_logger.close()

        # Save metrics history
        with open(self.exp_dir / "metrics" / "history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {"best_psnr": self.best_psnr}
