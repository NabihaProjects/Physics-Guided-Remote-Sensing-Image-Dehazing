"""Evaluation engine for testing models on paired or unpaired remote sensing imagery."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from src.metrics.metrics import MetricTracker
from src.metrics.profiler import profile_model
from src.utils.logger import setup_logger


class Evaluator:
    """Evaluates dehazing models, collects quantitative metrics, and saves outputs."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | None = None,
        output_dir: Path | str = "eval_results",
    ) -> None:
        """Initialize evaluator.

        Args:
            model: Dehazing model (RS-DCP, AOD-Net, RS-PGNet).
            device: PyTorch device.
            output_dir: Destination path for results and generated images.
        """
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("evaluator", self.output_dir / "eval.log")

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset: Any,
        save_images: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate model across an entire dataset.

        Args:
            dataset: Dataset yielding items with 'hazy', 'clean', and 'name'.
            save_images: Whether to save output dehazed images to disk.

        Returns:
            Dictionary containing aggregated metrics and sample counts.
        """
        tracker = MetricTracker()
        records: List[Dict[str, Any]] = []

        if save_images:
            images_dir = self.output_dir / "dehazed_images"
            images_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Evaluating %d test samples...", len(dataset))

        for idx in range(len(dataset)):
            sample = dataset[idx]
            hazy = sample["hazy"].unsqueeze(0).to(self.device)
            clean = sample["clean"].unsqueeze(0).to(self.device)
            name = sample.get("name", f"sample_{idx:04d}")

            pred = self.model(hazy)
            if isinstance(pred, dict):
                pred = pred["output"]

            tracker.update(pred, clean)

            if save_images:
                # Save dehazed image
                pred_np = pred[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
                pred_uint8 = (pred_np * 255.0).astype(np.uint8)
                Image.fromarray(pred_uint8).save(images_dir / f"{name}_dehazed.png")

        summary = tracker.summary()
        simple_metrics = tracker.to_dict()

        self.logger.info(
            "Evaluation complete: PSNR = %.2f dB | SSIM = %.4f | CIEDE2000 = %.2f",
            simple_metrics["psnr"],
            simple_metrics["ssim"],
            simple_metrics["ciede2000"],
        )

        # Export metrics JSON
        with open(self.output_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return simple_metrics

