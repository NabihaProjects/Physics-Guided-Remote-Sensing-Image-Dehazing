"""Benchmark evaluation script for SateHaze1k / Haze1k test sets.

Evaluates dehazing models against Section 4.2 metrics:
- PSNR (dB, skimage)
- SSIM (skimage)
- CIEDE2000 color difference (skimage)
- LPIPS perceptual similarity (AlexNet)
- SAM (Spectral Angle Mapper in degrees, Eq. 15)
- ERGAS (Relative Dimensionless Global Error, r=1, Eq. 16)
- UIQI (Universal Image Quality Index, Eq. 17)
- QNR (exp(-SAM/10) * exp(-ERGAS/100), Eq. 18)
- HIST (Histogram similarity, Eq. 20)
- NIQE (Natural Image Quality Evaluator, pyiqa CPU)

Outputs:
- Full metrics summary per model and per subset (thin, moderate, thick)
- CSV results table saved to experiments/benchmarks/
- Markdown summary table
- Qualitative comparison side-by-side figures
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.satehaze1k import SateHaze1kDataset
from src.metrics.metrics import (
    MetricTracker,
    calculate_psnr,
    calculate_ssim,
    calculate_ciede2000,
    calculate_lpips,
    calculate_sam,
    calculate_ergas,
    calculate_uiqi,
    calculate_qnr,
    calculate_hist,
    calculate_niqe,
)
from src.models.aodnet import AODNet
from src.models.dcp import DCP
from src.models.physdehaze_lite import PhysDehazeLite


def evaluate_model_on_dataset(
    model: nn.Module,
    dataset: Any,
    device: torch.device,
    compute_lpips: bool = True,
    compute_niqe: bool = False,
    max_samples: Optional[int] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Run full evaluation of a model on a dataset across all Section 4.2 metrics.

    Args:
        model: Dehazing model with forward(hazy) -> dehazed in [0, 1].
        dataset: Dataset returning {'hazy': tensor, 'clean': tensor, 'name': str}.
        device: PyTorch device ('mps' or 'cpu').
        compute_lpips: Whether to compute AlexNet LPIPS.
        compute_niqe: Whether to compute pyiqa NIQE.
        max_samples: Optional limit on number of test samples.
        save_dir: Optional directory to save output images.

    Returns:
        Dictionary of mean metric scores.
    """
    model.eval()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    tracker = MetricTracker()
    lpips_vals = []
    niqe_vals = []
    latencies = []

    num_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    with torch.no_grad():
        for i in range(num_samples):
            item = dataset[i]
            hazy = item["hazy"].unsqueeze(0).to(device, dtype=torch.float32)
            clean = item["clean"].unsqueeze(0).to(device, dtype=torch.float32)
            name = item.get("name", f"sample_{i:04d}")

            start_t = time.perf_counter()
            pred = model(hazy)
            if isinstance(pred, dict):
                pred = pred["output"]
            pred = torch.clamp(pred, 0.0, 1.0)
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(latency_ms)

            tracker.update(pred, clean)

            if compute_lpips:
                try:
                    lp_score = calculate_lpips(pred, clean)
                    lpips_vals.append(lp_score)
                except Exception:
                    pass

            if compute_niqe:
                try:
                    niqe_score = calculate_niqe(pred)
                    niqe_vals.append(niqe_score)
                except Exception:
                    pass

            if save_dir is not None and i < 10:
                # Save visual sample
                pred_np = (pred[0].cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(pred_np).save(save_dir / f"{name}_dehazed.png")

    summary = tracker.summary()
    results = {
        "psnr": float(summary["psnr"]["mean"]),
        "ssim": float(summary["ssim"]["mean"]),
        "ciede2000": float(summary["ciede2000"]["mean"]),
        "sam": float(summary["sam"]["mean"]),
        "ergas": float(summary["ergas"]["mean"]),
        "uiqi": float(summary["uiqi"]["mean"]),
        "qnr": float(summary["qnr"]["mean"]),
        "hist": float(summary["hist"]["mean"]),
        "mean_latency_ms": float(np.mean(latencies)),
        "fps": float(1000.0 / max(np.mean(latencies), 1e-4)),
    }

    if lpips_vals:
        results["lpips"] = float(np.mean(lpips_vals))
    if niqe_vals:
        results["niqe"] = float(np.mean(niqe_vals))

    return results


def save_visual_comparison(
    hazy: torch.Tensor,
    clean: torch.Tensor,
    dcp_out: torch.Tensor,
    phys_out: torch.Tensor,
    save_path: Path,
    title_suffix: str = "",
) -> None:
    """Save a 1x4 side-by-side comparison figure: Hazy | DCP | PhysDehaze-Lite | Clear GT."""
    import matplotlib.pyplot as plt

    def to_np(t: torch.Tensor) -> np.ndarray:
        if t.dim() == 4:
            t = t[0]
        return (t.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

    img_hazy = to_np(hazy)
    img_dcp = to_np(dcp_out)
    img_phys = to_np(phys_out)
    img_gt = to_np(clean)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(img_hazy)
    axes[0].set_title("Hazy Input", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(img_dcp)
    axes[1].set_title("DCP (He et al. 2010)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(img_phys)
    axes[2].set_title("PhysDehaze-Lite (Ours)", fontsize=12, fontweight="bold", color="darkgreen")
    axes[2].axis("off")

    axes[3].imshow(img_gt)
    axes[3].set_title("Ground Truth (Clear)", fontsize=12)
    axes[3].axis("off")

    plt.suptitle(f"Remote Sensing Dehazing Comparison {title_suffix}", fontsize=14, y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dehazing Baselines and PhysDehaze-Lite.")
    parser.add_argument("--data-dir", type=str, default="data/Haze1k", help="Path to Haze1k dataset.")
    parser.add_argument(
        "--physdehaze-ckpt",
        type=str,
        default="experiments/physdehaze_lite/checkpoints/best_model.pth",
        help="Path to trained PhysDehaze-Lite checkpoint.",
    )
    parser.add_argument(
        "--aodnet-ckpt",
        type=str,
        default="experiments/smoke_test_aodnet/checkpoints/best_model.pth",
        help="Path to trained AOD-Net checkpoint.",
    )
    parser.add_argument("--output-dir", type=str, default="experiments/benchmarks", help="Output directory.")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
    parser.add_argument("--samples-per-subset", type=int, default=None, help="Limit samples for speed.")
    parser.add_argument("--compute-lpips", action="store_true", default=True)
    parser.add_argument("--compute-niqe", action="store_true", default=False)
    args = parser.parse_args()

    # Device selection
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Benchmark Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir = out_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instantiate Models
    models: Dict[str, nn.Module] = {}

    # Classical DCP
    models["DCP (Classical)"] = DCP(patch_size=15, omega=0.95, t0=0.10)

    # AOD-Net
    aodnet = AODNet(in_channels=3)
    aod_ckpt_candidate = Path("experiments/aodnet/checkpoints/best_model.pth")
    if not aod_ckpt_candidate.is_file():
        aod_ckpt_candidate = Path(args.aodnet_ckpt)
    if aod_ckpt_candidate.is_file():
        ckpt = torch.load(aod_ckpt_candidate, map_location=device)
        aodnet.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"Loaded AOD-Net from: {aod_ckpt_candidate}")
    models["AOD-Net"] = aodnet.to(device)

    # PhysDehaze-Lite
    phys = PhysDehazeLite(in_channels=3, base_channels=32)
    if Path(args.physdehaze_ckpt).is_file():
        ckpt = torch.load(args.physdehaze_ckpt, map_location=device)
        phys.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"Loaded PhysDehaze-Lite from: {args.physdehaze_ckpt}")
    else:
        print(f"Warning: PhysDehaze-Lite checkpoint not found at {args.physdehaze_ckpt}")
    models["PhysDehaze-Lite (Ours)"] = phys.to(device)

    # Count parameters
    param_counts = {
        "DCP (Classical)": 0,
        "AOD-Net": sum(p.numel() for p in aodnet.parameters() if p.requires_grad),
        "PhysDehaze-Lite (Ours)": sum(p.numel() for p in phys.parameters() if p.requires_grad),
    }

    # 2. Datasets
    subsets = ["thin", "moderate", "thick"]
    all_benchmark_rows: List[Dict[str, Any]] = []

    for subset in subsets:
        print(f"\n{'='*60}\nEvaluating on Haze1k: {subset.upper()} Test Split\n{'='*60}")
        test_ds = SateHaze1kDataset(
            root_dir=args.data_dir,
            subset=subset,
            split="test",
            crop_size=None,  # Full 512x512 resolution for official evaluation
        )
        print(f"Found {len(test_ds)} test pairs for {subset}.")

        # Generate visual side-by-side comparison for the first 3 images
        for v_idx in range(min(3, len(test_ds))):
            sample = test_ds[v_idx]
            h_t = sample["hazy"].unsqueeze(0).to(device)
            c_t = sample["clean"].unsqueeze(0).to(device)
            with torch.no_grad():
                dcp_t = models["DCP (Classical)"](h_t)
                phys_t = models["PhysDehaze-Lite (Ours)"](h_t)
                if isinstance(phys_t, dict):
                    phys_t = phys_t["output"]

            save_visual_comparison(
                h_t, c_t, dcp_t, phys_t,
                visuals_dir / f"compare_{subset}_{v_idx:02d}.png",
                title_suffix=f"({subset.capitalize()} #{v_idx + 1})",
            )

        for model_name, model in models.items():
            print(f"--> Running {model_name}...")
            # For DCP, run on CPU for guided filter compatibility
            m_dev = torch.device("cpu") if "DCP" in model_name else device
            model_eval = model.to(m_dev)

            scores = evaluate_model_on_dataset(
                model=model_eval,
                dataset=test_ds,
                device=m_dev,
                compute_lpips=args.compute_lpips,
                compute_niqe=args.compute_niqe,
                max_samples=args.samples_per_subset,
                save_dir=out_dir / f"images_{subset}_{model_name.replace(' ', '_').lower()}",
            )

            row = {
                "subset": subset,
                "model": model_name,
                "params": param_counts[model_name],
                **scores,
            }
            all_benchmark_rows.append(row)
            print(
                f"    PSNR: {scores['psnr']:.2f} dB | SSIM: {scores['ssim']:.4f} | "
                f"CIEDE: {scores['ciede2000']:.2f} | SAM: {scores['sam']:.2f}° | "
                f"QNR: {scores['qnr']:.4f} | FPS: {scores['fps']:.1f}"
            )

    # 3. Save to CSV
    csv_path = out_dir / "benchmark_results.csv"
    fieldnames = [
        "subset", "model", "params", "psnr", "ssim", "ciede2000",
        "sam", "ergas", "uiqi", "qnr", "hist", "lpips", "mean_latency_ms", "fps",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in fieldnames if k in all_benchmark_rows[0]])
        writer.writeheader()
        for r in all_benchmark_rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames if k in r})

    print(f"\nSaved CSV benchmark results to: {csv_path}")

    # 4. Print Overall Summary Table
    print(f"\n{'='*90}")
    print(f"{'Model':<24} | {'Subset':<9} | {'Params':<8} | {'PSNR (dB)':<9} | {'SSIM':<6} | {'CIEDE':<6} | {'SAM (°)':<7} | {'QNR':<6}")
    print(f"{'-'*90}")
    for r in all_benchmark_rows:
        print(
            f"{r['model']:<24} | {r['subset']:<9} | {r['params']:<8} | "
            f"{r['psnr']:<9.2f} | {r['ssim']:<6.4f} | {r['ciede2000']:<6.2f} | "
            f"{r['sam']:<7.2f} | {r['qnr']:<6.4f}"
        )
    print(f"{'='*90}")


if __name__ == "__main__":
    main()

