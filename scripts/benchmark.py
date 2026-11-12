"""Benchmark runner comparing RS-DCP, AOD-Net, and RS-PGNet on Remote Sensing Dehazing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.data.dataset import RemoteSensingDehazeDataset
from src.metrics.metrics import MetricTracker
from src.metrics.profiler import profile_model
from src.models.aodnet import AODNet
from src.models.dcp import RSDCP
from src.models.rs_pgnet import RSPGNet
from src.utils.config import ExperimentConfig, load_config
from src.utils.logger import setup_logger
from src.utils.seed import get_device, set_seed


def evaluate_method(
    name: str,
    model: nn.Module,
    dataset: RemoteSensingDehazeDataset,
    device: torch.device,
    is_trainable: bool = True,
) -> Dict[str, Any]:
    """Evaluate a single dehazing method across the test dataset."""
    model = model.to(device)
    model.eval()

    tracker = MetricTracker()
    first_sample_visual = None

    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            hazy = item["hazy"].unsqueeze(0).to(device)
            clean = item["clean"].unsqueeze(0).to(device)

            out = model(hazy)
            if isinstance(out, dict):
                out = out["output"]

            tracker.update(out, clean)

            if first_sample_visual is None:
                first_sample_visual = {
                    "hazy": hazy[0].cpu(),
                    "pred": out[0].cpu(),
                    "clean": clean[0].cpu(),
                }

    metrics = tracker.to_dict()

    # Profile hardware efficiency
    if is_trainable:
        profile = profile_model(
            model=model,
            input_shape=(1, 3, dataset.image_size, dataset.image_size),
            device=device,
            num_runs=15,
        )
    else:
        # Non-parametric / classical method
        profile = {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "model_size_mb": 0.0,
            "gmacs": 0.0,
            "mean_latency_ms": 0.0,
            "std_latency_ms": 0.0,
            "fps": 0.0,
        }

    return {
        "method": name,
        "psnr": round(metrics["psnr"], 2),
        "ssim": round(metrics["ssim"], 4),
        "ciede2000": round(metrics["ciede2000"], 2),
        **profile,
        "_sample": first_sample_visual,
    }


def run_benchmark(
    config: ExperimentConfig,
    num_test_samples: int = 20,
    output_dir: str = "experiments/benchmark_results",
    checkpoint_dir: str | None = None,
) -> List[Dict[str, Any]]:
    """Run full reproducible benchmark suite."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("benchmark", out_path / "benchmark.log")

    set_seed(config.project.seed)
    device = get_device(config.hardware.device)
    logger.info("Running benchmark on device: %s with %d test samples", device, num_test_samples)

    # Prepare shared test dataset
    test_ds = RemoteSensingDehazeDataset(
        root_dir=config.data.data_dir,
        split="test",
        image_size=config.data.image_size,
        num_synthetic_samples=num_test_samples,
        seed=config.project.seed + 9999,
        augment=False,
    )

    results = []

    # 1. Unprocessed Hazy Input (Baseline Reference)
    class IdentityModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    logger.info("Evaluating [1/4]: Unprocessed Hazy Input (Lower Bound)...")
    res_input = evaluate_method("Hazy Input (Unprocessed)", IdentityModel(), test_ds, device, is_trainable=False)
    results.append(res_input)

    # 2. Classical RS-DCP Baseline
    logger.info("Evaluating [2/4]: RS-DCP (Classical Prior)...")
    rs_dcp = RSDCP(patch_size=15, filter_radius=15)
    res_dcp = evaluate_method("RS-DCP (He et al. Adapted)", rs_dcp, test_ds, device, is_trainable=False)
    results.append(res_dcp)

    # 3. Learning Baseline: AOD-Net
    logger.info("Evaluating [3/4]: AOD-Net (Deep Learning Baseline)...")
    aodnet = AODNet(in_channels=3)
    if checkpoint_dir and (Path(checkpoint_dir) / "aodnet_best.pth").exists():
        ckpt = torch.load(Path(checkpoint_dir) / "aodnet_best.pth", map_location=device)
        aodnet.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded pre-trained weights for AOD-Net.")
    res_aod = evaluate_method("AOD-Net (ICCV 2017)", aodnet, test_ds, device, is_trainable=True)
    results.append(res_aod)

    # 4. Proposed Method: RS-PGNet
    logger.info("Evaluating [4/4]: RS-PGNet (Proposed Physics-Guided Network)...")
    rs_pgnet = RSPGNet(in_channels=3, base_channels=config.model.base_channels, num_refine_blocks=3)
    if checkpoint_dir and (Path(checkpoint_dir) / "rs_pgnet_best.pth").exists():
        ckpt = torch.load(Path(checkpoint_dir) / "rs_pgnet_best.pth", map_location=device)
        rs_pgnet.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded pre-trained weights for RS-PGNet.")
    res_pgnet = evaluate_method("RS-PGNet (Proposed Novel)", rs_pgnet, test_ds, device, is_trainable=True)
    results.append(res_pgnet)

    # Save visual comparison grid
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    sample = res_pgnet["_sample"]
    axes[0].imshow(sample["hazy"].permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0].set_title("Hazy Input")
    axes[0].axis("off")

    axes[1].imshow(res_dcp["_sample"]["pred"].permute(1, 2, 0).clamp(0, 1).numpy())
    axes[1].set_title("RS-DCP")
    axes[1].axis("off")

    axes[2].imshow(res_aod["_sample"]["pred"].permute(1, 2, 0).clamp(0, 1).numpy())
    axes[2].set_title("AOD-Net")
    axes[2].axis("off")

    axes[3].imshow(sample["pred"].permute(1, 2, 0).clamp(0, 1).numpy())
    axes[3].set_title("RS-PGNet (Proposed)")
    axes[3].axis("off")

    axes[4].imshow(sample["clean"].permute(1, 2, 0).clamp(0, 1).numpy())
    axes[4].set_title("Ground Truth")
    axes[4].axis("off")

    plt.tight_layout()
    fig.savefig(out_path / "benchmark_comparison.png", dpi=150)
    plt.close(fig)

    # Clean internal samples from result dictionaries for export
    clean_results = []
    for r in results:
        entry = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_results.append(entry)

    # Export CSV
    csv_file = out_path / "benchmark_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "method",
            "psnr",
            "ssim",
            "ciede2000",
            "total_parameters",
            "model_size_mb",
            "mean_latency_ms",
            "fps",
            "gmacs",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean_results)

    # Export JSON
    json_file = out_path / "benchmark_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)

    # Print formatted Markdown table
    header = f"| {'Method':<28} | {'PSNR (dB) ↑':<11} | {'SSIM ↑':<8} | {'CIEDE2000 ↓':<12} | {'Params':<10} | {'Latency (ms) ↓':<14} |"
    sep = f"|{'-'*30}|{'-'*13}|{'-'*10}|{'-'*14}|{'-'*12}|{'-'*16}|"
    rows = [header, sep]
    for r in clean_results:
        p_str = f"{r['total_parameters']:,}" if r['total_parameters'] > 0 else "N/A"
        l_str = f"{r['mean_latency_ms']:.2f}" if r['mean_latency_ms'] > 0 else "N/A"
        rows.append(
            f"| {r['method']:<28} | {r['psnr']:<11.2f} | {r['ssim']:<8.4f} | {r['ciede2000']:<12.2f} | {p_str:<10} | {l_str:<14} |"
        )
    table_str = "\n".join(rows)
    logger.info("\nBenchmark Summary Table:\n%s", table_str)

    with open(out_path / "benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write("# Remote Sensing Dehazing Benchmark Results\n\n" + table_str + "\n")

    return clean_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RS Dehazing Benchmark.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--samples", type=int, default=15, help="Number of benchmark test samples.")
    parser.add_argument("--output-dir", type=str, default="experiments/benchmark_results")
    parser.add_argument("--checkpoints", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    run_benchmark(
        config=cfg,
        num_test_samples=args.samples,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoints,
    )

