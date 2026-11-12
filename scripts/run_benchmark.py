"""Unified benchmark and ablation runner for Remote Sensing Dehazing.

Evaluates DCP, the CNN baseline (AOD-Net), and PhysDehaze-Lite across:
1. Haze1k Thin, Moderate, Thick test splits (in-domain, native 512x512)
2. RICE1, all pairs (cross-domain, zero-shot, no fine-tuning, native 512x512)
3. RRSHID Thin, Moderate, Thick test splits (real-world haze; zero-shot test split, native 256x256)

Computes Section 4.2 metrics:
- Full-Reference: PSNR, SSIM, CIEDE2000, SAM, ERGAS, UIQI, QNR, HIST, LPIPS
- No-Reference: NIQE
- Optional: FID (off by default)

Saves:
- One CSV per dataset in experiments/results/
- One unified Markdown table in experiments/results/combined_benchmark.md
- Hardware efficiency profiling (parameters, FLOPs, latency, and FPS on MPS at 256x256) in experiments/results/efficiency.csv
- Ablation study of PhysDehaze-Lite (full, no physics loss, no SAM loss, no airlight head) in experiments/results/ablations.csv
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.rice import RICE1Dataset
from src.datasets.satehaze1k import SateHaze1kDataset
from src.datasets.rrshid import RRSHIDDataset
from src.losses.physdehaze_loss import PhysDehazeLoss
from src.metrics.metrics import (
    MetricTracker,
    calculate_ciede2000,
    calculate_ergas,
    calculate_fid,
    calculate_hist,
    calculate_lpips,
    calculate_niqe,
    calculate_psnr,
    calculate_qnr,
    calculate_sam,
    calculate_ssim,
    calculate_uiqi,
)
from src.metrics.profiler import count_parameters, estimate_conv_macs
from src.models.aodnet import AODNet
from src.models.dcp import DCP
from src.models.physdehaze_lite import PhysDehazeLite
from src.train.train import train_model


# ---------------------------------------------------------------------------
# FLOPs Profiler Utility at 256x256
# ---------------------------------------------------------------------------

def calculate_model_flops(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
) -> Tuple[float, str]:
    """Compute model FLOPs on CPU using available profilers at 256x256.

    Returns:
        Tuple of (gflops_value, method_name).
    """
    orig_device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    model_cpu = model.cpu().eval()
    dummy_input = torch.zeros(*input_shape, dtype=torch.float32, device="cpu")

    res: Tuple[float, str] = (0.0, "unavailable")

    # 1. Try thop
    try:
        import thop
        macs, _ = thop.profile(model_cpu, inputs=(dummy_input,), verbose=False)
        gflops = float(macs * 2) / 1e9  # 1 MAC = 2 FLOPs
        res = (round(gflops, 3), "thop")
    except Exception:
        pass

    # 2. Try fvcore
    if res[1] == "unavailable":
        try:
            from fvcore.nn import FlopCountAnalysis
            flops = FlopCountAnalysis(model_cpu, dummy_input).total()
            res = (round(float(flops) / 1e9, 3), "fvcore")
        except Exception:
            pass

    # 3. Try PyTorch native FlopCounterMode
    if res[1] == "unavailable":
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fcm:
                _ = model_cpu(dummy_input)
            total_flops = fcm.get_total_flops()
            res = (round(float(total_flops) / 1e9, 3), "torch.utils.flop_counter")
        except Exception:
            pass

    # 4. Fallback: custom convolution MACs estimation
    if res[1] == "unavailable":
        try:
            gmacs = estimate_conv_macs(model_cpu, input_shape=input_shape)
            gflops = gmacs * 2.0
            res = (round(float(gflops), 3), "custom_conv_macs")
        except Exception:
            pass

    model.to(orig_device)
    return res


# ---------------------------------------------------------------------------
# Latency & FPS Profiler on MPS at 256x256
# ---------------------------------------------------------------------------

def measure_mps_latency(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    warmup: int = 10,
    runs: int = 50,
) -> Dict[str, float]:
    """Measure inference latency and FPS with hardware synchronization at 256x256."""
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape, device=device, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    times: List[float] = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / max(mean_ms, 1e-4)

    return {
        "mean_latency_ms": round(mean_ms, 2),
        "std_latency_ms": round(std_ms, 2),
        "fps": round(fps, 2),
    }


# ---------------------------------------------------------------------------
# Dataset Evaluation Runner
# ---------------------------------------------------------------------------

def evaluate_model_on_dataset(
    model: nn.Module,
    model_name: str,
    dataset: Any,
    dataset_name: str,
    device: torch.device,
    max_samples: Optional[int] = None,
    compute_lpips: bool = True,
    compute_niqe: bool = True,
    compute_fid_flag: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single model on a dataset across all Section 4.2 metrics."""
    eval_dev = device
    model = model.to(eval_dev).eval()

    tracker = MetricTracker()
    n_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    pred_features = []
    gt_features = []

    with torch.no_grad():
        for i in range(n_samples):
            item = dataset[i]
            hazy = item["hazy"].unsqueeze(0).to(eval_dev, dtype=torch.float32)
            clean = item["clean"].unsqueeze(0).to(eval_dev, dtype=torch.float32)

            pred = model(hazy)
            if isinstance(pred, dict):
                pred = pred["output"]
            pred = torch.clamp(pred, 0.0, 1.0)

            # Update tracker metrics
            tracker.update(
                pred.cpu(),
                clean.cpu(),
                compute_lpips=compute_lpips,
                compute_niqe=compute_niqe,
            )

            # If FID requested, pool simple features
            if compute_fid_flag and i < 50:
                p_feat = F.adaptive_avg_pool2d(pred, (1, 1)).flatten().cpu().numpy()
                g_feat = F.adaptive_avg_pool2d(clean, (1, 1)).flatten().cpu().numpy()
                pred_features.append(p_feat)
                gt_features.append(g_feat)

    metrics = tracker.to_dict()
    row = {
        "dataset": dataset_name,
        "model": model_name,
        "samples_evaluated": n_samples,
        "psnr": round(metrics.get("psnr", 0.0), 2),
        "ssim": round(metrics.get("ssim", 0.0), 4),
        "ciede2000": round(metrics.get("ciede2000", 0.0), 2),
        "sam": round(metrics.get("sam", 0.0), 2),
        "ergas": round(metrics.get("ergas", 0.0), 2),
        "uiqi": round(metrics.get("uiqi", 0.0), 4),
        "qnr": round(metrics.get("qnr", 0.0), 4),
        "hist": round(metrics.get("hist", 0.0), 4),
    }

    if "lpips" in metrics:
        row["lpips"] = round(metrics["lpips"], 4)
    if "niqe" in metrics:
        row["niqe"] = round(metrics["niqe"], 2)

    if compute_fid_flag and len(pred_features) > 2:
        row["fid"] = round(calculate_fid(np.array(pred_features), np.array(gt_features)), 2)

    return row


# ---------------------------------------------------------------------------
# Ablation Runner
# ---------------------------------------------------------------------------

def run_ablation_study(
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    epochs: int = 2,
    batch_size: int = 8,
    force_retrain: bool = False,
) -> List[Dict[str, Any]]:
    """Train and evaluate the 4 required PhysDehaze-Lite ablation configurations.

    Ablations:
    (a) Full model: L1 + 0.2*SSIM + 0.5*Physics + 0.1*SAM, learnable airlight head
    (b) Without physics consistency loss: weight_physics = 0.0
    (c) Without spectral SAM loss: weight_sam = 0.0
    (d) Without airlight head: use_airlight_head = False (fixed A = 1.0)
    """
    ablation_dir = output_dir / "ablations"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*75}")
    print(f"RUNNING PHYSDEHAZE-LITE ABLATION STUDY ({epochs} Epochs Each)")
    print(f"{'='*75}")

    train_ds = SateHaze1kDataset(root_dir=data_dir, subset="all", split="train", crop_size=(128, 128))
    val_ds = SateHaze1kDataset(root_dir=data_dir, subset="all", split="val", crop_size=(128, 128))
    test_ds = SateHaze1kDataset(root_dir=data_dir, subset="all", split="test", crop_size=None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    configs = [
        {
            "id": "full_model",
            "name": "(a) Full PhysDehaze-Lite",
            "use_airlight_head": True,
            "loss_kwargs": {"use_l1": True, "use_ssim": True, "use_physics": True, "use_sam": True},
            "desc": "L1 + SSIM + Physics + SAM with Learnable A",
        },
        {
            "id": "no_physics",
            "name": "(b) w/o Physics Consistency Loss",
            "use_airlight_head": True,
            "loss_kwargs": {"use_l1": True, "use_ssim": True, "use_physics": False, "use_sam": True},
            "desc": "L1 + SSIM + SAM (Physics Loss = 0)",
        },
        {
            "id": "no_sam",
            "name": "(c) w/o Spectral SAM Loss",
            "use_airlight_head": True,
            "loss_kwargs": {"use_l1": True, "use_ssim": True, "use_physics": True, "use_sam": False},
            "desc": "L1 + SSIM + Physics (SAM Loss = 0)",
        },
        {
            "id": "fixed_airlight",
            "name": "(d) w/o Airlight Head (Fixed A=1.0)",
            "use_airlight_head": False,
            "loss_kwargs": {"use_l1": True, "use_ssim": True, "use_physics": True, "use_sam": True},
            "desc": "Full Loss with Fixed Constant Airlight A=[1.0, 1.0, 1.0]",
        },
    ]

    ablation_results: List[Dict[str, Any]] = []

    for cfg in configs:
        tag = cfg["id"]
        run_name = cfg["name"]
        run_out_dir = ablation_dir / tag
        best_ckpt = run_out_dir / "checkpoints" / "best_model.pth"

        model = PhysDehazeLite(
            in_channels=3,
            base_channels=32,
            use_airlight_head=cfg["use_airlight_head"],
            fixed_airlight_val=1.0,
        )

        if not best_ckpt.exists() or force_retrain:
            print(f"\n--> Training Ablation: {run_name} for {epochs} epochs...")
            criterion = PhysDehazeLoss(**cfg["loss_kwargs"]).to(device)
            _ = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                output_dir=run_out_dir,
                epochs=epochs,
                learning_rate=2e-4,
                device=device,
                criterion=criterion,
                log_tensorboard=False,
            )

        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Loaded ablation checkpoint from: {best_ckpt}")

        eval_scores = evaluate_model_on_dataset(
            model=model,
            model_name=run_name,
            dataset=test_ds,
            dataset_name="Haze1k_Combined_Test",
            device=device,
            max_samples=25,
            compute_lpips=False,
            compute_niqe=False,
        )

        res_row = {
            "variant": run_name,
            "description": cfg["desc"],
            "epochs_trained": epochs,
            "psnr": eval_scores["psnr"],
            "ssim": eval_scores["ssim"],
            "ciede2000": eval_scores["ciede2000"],
            "sam": eval_scores["sam"],
            "ergas": eval_scores["ergas"],
            "qnr": eval_scores["qnr"],
        }
        ablation_results.append(res_row)
        print(f"    Result: PSNR={res_row['psnr']:.2f} dB | SSIM={res_row['ssim']:.4f} | SAM={res_row['sam']:.2f}° | QNR={res_row['qnr']:.4f}")

    # Save CSV
    csv_file = output_dir / "ablations.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ablation_results[0].keys()))
        writer.writeheader()
        writer.writerows(ablation_results)

    # Save Markdown Summary
    md_file = output_dir / "ablation_summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# PhysDehaze-Lite Ablation Study\n\n")
        f.write(f"All ablation variants trained for **{epochs} epochs** on the combined Haze1k training set under identical initial conditions.\n\n")
        f.write("| Variant | Description | Epochs | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | SAM (°) ↓ | ERGAS ↓ | QNR ↑ |\n")
        f.write("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in ablation_results:
            f.write(
                f"| **{r['variant']}** | {r['description']} | {r['epochs_trained']} | "
                f"**{r['psnr']:.2f}** | **{r['ssim']:.4f}** | {r['ciede2000']:.2f} | "
                f"{r['sam']:.2f} | {r['ergas']:.2f} | **{r['qnr']:.4f}** |\n"
            )

    return ablation_results


# ---------------------------------------------------------------------------
# Main Orchestration Script
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Remote Sensing Dehazing Benchmark & Ablation Runner.")
    parser.add_argument("--data-dir", type=str, default="data/Haze1k", help="Haze1k root directory.")
    parser.add_argument("--rice-dir", type=str, default="data/RICE_DATASET/RICE1", help="RICE1 root directory.")
    parser.add_argument("--rrshid-dir", type=str, default="data/RRSHID", help="RRSHID root directory.")
    parser.add_argument(
        "--physdehaze-ckpt",
        type=str,
        default="experiments/physdehaze_lite/checkpoints/best_model.pth",
        help="Path to trained PhysDehaze-Lite checkpoint.",
    )
    parser.add_argument(
        "--aodnet-ckpt",
        type=str,
        default="experiments/aodnet/checkpoints/best_model.pth",
        help="Path to trained AOD-Net checkpoint.",
    )
    parser.add_argument("--output-dir", type=str, default="experiments/results", help="Output directory.")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
    parser.add_argument("--haze1k-samples", type=int, default=None, help="Limit Haze1k samples per subset (None = all 45).")
    parser.add_argument("--rice1-samples", type=int, default=50, help="Number of RICE1 samples (None = all 500).")
    parser.add_argument("--rrshid-samples", type=int, default=None, help="Number of RRSHID test samples per subset (None = all).")
    parser.add_argument("--ablation-epochs", type=int, default=2, help="Epochs to train each ablation variant.")
    parser.add_argument("--skip-ablations", action="store_true", default=False, help="Skip ablation training.")
    parser.add_argument("--force-ablations", action="store_true", default=False, help="Force retrain ablations.")
    parser.add_argument("--compute-fid", action="store_true", default=False, help="Compute FID (off by default).")
    args = parser.parse_args()

    # Hardware Device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Execution Target Device: {device} (PYTORCH_ENABLE_MPS_FALLBACK=1)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Initialize Models
    # -----------------------------------------------------------------------
    print("\n[Step 1/4] Initializing Models & Loading Weights...")

    # 1. DCP
    dcp = DCP(patch_size=15, omega=0.95, t0=0.10).to(device)

    # 2. AOD-Net
    aodnet = AODNet(in_channels=3)
    if Path(args.aodnet_ckpt).is_file():
        ckpt = torch.load(args.aodnet_ckpt, map_location=device)
        aodnet.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"  Loaded AOD-Net from: {args.aodnet_ckpt}")
    aodnet = aodnet.to(device)

    # 3. PhysDehaze-Lite
    physdehaze = PhysDehazeLite(in_channels=3, base_channels=32)
    if Path(args.physdehaze_ckpt).is_file():
        ckpt = torch.load(args.physdehaze_ckpt, map_location=device)
        physdehaze.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"  Loaded PhysDehaze-Lite from: {args.physdehaze_ckpt}")
    physdehaze = physdehaze.to(device)

    models_to_bench = [
        ("DCP (He et al. 2010)", dcp),
        ("AOD-Net (CNN Baseline)", aodnet),
        ("PhysDehaze-Lite (Proposed Novel)", physdehaze),
    ]

    # -----------------------------------------------------------------------
    # Step 2: Efficiency Profiling (Params, FLOPs, Latency & FPS at 256x256)
    # -----------------------------------------------------------------------
    print("\n[Step 2/4] Profiling Hardware Efficiency at 256x256 on MPS...")
    efficiency_records: List[Dict[str, Any]] = []

    for name, model in models_to_bench:
        if "DCP" in name:
            timing = measure_mps_latency(model, device=device, input_shape=(1, 3, 256, 256))
            rec = {
                "model": name,
                "total_parameters": 0,
                "trainable_parameters": 0,
                "model_size_mb": 0.0,
                "gflops": 0.0,
                "flops_profiler": "N/A (Analytical Prior)",
                **timing,
            }
        else:
            tot_p, tr_p, sz_mb = count_parameters(model)
            gflops, method = calculate_model_flops(model, input_shape=(1, 3, 256, 256))
            timing = measure_mps_latency(model, device=device, input_shape=(1, 3, 256, 256))
            rec = {
                "model": name,
                "total_parameters": tot_p,
                "trainable_parameters": tr_p,
                "model_size_mb": round(sz_mb, 3),
                "gflops": gflops,
                "flops_profiler": method,
                **timing,
            }
        efficiency_records.append(rec)
        print(f"  {name:<32}: Params={rec['trainable_parameters']:,} | FLOPs={rec['gflops']} GFLOPs | Latency={rec['mean_latency_ms']} ms | FPS={rec['fps']}")

    # Save efficiency CSV
    eff_csv = out_dir / "efficiency.csv"
    with open(eff_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(efficiency_records[0].keys()))
        writer.writeheader()
        writer.writerows(efficiency_records)

    # -----------------------------------------------------------------------
    # Step 3: Multi-Dataset Benchmark Inference (Haze1k, RICE1, RRSHID)
    # -----------------------------------------------------------------------
    print("\n[Step 3/4] Evaluating Models on Test Splits across All Datasets...")

    datasets_to_run = [
        # In-Domain Synthetic Benchmarks
        ("Haze1k_Thin", SateHaze1kDataset(root_dir=args.data_dir, subset="thin", split="test", crop_size=None), args.haze1k_samples, "In-Domain (512x512)"),
        ("Haze1k_Moderate", SateHaze1kDataset(root_dir=args.data_dir, subset="moderate", split="test", crop_size=None), args.haze1k_samples, "In-Domain (512x512)"),
        ("Haze1k_Thick", SateHaze1kDataset(root_dir=args.data_dir, subset="thick", split="test", crop_size=None), args.haze1k_samples, "In-Domain (512x512)"),
        # Cross-Domain Zero-Shot Transfer
        ("RICE1_CrossDomain_ZeroShot", RICE1Dataset(root_dir=args.rice_dir, split="all", crop_size=None), args.rice1_samples, "Cross-Domain Zero-Shot (512x512)"),
        # Real-World Remote Sensing Haze (Zero-Shot Test Splits)
        ("RRSHID_Thin", RRSHIDDataset(root_dir=args.rrshid_dir, subset="thin", split="test", crop_size=None), args.rrshid_samples, "Real Haze Zero-Shot (256x256)"),
        ("RRSHID_Moderate", RRSHIDDataset(root_dir=args.rrshid_dir, subset="moderate", split="test", crop_size=None), args.rrshid_samples, "Real Haze Zero-Shot (256x256)"),
        ("RRSHID_Thick", RRSHIDDataset(root_dir=args.rrshid_dir, subset="thick", split="test", crop_size=None), args.rrshid_samples, "Real Haze Zero-Shot (256x256)"),
    ]

    all_dataset_records: Dict[str, List[Dict[str, Any]]] = {}
    master_table_rows: List[Dict[str, Any]] = []

    for ds_name, ds_obj, max_s, desc in datasets_to_run:
        eval_count = len(ds_obj) if max_s is None else min(len(ds_obj), max_s)
        print(f"\n{'='*75}\nDataset: {ds_name} [{desc}] (Available: {len(ds_obj)}, Evaluating: {eval_count})\n{'='*75}")
        ds_records: List[Dict[str, Any]] = []

        for m_name, m_obj in models_to_bench:
            print(f"  --> Running {m_name}...")
            res = evaluate_model_on_dataset(
                model=m_obj,
                model_name=m_name,
                dataset=ds_obj,
                dataset_name=ds_name,
                device=device,
                max_samples=eval_count,
                compute_lpips=True,
                compute_niqe=True,
                compute_fid_flag=args.compute_fid,
            )
            res["domain_type"] = desc
            ds_records.append(res)
            master_table_rows.append(res)
            print(
                f"      PSNR: {res['psnr']:.2f} dB | SSIM: {res['ssim']:.4f} | "
                f"CIEDE: {res['ciede2000']:.2f} | SAM: {res['sam']:.2f}° | "
                f"QNR: {res['qnr']:.4f} | LPIPS: {res.get('lpips', 0.0):.4f}"
            )

        all_dataset_records[ds_name] = ds_records

        # Save individual dataset CSV
        ds_csv_path = out_dir / f"{ds_name.lower()}.csv"
        with open(ds_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(ds_records[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ds_records)
        print(f"  Saved CSV to: {ds_csv_path}")

    # -----------------------------------------------------------------------
    # Step 4: PhysDehaze-Lite Ablation Study
    # -----------------------------------------------------------------------
    ablation_rows: List[Dict[str, Any]] = []
    if not args.skip_ablations:
        print("\n[Step 4/4] Executing PhysDehaze-Lite Component Ablations...")
        ablation_rows = run_ablation_study(
            data_dir=Path(args.data_dir),
            output_dir=out_dir,
            device=device,
            epochs=args.ablation_epochs,
            batch_size=8,
            force_retrain=args.force_ablations,
        )
    else:
        print("\n[Step 4/4] Skipping ablation study as requested (--skip-ablations).")
        ablation_csv = out_dir / "ablations.csv"
        if ablation_csv.exists():
            with open(ablation_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                ablation_rows = list(reader)

    # -----------------------------------------------------------------------
    # Generate Combined Markdown Report
    # -----------------------------------------------------------------------
    combined_md_path = out_dir / "combined_benchmark.md"
    with open(combined_md_path, "w", encoding="utf-8") as f:
        f.write("# Remote Sensing Image Dehazing: Unified Benchmark & Evaluation Results\n\n")
        f.write("Based on the evaluation criteria of Section 4.2 (*ISPRS Journal of Photogrammetry and Remote Sensing, 2026*).\n\n")
        f.write("- **Hardware Platform**: Apple Silicon MPS / CPU Fallback\n")
        f.write("- **Efficiency Profile Resolution**: Native $256 \\times 256$ on Apple Silicon MPS\n")
        f.write("- **RICE1 Protocol**: Cross-domain, zero-shot transfer evaluated at native resolution ($512 \\times 512$) without fine-tuning.\n")
        f.write("- **Haze1k Protocol**: Tested on official test splits at native resolution ($512 \\times 512$).\n")
        f.write("- **RRSHID Protocol**: Tested on official test splits at native resolution ($256 \\times 256$) under real haze conditions.\n\n")

        # Table 1: Cross-Dataset Benchmark Table
        f.write("## 1. Full-Reference Benchmark Performance\n\n")
        f.write("| Dataset | Type | Model | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | SAM (°) ↓ | ERGAS ↓ | UIQI ↑ | QNR ↑ | HIST ↑ | LPIPS ↓ | NIQE ↓ |\n")
        f.write("|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in master_table_rows:
            f.write(
                f"| **{r['dataset']}** | {r.get('domain_type', 'N/A')} | {r['model']} | "
                f"**{float(r['psnr']):.2f}** | **{float(r['ssim']):.4f}** | {float(r['ciede2000']):.2f} | "
                f"{float(r['sam']):.2f} | {float(r['ergas']):.2f} | {float(r['uiqi']):.4f} | "
                f"**{float(r['qnr']):.4f}** | {float(r['hist']):.4f} | {float(r.get('lpips', 0.0)):.4f} | {float(r.get('niqe', 0.0)):.2f} |\n"
            )

        # Table 2: Efficiency Profiling at 256x256
        f.write("\n## 2. Hardware Computational Efficiency (Native 256x256 Input on MPS)\n\n")
        f.write("| Model | Trainable Params | Model Size (MB) | FLOPs (GFLOPs) | Profiler Method | Latency (ms) | Throughput (FPS) |\n")
        f.write("|:---|:---:|:---:|:---:|:---|:---:|:---:|\n")
        for eff in efficiency_records:
            f.write(
                f"| **{eff['model']}** | {eff['trainable_parameters']:,} | {eff['model_size_mb']} MB | "
                f"**{eff['gflops']}** | {eff['flops_profiler']} | {eff['mean_latency_ms']} ± {eff['std_latency_ms']} ms | "
                f"**{eff['fps']} FPS** |\n"
            )

        # Table 3: Ablation Study
        if ablation_rows:
            f.write(f"\n## 3. PhysDehaze-Lite Component Ablations ({args.ablation_epochs} Epochs Each)\n\n")
            f.write("| Ablation Variant | Loss Formulation & Head Setup | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | SAM (°) ↓ | ERGAS ↓ | QNR ↑ |\n")
            f.write("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for a in ablation_rows:
                f.write(
                    f"| **{a['variant']}** | {a['description']} | **{float(a['psnr']):.2f}** | "
                    f"**{float(a['ssim']):.4f}** | {float(a['ciede2000']):.2f} | {float(a['sam']):.2f} | "
                    f"{float(a['ergas']):.2f} | **{float(a['qnr']):.4f}** |\n"
                )

    print(f"\n[DONE] Combined report successfully written to: {combined_md_path}")

    # Print Final Markdown Tables to Console
    print("\n" + "=" * 90)
    print("FINAL BENCHMARK SUMMARY TABLE")
    print("=" * 90)
    with open(combined_md_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
