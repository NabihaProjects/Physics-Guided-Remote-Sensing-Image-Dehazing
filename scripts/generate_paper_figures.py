"""Generate publication-quality analysis figures inspired by the survey paper:

1. Feature A (Fig. 16 style): Epistemic Uncertainty Maps using MC Dropout (N=10)
   - Evaluates on 3 Haze1k test scenes and 3 RICE1 test scenes
   - Displays: Hazy | Dehazed (Mean) | Clear Ground Truth | Uncertainty Heatmap (Std Dev)
   - Saved as PNG at 200 dpi in experiments/figures/

2. Feature B (Fig. 12 style): Radiometric Consistency Scatter Plot
   - Predicted vs. Ground-Truth pixel intensities (5,000 sampled pixels per haze subset)
   - Plotted for DCP, AOD-Net (CNN baseline), and PhysDehaze-Lite
   - Colored by haze level (Thin, Moderate, Thick) with 1:1 reference line and R^2 / RMSE metrics
   - Saved as PNG at 200 dpi in experiments/figures/

3. Visual Comparison Grid:
   - 5-column grid: Hazy | DCP | AOD-Net | PhysDehaze-Lite | Clear
   - 5 diverse remote sensing test images across Haze1k and RICE1
   - Saved as experiments/figures/visual_comparison_grid.png at 200 dpi
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.rice import RICE1Dataset
from src.datasets.satehaze1k import SateHaze1kDataset
from src.models.aodnet import AODNet
from src.models.dcp import DCP
from src.models.physdehaze_lite import PhysDehazeLite


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) or (1, C, H, W) tensor in [0, 1] to (H, W, C) numpy array."""
    if t.dim() == 4:
        t = t[0]
    return t.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy()


# ---------------------------------------------------------------------------
# Feature A: MC Dropout Uncertainty Map (Fig. 16 Style)
# ---------------------------------------------------------------------------

def compute_mc_dropout_uncertainty(
    model: PhysDehazeLite,
    hazy_tensor: torch.Tensor,
    num_samples: int = 10,
    dropout_rate: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run MC Dropout inference N times and compute per-pixel mean and standard deviation.

    Args:
        model: PhysDehazeLite model.
        hazy_tensor: (1, 3, H, W) input tensor on compute device.
        num_samples: Number of stochastic forward passes (N=10).
        dropout_rate: Probability of feature dropping.

    Returns:
        Tuple of (mean_prediction_rgb, uncertainty_map_2d).
    """
    model.eval()
    model.enable_mc_dropout(dropout_rate=dropout_rate)

    predictions: List[np.ndarray] = []

    with torch.no_grad():
        for _ in range(num_samples):
            pred = model(hazy_tensor)
            if isinstance(pred, dict):
                pred = pred["output"]
            pred = pred.clamp(0.0, 1.0)
            predictions.append(pred[0].cpu().permute(1, 2, 0).numpy())

    model.disable_mc_dropout()
    model.eval()

    # Stack predictions: shape (N, H, W, 3)
    preds_stack = np.stack(predictions, axis=0)

    # Mean predicted dehazed image: (H, W, 3)
    mean_pred = np.mean(preds_stack, axis=0)

    # Per-pixel standard deviation across passes: (H, W, 3) -> averaged across RGB channels (H, W)
    std_map = np.std(preds_stack, axis=0)
    uncertainty_map = np.mean(std_map, axis=-1)

    return mean_pred, uncertainty_map


def plot_single_uncertainty(
    hazy_np: np.ndarray,
    dehazed_np: np.ndarray,
    clear_np: np.ndarray,
    uncertainty_np: np.ndarray,
    title_prefix: str,
    save_path: Path,
) -> None:
    """Save 1x4 figure: Hazy | Dehazed | Clear | Uncertainty Heatmap."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)

    # 1. Hazy
    axes[0].imshow(hazy_np)
    axes[0].set_title("Hazy Input $I(x)$", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # 2. Dehazed
    axes[1].imshow(dehazed_np)
    axes[1].set_title("PhysDehaze-Lite Mean $\\bar{J}(x)$", fontsize=13, fontweight="bold", color="darkgreen")
    axes[1].axis("off")

    # 3. Ground Truth Clear
    axes[2].imshow(clear_np)
    axes[2].set_title("Ground Truth (Clear)", fontsize=13, fontweight="bold")
    axes[2].axis("off")

    # 4. Uncertainty Heatmap
    im = axes[3].imshow(uncertainty_np, cmap="inferno", vmin=0.0, vmax=max(float(np.percentile(uncertainty_np, 99.5)), 0.05))
    axes[3].set_title("Uncertainty Heatmap $\\sigma(x)$ [N=10]", fontsize=13, fontweight="bold", color="crimson")
    axes[3].axis("off")
    cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Epistemic Std Dev $\\sigma$", fontsize=11)

    plt.suptitle(f"{title_prefix} — Epistemic Uncertainty Estimation (Paper Fig. 16)", fontsize=15, y=1.03)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved uncertainty map to: {save_path}")


def run_uncertainty_analysis(
    model: PhysDehazeLite,
    haze1k_ds: Any,
    rice1_ds: Any,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Generate uncertainty maps for 3 Haze1k test images and 3 RICE1 test images."""
    print("\n--> Generating Feature A: Uncertainty Maps (Paper Fig. 16)...")

    # 1. 3 Haze1k test images (Thin, Moderate, Thick)
    haze1k_indices = [0, 15, 30]  # Representative samples
    for i, idx in enumerate(haze1k_indices):
        item = haze1k_ds[idx]
        hazy_t = item["hazy"].unsqueeze(0).to(device)
        clear_np = tensor_to_numpy(item["clean"])
        hazy_np = tensor_to_numpy(item["hazy"])

        dehazed_mean, uncert = compute_mc_dropout_uncertainty(model, hazy_t, num_samples=10, dropout_rate=0.15)

        out_file = output_dir / f"uncertainty_haze1k_sample_{i + 1}.png"
        plot_single_uncertainty(
            hazy_np, dehazed_mean, clear_np, uncert,
            f"Haze1k Test Sample #{i + 1}", out_file
        )

    # 2. 3 RICE1 test images
    rice1_indices = [2, 12, 22]
    for i, idx in enumerate(rice1_indices):
        item = rice1_ds[idx]
        hazy_t = item["hazy"].unsqueeze(0).to(device)
        clear_np = tensor_to_numpy(item["clean"])
        hazy_np = tensor_to_numpy(item["hazy"])

        dehazed_mean, uncert = compute_mc_dropout_uncertainty(model, hazy_t, num_samples=10, dropout_rate=0.15)

        out_file = output_dir / f"uncertainty_rice1_sample_{i + 1}.png"
        plot_single_uncertainty(
            hazy_np, dehazed_mean, clear_np, uncert,
            f"RICE1 Zero-Shot Sample #{i + 1}", out_file
        )

    # Combined 6-panel summary figure
    fig, axes = plt.subplots(6, 4, figsize=(16, 22), constrained_layout=True)
    all_samples = [("Haze1k", haze1k_ds, haze1k_indices), ("RICE1", rice1_ds, rice1_indices)]

    row_idx = 0
    for name, ds, indices in all_samples:
        for idx in indices:
            item = ds[idx]
            hazy_t = item["hazy"].unsqueeze(0).to(device)
            clear_np = tensor_to_numpy(item["clean"])
            hazy_np = tensor_to_numpy(item["hazy"])

            dehazed_mean, uncert = compute_mc_dropout_uncertainty(model, hazy_t, num_samples=10, dropout_rate=0.15)

            axes[row_idx, 0].imshow(hazy_np)
            axes[row_idx, 0].set_ylabel(f"{name} #{row_idx + 1}", fontsize=11, fontweight="bold")
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(dehazed_mean)
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(clear_np)
            axes[row_idx, 2].axis("off")

            im = axes[row_idx, 3].imshow(uncert, cmap="inferno", vmin=0.0, vmax=max(float(np.percentile(uncert, 99.5)), 0.05))
            axes[row_idx, 3].axis("off")

            if row_idx == 0:
                axes[row_idx, 0].set_title("Hazy Input $I(x)$", fontsize=12, fontweight="bold")
                axes[row_idx, 1].set_title("Dehazed $\\bar{J}(x)$", fontsize=12, fontweight="bold", color="darkgreen")
                axes[row_idx, 2].set_title("Ground Truth", fontsize=12, fontweight="bold")
                axes[row_idx, 3].set_title("Uncertainty Heatmap $\\sigma(x)$", fontsize=12, fontweight="bold", color="crimson")

            row_idx += 1

    fig.suptitle("Epistemic Uncertainty Estimation via MC Dropout (Paper Fig. 16)", fontsize=16, y=1.01)
    combined_uncert_path = output_dir / "uncertainty_maps_combined.png"
    fig.savefig(combined_uncert_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved combined uncertainty figure to: {combined_uncert_path}")


# ---------------------------------------------------------------------------
# Feature B: Radiometric Consistency Scatter Plot (Fig. 12 Style)
# ---------------------------------------------------------------------------

def run_radiometric_consistency_analysis(
    models: Dict[str, nn.Module],
    subsets_data: Dict[str, Any],
    device: torch.device,
    output_dir: Path,
    samples_per_subset: int = 5000,
) -> None:
    """Plot predicted vs. ground-truth pixel values sampled across haze levels with 1:1 line.

    Args:
        models: Dict of models {'DCP': dcp, 'AOD-Net': aodnet, 'PhysDehaze-Lite': phys}.
        subsets_data: Dict of subsets {'thin': ds, 'moderate': ds, 'thick': ds}.
        device: PyTorch device.
        output_dir: Directory to save figure.
        samples_per_subset: Number of pixels to sample per subset (default 5000).
    """
    print("\n--> Generating Feature B: Radiometric Consistency Scatter Plot (Paper Fig. 12)...")

    # Subset color mapping
    palette = {
        "thin": {"color": "#27ae60", "label": "Thin Haze", "alpha": 0.45},
        "moderate": {"color": "#e67e22", "label": "Moderate Haze", "alpha": 0.45},
        "thick": {"color": "#c0392b", "label": "Thick Haze", "alpha": 0.45},
    }

    # Collect pixel samples for each model and subset
    sampled_points: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {
        m_name: {} for m_name in models.keys()
    }

    np.random.seed(42)

    for subset_name, dataset in subsets_data.items():
        print(f"  Sampling {samples_per_subset} pixels from Haze1k {subset_name.upper()}...")
        # Gather pixels from the first 5 test images of this subset
        n_imgs = min(5, len(dataset))
        pixels_per_img = samples_per_subset // n_imgs

        for i in range(n_imgs):
            item = dataset[i]
            hazy_t = item["hazy"].unsqueeze(0)
            clean_t = item["clean"].unsqueeze(0)

            # Ground truth pixels
            gt_flat = clean_t[0].permute(1, 2, 0).numpy().reshape(-1, 3)

            for m_name, model in models.items():
                m_dev = torch.device("cpu") if "DCP" in m_name else device
                with torch.no_grad():
                    pred_t = model(hazy_t.to(m_dev))
                    if isinstance(pred_t, dict):
                        pred_t = pred_t["output"]
                    pred_t = pred_t.clamp(0.0, 1.0)
                pred_flat = pred_t[0].cpu().permute(1, 2, 0).numpy().reshape(-1, 3)

                # Randomly sample pixel indices
                idx_choice = np.random.choice(len(gt_flat), size=pixels_per_img, replace=False)
                # Average RGB channels for radiometric intensity
                p_sample = np.mean(pred_flat[idx_choice], axis=-1)
                g_sample = np.mean(gt_flat[idx_choice], axis=-1)

                if subset_name not in sampled_points[m_name]:
                    sampled_points[m_name][subset_name] = ([], [])
                sampled_points[m_name][subset_name][0].extend(g_sample)
                sampled_points[m_name][subset_name][1].extend(p_sample)

    # Plot 1x3 scatter figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    for ax_idx, (m_name, _) in enumerate(models.items()):
        ax = axes[ax_idx]

        all_gt: List[float] = []
        all_pred: List[float] = []

        for subset_name, (gt_pts, pred_pts) in sampled_points[m_name].items():
            gt_arr = np.array(gt_pts[:samples_per_subset])
            pred_arr = np.array(pred_pts[:samples_per_subset])
            all_gt.extend(gt_arr)
            all_pred.extend(pred_arr)

            style = palette[subset_name]
            ax.scatter(
                gt_arr, pred_arr,
                c=style["color"],
                label=style["label"],
                alpha=style["alpha"],
                s=10,
                edgecolors="none",
            )

        all_gt_np = np.array(all_gt)
        all_pred_np = np.array(all_pred)

        # 1:1 reference line
        ax.plot([0.0, 1.0], [0.0, 1.0], "k--", lw=1.8, label="1:1 Perfect Fidelity")

        # Regression metrics
        r_corr = float(np.corrcoef(all_gt_np, all_pred_np)[0, 1])
        r2 = r_corr ** 2
        rmse = float(np.sqrt(np.mean((all_gt_np - all_pred_np) ** 2)))

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("Ground-Truth Reflectance (Clear)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted Reflectance (Dehazed)", fontsize=11, fontweight="bold")
        ax.set_title(f"{m_name}", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.5)

        # Annotate R^2 and RMSE box
        stats_text = f"$R^2 = {r2:.4f}$\n$r = {r_corr:.4f}$\n$\\mathrm{{RMSE}} = {rmse:.4f}$"
        ax.text(
            0.05, 0.92, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#ccc"),
        )

        if ax_idx == 0:
            ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    fig.suptitle(
        "Radiometric Consistency: Predicted vs. Ground-Truth Reflectance (Paper Fig. 12)",
        fontsize=15,
        fontweight="bold",
        y=1.04,
    )
    scatter_save_path = output_dir / "radiometric_consistency_scatter.png"
    fig.savefig(scatter_save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved radiometric consistency scatter plot to: {scatter_save_path}")


# ---------------------------------------------------------------------------
# Visual Comparison Grid (5 Test Images)
# ---------------------------------------------------------------------------

def run_visual_comparison_grid(
    models: Dict[str, nn.Module],
    test_samples: List[Tuple[str, Dict[str, torch.Tensor]]],
    device: torch.device,
    output_dir: Path,
) -> None:
    """Generate 5-image x 5-column visual comparison grid.

    Columns: Hazy | DCP | AOD-Net | PhysDehaze-Lite | Clear
    """
    print("\n--> Generating Visual Comparison Grid (5 Test Images)...")

    n_rows = len(test_samples)
    fig, axes = plt.subplots(n_rows, 5, figsize=(18, 3.6 * n_rows), constrained_layout=True)

    headers = ["Hazy Input", "DCP (He et al.)", "AOD-Net (Baseline)", "PhysDehaze-Lite (Ours)", "Ground Truth"]

    for row_idx, (sample_label, sample_data) in enumerate(test_samples):
        hazy_t = sample_data["hazy"].unsqueeze(0)
        clean_t = sample_data["clean"].unsqueeze(0)

        img_hazy = tensor_to_numpy(hazy_t)
        img_clean = tensor_to_numpy(clean_t)

        with torch.no_grad():
            # DCP
            dcp_out = tensor_to_numpy(models["DCP (Classical)"](hazy_t.cpu()))
            # AOD-Net
            aod_out = tensor_to_numpy(models["AOD-Net (Baseline)"](hazy_t.to(device)))
            # PhysDehaze-Lite
            phys_res = models["PhysDehaze-Lite (Proposed Novel)"](hazy_t.to(device))
            if isinstance(phys_res, dict):
                phys_res = phys_res["output"]
            phys_out = tensor_to_numpy(phys_res)

        images = [img_hazy, dcp_out, aod_out, phys_out, img_clean]

        for col_idx, img in enumerate(images):
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.axis("off")

            if row_idx == 0:
                is_bold = col_idx in (0, 3, 4)
                color = "darkgreen" if col_idx == 3 else "black"
                ax.set_title(headers[col_idx], fontsize=13, fontweight="bold" if is_bold else "normal", color=color)

            if col_idx == 0:
                ax.text(
                    -0.08, 0.5, sample_label,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=90,
                )

    fig.suptitle(
        "Remote Sensing Dehazing Visual Comparison Grid",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    grid_save_path = output_dir / "visual_comparison_grid.png"
    fig.savefig(grid_save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved visual comparison grid to: {grid_save_path}")


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper Analysis Figures (Uncertainty, Scatter, Visual Grid).")
    parser.add_argument("--data-dir", type=str, default="data/Haze1k", help="Haze1k dataset root.")
    parser.add_argument("--rice-dir", type=str, default="data/RICE_DATASET/RICE1", help="RICE1 dataset root.")
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
    parser.add_argument("--output-dir", type=str, default="experiments/figures", help="Output directory for figures.")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
    args = parser.parse_args()

    # Device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Analysis Compute Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Models
    print("\nLoading Pre-Trained Benchmark Models...")
    dcp = DCP(patch_size=15, omega=0.95, t0=0.10)

    aodnet = AODNet(in_channels=3)
    if Path(args.aodnet_ckpt).is_file():
        ckpt_a = torch.load(args.aodnet_ckpt, map_location=device)
        aodnet.load_state_dict(ckpt_a.get("model_state_dict", ckpt_a))
        print(f"  Loaded AOD-Net from: {args.aodnet_ckpt}")
    aodnet = aodnet.to(device).eval()

    physdehaze = PhysDehazeLite(in_channels=3, base_channels=32)
    if Path(args.physdehaze_ckpt).is_file():
        ckpt_p = torch.load(args.physdehaze_ckpt, map_location=device)
        physdehaze.load_state_dict(ckpt_p.get("model_state_dict", ckpt_p))
        print(f"  Loaded PhysDehaze-Lite from: {args.physdehaze_ckpt}")
    physdehaze = physdehaze.to(device).eval()

    models_dict = {
        "DCP (Classical)": dcp,
        "AOD-Net (Baseline)": aodnet,
        "PhysDehaze-Lite (Proposed Novel)": physdehaze,
    }

    # 2. Datasets
    print("\nLoading Test Datasets (Haze1k Subsets & RICE1)...")
    ds_thin = SateHaze1kDataset(root_dir=args.data_dir, subset="thin", split="test", crop_size=None)
    ds_mod = SateHaze1kDataset(root_dir=args.data_dir, subset="moderate", split="test", crop_size=None)
    ds_thick = SateHaze1kDataset(root_dir=args.data_dir, subset="thick", split="test", crop_size=None)
    ds_rice = RICE1Dataset(root_dir=args.rice_dir, split="test", crop_size=None)

    # Feature A: Uncertainty Maps (3 Haze1k + 3 RICE1)
    ds_all_haze1k = SateHaze1kDataset(root_dir=args.data_dir, subset="all", split="test", crop_size=None)
    run_uncertainty_analysis(
        model=physdehaze,
        haze1k_ds=ds_all_haze1k,
        rice1_ds=ds_rice,
        device=device,
        output_dir=out_dir,
    )

    # Feature B: Radiometric Consistency Scatter Plot (5000 pixels per subset)
    run_radiometric_consistency_analysis(
        models=models_dict,
        subsets_data={"thin": ds_thin, "moderate": ds_mod, "thick": ds_thick},
        device=device,
        output_dir=out_dir,
        samples_per_subset=5000,
    )

    # Visual Comparison Grid (5 images: Thin #1, Thin #2, Moderate #1, Thick #1, RICE1 #1)
    test_5_samples = [
        ("Thin Haze #1", ds_thin[0]),
        ("Thin Haze #2", ds_thin[4]),
        ("Moderate Haze #1", ds_mod[2]),
        ("Thick Haze #1", ds_thick[1]),
        ("RICE1 Cloud #1", ds_rice[0]),
    ]
    run_visual_comparison_grid(
        models=models_dict,
        test_samples=test_5_samples,
        device=device,
        output_dir=out_dir,
    )

    print(f"\n{'='*75}")
    print(f"ALL PUBLICATION-QUALITY FIGURES GENERATED SUCCESSFULLY IN {out_dir}")
    print(f"{'='*75}")
    for fig_path in sorted(out_dir.glob("*.png")):
        size_kb = fig_path.stat().st_size / 1024.0
        print(f"  - {fig_path.name:<40} ({size_kb:.1f} KB, 200 DPI)")


if __name__ == "__main__":
    main()

