"""CLI demo script to dehaze an image or run an interactive procedural demonstration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF

from src.datasets.dataset import create_synthetic_rs_sample
from src.models.aodnet import AODNet
from src.models.dcp import DCP
from src.models.physdehaze_lite import PhysDehazeLite
from src.physics.atmospheric_model import generate_nonuniform_transmission, synthesize_haze
from src.utils.seed import get_device, set_seed


def run_demo(
    image_path: str | None = None,
    checkpoint: str | None = None,
    output_path: str = "demo_result.png",
) -> None:
    """Run single image dehazing demo across all methods."""
    set_seed(42)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Running demo on device: {device}")

    if image_path and Path(image_path).exists():
        with Image.open(image_path) as img:
            hazy_tensor = TF.to_tensor(img.convert("RGB")).unsqueeze(0).to(device)
        print(f"Loaded input image from: {image_path}")
    else:
        # Generate realistic procedural RS sample
        print("No image provided. Generating realistic procedural remote sensing scene...")
        clean = create_synthetic_rs_sample(256, 256, seed=42)
        trans = generate_nonuniform_transmission(256, 256, t_min=0.25, t_max=0.90)
        hazy_tensor = synthesize_haze(clean.unsqueeze(0), trans, atmospheric_light=0.88).to(device)

    # Initialize models
    dcp = DCP(patch_size=15, omega=0.95, t0=0.10)
    aodnet = AODNet(in_channels=3).to(device).eval()
    physdehaze = PhysDehazeLite(in_channels=3, base_channels=32).to(device).eval()

    # Load AOD-Net checkpoint if present
    aod_ckpt = Path("experiments/smoke_test_aodnet/checkpoints/best_model.pth")
    if aod_ckpt.exists():
        ckpt = torch.load(aod_ckpt, map_location=device)
        aodnet.load_state_dict(ckpt.get("model_state_dict", ckpt))

    # Load PhysDehaze-Lite checkpoint
    phys_ckpt_path = checkpoint or "experiments/physdehaze_lite/checkpoints/best_model.pth"
    if not Path(phys_ckpt_path).exists():
        phys_ckpt_path = "experiments/smoke_test_physdehaze_lite/checkpoints/best_model.pth"
    if Path(phys_ckpt_path).exists():
        ckpt = torch.load(phys_ckpt_path, map_location=device)
        physdehaze.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"Loaded weights for PhysDehaze-Lite from {phys_ckpt_path}")

    with torch.no_grad():
        out_dcp = dcp(hazy_tensor.cpu()).to(device)
        out_aod = aodnet(hazy_tensor)
        res_phys = physdehaze(hazy_tensor, return_dict=True)

    out_phys = res_phys["output"]
    trans_est = res_phys["transmission"]
    j_phys = res_phys["j_phys"]

    # Plot comparison
    fig, axes = plt.subplots(1, 6, figsize=(22, 3.8))
    axes[0].imshow(hazy_tensor[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[0].set_title("Hazy Input $I(x)$", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(trans_est[0, 0].cpu().numpy(), cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Est. Transmission $t(x)$", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(j_phys[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[2].set_title("Direct ASM $J_{\\text{phys}}(x)$", fontsize=11)
    axes[2].axis("off")

    axes[3].imshow(out_dcp[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[3].set_title("DCP (He et al. 2010)", fontsize=11)
    axes[3].axis("off")

    axes[4].imshow(out_aod[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[4].set_title("AOD-Net (Baseline CNN)", fontsize=11)
    axes[4].axis("off")

    axes[5].imshow(out_phys[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
    axes[5].set_title("PhysDehaze-Lite (Ours)", fontsize=11, fontweight="bold", color="darkgreen")
    axes[5].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Demo result successfully saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dehazing demo on an image.")
    parser.add_argument("--image", type=str, default=None, help="Path to input hazy image.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PhysDehaze-Lite checkpoint.")
    parser.add_argument("--output", type=str, default="demo_result.png", help="Output file path.")
    args = parser.parse_args()

    run_demo(image_path=args.image, checkpoint=args.checkpoint, output_path=args.output)


if __name__ == "__main__":
    main()
