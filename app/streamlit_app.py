"""Streamlit Web Application for Remote Sensing Image Dehazing.

Features:
- Upload hazy satellite/aerial imagery (PNG/JPG).
- Choose between models: DCP (He et al. 2010), CNN (AOD-Net), and PhysDehaze-Lite.
- Side-by-side display of Hazy Input and Dehazed Output.
- Estimated spatial Transmission Map t(x) for PhysDehaze-Lite.
- Optional upload of clear ground-truth reference image to display PSNR, SSIM, and SAM.
- Auto-detects Apple Silicon MPS or CPU device.
- Loads best pre-trained checkpoints from experiments/.
"""

from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import io
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.metrics.metrics import calculate_ciede2000, calculate_psnr, calculate_qnr, calculate_sam, calculate_ssim
from src.models.aodnet import AODNet
from src.models.dcp import DCP
from src.models.physdehaze_lite import PhysDehazeLite

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Remote Sensing Dehazing Studio",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_dehazing_models() -> Tuple[Dict[str, Any], torch.device]:
    """Load all dehazing models and pre-trained checkpoints from experiments/."""
    # Hardware device detection: MPS -> CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    models: Dict[str, Any] = {}

    # 1. Classical DCP (runs on CPU for guided filter)
    models["DCP"] = DCP(patch_size=15, omega=0.95, t0=0.10)

    # 2. CNN Baseline (AOD-Net)
    aodnet = AODNet(in_channels=3).to(device).eval()
    aod_ckpt_candidates = [
        REPO_ROOT / "experiments" / "aodnet" / "checkpoints" / "best_model.pth",
        REPO_ROOT / "experiments" / "smoke_test_aodnet" / "checkpoints" / "best_model.pth",
    ]
    for ckpt_path in aod_ckpt_candidates:
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location=device)
            aodnet.load_state_dict(ckpt.get("model_state_dict", ckpt))
            break
    models["CNN"] = aodnet

    # 3. PhysDehaze-Lite (Proposed Novel)
    phys = PhysDehazeLite(in_channels=3, base_channels=32).to(device).eval()
    phys_ckpt_candidates = [
        REPO_ROOT / "experiments" / "physdehaze_lite" / "checkpoints" / "best_model.pth",
        REPO_ROOT / "experiments" / "smoke_test_physdehaze_lite" / "checkpoints" / "best_model.pth",
    ]
    for ckpt_path in phys_ckpt_candidates:
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location=device)
            phys.load_state_dict(ckpt.get("model_state_dict", ckpt))
            break
    models["PhysDehaze-Lite"] = phys

    return models, device


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) or (1, C, H, W) float tensor in [0, 1] to (H, W, C) uint8 numpy array."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    np_arr = tensor.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    return (np_arr * 255.0).round().astype(np.uint8)


def main() -> None:
    # Title & Header
    st.title("🛰️ Remote Sensing Image Dehazing Studio")
    st.markdown(
        """
        Interactive evaluation of physics-guided and baseline dehazing models for satellite and aerial imagery, 
        based on the atmospheric scattering model:  
        $$\\mathbf{I}(x) = \\mathbf{J}(x) \\cdot t(x) + \\mathbf{A} \\cdot (1 - t(x))$$
        """
    )

    # Load Models
    models, device = load_dehazing_models()

    # Sidebar: Model Selection and Settings
    st.sidebar.header("⚙️ Model Configuration")
    model_choice = st.sidebar.selectbox(
        "Select Dehazing Model",
        options=["PhysDehaze-Lite", "CNN", "DCP"],
        index=0,
        help="DCP: Dark Channel Prior (He et al. 2010); CNN: AOD-Net baseline; PhysDehaze-Lite: Proposed physics-guided depthwise U-Net.",
    )

    st.sidebar.markdown(f"**Compute Device:** `{device.type.upper()}`")
    if model_choice == "PhysDehaze-Lite":
        st.sidebar.info("💡 **PhysDehaze-Lite** estimates both scene radiance $J(x)$ and transmission map $t(x)$ via atmospheric physical inversion.")
    elif model_choice == "CNN":
        st.sidebar.info("💡 **CNN (AOD-Net)** directly estimates the unified physical coefficient $K(x) = \\frac{1/t(x) \\cdot (I - A) + A}{I - 1}$.")
    else:
        st.sidebar.info("💡 **DCP** is a classical non-learning prior utilizing local dark channel heuristics and guided filtering.")

    # Image Upload Section
    st.sidebar.header("📁 Image Upload")
    hazy_file = st.sidebar.file_uploader(
        "Upload Hazy Image (Required)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Select a hazy or cloudy satellite/aerial image.",
    )

    clear_file = st.sidebar.file_uploader(
        "Upload Clear Ground-Truth Reference (Optional)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        help="Upload the matching haze-free reference image to compute quantitative metrics (PSNR, SSIM, SAM).",
    )

    # Optional Preset Sample Loader if user has not uploaded a file yet
    if hazy_file is None:
        st.info("👆 Please upload a hazy image in the sidebar to begin, or click below to load a sample test scene.")
        sample_col1, sample_col2, sample_col3 = st.columns(3)

        sample_paths = [
            ("Haze1k Thin Sample", REPO_ROOT / "data" / "Haze1k" / "Haze1k" / "Haze1k_thin" / "dataset" / "test" / "input" / "0.png",
             REPO_ROOT / "data" / "Haze1k" / "Haze1k" / "Haze1k_thin" / "dataset" / "test" / "target" / "0.png"),
            ("Haze1k Moderate Sample", REPO_ROOT / "data" / "Haze1k" / "Haze1k" / "Haze1k_moderate" / "dataset" / "test" / "input" / "0.png",
             REPO_ROOT / "data" / "Haze1k" / "Haze1k" / "Haze1k_moderate" / "dataset" / "test" / "target" / "0.png"),
            ("RICE1 Cloud Sample", REPO_ROOT / "data" / "RICE_DATASET" / "RICE1" / "cloud" / "0.png",
             REPO_ROOT / "data" / "RICE_DATASET" / "RICE1" / "label" / "0.png"),
        ]

        load_sample = None
        with sample_col1:
            if st.button("Load Haze1k Thin Scene"):
                load_sample = sample_paths[0]
        with sample_col2:
            if st.button("Load Haze1k Moderate Scene"):
                load_sample = sample_paths[1]
        with sample_col3:
            if st.button("Load RICE1 Cloud Scene"):
                load_sample = sample_paths[2]

        if load_sample and load_sample[1].is_file():
            hazy_pil = Image.open(load_sample[1]).convert("RGB")
            clear_pil = Image.open(load_sample[2]).convert("RGB") if load_sample[2].is_file() else None
        else:
            return
    else:
        hazy_pil = Image.open(hazy_file).convert("RGB")
        clear_pil = Image.open(clear_file).convert("RGB") if clear_file is not None else None

    # Preprocess Hazy Input
    hazy_tensor = TF.to_tensor(hazy_pil).unsqueeze(0)
    orig_w, orig_h = hazy_pil.size

    # Inference Execution
    selected_model = models[model_choice]

    with st.spinner(f"Running inference with {model_choice}..."):
        with torch.no_grad():
            if model_choice == "PhysDehaze-Lite":
                hazy_in = hazy_tensor.to(device)
                res = selected_model(hazy_in, return_dict=True)
                dehazed_tensor = res["output"].cpu()
                trans_map_tensor = res["transmission"].cpu()
                airlight_tensor = res.get("airlight", None)
            elif model_choice == "CNN":
                hazy_in = hazy_tensor.to(device)
                out = selected_model(hazy_in)
                dehazed_tensor = out.cpu().clamp(0.0, 1.0)
                trans_map_tensor = None
                airlight_tensor = None
            else:  # DCP
                hazy_in = hazy_tensor.cpu()
                out = selected_model(hazy_in)
                dehazed_tensor = out.cpu().clamp(0.0, 1.0)
                trans_map_tensor = None
                airlight_tensor = None

    dehazed_np = tensor_to_numpy_image(dehazed_tensor)

    # -----------------------------------------------------------------------
    # Section 1: Side-by-Side Visual Comparison
    # -----------------------------------------------------------------------
    st.subheader("1. Side-by-Side Dehazing Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Hazy Input Image $\\mathbf{I}(x)$**")
        st.image(hazy_pil, use_container_width=True, caption=f"Resolution: {orig_w}x{orig_h}")

    with col2:
        st.markdown(f"**Dehazed Output $\\mathbf{J}(x)$ — {model_choice}**")
        st.image(dehazed_np, use_container_width=True, caption=f"Processed with {model_choice}")

    # -----------------------------------------------------------------------
    # Section 2: Estimated Transmission Map for PhysDehaze-Lite
    # -----------------------------------------------------------------------
    if trans_map_tensor is not None:
        st.subheader("2. Estimated Transmission Map $t(x)$ (PhysDehaze-Lite)")
        st.markdown(
            "PhysDehaze-Lite predicts the spatially continuous transmission map $t(x) \\in (0, 1]$ "
            "governing optical transmittance through the atmosphere."
        )

        t_np = trans_map_tensor[0, 0].numpy()
        mean_t = float(np.mean(t_np))
        min_t = float(np.min(t_np))
        max_t = float(np.max(t_np))

        t_col1, t_col2 = st.columns([2, 1])

        with t_col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(t_np, cmap="inferno", vmin=0.0, vmax=1.0)
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Optical Transmission $t(x)$", fontsize=11)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with t_col2:
            st.markdown("### Transmission Statistics")
            st.metric("Mean Transmittance $\\bar{t}$", f"{mean_t:.3f}", help="Closer to 1.0 indicates clear atmosphere.")
            st.metric("Minimum Transmittance $t_{\\min}$", f"{min_t:.3f}", help="Lower values correspond to thickest haze.")
            st.metric("Maximum Transmittance $t_{\\max}$", f"{max_t:.3f}")
            if airlight_tensor is not None:
                a_vals = airlight_tensor.squeeze().cpu().numpy()
                st.markdown(f"**Estimated Airlight $\\mathbf{{A}}$:** `[{a_vals[0]:.3f}, {a_vals[1]:.3f}, {a_vals[2]:.3f}]`")

    # -----------------------------------------------------------------------
    # Section 3: Quantitative Metrics (If Reference Uploaded)
    # -----------------------------------------------------------------------
    if clear_pil is not None:
        st.subheader("3. Quantitative Evaluation Against Ground Truth")
        st.markdown("Computed using the survey paper's Section 4.2 evaluation metrics:")

        # Align reference image if sizes differ
        if clear_pil.size != (orig_w, orig_h):
            clear_pil = clear_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)

        clean_tensor = TF.to_tensor(clear_pil).unsqueeze(0)

        # Compute Required Metrics: PSNR, SSIM, SAM
        psnr_val = calculate_psnr(dehazed_tensor, clean_tensor)
        ssim_val = calculate_ssim(dehazed_tensor, clean_tensor)
        sam_val = calculate_sam(dehazed_tensor, clean_tensor)

        # Compute Supplementary Metrics: CIEDE2000, QNR
        ciede_val = calculate_ciede2000(dehazed_tensor, clean_tensor)
        qnr_val = calculate_qnr(dehazed_tensor, clean_tensor)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("PSNR (Peak SNR)", f"{psnr_val:.2f} dB", help="Higher is better.")
        m2.metric("SSIM (Structural)", f"{ssim_val:.4f}", help="Higher is better (max 1.0).")
        m3.metric("SAM (Spectral Angle)", f"{sam_val:.2f}°", help="Spectral Angle Mapper. Lower is better (0.0° ideal).")
        m4.metric("CIEDE2000 (Color)", f"{ciede_val:.2f}", help="Color difference. Lower is better.")
        m5.metric("QNR (Quality)", f"{qnr_val:.4f}", help="Survey Eq. 18. Higher is better (max 1.0).")

    # -----------------------------------------------------------------------
    # Download Processed Result
    # -----------------------------------------------------------------------
    st.divider()
    buf = io.BytesIO()
    Image.fromarray(dehazed_np).save(buf, format="PNG")
    st.download_button(
        label=f"💾 Download Dehazed Image ({model_choice})",
        data=buf.getvalue(),
        file_name=f"dehazed_{model_choice.lower().replace('-', '_')}.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
