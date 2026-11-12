"""Evaluation metrics for remote sensing image dehazing benchmark.

Following Section 4.2 of the survey paper:
"Remote sensing image dehazing: A systematic review of progress, challenges, and prospects"
(ISPRS Journal of Photogrammetry and Remote Sensing 235, 2026).

All functions accept float tensors scaled to [0, 1] with shape (B, 3, H, W) or (3, H, W),
and are fully compatible with CPU and Apple Silicon ('mps') without CUDA-only constraints.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Cached model instances to avoid redundant re-instantiations
_LPIPS_CACHE: Dict[Tuple[str, str], Any] = {}
_NIQE_CACHE: Dict[str, Any] = {}


def _to_4d_tensor(img: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 4D float tensor of shape (B, C, H, W).

    Args:
        img: Input tensor of shape (C, H, W) or (B, C, H, W) with pixel values in [0, 1].

    Returns:
        Tensor of shape (B, C, H, W) as float32.
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    elif img.dim() != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got tensor with {img.dim()} dimensions")
    return img.float()


def _tensor_to_np_img(img: torch.Tensor) -> np.ndarray:
    """Convert tensor of shape (C, H, W) or (1, C, H, W) in [0, 1] to (H, W, C) float64."""
    if img.dim() == 4:
        if img.size(0) != 1:
            raise ValueError(f"Expected single image (B=1), got batch size {img.size(0)}")
        img = img.squeeze(0)
    arr = img.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0).astype(np.float64)


def get_default_device() -> torch.device:
    """Get the preferred compute device (MPS if available on macOS, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. PSNR (Peak Signal-to-Noise Ratio)
# ---------------------------------------------------------------------------

def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        pred: Predicted image tensor in range [0, 1] with shape (B, 3, H, W) or (3, H, W).
        target: Ground truth image tensor in range [0, 1] with shape (B, 3, H, W) or (3, H, W).
        max_val: Dynamic range of pixel values (default 1.0).

    Returns:
        PSNR value in dB. Returns 100.0 dB if images are numerically identical.
    """
    pred_4d = _to_4d_tensor(pred)
    target_4d = _to_4d_tensor(target)

    batch_psnrs: List[float] = []
    for i in range(pred_4d.size(0)):
        mse = F.mse_loss(pred_4d[i], target_4d[i]).item()
        if mse < 1e-10:
            batch_psnrs.append(100.0)
        else:
            batch_psnrs.append(float(10.0 * np.log10((max_val ** 2) / mse)))

    return float(np.mean(batch_psnrs))


# ---------------------------------------------------------------------------
# 2. SSIM (Structural Similarity Index)
# ---------------------------------------------------------------------------

def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    win_size: int = 7,
) -> float:
    """Calculate Structural Similarity Index (SSIM) across RGB channels using scikit-image.

    Args:
        pred: Predicted image tensor in range [0, 1] with shape (B, 3, H, W) or (3, H, W).
        target: Target image tensor in range [0, 1] with shape (B, 3, H, W) or (3, H, W).
        win_size: Side length of the sliding Gaussian window.

    Returns:
        Mean SSIM score in range [-1.0, 1.0]. Returns 1.0 for identical images.
    """
    pred_4d = _to_4d_tensor(pred)
    target_4d = _to_4d_tensor(target)

    batch_ssims: List[float] = []
    for i in range(pred_4d.size(0)):
        np_pred = _tensor_to_np_img(pred_4d[i])
        np_target = _tensor_to_np_img(target_4d[i])

        min_dim = min(np_pred.shape[0], np_pred.shape[1])
        actual_win_size = min(win_size if win_size % 2 == 1 else win_size - 1, min_dim)
        if actual_win_size % 2 == 0:
            actual_win_size -= 1
        actual_win_size = max(actual_win_size, 3)

        score = structural_similarity(
            np_pred,
            np_target,
            win_size=actual_win_size,
            channel_axis=2,
            data_range=1.0,
        )
        batch_ssims.append(float(score))

    return float(np.mean(batch_ssims))


# ---------------------------------------------------------------------------
# 3. CIEDE2000 (CIE Delta E 2000 Color Difference)
# ---------------------------------------------------------------------------

def calculate_ciede2000(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Calculate CIE Delta E 2000 color difference on CIELAB representation.

    Measures spectral and perceptual color distortion across land cover types.
    Lower is better; returns 0.0 for identical images.

    Args:
        pred: Predicted RGB image tensor in range [0, 1] with shape (B, 3, H, W) or (3, H, W).
        target: Reference RGB image tensor in range [0, 1] with shape (B, 3, H, W) or (3, H, W).

    Returns:
        Mean CIEDE2000 error across all spatial pixels.
    """
    pred_4d = _to_4d_tensor(pred)
    target_4d = _to_4d_tensor(target)

    batch_delta: List[float] = []
    for i in range(pred_4d.size(0)):
        np_pred = _tensor_to_np_img(pred_4d[i])
        np_target = _tensor_to_np_img(target_4d[i])

        lab_pred = rgb2lab(np_pred)
        lab_target = rgb2lab(np_target)

        delta_e = deltaE_ciede2000(lab_pred, lab_target)
        batch_delta.append(float(np.mean(delta_e)))

    return float(np.mean(batch_delta))


# ---------------------------------------------------------------------------
# 4. LPIPS (Learned Perceptual Image Patch Similarity)
# ---------------------------------------------------------------------------

def get_lpips_model(
    net: str = "alex",
    device: Optional[Union[str, torch.device]] = None,
) -> Any:
    """Retrieve or initialize a cached LPIPS model instance.

    Args:
        net: Perceptual backbone architecture ('alex' per survey benchmark).
        device: Target compute device.

    Returns:
        Configured LPIPS evaluation model.
    """
    global _LPIPS_CACHE
    if device is None:
        target_dev = get_default_device()
    else:
        target_dev = torch.device(device)

    dev_str = str(target_dev)
    cache_key = (net, dev_str)
    if cache_key not in _LPIPS_CACHE:
        import lpips
        # Suppress verbose printouts during initialization
        model = lpips.LPIPS(net=net, verbose=False)
        model = model.to(target_dev)
        model.eval()
        _LPIPS_CACHE[cache_key] = model

    return _LPIPS_CACHE[cache_key]


def calculate_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
    lpips_model: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> float:
    """Calculate Learned Perceptual Image Patch Similarity (LPIPS) with AlexNet backbone.

    Args:
        pred: Predicted image tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
        target: Ground truth image tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
        net: Perceptual feature extractor ('alex').
        lpips_model: Optional pre-loaded LPIPS instance.
        device: Device to run LPIPS on (MPS or CPU).

    Returns:
        Mean perceptual distance score. Returns ~0.0 for identical images.
    """
    pred_4d = _to_4d_tensor(pred)
    target_4d = _to_4d_tensor(target)

    model = lpips_model if lpips_model is not None else get_lpips_model(net=net, device=device)
    model_device = next(model.parameters()).device

    # LPIPS expects input tensors rescaled from [0, 1] to [-1, 1]
    p_norm = (pred_4d.to(model_device) * 2.0 - 1.0).clamp(-1.0, 1.0)
    t_norm = (target_4d.to(model_device) * 2.0 - 1.0).clamp(-1.0, 1.0)

    with torch.no_grad():
        dist = model(p_norm, t_norm)

    return float(dist.mean().item())


# ---------------------------------------------------------------------------
# 5. SAM (Spectral Angle Mapper in degrees, Eq. 15)
# ---------------------------------------------------------------------------

def calculate_sam(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Calculate Spectral Angle Mapper (SAM) in degrees (Survey Eq. 15).

    Evaluates spectral distortion by computing the angle in degrees between
    the predicted and reference spectral vectors across channels at each pixel.

    Formula (Eq. 15):
        SAM(x, y) = arccos( (x . y) / (||x||_2 * ||y||_2 + eps) ) * (180 / pi)

    Args:
        pred: Predicted image tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
        target: Reference image tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
        eps: Small constant for numerical stability.

    Returns:
        Mean spectral angle in degrees across all pixels (0.0 degrees for identical images).
    """
    pred_4d = _to_4d_tensor(pred)
    target_4d = _to_4d_tensor(target)

    # Identical images shortcut to guarantee exact 0.0 without numerical jitter
    if F.mse_loss(pred_4d, target_4d).item() < 1e-10:
        return 0.0

    # Compute in double precision on CPU to prevent float32 arccos(1 - eps) blow-up
    p_64 = pred_4d.detach().cpu().double()
    t_64 = target_4d.detach().cpu().double()

    # Dot product along channel dimension (B, H, W)
    dot = torch.sum(p_64 * t_64, dim=1)
    norm_p = torch.norm(p_64, p=2, dim=1)
    norm_t = torch.norm(t_64, p=2, dim=1)

    denom = norm_p * norm_t
    valid_mask = denom > eps

    # For pixels where both vectors have valid norm, compute cos theta;
    # for black pixels in both (norm <= eps), the angle is 0.0 (cos = 1.0)
    cos_theta = torch.where(
        valid_mask,
        dot / torch.clamp(denom, min=eps),
        torch.where((norm_p <= eps) & (norm_t <= eps), torch.ones_like(dot), torch.zeros_like(dot)),
    )
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    angle_rad = torch.acos(cos_theta)
    angle_deg = angle_rad * (180.0 / np.pi)

    return float(angle_deg.mean().item())


# ---------------------------------------------------------------------------
# 6. ERGAS (Relative Dimensionless Global Error in Synthesis, Eq. 16)
# ---------------------------------------------------------------------------

def calculate_ergas(
    pred: torch.Tensor,
    target: torch.Tensor,
    r: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """Calculate ERGAS (Relative Dimensionless Global Error in Synthesis, Survey Eq. 16).

    Formula (Eq. 16):
        ERGAS = 100 * r * sqrt( (1 / C) * sum_{c=1}^C [ MSE(I^c, J^c) / ((mu(J^c))^2 + eps) ] )

    where r is the spatial resolution ratio (r = 1.0 for dehazing since prediction and
    ground-truth share identical spatial resolution), C is the number of spectral bands (3),
    and mu(J^c) is the mean intensity of band c of the ground-truth image.

    Args:
        pred: Predicted image tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
        target: Reference image tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
        r: Resolution ratio (r = 1.0 for dehazing).
        eps: Small constant to prevent division by zero for dark reference bands.

    Returns:
        ERGAS metric value (lower is better, 0.0 for identical images).
    """
    pred_4d = _to_4d_tensor(pred)
    target_4d = _to_4d_tensor(target)

    batch_ergas: List[float] = []
    for b in range(pred_4d.size(0)):
        p_b = pred_4d[b]
        t_b = target_4d[b]
        c_terms: List[float] = []
        for c in range(p_b.size(0)):
            mse_c = F.mse_loss(p_b[c], t_b[c]).item()
            mu_c = t_b[c].mean().item()
            term = mse_c / ((mu_c ** 2) + eps)
            c_terms.append(term)

        ergas_val = 100.0 * r * np.sqrt(float(np.mean(c_terms)))
        batch_ergas.append(ergas_val)

    return float(np.mean(batch_ergas))


# ---------------------------------------------------------------------------
# 7. UIQI (Universal Image Quality Index, Eq. 17)
# ---------------------------------------------------------------------------

def calculate_uiqi(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Calculate Universal Image Quality Index (UIQI, Survey Eq. 17).

    Wang & Bovik (2002) quality index decomposing similarity into loss of correlation,
    luminance distortion, and contrast distortion. Computed per spectral band and averaged.

    Formula (Eq. 17):
        Q = (4 * sigma_{xy} * mu_x * mu_y) / ( (sigma_x^2 + sigma_y^2) * (mu_x^2 + mu_y^2) )

    Args:
        pred: Predicted image tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
        target: Reference image tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
        eps: Small constant for numerical stability on flat constant patches.

    Returns:
        UIQI quality score in [-1.0, 1.0] (1.0 for identical images).
    """
    pred_4d = _to_4d_tensor(pred)
    target_4d = _to_4d_tensor(target)

    batch_uiqi: List[float] = []
    for b in range(pred_4d.size(0)):
        p_b = pred_4d[b]
        t_b = target_4d[b]
        channel_q: List[float] = []
        for c in range(p_b.size(0)):
            x = p_b[c]
            y = t_b[c]
            mu_x = x.mean().item()
            mu_y = y.mean().item()

            sigma_x2 = ((x - mu_x) ** 2).mean().item()
            sigma_y2 = ((y - mu_y) ** 2).mean().item()
            sigma_xy = ((x - mu_x) * (y - mu_y)).mean().item()

            denom = (sigma_x2 + sigma_y2) * (mu_x ** 2 + mu_y ** 2)
            if denom < eps:
                # Handle completely flat / uniform images
                if F.mse_loss(x, y).item() < eps:
                    q = 1.0
                else:
                    q = 0.0
            else:
                q = (4.0 * sigma_xy * mu_x * mu_y) / denom
            channel_q.append(float(np.clip(q, -1.0, 1.0)))

        batch_uiqi.append(float(np.mean(channel_q)))

    return float(np.mean(batch_uiqi))


# ---------------------------------------------------------------------------
# 8. QNR (Quality with No Reference, Eq. 18)
# ---------------------------------------------------------------------------

def calculate_qnr(
    sam: float,
    ergas: float,
) -> float:
    """Calculate QNR metric following the survey's exact formulation (Survey Eq. 18).

    Formula (Eq. 18):
        QNR = exp(-SAM / 10) * exp(-ERGAS / 100)

    Note:
        This definition is taken directly from Section 4.2 Eq. 18 of the base review paper:
        'Remote sensing image dehazing: A systematic review of progress, challenges,
        and prospects' (ISPRS J. Photogramm. Remote Sens. 235, 2026).
        It combines spectral angle distortion (SAM in degrees) and relative global
        radiometric error (ERGAS) via exponential decay into a unified index in (0, 1],
        where 1.0 indicates perfect spectral and radiometric fidelity.
        This formulation differs fundamentally from the classic pansharpening QNR proposed
        by Alparone et al. (2008), which evaluates inter-band spectral distortion and
        spatial detail distortion across resolution scales using UIQI.

    Args:
        sam: Spectral Angle Mapper score in degrees (>= 0.0).
        ergas: ERGAS error score (>= 0.0).

    Returns:
        QNR score in (0.0, 1.0]. Returns 1.0 when SAM=0 and ERGAS=0.
    """
    return float(np.exp(-sam / 10.0) * np.exp(-ergas / 100.0))


# ---------------------------------------------------------------------------
# 9. HIST (Histogram Similarity, Eq. 20)
# ---------------------------------------------------------------------------

def calculate_hist(
    pred: torch.Tensor,
    target: torch.Tensor,
    bins: int = 256,
) -> float:
    """Calculate Histogram Similarity (HIST) across 256 normalized bins (Survey Eq. 20).

    Formula (Eq. 20):
        HIST = 1 - 0.5 * sum_{i=1}^{256} |h_i - g_i|

    where h and g are normalized 256-bin histograms of the predicted and ground-truth images
    computed over the intensity range [0, 1], normalized such that sum(h) = sum(g) = 1.
    Computed per channel and averaged across RGB channels.

    Args:
        pred: Predicted image tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
        target: Reference image tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
        bins: Number of histogram bins (default 256).

    Returns:
        Histogram similarity in [0.0, 1.0]. Returns 1.0 for identical images.
    """
    pred_4d = _to_4d_tensor(pred)
    target_4d = _to_4d_tensor(target)

    batch_hist: List[float] = []
    for b in range(pred_4d.size(0)):
        p_b = pred_4d[b].detach().cpu().numpy()
        t_b = target_4d[b].detach().cpu().numpy()
        c_sims: List[float] = []
        for c in range(p_b.shape[0]):
            p_flat = np.clip(p_b[c].flatten(), 0.0, 1.0)
            t_flat = np.clip(t_b[c].flatten(), 0.0, 1.0)

            h, _ = np.histogram(p_flat, bins=bins, range=(0.0, 1.0))
            g, _ = np.histogram(t_flat, bins=bins, range=(0.0, 1.0))

            # Normalize to probability distributions
            h_norm = h.astype(np.float64) / max(float(h.sum()), 1.0)
            g_norm = g.astype(np.float64) / max(float(g.sum()), 1.0)

            sim = 1.0 - 0.5 * float(np.sum(np.abs(h_norm - g_norm)))
            c_sims.append(sim)

        batch_hist.append(float(np.mean(c_sims)))

    return float(np.mean(batch_hist))


# ---------------------------------------------------------------------------
# 10. NIQE (Natural Image Quality Evaluator, No-Reference)
# ---------------------------------------------------------------------------

def get_niqe_model(
    device: Optional[Union[str, torch.device]] = None,
) -> Optional[Any]:
    """Retrieve or initialize a cached NIQE model instance via pyiqa.

    Note:
        NIQE internal feature estimation requires float64 tensor operations, which
        Apple Silicon MPS does not support. Therefore, NIQE is always initialized
        and evaluated on CPU for cross-platform stability.

    Args:
        device: Target compute device (CPU enforced for NIQE float64 compatibility).

    Returns:
        Configured NIQE model instance, or None if pyiqa is unavailable.
    """
    global _NIQE_CACHE
    # NIQE architecture casts tensors to torch.float64, which is unsupported on MPS.
    cpu_dev = torch.device("cpu")
    dev_str = "cpu"

    if dev_str not in _NIQE_CACHE:
        try:
            import pyiqa
            model = pyiqa.create_metric("niqe", device=cpu_dev)
            _NIQE_CACHE[dev_str] = model
        except Exception as exc:
            logger.warning("Could not initialize NIQE metric via pyiqa: %s", exc)
            return None

    return _NIQE_CACHE.get(dev_str)


def calculate_niqe(
    img: torch.Tensor,
    device: Optional[Union[str, torch.device]] = None,
) -> Optional[float]:
    """Calculate Natural Image Quality Evaluator (NIQE) score (no-reference).

    Lower NIQE score indicates better natural statistical quality and fewer artifacts.
    Note: Executes on CPU due to NIQE's float64 requirement (MPS unsupported).
    If spatial dimensions are smaller than NIQE's minimal patch scale (168x168),
    the image is bilinearly upsampled to ensure valid feature computation.

    Args:
        img: Input image tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
        device: Target device (CPU enforced for NIQE float64 compatibility).

    Returns:
        NIQE score as float, or None if pyiqa is not installed/accessible.
    """
    img_4d = _to_4d_tensor(img)
    model = get_niqe_model(device=device)
    if model is None:
        return None

    # Always feed CPU float32 tensor
    img_cpu = img_4d.detach().cpu().clamp(0.0, 1.0)

    # NIQE multi-scale Gaussian decomposition requires minimum spatial size (>= 168)
    _, _, h, w = img_cpu.shape
    if h < 168 or w < 168:
        new_h = max(h, 168)
        new_w = max(w, 168)
        img_cpu = F.interpolate(img_cpu, size=(new_h, new_w), mode="bilinear", align_corners=False)

    try:
        with torch.no_grad():
            score = model(img_cpu)
        return float(score.mean().item())
    except Exception as exc:
        logger.warning("NIQE evaluation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# 11. FID (Fréchet Inception Distance, Optional)
# ---------------------------------------------------------------------------

def calculate_fid(
    features_pred: np.ndarray,
    features_target: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Calculate Fréchet Inception Distance between two feature distributions.

    Formula:
        FID = ||mu_1 - mu_2||^2 + Tr(Sigma_1 + Sigma_2 - 2 * (Sigma_1 * Sigma_2)^(1/2))

    Args:
        features_pred: Feature activations of predicted images with shape (N, D).
        features_target: Feature activations of reference images with shape (N, D).
        eps: Small constant for matrix square root numerical stability.

    Returns:
        FID distance score (>= 0.0; returns 0.0 for identical distributions).
    """
    from scipy import linalg

    act1 = np.asarray(features_pred, dtype=np.float64)
    act2 = np.asarray(features_target, dtype=np.float64)

    mu1 = np.mean(act1, axis=0)
    mu2 = np.mean(act2, axis=0)

    sigma1 = np.cov(act1, rowvar=False)
    sigma2 = np.cov(act2, rowvar=False)

    if sigma1.ndim == 0:
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid_val = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    return max(fid_val, 0.0)


# ---------------------------------------------------------------------------
# Metric Tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """Aggregates and tracks evaluation metrics over validation and test batches."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._storage: Dict[str, List[float]] = {
            "psnr": [],
            "ssim": [],
            "ciede2000": [],
            "sam": [],
            "ergas": [],
            "uiqi": [],
            "qnr": [],
            "hist": [],
        }

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        compute_lpips: bool = False,
        compute_niqe: bool = False,
        extra_metrics: Optional[Dict[str, float]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Compute and accumulate Section 4.2 metrics for a batch.

        Args:
            pred: Predicted batch tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
            target: Ground truth batch tensor in [0, 1] with shape (B, 3, H, W) or (3, H, W).
            compute_lpips: Whether to compute AlexNet LPIPS perceptual distance.
            compute_niqe: Whether to compute no-reference NIQE quality.
            extra_metrics: Optional additional scalar values (e.g. loss).
            device: Compute device.
        """
        pred_4d = _to_4d_tensor(pred)
        target_4d = _to_4d_tensor(target)

        batch_size = pred_4d.size(0)
        for i in range(batch_size):
            p = pred_4d[i]
            t = target_4d[i]

            sam_val = calculate_sam(p, t)
            ergas_val = calculate_ergas(p, t, r=1.0)

            self._storage["psnr"].append(calculate_psnr(p, t))
            self._storage["ssim"].append(calculate_ssim(p, t))
            self._storage["ciede2000"].append(calculate_ciede2000(p, t))
            self._storage["sam"].append(sam_val)
            self._storage["ergas"].append(ergas_val)
            self._storage["uiqi"].append(calculate_uiqi(p, t))
            self._storage["qnr"].append(calculate_qnr(sam_val, ergas_val))
            self._storage["hist"].append(calculate_hist(p, t))

            if compute_lpips:
                if "lpips" not in self._storage:
                    self._storage["lpips"] = []
                self._storage["lpips"].append(calculate_lpips(p, t, device=device))

            if compute_niqe:
                score = calculate_niqe(p, device=device)
                if score is not None:
                    if "niqe" not in self._storage:
                        self._storage["niqe"] = []
                    self._storage["niqe"].append(score)

        if extra_metrics:
            for k, v in extra_metrics.items():
                if k not in self._storage:
                    self._storage[k] = []
                self._storage[k].append(v)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return mean, standard deviation, and sample count for each tracked metric."""
        res: Dict[str, Dict[str, float]] = {}
        for k, values in self._storage.items():
            if len(values) == 0:
                continue
            res[k] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": float(len(values)),
            }
        return res

    def to_dict(self) -> Dict[str, float]:
        """Return a simplified dictionary of mean metric values."""
        return {k: float(np.mean(v)) for k, v in self._storage.items() if len(v) > 0}


# ---------------------------------------------------------------------------
# Benchmark Evaluation Pipeline: evaluate(pred_dir, gt_dir)
# ---------------------------------------------------------------------------

def evaluate(
    pred_dir: Union[str, Path],
    gt_dir: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    compute_lpips: bool = True,
    compute_niqe: bool = True,
    compute_fid: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """Loop over predictions and ground truth images, evaluate metrics, and save CSV.

    Args:
        pred_dir: Directory containing predicted (dehazed) images.
        gt_dir: Directory containing ground truth (clear) reference images.
        output_csv: Optional file path to save detailed results CSV.
        compute_lpips: Whether to compute AlexNet LPIPS perceptual distance.
        compute_niqe: Whether to compute no-reference NIQE score.
        compute_fid: Whether to compute FID (off by default as it is slow and heavy).
        device: Device to run perceptual evaluations on (MPS or CPU).

    Returns:
        Dictionary mapping metric names to their dataset-level mean values.
    """
    pred_path = Path(pred_dir)
    gt_path = Path(gt_dir)

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions directory not found: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_path}")

    # Discover candidate image files in pred_dir
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    pred_files = sorted([p for p in pred_path.iterdir() if p.suffix.lower() in valid_exts])

    if len(pred_files) == 0:
        raise ValueError(f"No image files found in {pred_path}")

    # Build reference lookup mapping by stem and filename
    gt_lookup: Dict[str, Path] = {}
    for g in gt_path.iterdir():
        if g.suffix.lower() in valid_exts:
            gt_lookup[g.name] = g
            gt_lookup[g.stem] = g

    # Shared LPIPS model if enabled
    lpips_model = get_lpips_model(net="alex", device=device) if compute_lpips else None

    records: List[Dict[str, Any]] = []
    tracker = MetricTracker()

    for p_file in pred_files:
        # Find corresponding reference image
        ref_file = gt_lookup.get(p_file.name) or gt_lookup.get(p_file.stem)
        if ref_file is None:
            logger.warning("No ground-truth match found for %s, skipping", p_file.name)
            continue

        # Load images and convert to [0, 1] tensors
        with Image.open(p_file) as p_img, Image.open(ref_file) as t_img:
            p_rgb = p_img.convert("RGB")
            t_rgb = t_img.convert("RGB")

            # If sizes differ, resize prediction to match reference
            if p_rgb.size != t_rgb.size:
                p_rgb = p_rgb.resize(t_rgb.size, Image.Resampling.BILINEAR)

            p_arr = np.array(p_rgb).astype(np.float32) / 255.0
            t_arr = np.array(t_rgb).astype(np.float32) / 255.0

        p_tensor = torch.from_numpy(p_arr).permute(2, 0, 1)  # (3, H, W)
        t_tensor = torch.from_numpy(t_arr).permute(2, 0, 1)  # (3, H, W)

        # Compute full-reference metrics
        psnr_val = calculate_psnr(p_tensor, t_tensor)
        ssim_val = calculate_ssim(p_tensor, t_tensor)
        ciede_val = calculate_ciede2000(p_tensor, t_tensor)
        sam_val = calculate_sam(p_tensor, t_tensor)
        ergas_val = calculate_ergas(p_tensor, t_tensor, r=1.0)
        uiqi_val = calculate_uiqi(p_tensor, t_tensor)
        qnr_val = calculate_qnr(sam_val, ergas_val)
        hist_val = calculate_hist(p_tensor, t_tensor)

        row: Dict[str, Any] = {
            "image": p_file.name,
            "psnr": psnr_val,
            "ssim": ssim_val,
            "ciede2000": ciede_val,
            "sam": sam_val,
            "ergas": ergas_val,
            "uiqi": uiqi_val,
            "qnr": qnr_val,
            "hist": hist_val,
        }

        # Track into MetricTracker
        tracker._storage["psnr"].append(psnr_val)
        tracker._storage["ssim"].append(ssim_val)
        tracker._storage["ciede2000"].append(ciede_val)
        tracker._storage["sam"].append(sam_val)
        tracker._storage["ergas"].append(ergas_val)
        tracker._storage["uiqi"].append(uiqi_val)
        tracker._storage["qnr"].append(qnr_val)
        tracker._storage["hist"].append(hist_val)

        if compute_lpips and lpips_model is not None:
            lpips_val = calculate_lpips(p_tensor, t_tensor, lpips_model=lpips_model, device=device)
            row["lpips"] = lpips_val
            if "lpips" not in tracker._storage:
                tracker._storage["lpips"] = []
            tracker._storage["lpips"].append(lpips_val)

        if compute_niqe:
            niqe_val = calculate_niqe(p_tensor, device=device)
            if niqe_val is not None:
                row["niqe"] = niqe_val
                if "niqe" not in tracker._storage:
                    tracker._storage["niqe"] = []
                tracker._storage["niqe"].append(niqe_val)

        records.append(row)

    if len(records) == 0:
        raise ValueError("No matching image pairs were found between pred_dir and gt_dir")

    # Compute dataset means
    mean_results = tracker.to_dict()

    # Save detailed CSV if path is specified
    if output_csv is not None:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["image"] + [k for k in mean_results.keys()]

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)

            # Summary mean row
            summary_row: Dict[str, Any] = {"image": "MEAN"}
            summary_row.update(mean_results)
            writer.writerow(summary_row)

    return mean_results
