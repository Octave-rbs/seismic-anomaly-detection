"""
Diffusion residual computation module for seismic anomaly detection.

Purifies spectrogram patches using a diffusion model (DDPM with DDIM reverse steps)
and extracts the positive normalized anomaly residual map:
    residual = max(0, x0 - x0_hat) / (1 + |x0_hat|)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Tuple, Union

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch

from diffusion import DiffusionSchedule, diffuse
from inference import (
    load_diffusion_model,
    make_ddim_timesteps,
    reverse_diffusion_ddim,
)

# -----------------------------------------------------------------------------
# Global model cache (Singleton pattern to avoid reloading weights per patch)
# -----------------------------------------------------------------------------
_MODEL_CACHE: dict[Tuple[str, str], Tuple[torch.nn.Module, DiffusionSchedule]] = {}


def resolve(path: str) -> str:
    """Resolve a path relative to the project root if it is not absolute."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def get_model_and_schedule(
    model_name_or_path: str = "models/unet32.pt",
    device: str | torch.device | None = None,
) -> Tuple[torch.nn.Module, DiffusionSchedule]:
    """
    Get a cached diffusion model and its schedule, loading them if necessary.

    Parameters
    ----------
    model_name_or_path : str
        Relative or absolute path to the .pt checkpoint. Default is 'models/unet32.pt'.
    device : str or torch.device, optional
        Target device ('cuda' or 'cpu'). If None, automatically detects CUDA.

    Returns
    -------
    model : UNetDiffusion
    schedule : DiffusionSchedule
    """
    resolved_path = resolve(model_name_or_path)
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = str(device)

    cache_key = (resolved_path, device_str)
    if cache_key not in _MODEL_CACHE:
        model, schedule = load_diffusion_model(resolved_path, device=device_str)
        _MODEL_CACHE[cache_key] = (model, schedule)

    return _MODEL_CACHE[cache_key]


def load_patch(
    patch_or_path: Union[str, os.PathLike, np.ndarray, torch.Tensor],
    mean: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    """
    Load and standardize a patch to zero mean and unit variance (Eq. 9 of challenge note).

    Parameters
    ----------
    patch_or_path : str, Path, np.ndarray or torch.Tensor
        Either the file path to a .npy patch or an array/tensor of shape (H, W) or (1, H, W).
    mean : float, optional
        Predefined mean for normalization. If None, computes patch.mean().
    std : float, optional
        Predefined std for normalization. If None, computes patch.std().

    Returns
    -------
    normalized_patch : np.ndarray of shape (H, W), dtype float32
    """
    if isinstance(patch_or_path, (str, os.PathLike)):
        patch = np.load(str(patch_or_path)).astype(np.float32).squeeze()
    elif isinstance(patch_or_path, torch.Tensor):
        patch = patch_or_path.detach().cpu().numpy().astype(np.float32).squeeze()
    elif isinstance(patch_or_path, np.ndarray):
        patch = patch_or_path.astype(np.float32).squeeze()
    else:
        raise TypeError(f"Unsupported patch input type: {type(patch_or_path)}")

    if patch.ndim != 2:
        raise ValueError(f"Invalid patch shape {patch.shape}; expected a 2D patch.")

    if mean is None or std is None:
        p_mean = patch.mean() if mean is None else mean
        p_std = patch.std() if std is None else std
        return (patch - p_mean) / (p_std + 1e-8)

    return (patch - mean) / (std + 1e-8)


@torch.no_grad()
def generate_purified_reconstruction(
    patch_or_path: Union[str, os.PathLike, np.ndarray, torch.Tensor],
    model: torch.nn.Module | None = None,
    schedule: DiffusionSchedule | None = None,
    model_name_or_path: str = "models/unet32.pt",
    t_start: int = 600,
    ddim_steps: int = 6,
    num_samples: int = 1,
    seed: int | None = 0,
    device: str | torch.device | None = None,
    mean: float | None = None,
    std: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Purify a patch using forward diffusion to t_start followed by DDIM reconstruction.

    Parameters
    ----------
    patch_or_path : str, Path, np.ndarray or torch.Tensor
        Input patch (file path or array).
    model : nn.Module, optional
        Pre-loaded model. If None, loaded from cache using `model_name_or_path`.
    schedule : DiffusionSchedule, optional
        Pre-loaded schedule matching model.
    model_name_or_path : str
        Model checkpoint to use if `model` is not provided.
    t_start : int
        Intermediate corruption timestep (default: 600).
    ddim_steps : int
        Number of deterministic reverse steps (default: 6).
    num_samples : int
        Number of independent stochastic noise realizations to average (default: 1).
    seed : int, optional
        Random seed for reproducibility. If None, uses non-deterministic RNG.
    device : str or torch.device, optional
        Execution device.
    mean, std : float, optional
        Normalization statistics.

    Returns
    -------
    x0_np : np.ndarray (H, W)
        Normalized original patch.
    x0_hat_np : np.ndarray (H, W)
        Purified background reconstruction.
    """
    if model is None or schedule is None:
        model, schedule = get_model_and_schedule(model_name_or_path, device=device)

    dev = next(model.parameters()).device
    x0_np = load_patch(patch_or_path, mean=mean, std=std)

    x0_single = torch.from_numpy(x0_np)[None, None].to(dev)  # (1, 1, H, W)
    batch_size = max(1, num_samples)
    x0_batch = x0_single.repeat(batch_size, 1, 1, 1)

    t_clamped = min(t_start, schedule.timesteps - 1)
    t_tensor = torch.full((batch_size,), t_clamped, device=dev, dtype=torch.long)

    if seed is not None:
        torch.manual_seed(seed)

    # 1. Forward corruption to t_start
    xt, _ = diffuse(x0_batch, t_tensor, schedule)

    # 2. Reverse DDIM reconstruction
    k_steps = min(ddim_steps, t_clamped + 1)
    timesteps = make_ddim_timesteps(k_steps, schedule.timesteps, t_clamped)
    x0_hat_batch, _ = reverse_diffusion_ddim(xt, timesteps, model, schedule)

    # 3. Aggregate multiple samples if requested
    if batch_size > 1:
        x0_hat = x0_hat_batch.mean(dim=0, keepdim=True)
    else:
        x0_hat = x0_hat_batch

    x0_hat_np = x0_hat.squeeze().cpu().numpy()
    return x0_np, x0_hat_np


@torch.no_grad()
def compute_diffusion_residual(
    patch_or_path: Union[str, os.PathLike, np.ndarray, torch.Tensor],
    model: torch.nn.Module | None = None,
    schedule: DiffusionSchedule | None = None,
    model_name_or_path: str = "models/unet32.pt",
    t_start: int = 800,
    ddim_steps: int = 6,
    num_samples: int = 1,
    seed: int | None = 0,
    device: str | torch.device | None = None,
    mean: float | None = None,
    std: float | None = None,
) -> np.ndarray:
    """
    Compute the positive normalized diffusion anomaly residual map for a patch:
        residual = max(0, x0 - x0_hat) / (1 + |x0_hat|)

    Highlights physical emissions (additive energy) while suppressing background troughs.

    Parameters
    ----------
    patch_or_path : str, Path, np.ndarray or torch.Tensor
        Input patch to purify.
    model, schedule : optional
        Cached or preloaded diffusion model and schedule.
    model_name_or_path : str
        Default 'models/unet32.pt'.
    t_start : int
        Starting corruption timestep (default: 600).
    ddim_steps : int
        Number of reverse steps (default: 6).
    num_samples : int
        Number of noise seeds to average (default: 1).
    seed : int, optional
        RNG seed (default: 0).
    device : str or torch.device, optional
        Device to run inference on.
    mean, std : float, optional
        Normalization parameters.

    Returns
    -------
    residual : np.ndarray of shape (H, W), dtype float32
    """
    x0_np, x0_hat_np = generate_purified_reconstruction(
        patch_or_path=patch_or_path,
        model=model,
        schedule=schedule,
        model_name_or_path=model_name_or_path,
        t_start=t_start,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        seed=seed,
        device=device,
        mean=mean,
        std=std,
    )

    diff = x0_np - x0_hat_np
    pos_diff = np.maximum(0.0, diff)
    residual = pos_diff / (1.0 + np.abs(x0_hat_np))
    smoothed_residual=apply_mean_smoothing(residual,3)
    return residual +2*(smoothed_residual.astype(np.float32))
    return residual.astype(np.float32)


def compute_residual_with_details(
    patch_or_path: Union[str, os.PathLike, np.ndarray, torch.Tensor],
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function returning (residual, original_normalized, reconstruction).
    """
    x0_np, x0_hat_np = generate_purified_reconstruction(patch_or_path, **kwargs)

    diff = x0_np - x0_hat_np
    pos_diff = np.maximum(0.0, diff)
    residual = pos_diff / (1.0 + np.abs(x0_hat_np))

    return residual.astype(np.float32), x0_np, x0_hat_np

# -----------------------------------------------------------------------------
# Smoothing functions
# -----------------------------------------------------------------------------

from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import uniform_filter

def apply_gaussian_smoothing(residual: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return gaussian_filter(residual, sigma=sigma)

def apply_median_smoothing(residual: np.ndarray, size: int = 3) -> np.ndarray:
    return median_filter(residual, size=size)

def apply_mean_smoothing(residual: np.ndarray, size: int = 3) -> np.ndarray:
    return uniform_filter(residual, size=size)

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_diffusion_result(
    patch_or_path: Union[str, os.PathLike, np.ndarray, torch.Tensor],
    model: torch.nn.Module | None = None,
    schedule: DiffusionSchedule | None = None,
    model_name_or_path: str = "models/unet32.pt",
    t_start: int = 750,
    ddim_steps: int = 6,
    num_samples: int = 1,
    seed: int | None = 0,
    device: str | torch.device | None = None,
    show: bool = True,
) -> Tuple[plt.Figure, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Purify a patch and display a 4-panel comparative figure:
    1. Original normalized spectrogram patch (frequency y vs time x)
    2. DDIM background reconstruction
    3. Positive normalized anomaly residual map
    4. Smoothed residual map (Median filter)

    Parameters
    ----------
    patch_or_path : str, Path, np.ndarray or torch.Tensor
        Path to .npy patch or 2D array.
    model, schedule : optional
        Pre-loaded model and schedule. If None, retrieved from cache.
    model_name_or_path : str
        Model checkpoint to use if model is None (default: 'models/unet32.pt').
    t_start : int
        Intermediate noising timestep (default: 600).
    ddim_steps : int
        Number of reverse DDIM steps (default: 6).
    num_samples : int
        Number of noise realizations to average (default: 1).
    seed : int, optional
        RNG seed (default: 0).
    device : str or torch.device, optional
        Target device ('cuda' or 'cpu').
    show : bool
        Whether to call plt.show() (default: True).

    Returns
    -------
    fig : matplotlib.figure.Figure
    (residual, smooth_residual, original_normalized, reconstruction) : Tuple of np.ndarray
    """
    residual, x0_np, x0_hat_np = compute_residual_with_details(
        patch_or_path=patch_or_path,
        model=model,
        schedule=schedule,
        model_name_or_path=model_name_or_path,
        t_start=t_start,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        seed=seed,
        device=device,
    )

    # Compute the smoothed residual
    smooth_residual = residual + apply_mean_smoothing(residual, size=3)

    # Check for annotations to enrich figure title
    label_info = ""
    patch_name = "Patch"
    if isinstance(patch_or_path, (str, os.PathLike)):
        patch_name = os.path.basename(str(patch_or_path))
        annot_file = resolve("dataset/annotations.json")
        if os.path.exists(annot_file):
            try:
                with open(annot_file, "r") as f:
                    annotations = json.load(f)
                if patch_name in annotations:
                    a = annotations[patch_name]
                    labels = [k for k in ("horizontal", "other") if a.get(k)] or ["aucun / bruit pur"]
                    label_info = f" | Vérité terrain: {' + '.join(labels)}"
            except Exception:
                pass

    # Changed from 1x3 to 1x4 and increased width to 18
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

    # Shared dynamic range for original and reconstruction
    vmin = min(x0_np.min(), x0_hat_np.min())
    vmax = max(x0_np.max(), x0_hat_np.max())

    im0 = axes[0].imshow(x0_np, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title("1. Original (normalisé)")
    axes[0].set_xlabel("Temps (x)")
    axes[0].set_ylabel("Fréquence (y)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(x0_hat_np, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title(f"2. Reconstruction DDIM (t={t_start})")
    axes[1].set_xlabel("Temps (x)")
    axes[1].set_ylabel("Fréquence (y)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(residual, origin="lower", cmap="magma")
    axes[2].set_title("3. Résidu positif normalisé")
    axes[2].set_xlabel("Temps (x)")
    axes[2].set_ylabel("Fréquence (y)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4th panel for smoothed residual
    im3 = axes[3].imshow(smooth_residual, origin="lower", cmap="magma")
    axes[3].set_title("4. Résidu lissé")
    axes[3].set_xlabel("Temps (x)")
    axes[3].set_ylabel("Fréquence (y)")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(f"{patch_name}{label_info}\n(Modèle: {os.path.basename(model_name_or_path)}, t_start={t_start}, steps={ddim_steps})", fontsize=11)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, (residual, smooth_residual, x0_np, x0_hat_np)


# -----------------------------------------------------------------------------
# Quick demo / test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Test patch with horizontal anomaly
    test_patch = resolve("dataset/test/patch_9099_13.npy")
    if not os.path.exists(test_patch):
        test_dir = resolve("dataset/test")
        all_files = [f for f in os.listdir(test_dir) if f.endswith(".npy")]
        test_patch = os.path.join(test_dir, all_files[0])

    print(f"[INFO] Exécution de plot_diffusion_result sur : {os.path.basename(test_patch)}")

    # Make sure to unpack 4 variables now instead of 3
    fig, (res, res_smooth, orig, reco) = plot_diffusion_result(
        test_patch,
        model_name_or_path="models/unet64.pt",
        t_start=700,
        ddim_steps=4,
        num_samples=1,
        show=True,
    )

    print(f"[INFO] Dimensions : {res.shape}")
    print(f"[INFO] Min résidu: {res.min():.4f}, Max: {res.max():.4f}, Moyenne: {res.mean():.4f}")
    print(f"[INFO] Min résidu lissé: {res_smooth.min():.4f}, Max: {res_smooth.max():.4f}, Moyenne: {res_smooth.mean():.4f}")
    print("[INFO] Démonstration terminée avec succès !")

    test_patch = resolve("dataset/test/patch_9107_22.npy")
    if not os.path.exists(test_patch):
        test_dir = resolve("dataset/test")
        all_files = [f for f in os.listdir(test_dir) if f.endswith(".npy")]
        test_patch = os.path.join(test_dir, all_files[0])

    print(f"[INFO] Exécution de plot_diffusion_result sur : {os.path.basename(test_patch)}")

    # Make sure to unpack 4 variables now instead of 3
    fig, (res, res_smooth, orig, reco) = plot_diffusion_result(
        test_patch,
        model_name_or_path="models/unet64.pt",
        t_start=700,
        ddim_steps=4,
        num_samples=1,
        show=True,
    )

    print(f"[INFO] Dimensions : {res.shape}")
    print(f"[INFO] Min résidu: {res.min():.4f}, Max: {res.max():.4f}, Moyenne: {res.mean():.4f}")
    print(f"[INFO] Min résidu lissé: {res_smooth.min():.4f}, Max: {res_smooth.max():.4f}, Moyenne: {res_smooth.mean():.4f}")
    print("[INFO] Démonstration terminée avec succès !")    