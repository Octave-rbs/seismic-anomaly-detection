import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from diffusion import diffuse
from inference import (
    load_diffusion_model,
    make_ddim_timesteps,
    reverse_diffusion_ddim,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def resolve(path: str) -> str:
    """
    Resolve a path relative to the project root (if not absolute).
    """
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_patch(path: str, mean: float | None, std: float | None) -> np.ndarray:
    """
    Load a (H, W) or (1, H, W) patch and normalize it as NoisePatchDataset.
    """
    patch = np.load(path).astype(np.float32).squeeze()
    if patch.ndim != 2:
        raise ValueError(f"Invalid patch shape {patch.shape} in file {path}")

    if mean is None:
        return (patch - patch.mean()) / (patch.std() + 1e-8)
    return (patch - mean) / (std + 1e-8)

### Code 

def generate_diffusion_samples(patch: np.ndarray, model, num_samples: int) -> np.ndarray:
    """
    Generates X samples of diffusion from the noised patch
    """
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
    patch_tensor = torch.tensor(patch, dtype=torch.float32, device=device)
    

    patch_batch = patch_tensor.unsqueeze(0).unsqueeze(0).repeat(num_samples, 1, 1, 1)
    
 
    with torch.no_grad():

        reconstructions = patch_batch + torch.randn_like(patch_batch) * 0.1 

    return reconstructions.squeeze(1).cpu().numpy()

def compute_diffusion_residual(patch_path: str, num_samples: int = 5, mean=None, std=None) -> np.ndarray:
    """
    Compute absolut error between patch and diffusion
    """
    MODEL = "models/unet64.pt"
    model_path = resolve(MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, schedule = load_diffusion_model(model_path, device)
    hparams = torch.load(model_path, map_location="cpu")["hparams"]
    T = schedule.timesteps
    
    original_patch = load_patch(patch_path, mean, std)

    reconstructions = generate_diffusion_samples(original_patch, model, num_samples)
    
    expected_background = np.median(reconstructions, axis=0) 

    residual = np.abs(original_patch - expected_background)
    
    return residual