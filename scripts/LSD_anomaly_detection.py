"""
Implements a line detection algorithm based on the frequency of max intensity for every timestep 
"""
import numpy as np
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

#Utilities
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


def resolve(path: str) -> str:
    """
    Resolve a path relative to the project root (if not absolute).
    """
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


TEST_DIR = "dataset/test"
#IMAGE = "patch_9099_13.npy"
#IMAGE = "patch_9107_22.npy" # Horizontal line
IMAGE = "patch_9150_13.npy" # Other

test_dir = resolve(TEST_DIR)
image_path = os.path.join(test_dir, IMAGE)
def gradient_calculator(patch):
    gx = np.zeros_like(patch, dtype=np.float32)
    gy = np.zeros_like(patch, dtype=np.float32)
    

    tl = patch[:-1, :-1] 
    tr = patch[:-1, 1:]  
    bl = patch[1:, :-1]  
    br = patch[1:, 1:]   
    
    # Gradient
    gx[:-1, :-1] = (tr + br - tl - bl) / 2.0
    gy[:-1, :-1] = (bl + br - tl - tr) / 2.0
    
    norm = np.sqrt(gx**2+gy**2)
    angle = np.arctan2(gy, gx)
    return norm, angle

def diff_angle(angle_1,angle_2):
    #angle modulo 2 pi
    return min(np.abs(angle_1-angle_2),2*np.pi - np.abs(angle_1-angle_2))

def region_growing(mag, angle):
    """Regroupe les pixels adjacents ayant une orientation de gradient similaire."""
    tau=np.pi/8
    f, t = mag.shape
    used = np.zeros((f, t), dtype=bool)
    regions = []
    
    
    flat_indices = np.argsort(-mag.ravel())
    y_coords, x_coords = np.unravel_index(flat_indices, (f, t))

    #We ignore lowest magnitude gradient as probably not in a defined line
    min_mag = np.percentile(mag, 50) 

    #Compute the theoretical limit of significant region size (vs random noise region size)

    N_test = (f * t) ** 2.5 
    
    # Random alignement probability
    p = tau / np.pi 
    
    # NFA lower bound (Number of false Alarm) we want to have average number of false region <1
    min_pixels_nfa = int(np.ceil(-np.log10(N_test) / np.log10(p)))
    
    for y, x in zip(y_coords, x_coords):
        if used[y, x] or mag[y, x] < min_mag:
            continue
            
        seed_angle = angle[y, x]
        region = []
        queue = [(y, x)]
        used[y, x] = True
        
        #BFS region 
        while queue:
            cy, cx = queue.pop(0)
            region.append((cy, cx))
            
            # Vérification des 8 voisins
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < f and 0 <= nx < t and not used[ny, nx]:
                        if diff_angle(angle[ny, nx], seed_angle) <= tau:
                            used[ny, nx] = True
                            queue.append((ny, nx))
        if len(region) >= min_pixels_nfa:
            regions.append(np.array(region))

    return regions

import numpy as np

def is_region_horizontal(region, angles_matrix, tol_rad=np.pi/8):
    """Detect horizontal region by looking at gradient is vertical"""
    region_angles = angles_matrix[region[:, 0], region[:, 1]]

    median_angle = np.median(region_angles)
    diff_from_vertical = np.abs(np.abs(median_angle) - np.pi/2)
    return diff_from_vertical <= tol_rad


def anamoly_detection_LSD(patch_path: str, mean=None, std=None) -> dict:
    patch = load_patch(patch_path, mean, std)
    norm, angle = gradient_calculator(patch)
    regions = region_growing(norm, angle)
    
    has_horizontal = False
    has_other = False
    
    for region in regions:
        if is_region_horizontal(region, angle, tol_rad=15):
            has_horizontal = True
        else:
            has_other = True
            
        if has_horizontal and has_other:
            break
            
    return {"horizontal": has_horizontal, "other": has_other}