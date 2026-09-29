"""
Detect horizontal / oblique lines from the reconstruction error
of the diffusion model, with Hough transform.

Pipeline:
    test patch
        -> diffusion
        -> DDIM reconstruction
        -> absolute error
        -> threshold
        -> Hough line detection

Possible outputs:
    - nothing
    - horizontal
    - other

python scripts/hough_transform.py
"""

import sys
import os
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from skimage.transform import probabilistic_hough_line

from diffusion import diffuse
from inference import (
    load_diffusion_model,
    make_ddim_timesteps,
    reverse_diffusion_ddim,
)

from reconstruct_test_patch import resolve, load_patch


### Parameters ###

MODEL = "models/unet64.pt"
TEST_DIR = "dataset/test"
# IMAGE = "patch_9107_22.npy"  # Horizontal
IMAGE = "patch_9150_13.npy"  # Other
T_START = 600
DDIM_STEPS = 6
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
test_dir = resolve(TEST_DIR)
if IMAGE is None:
    files = sorted(f for f in os.listdir(test_dir) if f.endswith(".npy"))
    image_path = os.path.join(test_dir, np.random.default_rng(SEED).choice(files))

else:
    image_path = os.path.join(test_dir, IMAGE)

size_image = load_patch(image_path, 0, 1).shape[0]

# Hyperparameters for Hough transform, to be optimized with test_methods.py ?
HOUGH_THRESHOLD = 3  # minimum number of perfectly aligned pixels to consider a line
HOUGH_LINE_LENGTH = size_image // 3  # minimum length of a line segment to be accepted
HOUGH_LINE_GAP = 6  # maximum gap between points on a line to be the same line
BACKGROUND_THRESHOLD_PERCENTILE = 97  # threshold percentile to create the binary image


def detect_lines(error):

    # 1. Thresholding to create the binary image for Hough transform
    # TODO: Apply first preprocessing if we find one useful
    threshold = np.percentile(error, BACKGROUND_THRESHOLD_PERCENTILE)
    binary = error > threshold

    # 2. Hough
    lines = probabilistic_hough_line(
        binary,
        threshold=HOUGH_THRESHOLD,
        line_length=HOUGH_LINE_LENGTH,
        line_gap=HOUGH_LINE_GAP,
    )

    horizontal = []
    other = []

    # 3. Classify lines
    for line in lines:
        (x1, y1), (x2, y2) = line
        dx = x2 - x1
        dy = y2 - y1

        angle = np.degrees(np.arctan2(dy, dx))

        # ramener dans [-90, 90]
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180

        length = np.sqrt(dx**2 + dy**2)

        if abs(angle) <= 10:
            horizontal.append((line, angle, length))

        elif abs(angle) < 80:
            other.append((line, angle, length))

    return binary, horizontal, other


def draw_detection(
    error,
    binary,
    horizontal,
    other,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im = axes[0].imshow(error, origin="lower", cmap="magma")
    axes[0].set_title("Absolute error")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].imshow(binary, origin="lower", cmap="gray")
    axes[1].set_title("Thresholded error")
    axes[1].axis("off")

    axes[2].imshow(error, origin="lower", cmap="magma")
    for line, _, _ in horizontal:
        (x1, y1), (x2, y2) = line
        axes[2].plot((x1, x2), (y1, y2), color="cyan", linewidth=2)
    for line, _, _ in other:
        (x1, y1), (x2, y2) = line
        axes[2].plot((x1, x2), (y1, y2), color="yellow", linewidth=2)
    axes[2].set_title(
        f"Detected lines (horizontal={len(horizontal)}, other={len(other)})"
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def identify_lines_with_hough(
    model_path: str,
    image_path: str,
    t_start: int,
    ddim_steps: int,
    device: str,
    seed: int,
    visualize: bool = True,
):

    model, schedule = load_diffusion_model(model_path, device)
    hparams = torch.load(model_path, map_location="cpu")["hparams"]
    T = schedule.timesteps

    image = load_patch(image_path, hparams["data_mean"], hparams["data_std"])
    x0 = torch.from_numpy(image)[None, None].to(device)
    torch.manual_seed(seed)
    t = torch.tensor([t_start], device=device)
    xt, _ = diffuse(x0, t, schedule)

    ddim_steps = min(ddim_steps, t_start + 1)
    timesteps = make_ddim_timesteps(ddim_steps, T, t_start)
    x0_hat, _ = reverse_diffusion_ddim(xt, timesteps, model, schedule)

    # Reconstruction error
    x0_np = x0.squeeze().cpu().numpy()
    x0_hat_np = x0_hat.squeeze().cpu().numpy()
    error = np.abs(x0_np - x0_hat_np)

    # Hough detection
    binary, horizontal, other = detect_lines(error)

    # Visualization
    if visualize:
        draw_detection(
            error,
            binary,
            horizontal,
            other,
        )
    return {image_path: {"horizontal": len(horizontal) > 0, "other": len(other) > 0}}


if __name__ == "__main__":
    identify_lines_with_hough(
        model_path=resolve(MODEL),
        image_path=image_path,
        t_start=T_START,
        ddim_steps=DDIM_STEPS,
        device=DEVICE,
        seed=SEED,
        visualize=True,
    )
