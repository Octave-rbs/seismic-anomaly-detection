import sys
import os
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from skimage.transform import hough_line, hough_line_peaks

from diffusion import diffuse
from inference import (
    load_diffusion_model,
    make_ddim_timesteps,
    reverse_diffusion_ddim,
)

from reconstruct_test_patch import resolve, load_patch
from test_methods import compare_method_to_annotations
from image_postprocessing import box_convolution


### Parameters ###

MODEL = "models/unet64.pt"
TEST_DIR = "dataset/test"
# IMAGE = "patch_9107_22.npy"  # Horizontal
# IMAGE = "patch_9150_13.npy"  # Other
# IMAGE = "patch_9998_20.npy"  # Nothing
# IMAGE = "patch_9998_11.npy"  # Other
IMAGE = "patch_9103_10.npy"
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

# Hyperparameters for Hough transform
HOUGH_THRESHOLD = 10
HOUGH_NUM_PEAKS = 10
BACKGROUND_THRESHOLD_PERCENTILE = 97  # threshold percentile to create the binary image


def detect_lines(error):
    threshold = np.percentile(error, BACKGROUND_THRESHOLD_PERCENTILE)

    binary = error > threshold

    # Hough classique
    hspace, angles, distances = hough_line(binary)

    # Recherche des pics dans l'accumulateur
    accum, peak_angles, peak_distances = hough_line_peaks(
        hspace,
        angles,
        distances,
        threshold=HOUGH_THRESHOLD,
        num_peaks=HOUGH_NUM_PEAKS,
    )

    horizontal = []
    other = []

    for score, theta, rho in zip(accum, peak_angles, peak_distances):
        # theta = angle de la normale
        # donc angle de la ligne :
        line_angle = np.degrees(theta + np.pi / 2)

        # ramener entre -90° et +90°
        if line_angle > 90:
            line_angle -= 180
        elif line_angle < -90:
            line_angle += 180

        print(
            f"rho={rho:.1f}, "
            f"theta={np.degrees(theta):.1f}°, "
            f"line_angle={line_angle:.1f}°, "
            f"votes={score}"
        )

        if abs(line_angle) <= 10:
            horizontal.append((line_angle, theta, rho, score))

        elif abs(line_angle) < 80:
            other.append((line_angle, theta, rho, score))

    return binary, horizontal, other


def draw_detection(
    error,
    binary,
    horizontal,
    other,
    peak_lines=None,
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

    def draw_hough_line(theta, rho, line_type):
        height, width = error.shape

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        points = []

        if abs(sin_theta) > 1e-8:
            y = rho / sin_theta
            if 0 <= y <= height - 1:
                points.append((0, y))

        if abs(sin_theta) > 1e-8:
            x = width - 1
            y = (rho - x * cos_theta) / sin_theta
            if 0 <= y <= height - 1:
                points.append((x, y))

        if abs(cos_theta) > 1e-8:
            x = rho / cos_theta
            if 0 <= x <= width - 1:
                points.append((x, 0))

        if abs(cos_theta) > 1e-8:
            y = height - 1
            x = (rho - y * sin_theta) / cos_theta
            if 0 <= x <= width - 1:
                points.append((x, y))

        # Éliminer les doublons
        unique_points = []
        for p in points:
            if not any(np.allclose(p, q) for q in unique_points):
                unique_points.append(p)

        if len(unique_points) < 2:
            return

        p1, p2 = unique_points[:2]
        color = "cyan" if line_type == "horizontal" else "yellow"

        axes[2].plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            color=color,
            linewidth=2,
        )

    # Horizontal lines
    for line_angle, theta, rho, score in horizontal:
        draw_hough_line(theta, rho, "horizontal")

    # Oblique lines
    for line_angle, theta, rho, score in other:
        draw_hough_line(theta, rho, "other")

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
    preprocess: bool = False,
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

    if preprocess:
        error = (
            box_convolution(error, kernel_size=10, theta=0.0, sigma=100).cpu().numpy()
        )

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
    return {"horizontal": len(horizontal) > 0, "other": len(other) > 0}


def evaluate_hough_for_image(image):
    result = identify_lines_with_hough(
        model_path=resolve(MODEL),
        image_path=image,
        t_start=T_START,
        ddim_steps=DDIM_STEPS,
        device=DEVICE,
        seed=SEED,
        visualize=False,
    )
    return result


def evaluate_hough_method():
    results = compare_method_to_annotations(method=evaluate_hough_for_image, show=True)
    return results


if __name__ == "__main__":
    identify_lines_with_hough(
        model_path=resolve(MODEL),
        image_path=image_path,
        t_start=T_START,
        ddim_steps=DDIM_STEPS,
        device=DEVICE,
        seed=SEED,
        preprocess=False,
        visualize=True,
    )
    # evaluate_hough_method()
