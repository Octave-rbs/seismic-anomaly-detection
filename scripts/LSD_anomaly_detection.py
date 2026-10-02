"""
LSD (Line Segment Detector) Anomaly Detection on Spectrograms and Diffusion Residuals.

Detects linear structures (horizontal and oblique segments) in spectrograms
or diffusion residual maps, extracts geometric coordinates (x1, y1) -> (x2, y2),
classifies segment orientations, and displays comparative visualizations.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple, Union

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
SCRIPTS_DIR = os.path.join(ROOT, "scripts")

for d in (SRC_DIR, SCRIPTS_DIR):
    if d not in sys.path:
        sys.path.append(d)

import matplotlib.pyplot as plt
import numpy as np

from diffusion_residual import compute_diffusion_residual, load_patch, resolve


# -----------------------------------------------------------------------------
# Gradient and Angle Utilities
# -----------------------------------------------------------------------------
def compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient magnitude and gradient orientation angle using central differences.

    Parameters
    ----------
    image : np.ndarray (H, W)
        Input 2D image (spectrogram patch or residual map).

    Returns
    -------
    mag : np.ndarray (H, W)
        Gradient magnitude.
    angle : np.ndarray (H, W)
        Gradient orientation angle in radians [-pi, pi].
    """
    gy, gx = np.gradient(image.astype(np.float32))
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    return mag, angle


def angular_diff(a1: float, a2: float) -> float:
    """Compute the minimal angular difference modulo pi (gradient line orientation)."""
    d = np.abs(a1 - a2) % np.pi
    return min(d, np.pi - d)

# -----------------------------------------------------------------------------
# Region Growing (LSD Line Support Regions)
# -----------------------------------------------------------------------------
def region_growing(
    mag: np.ndarray,
    angle: np.ndarray,
    tau: float = np.pi / 8,
    min_pixels_nfa: int = 10,
) -> List[np.ndarray]:

    h, w = mag.shape
    used = np.zeros((h, w), dtype=bool)
    regions = []

    # Threshold gradient magnitude to eliminate uniform background noise
    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag))
    threshold_mag = mag_mean + mag_std

    # Start region growing from highest gradient magnitude pixels
    sorted_indices = np.argsort(-mag.ravel())
    y_coords, x_coords = np.unravel_index(sorted_indices, (h, w))

    for sy, sx in zip(y_coords, x_coords):
        if used[sy, sx] or mag[sy, sx] < threshold_mag:
            continue

        seed_angle = angle[sy, sx]
        region_pixels = []
        queue = [(sy, sx)]
        used[sy, sx] = True

        while queue:
            cy, cx = queue.pop(0)
            region_pixels.append((cy, cx))

            # Check 8-neighborhood
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not used[ny, nx]:
                        if mag[ny, nx] >= threshold_mag * 0.5:
                            if angular_diff(angle[ny, nx], seed_angle) <= tau:
                                used[ny, nx] = True
                                queue.append((ny, nx))

        if len(region_pixels) >= min_pixels_nfa:
            regions.append(np.array(region_pixels))

    return regions


# -----------------------------------------------------------------------------
# Segment Extraction: (x1, y1) -> (x2, y2)
# -----------------------------------------------------------------------------
def fit_segment_to_region(
    region: np.ndarray,
    mag: np.ndarray,
    min_length: float = 7,
    min_elongation: float = 2,
) -> Dict[str, Any] | None:
    """
    Extract a geometric line segment (x1, y1) -> (x2, y2) from a pixel region via PCA.

    Parameters
    ----------
    region : np.ndarray of shape (N, 2)
        Pixel coordinates (y, x).
    mag : np.ndarray (H, W)
        Gradient magnitude used for weighting.
    min_length : float
        Minimum Euclidean segment length in pixels.
    min_elongation : float
        Minimum ratio between longitudinal and transverse spread (sqrt(lambda1 / lambda2)).

    Returns
    -------
    segment_info : dict or None
        Dictionary with keys:
            'p1': (x1, y1)
            'p2': (x2, y2)
            'angle': float in [-90, 90] degrees (angle of the segment with x-axis)
            'length': float
            'elongation': float
            'type': 'horizontal' | 'other' | 'vertical'
    """
    if len(region) < 3:
        return None

    ys = region[:, 0].astype(np.float64)
    xs = region[:, 1].astype(np.float64)
    weights = mag[region[:, 0], region[:, 1]].astype(np.float64)
    w_sum = np.sum(weights)

    if w_sum <= 1e-8:
        weights = np.ones_like(ys)
        w_sum = float(len(region))

    # Center of mass
    cx = np.sum(xs * weights) / w_sum
    cy = np.sum(ys * weights) / w_sum

    # Central second moments (spatial inertia tensor)
    dx = xs - cx
    dy = ys - cy
    mu_xx = np.sum(weights * (dx**2)) / w_sum
    mu_yy = np.sum(weights * (dy**2)) / w_sum
    mu_xy = np.sum(weights * dx * dy) / w_sum

    # Principal direction angle
    theta = 0.5 * np.arctan2(2.0 * mu_xy, mu_xx - mu_yy)
    ux = np.cos(theta)
    uy = np.sin(theta)

    # Eigenvalues of covariance matrix
    delta = np.sqrt((mu_xx - mu_yy) ** 2 + 4.0 * (mu_xy**2))
    lambda1 = 0.5 * (mu_xx + mu_yy + delta)
    lambda2 = max(0.5 * (mu_xx + mu_yy - delta), 1e-6)
    elongation = np.sqrt(lambda1 / lambda2)

    # Project coordinates on principal axis to find segment endpoints
    projections = dx * ux + dy * uy
    s_min = np.min(projections)
    s_max = np.max(projections)

    length = s_max - s_min
    if length < min_length:
        return None
    if elongation < min_elongation:
        return None
    # Compute endpoint coordinates
    x1 = cx + s_min * ux
    y1 = cy + s_min * uy
    x2 = cx + s_max * ux
    y2 = cy + s_max * uy

    # Segment angle with horizontal axis in degrees in [-90, 90]
    segment_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    if segment_angle > 90:
        segment_angle -= 180
    elif segment_angle < -90:
        segment_angle += 180

    # Classify orientation
    abs_angle = abs(segment_angle)
    if abs_angle <= np.degrees(np.pi/8):
        seg_type = "horizontal"
    elif abs_angle >=np.degrees(np.pi/2-np.pi/8):
        seg_type = "vertical"  # Non-target event (impulsive/vertical)
    else:
        seg_type = "other"  # Oblique / chirping line

    return {
        "p1": (float(x1), float(y1)),
        "p2": (float(x2), float(y2)),
        "angle": float(segment_angle),
        "length": float(length),
        "elongation": float(elongation),
        "type": seg_type,
    }


def detect_segments_lsd(
    image: np.ndarray,
    tau: float = np.pi / 8,
    min_pixels_nfa: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Run LSD line segment detection on an image (patch or residual).

    Returns
    -------
    segments : List[dict]
        List of detected segments with endpoints, length, angle, and type.
    """
    mag, angle = compute_gradients(image)
    regions = region_growing(
        mag=mag,
        angle=angle,
        tau=tau,
        min_pixels_nfa=int(min_pixels_nfa),
    )

    segments = []
    for region in regions:
        seg = fit_segment_to_region(
            region=region,
            mag=mag,
            min_length=min_pixels_nfa
        )
        if seg is not None:
            segments.append(seg)

    return segments


# -----------------------------------------------------------------------------
# Decision Pipeline
# -----------------------------------------------------------------------------
def predict_lsd(
    patch_or_path: Union[str, os.PathLike, np.ndarray],
    use_diffusion: bool = True,
    t_start: int = 800,
    ddim_steps: int = 6,
    model_name_or_path: str = "models/unet32.pt",
    min_length: int=7
) -> Dict[str, bool]:
    """
    Predict presence of 'horizontal' and 'other' line anomalies for a patch.

    Parameters
    ----------
    patch_or_path : str, Path or np.ndarray
        Input patch to evaluate.
    use_diffusion : bool
        If True, runs LSD on the diffusion residual map (recommended).
        If False, runs LSD on the raw normalized patch.

    Returns
    -------
    decision : dict
        {'horizontal': bool, 'other': bool}
    """
    if use_diffusion:
        input_map = compute_diffusion_residual(
            patch_or_path,
            model_name_or_path=model_name_or_path,
            t_start=t_start,
            ddim_steps=ddim_steps,
        )
    else:
        input_map = load_patch(patch_or_path)

    segments = detect_segments_lsd(input_map,min_pixels_nfa=min_length)

    has_horizontal = any(s["type"] == "horizontal" for s in segments)
    has_other = any(s["type"] == "other" for s in segments)

    return {"horizontal": has_horizontal, "other": has_other}


# -----------------------------------------------------------------------------
# Visualization (Same 3-panel style as diffusion)
# -----------------------------------------------------------------------------
def plot_lsd_detection(
    patch_or_path: Union[str, os.PathLike, np.ndarray],
    model_name_or_path: str = "models/unet32.pt",
    t_start: int = 600,
    ddim_steps: int = 6,
    show: bool = True,
) -> Tuple[plt.Figure, List[Dict[str, Any]]]:
    """
    Purify a patch, run LSD line segment detection, and display a 3-panel figure:
    1. Original normalized spectrogram patch
    2. Positive normalized diffusion residual
    3. Anomaly residual with detected line segments overlaid (cyan=horizontal, yellow=other)

    Parameters
    ----------
    patch_or_path : str, Path or np.ndarray
        Path to .npy patch or 2D array.
    model_name_or_path : str
        Diffusion model checkpoint (default: 'models/unet32.pt').
    t_start : int
        Diffusion intermediate noising step (default: 600).
    ddim_steps : int
        Number of DDIM reverse steps (default: 6).
    show : bool
        Whether to call plt.show() (default: True).

    Returns
    -------
    fig : matplotlib.figure.Figure
    segments : List[dict]
    """
    # 1. Load original patch
    x0_np = load_patch(patch_or_path)

    # 2. Compute diffusion residual
    residual = compute_diffusion_residual(
        patch_or_path,
        model_name_or_path=model_name_or_path,
        t_start=t_start,
        ddim_steps=ddim_steps,
    )

    # 3. Detect segments on the residual map
    segments = detect_segments_lsd(residual)

    horizontal_segs = [s for s in segments if s["type"] == "horizontal"]
    other_segs = [s for s in segments if s["type"] == "other"]
    vertical_segs = [s for s in segments if s["type"] == "vertical"]

    # 4. Check for ground truth annotations
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

    # 5. Build 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Original
    im0 = axes[0].imshow(x0_np, origin="lower", cmap="viridis")
    axes[0].set_title("1. Original (normalisé)")
    axes[0].set_xlabel("Temps (x)")
    axes[0].set_ylabel("Fréquence (y)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel 2: Diffusion residual
    im1 = axes[1].imshow(residual, origin="lower", cmap="magma")
    axes[1].set_title("2. Résidu positif normalisé et lissé")
    axes[1].set_xlabel("Temps (x)")
    axes[1].set_ylabel("Fréquence (y)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlaid detected segments
    axes[2].imshow(residual, origin="lower", cmap="magma")
    h_count = 0
    o_count = 0

    for seg in horizontal_segs:
        (x1, y1), (x2, y2) = seg["p1"], seg["p2"]
        label = "Horizontal (cyan)" if h_count == 0 else None
        axes[2].plot([x1, x2], [y1, y2], color="cyan", linewidth=2.5, label=label)
        axes[2].scatter([x1, x2], [y1, y2], color="cyan", s=15, zorder=3)
        h_count += 1

    for seg in other_segs:
        (x1, y1), (x2, y2) = seg["p1"], seg["p2"]
        label = "Other / oblique (jaune)" if o_count == 0 else None
        axes[2].plot([x1, x2], [y1, y2], color="yellow", linewidth=2.5, label=label)
        axes[2].scatter([x1, x2], [y1, y2], color="yellow", s=15, zorder=3)
        o_count += 1

    axes[2].set_xlim(0, x0_np.shape[1] - 1)
    axes[2].set_ylim(0, x0_np.shape[0] - 1)
    axes[2].set_title(f"3. LSD: horizontal={len(horizontal_segs)}, other={len(other_segs)}")
    axes[2].set_xlabel("Temps (x)")
    axes[2].set_ylabel("Fréquence (y)")
    if h_count > 0 or o_count > 0:
        axes[2].legend(loc="upper right", fontsize=8)

    det_summary = f"Détection: horizontal={len(horizontal_segs) > 0}, other={len(other_segs) > 0}"
    fig.suptitle(f"{patch_name}{label_info}\n[{det_summary}]", fontsize=11)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, segments

# -----------------------------------------------------------------------------
# Quick demo / Main
# -----------------------------------------------------------------------------
if False:#__name__ == "__main__":
    # Test on an annotated horizontal patch
    demo_image1 = resolve("dataset/test/patch_9107_22.npy")
    # Alternate patches for testing:
    demo_image2 = resolve("dataset/test/patch_9150_13.npy")  # Other / oblique
    demo_image3 = resolve("dataset/test/patch_9099_13.npy")  # Pure noise / no line

    #print(f"[INFO] Exécution de plot_lsd_detection sur : {os.path.basename(demo_image)}")
    fig, detected_segments = plot_lsd_detection(
        patch_or_path=demo_image1,
        model_name_or_path="models/unet32.pt",
        t_start=800,
        ddim_steps=6,
        show=True,
    )
    fig, detected_segments = plot_lsd_detection(
            patch_or_path=demo_image2,
            model_name_or_path="models/unet32.pt",
            t_start=800,
            ddim_steps=4,
            show=True,
    )
    fig, detected_segments = plot_lsd_detection(
            patch_or_path=demo_image3,
            model_name_or_path="models/unet32.pt",
            t_start=800,
            ddim_steps=4,
            show=True,
    )




    print(f"[INFO] Nombre total de segments détectés : {len(detected_segments)}")
    for i, seg in enumerate(detected_segments, 1):
        (x1, y1), (x2, y2) = seg["p1"], seg["p2"]
        print(
            f"  - Segment {i} [{seg['type'].upper()}]: "
            f"({x1:.1f}, {y1:.1f}) -> ({x2:.1f}, {y2:.1f}), "
            f"angle={seg['angle']:.1f}°, longueur={seg['length']:.1f}px, "
            f"élongation={seg['elongation']:.1f}"
        )

from test_method_enhanced import compare_method_to_annotations
from test_anomaly_detection import compare_method_to_annotations as anomaly_evaluation


def method_lsd(patch_or_path: Union[str, os.PathLike, np.ndarray]):
    use_diffusion= True
    t_start= 650
    ddim_steps = 3
    model_name_or_path = "models/unet32.pt"
    min_length=11
    return predict_lsd(patch_or_path,use_diffusion, t_start,ddim_steps, model_name_or_path,min_length)

#compare_method_to_annotations(method_lsd)
anomaly_evaluation(method_lsd)