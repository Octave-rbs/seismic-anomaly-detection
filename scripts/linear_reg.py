import os

import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from diffusion import diffuse
from inference import make_ddim_timesteps, reverse_diffusion_ddim
from reconstruct_test_patch import load_diffusion_model, resolve, load_patch

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
model_path = resolve("models/unet64.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 0
test_dir = "dataset/test"
T_start = 600
ddim_steps = 6

VISUALIZE = False   # output plot

# dev -----
files = sorted(f for f in os.listdir(test_dir) if f.endswith(".npy"))
image_path = os.path.join(
    test_dir,
    # "patch_91_20.npy"   # Not anomaly
    # "patch_9150_13.npy"   # Other
    # "patch_9107_22.npy" # Horizontal line
    # "patch_9100_23.npy" # Other très complexe
    "patch_9330_12.npy" # Other + horizontal
    # np.random.default_rng(seed).choice(files)
)
# ---------


# -----------------------------------------------------------------------------
# Diffusion helpers
# -----------------------------------------------------------------------------
def get_original_image_and_reconstructed(
    model_path: str,
    image_path: str,
    t_start: int = T_start,
    ddim_steps: int = ddim_steps,
    device: str = device,
    seed: int = seed,
):
    """Diffuse x0 to t_start then reconstruct with DDIM."""
    model, schedule = load_diffusion_model(model_path, device)
    hparams = torch.load(model_path, map_location="cpu")["hparams"]
    T = schedule.timesteps

    print(f"[INFO] Loaded {model_path} (T={T}), "
          f"base_channels={hparams['base_channels']}")

    image = load_patch(image_path, hparams["data_mean"], hparams["data_std"])
    x0 = torch.from_numpy(image)[None, None].to(device)  # (1,1,H,W)
    print(f"[INFO] Image: {image_path} {tuple(image.shape)}")

    if t_start is None:
        t_start = T // 4
    ddim_steps = min(ddim_steps, t_start + 1)

    torch.manual_seed(seed)
    t = torch.tensor([t_start], device=device)
    xt, _ = diffuse(x0, t, schedule)

    timesteps = make_ddim_timesteps(ddim_steps, T, t_start)
    x0_hat, _ = reverse_diffusion_ddim(xt, timesteps, model, schedule)

    x0_hat = x0_hat.squeeze().cpu().numpy()
    return x0, x0_hat


# -----------------------------------------------------------------------------
# Tabular conversion
# -----------------------------------------------------------------------------
def get_tabular_data_from_img(image: torch.Tensor) -> pl.DataFrame:
    """
    Convert a (1,1,H,W) or (H,W) image tensor to a long DataFrame with
    columns [y, x, intensity, weights], keeping only pixels with intensity > 0.
    """
    image = image.reshape(48, 48).numpy() if isinstance(image, torch.Tensor) else image.reshape(48, 48)

    schema = [f"x={i}" for i in range(image.shape[1])]

    return (
        pl.DataFrame(data=image, schema=schema)
        .with_row_index(name="y")
        .unpivot(index="y", variable_name="x", value_name="intensity")
        .with_columns(pl.col("x").str.split(by="=").list[1].alias("x"))
        .with_columns(
            (pl.max_horizontal(pl.col("intensity"), 0) ** 2).alias("weights")
        )
        .filter(pl.col("weights") > 0) # TODO: voir une manière encore plus stricte de filtrer.
    )

# -----------------------------------------------------------------------------
# Visualisation
# -----------------------------------------------------------------------------
def visualize(
    x0_np: np.ndarray,
    x0_hat_np: np.ndarray,
    Xy_origin: pl.DataFrame,
    reg_origin: LinearRegression,
    Xy_reco: pl.DataFrame,
    image_path: str,
) -> None:
    """
    4-panel figure:
      1. Original image
      2. Reconstructed image
      3. Filtered points (intensity > 0) overlaid on a blank canvas
      4. Scatter of filtered points + linear regression lines for both images
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    vmin, vmax = x0_np.min(), x0_np.max()

    # --- Panel 1: original ---
    im = axes[0].imshow(x0_np, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Original (normalized)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # --- Panel 2: reconstruction ---
    im = axes[1].imshow(x0_hat_np, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Reconstruction")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    # --- Panel 3: filtered points ---
    xs_orig = Xy_origin["x"].cast(pl.UInt32).to_numpy()
    ys_orig = Xy_origin["y"].to_numpy()
    intensities = Xy_origin["intensity"].to_numpy()

    filtered_grid = np.zeros_like(x0_np)
    filtered_grid[ys_orig, xs_orig] = intensities

    im = axes[2].imshow(filtered_grid, origin="lower", vmin=vmin, vmax=vmax)
    axes[2].set_title(
        f"Filtered points (intensity > 0)\n{len(xs_orig)}/{x0_np.size} pixels"
    )
    fig.colorbar(im, ax=axes[2], fraction=0.046)

    # --- Panel 4: scatter + regression lines ---
    ax = axes[3]

    xs_reco = Xy_reco["x"].cast(pl.UInt32).to_numpy()
    ys_reco = Xy_reco["y"].to_numpy()

    ax.scatter(xs_orig, ys_orig, s=2, alpha=0.4, color="steelblue",
               label="Original (filtered)")
    ax.scatter(xs_reco, ys_reco, s=2, alpha=0.4, color="coral",
               label="Reconstructed (filtered)")

    x_range = np.linspace(0, x0_np.shape[1] - 1, 200).reshape(-1, 1)
    ax.plot(x_range, reg_origin.predict(x_range), color="steelblue", linewidth=2,
            label=f"Reg. original (slope={reg_origin.coef_[0]:.3f})")

    ax.set_xlim(0, x0_np.shape[1] - 1)
    ax.set_ylim(0, x0_np.shape[0] - 1)
    ax.set_xlabel("x (column)")
    ax.set_ylabel("y (row)")
    ax.set_title("Linear regression: x → y")
    ax.legend(fontsize=7, markerscale=3)

    fig.suptitle(os.path.basename(image_path))
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    plt.show()


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------
def predict_sample(original_image, reconstructed_image):
    Xy_origin = get_tabular_data_from_img(original_image)
    Xy_reco = get_tabular_data_from_img(torch.tensor(reconstructed_image))

    X_origin = Xy_origin["x"].cast(pl.Float32).to_numpy().reshape(-1, 1)
    y_origin = Xy_origin["y"].cast(pl.Float32).to_numpy()
    w_origin = Xy_origin["weights"].to_numpy()

    X_reco = Xy_reco["x"].cast(pl.Float32).to_numpy().reshape(-1, 1)
    y_reco = Xy_reco["y"].cast(pl.Float32).to_numpy()

    reg_origin = LinearRegression()
    reg_origin.fit(X=X_origin, y=y_origin, sample_weight=w_origin)
    original_error = root_mean_squared_error(
        y_origin, reg_origin.predict(X_origin), sample_weight=w_origin
    )
    
    # ici réfléchir au fait qu'on weight l'origin mais pas la reco
    reconstructed_error = root_mean_squared_error(y_reco, reg_origin.predict(X_reco))
    
    diff = abs(round(reconstructed_error - original_error, 3))
    slope = round(reg_origin.coef_[0], 3)

    print(f"Difference between both errors: {diff}")
    print(f"Original   — intercept: {reg_origin.intercept_:.2f}, "
          f"slope: {slope}, RMSE: {original_error:.4f}")

    print(image_path.split('\\')[-1])
    print("\n")

    if (diff > 0.1) & (slope > 0.1):
        h_pred = False
        o_pred = False
    
    else:
        h_pred = False
        o_pred = False

    prediction = dict(
        {
            f"{image_path.split("\\")[-1]}":{
                "horizontal":h_pred,
                "other": o_pred
            }
        }
    )

    return Xy_origin, reg_origin, Xy_reco, prediction


def prediction_pipeline(
    image_path: str,
    show: bool = False,
) -> dict[str, dict[str, bool]]:
    x0, x0_hat = get_original_image_and_reconstructed(
        model_path=model_path,
        image_path=image_path,
    )

    Xy_origin, reg_origin, Xy_reco, prediction = predict_sample(
        original_image=x0,
        reconstructed_image=x0_hat,
    )

    if show:
        x0_np = x0.squeeze().cpu().numpy()
        visualize(
            x0_np=x0_np,
            x0_hat_np=x0_hat,
            Xy_origin=Xy_origin,
            reg_origin=reg_origin,
            Xy_reco=Xy_reco,
            image_path=image_path,
        )

    return prediction


if __name__ == "__main__":
    VISUALIZE = True
    prediction_pipeline(
        image_path=image_path, 
        show=VISUALIZE, 
        )