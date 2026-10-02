import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line

# 1. Create the binary image
array = np.zeros((10, 10))
array[5, 2:8] = 1
array[2, 3] = 1

# 2. Compute the Hough Transform
hspace, angles, dists = hough_line(array)

# 3. Create the figure
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# --- Left: Original image ---
ax[0].imshow(
    array, cmap="gray_r", origin="upper", interpolation="nearest", vmin=0, vmax=1
)
ax[0].set_xticks(range(10))
ax[0].set_yticks(range(10))
ax[0].set_xlabel("x (column)")
ax[0].set_ylabel("y (row)")
ax[0].set_title("Input image")
ax[0].grid(True, linewidth=0.5)
ax[0].set_xlim(-0.5, 9.5)
ax[0].set_ylim(9.5, -0.5)

# --- Right: Hough space ---
theta_curve = np.linspace(-np.pi / 2, np.pi / 2, 500)

# Get coordinates of all active pixels
rows, cols = np.nonzero(array)

# Plot one curve per pixel
for x, y in zip(cols, rows):
    rho_curve = x * np.cos(theta_curve) + y * np.sin(theta_curve)
    ax[1].plot(np.degrees(theta_curve), rho_curve, linewidth=1.5)

# Display the Hough accumulator as a faint background
ax[1].imshow(
    hspace,
    cmap="gray_r",
    aspect="auto",
    extent=[np.degrees(angles[0]), np.degrees(angles[-1]), dists[-1], dists[0]],
    alpha=0.2,
    interpolation="nearest",
)

# Highlight the intersection of the 6 horizontal pixels
ax[1].scatter([-90], [-5], marker="*", s=90)
ax[1].annotate(
    "Shared peak from the 6 aligned pixels",
    xy=(-89.5, -5),
    xytext=(-55, -8),
    arrowprops={"arrowstyle": "->"},
)

ax[1].set_xlim(-90, 90)
ax[1].set_ylim(-10, 10)
ax[1].set_xlabel(r"$\theta$ (degrees)")
ax[1].set_ylabel(r"$\rho$")
ax[1].set_title("Hough parameter space")
ax[1].grid(True, linewidth=0.4)

fig.suptitle("Hough Transform: pixels become curves in parameter space")
fig.tight_layout()
plt.savefig("hough_space_visualization.png", dpi=200, bbox_inches="tight")
plt.show()
