"""Take in normalized pixel images and tries to
increase the intensity of high intensity pixel clusters and
reduce the intensity of background noise."""

import torch
import torch.nn.functional as F
import os
import json
import numpy as np
import matplotlib.pyplot as plt


DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"
)
FILE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset/test"
)


def load_patch(path: str) -> np.ndarray:
    """
    Load a patch and standardize it (zero mean, unit variance).
    """
    patch = np.load(path).astype(np.float32).squeeze()
    return (patch - patch.mean()) / (patch.std() + 1e-8)


def box_convolution(image, kernel_size=10, theta=0.0, sigma=100):
    """
    Apply a 2D box convolution.
    This convolution will highlight high intensity pixels aligned with (x=cos(theta),y=sin(theta)).
    Sigma is a strength parameter applied to the scalar product.

    Args:
        image: 2D torch tensor or numpy array (H, W)
        kernel_size: size of the square kernel
        theta: angle researched
        sigma: strength parameter

    Returns:
        Filtered image (H, W)
    """
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image)
    kernel = [
        [
            0.0
            if ((i == 0) and (j == 0))
            else (i * np.cos(theta) - j * np.sin(theta)) ** sigma
            / np.sqrt(i**2 + j**2) ** sigma
            for i in range(-kernel_size // 2, kernel_size // 2 + 1)
        ]
        for j in range(-kernel_size // 2, kernel_size // 2 + 1)
    ]
    kernel = torch.tensor(kernel, dtype=image.dtype) / (np.sum(kernel))
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    image = image.unsqueeze(0).unsqueeze(0)
    padding = kernel_size // 3
    output = F.conv2d(image, kernel, padding=padding)
    return output.squeeze(0).squeeze(0)


def load_dataset():
    path = os.path.join(DATASET_DIR, "annotations.json")
    with open(path, "r") as f:
        annotations = json.load(f)
        return [
            load_patch(os.path.join(FILE_DIR, filename))
            for filename in annotations.keys()
        ], annotations.values()


def filter_dataset(dataset, kernel_size, theta):
    return [box_convolution(image, kernel_size, theta) for image in dataset]


def view_on_dataset(kernel_size=10, theta=0.0):
    dataset, annotations = load_dataset()
    for image, annot in zip(dataset, annotations):
        filtered_image = box_convolution(image)
        filtered_twice = box_convolution(filtered_image)
        fig, ax = plt.subplots(1, 3)
        fig.suptitle(str(annot))
        ax[0].imshow(image)
        ax[0].set_title("base image")
        ax[1].imshow(filtered_image)
        ax[1].set_title("filtered image")
        ax[2].imshow(filtered_twice)
        ax[2].set_title("filtered twice")
        plt.show()


if __name__ == "__main__":
    view_on_dataset()
