import json
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# from dataset import NoisePatchDataset

from diffusion import diffuse
from inference import (
    load_diffusion_model,
    make_ddim_timesteps,
    reverse_diffusion_ddim,
)

from reconstruct_test_patch import resolve, load_patch

def get_linked_dataset(dataset_dir: str):
    """
    Loads annotations and links them with their respective image patch data.
    """
    annotations_path = os.path.join(dataset_dir, "annotations.json")
    test_dir = os.path.join(dataset_dir, "test")

    # 1. Load the annotations dictionary {filename: {"horizontal": bool, "other": bool}}
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    linked_data = []
    
    # 2. Iterate through each annotated patch file
    for filename, labels in annotations.items():
        patch_path = os.path.join(test_dir, filename)
        
        if os.path.exists(patch_path):
            # 3. Load the patch data from the .npy file
            patch = np.load(patch_path)
            
            if patch.ndim == 2:
                patch = patch[None, :, :]
            elif patch.ndim != 3 or patch.shape[0] != 1:
                raise ValueError(
                    f"Invalid patch shape {patch.shape} in file {patch_path}"
                )
    
            patch = patch.astype(np.float32)
            patch -= patch.mean()
            patch /= (patch.std() + 1e-8)
    
            patch =  torch.from_numpy(patch)
            
            # Combine image data and annotations together
            linked_data.append({
                "filename": filename,
                "image": patch,
                "horizontal": labels["horizontal"],
                "other": labels["other"]
            })
            
    return linked_data

def detect_horizontal_lines(image, factor=2,  line_completion_threshold=0.8, use_kernel=True):
    image = image.unsqueeze(0)

    kernel = torch.tensor([
        [-2, -2, -2],
        [ 2,  2,  2],
        [-2, -2, -2]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    kernel = kernel.to(device=image.device, dtype=image.dtype)
    if use_kernel:
        response = F.conv2d(image, kernel, padding=1)
    else:
        response = image

    sum_by_lines = response.sum(dim=3, keepdim=True)

    mean, std = sum_by_lines.mean(), sum_by_lines.std()
    lines_bool = torch.where(line_completion_threshold * sum_by_lines > mean + factor*std, sum_by_lines, 0)

    
    # print(lines_bool)
    number = (lines_bool > 0).sum().item()

    return response, sum_by_lines, number

@torch.no_grad
def run_model_inference(model_path: str,
    image_path: str,
    t_start: int,
    ddim_steps: int,
    device: str,
    seed: int,
    visualize: bool = True,):

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

    return x0_hat_np, error

def test_rotated_image(image, num, factor=2, use_kernel=True):

    angles = np.linspace(0,-90, num)
    results = []
    for angle in angles:
        rotated_image = TF.rotate(image, angle)
        _,_,num = detect_horizontal_lines(rotated_image, factor=factor, use_kernel=use_kernel)
        results.append(num)

    

if __name__ == "__main__":

    dataset = get_linked_dataset("./dataset/")
    MODEL = "models/unet64.pt"
    TEST_DIR = "dataset/test"
    # IMAGE = "patch_9107_22.npy"  # Horizontal
    # IMAGE = "patch_9150_13.npy"  # Other
    # IMAGE = "patch_9998_20.npy"  # Nothing
    # IMAGE = "patch_9998_11.npy"
    T_START = 600
    DDIM_STEPS = 6
    SEED = 0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    test_dir = resolve(TEST_DIR)

    N = 50
    indexes = np.random.randint(len(dataset), size=N)
    hlines_found_expected_diff = []
    hlines_found_unexpected_diff = []
    hlines_found_expected_nodiff = []
    hlines_found_unexpected_nodiff = []
    
    for j in range(len(indexes)):
        i = indexes[j]
        if ((100*j)//N % 10) == 0:
            print(f"step {j} out of {N}")

        path_to_image = os.path.join(TEST_DIR, dataset[i]["filename"])

        _, error = run_model_inference(MODEL, path_to_image, T_START, DDIM_STEPS, DEVICE, SEED)
        error = torch.from_numpy(error)[None]

        _, _, number_diff = detect_horizontal_lines(error)
        _, _, number_nodiff = detect_horizontal_lines(dataset[i]["image"])

        if dataset[i]["horizontal"]:
            hlines_found_expected_diff.append(number_diff)
            hlines_found_expected_nodiff.append(number_nodiff)
        else:
            hlines_found_unexpected_diff.append(number_diff)
            hlines_found_unexpected_nodiff.append(number_nodiff)


        



    fig, ax = plt.subplots(2,2)
    ax[0, 0].bar(x=np.arange(max(hlines_found_expected_diff)+1), height=np.bincount(hlines_found_expected_diff), label="")
    ax[0, 0].set_title("hlines found in error after diffusion, (at least one expected)")
    ax[0, 1].bar(x=np.arange(max(hlines_found_unexpected_diff)+1), height=np.bincount(hlines_found_unexpected_diff), label="", alpha = 0.5)
    ax[0, 1].set_title("hlines found in error after diffusion, (none expected)")

    ax[1, 0].bar(x=np.arange(max(hlines_found_expected_nodiff)+1), height=np.bincount(hlines_found_expected_nodiff), label="", alpha = 0.5)
    ax[1, 0].set_title("hlines found in error before diffusion, (at least one expected)")
    ax[1, 1].bar(x=np.arange(max(hlines_found_unexpected_nodiff)+1), height=np.bincount(hlines_found_unexpected_nodiff), label="", alpha = 0.5)
    ax[1, 1].set_title("hlines found in error before diffusion, (none expected)")

    plt.tight_layout()
    fig.show()


