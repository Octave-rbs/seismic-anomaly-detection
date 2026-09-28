"""
Display the annotated test patches together with their annotations.

Steps:
1. Load the annotations (dataset/annotations.json)
2. Print the number of annotated patches for each combination of labels
3. Select the patches to display (all, or those with given labels)
4. Display them page by page, each patch titled with its labels

Usage: set the parameters in the "Parameters" section below, then run
    python dataset/view_annotations.py
Close a figure window to display the next page.
"""
import os
import json

import numpy as np
import matplotlib.pyplot as plt

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def load_annotations(path: str) -> dict:
    """
    Load the annotations: {filename: {"horizontal": bool, "other": bool}}.
    """
    with open(path, "r") as f:
        return json.load(f)


def print_statistics(annotations: dict, n_test: int) -> None:
    """
    Print the number of annotated patches for each combination of labels.
    """
    h = np.array([a["horizontal"] for a in annotations.values()], dtype=bool)
    o = np.array([a["other"] for a in annotations.values()], dtype=bool)

    print(f"Annotated patches      : {len(annotations)} / {n_test}")
    print(f"  No line              : {np.sum(~h & ~o)}")
    print(f"  Horizontal only      : {np.sum(h & ~o)}")
    print(f"  Other only           : {np.sum(~h & o)}")
    print(f"  Horizontal and other : {np.sum(h & o)}")
    print(f"  Total horizontal     : {np.sum(h)}")
    print(f"  Total other          : {np.sum(o)}")


def select(annotations: dict, horizontal: bool | None,
           other: bool | None) -> list:
    """
    Filenames whose labels match the filter (None means "any value").
    """
    return sorted(
        name for name, a in annotations.items()
        if (horizontal is None or a["horizontal"] == horizontal)
        and (other is None or a["other"] == other)
    )


def load_patch(path: str) -> np.ndarray:
    """
    Load a patch and standardize it (zero mean, unit variance).
    """
    patch = np.load(path).astype(np.float32).squeeze()
    return (patch - patch.mean()) / (patch.std() + 1e-8)


def title(name: str, a: dict) -> str:
    """
    Figure title: patch name and its labels.
    """
    labels = [k for k in ("horizontal", "other") if a[k]] or ["no line"]
    return f"{name.removesuffix('.npy')}\n{' + '.join(labels)}"


def show_page(names: list, annotations: dict, test_dir: str,
              rows: int, cols: int, page: int, n_pages: int) -> None:
    """
    Display one page of patches on a rows x cols grid.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.1 * rows),
                             squeeze=False)
    colors = {"no line": "black", "horizontal": "tab:blue",
              "other": "tab:red", "horizontal + other": "tab:purple"}

    for ax in axes.flat:
        ax.axis("off")

    for ax, name in zip(axes.flat, names):
        a = annotations[name]
        ax.imshow(load_patch(os.path.join(test_dir, name)), origin="lower")
        t = title(name, a)
        ax.set_title(t, fontsize=9, color=colors[t.split("\n")[1]])

    fig.suptitle(f"Annotated patches - page {page + 1}/{n_pages} "
                 f"(y: frequency, x: time)")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # ---------------------------- Parameters ---------------------------------
    ANNOTATIONS = os.path.join(DATASET_DIR, "annotations.json")
    TEST_DIR = os.path.join(DATASET_DIR, "test")

    # Label filter: True, False, or None (any value). For instance,
    # HORIZONTAL = None, OTHER = True displays all patches with another line
    HORIZONTAL = None
    OTHER = None

    # Grid of patches displayed on each page
    ROWS, COLS = 4, 6

    # Shuffle the patches (with this seed) or None to sort them by name
    SEED = None
    # -------------------------------------------------------------------------

    annotations = load_annotations(ANNOTATIONS)
    n_test = len([f for f in os.listdir(TEST_DIR) if f.endswith(".npy")])
    print_statistics(annotations, n_test)

    names = select(annotations, HORIZONTAL, OTHER)
    if SEED is not None:
        names = list(np.random.default_rng(SEED).permutation(names))
    print(f"\nSelected patches       : {len(names)}")

    per_page = ROWS * COLS
    n_pages = (len(names) + per_page - 1) // per_page
    for page in range(n_pages):
        show_page(names[page * per_page:(page + 1) * per_page], annotations,
                  TEST_DIR, ROWS, COLS, page, n_pages)
