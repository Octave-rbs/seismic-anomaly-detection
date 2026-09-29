import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from linear_reg import prediction_pipeline

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"dataset")
FILE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"dataset/test")

# The four mutually exclusive classes: (horizontal, other) -> class name
COMBINATIONS = {
    (False, False): "no anomaly",
    (False, True): "other",
    (True, False): "horizontal",
    (True, True): "horizontal+other",
}
CLASSES = tuple(COMBINATIONS.values())

SHOW = False


def load_annotations(path: str) -> dict:
    """
    Load the annotations: {filename: {"horizontal": bool, "other": bool}}.
    """
    with open(path, "r") as f:
        return json.load(f)

def load_patch(path: str) -> np.ndarray:
    """
    Load a patch and standardize it (zero mean, unit variance).
    """
    patch = np.load(path).astype(np.float32).squeeze()
    return (patch - patch.mean()) / (patch.std() + 1e-8)

def to_class(horizontal: bool, other: bool) -> str:
    """Map a (horizontal, other) pair to its combination class name."""
    return COMBINATIONS[(bool(horizontal), bool(other))]

def safe_div(num: float, den: float) -> float:
    """Division returning 0.0 instead of raising when the denominator is 0."""
    return float(num) / float(den) if den else 0.0

def compute_f1(tp: int, fp: int, fn: int) -> float:
    """F1-score of a single class from its confusion matrix counts."""
    return safe_div(2 * tp, 2 * tp + fp + fn)

def compare_method_to_annotations(
    method = lambda x : {'horizontal':False,'other':False},
    show: bool = False,
) -> dict[str, dict]:
    """
    Run a method on every annotated patch and report the confusion matrix
    (TP/TN/FP/FN), the precision, the recall and the F1-score of each of the
    four classes (horizontal, other), then the macro-avg and weighted-avg F1.
    """
    ANNOTATIONS = os.path.join(DATASET_DIR, "annotations.json")
    annotations = load_annotations(ANNOTATIONS)
    n_samples = len(annotations)

    results = {
        cls: {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "FP_paths": [], "FN_paths": []}
        for cls in CLASSES
    }
    binary = {flag: 0 for flag in ("horizontal", "other")}

    for file, annot in tqdm(annotations.items(), desc="Processing annotations"):
        path = os.path.join(FILE_DIR, file)
        pred = method(path)
        pred_cls = to_class(pred["horizontal"], pred["other"])
        true_cls = to_class(annot["horizontal"], annot["other"])

        for flag in binary:
            if bool(pred[flag]) == bool(annot[flag]):
                binary[flag] += 1

        for cls in CLASSES:
            if pred_cls == cls and true_cls == cls:
                results[cls]["TP"] += 1
            elif pred_cls == cls and true_cls != cls:
                results[cls]["FP"] += 1
                results[cls]["FP_paths"].append(path)
            elif pred_cls != cls and true_cls == cls:
                results[cls]["FN"] += 1
                results[cls]["FN_paths"].append(path)
            else:
                results[cls]["TN"] += 1

    f1_scores = {}
    supports = {}

    for cls in CLASSES:
        tp, tn = results[cls]["TP"], results[cls]["TN"]
        fp, fn = results[cls]["FP"], results[cls]["FN"]
        f1_scores[cls] = compute_f1(tp, fp, fn)
        supports[cls] = tp + fn

    print("\n" + "=" * 78)
    print(f"Per-class results over {n_samples} annotated patches")
    print("=" * 78)
    print(
        f"{'class':<17}{'TP':>6}{'TN':>6}{'FP':>6}{'FN':>6}"
        f"{'support':>9}{'prec':>8}{'rec':>8}{'F1':>8}"
    )
    print("-" * 78)
    for cls in CLASSES:
        tp, tn = results[cls]["TP"], results[cls]["TN"]
        fp, fn = results[cls]["FP"], results[cls]["FN"]
        print(
            f"{cls:<17}{tp:>6}{tn:>6}{fp:>6}{fn:>6}{tp + fn:>9}"
            f"{safe_div(tp, tp + fp):>8.4f}{safe_div(tp, tp + fn):>8.4f}{f1_scores[cls]:>8.4f}"
        )

    macro_f1 = float(np.mean(list(f1_scores.values())))
    weighted_f1 = float(np.average(list(f1_scores.values()), weights=list(supports.values())))

    print("\n" + "=" * 78)
    print("F1-score")
    print("=" * 78)
    for cls in CLASSES:
        print(f"{cls:<17}F1 : {f1_scores[cls]:.4f}")
    print("-" * 78)
    print(f"{'macro-avg':<17}F1 : {macro_f1:.4f}")
    print(f"{'weighted-avg':<17}F1 : {weighted_f1:.4f}")

    print("\n" + "=" * 78)
    print("Binary breakdown (per flag, all patches)")
    print("=" * 78)
    for flag, correct in binary.items():
        print(f"{flag:<17}accuracy : {safe_div(correct, n_samples):.4f}")
    print("-" * 78)

    if show:
        for cls in CLASSES:
            for kind in ("FP", "FN"):
                paths = results[cls][f"{kind}_paths"]
                if paths:
                    path = paths[int(np.random.rand() * len(paths))]
                    plt.title(f"true={annotations[os.path.basename(path)]} pred={cls} ({kind})")
                    plt.imshow(load_patch(path), origin="lower")
                    plt.show()

    return results


if __name__ == "__main__":
    compare_method_to_annotations(method=prediction_pipeline, show=SHOW)
