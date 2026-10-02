"""
Anomaly Detection Evaluation Module for Spectrogram Patches.

Evaluates whether a model/pipeline can distinguish anomalous patches
(containing horizontal and/or other line structures) from normal noise patches.

Ground truth definition:
- Anomaly (Positive): annot["horizontal"] OR annot["other"] (138 patches)
- Normal / Noise (Negative): NOT annot["horizontal"] AND NOT annot["other"] (364 patches)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Paths configuration
# -----------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
FILE_DIR = os.path.join(DATASET_DIR, "test")
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.json")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)


# -----------------------------------------------------------------------------
# Data Loaders & Helpers
# -----------------------------------------------------------------------------
def load_annotations(path: str = ANNOTATIONS_FILE) -> Dict[str, Dict[str, bool]]:
    """
    Load dataset annotations: {filename: {"horizontal": bool, "other": bool}}.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Annotations file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_patch(path: str) -> np.ndarray:
    """
    Load a spectrogram patch from a .npy file and standardize it (zero mean, unit variance).
    """
    patch = np.load(path).astype(np.float32).squeeze()
    return (patch - patch.mean()) / (patch.std() + 1e-8)


def parse_prediction(
    pred: Any,
    filename: str,
    threshold: float = 0.5,
) -> bool:
    """
    Normalize any method's prediction into a boolean anomaly decision (True = Anomaly, False = Normal).

    Supported return formats from method:
    1. bool: True (anomaly) or False (normal)
    2. dict with 'anomaly' or 'is_anomaly': {"anomaly": True}
    3. dict with 'horizontal' and 'other': {"horizontal": bool, "other": bool} -> horizontal or other
    4. dict keyed by filename/basename: {"patch_xxx.npy": {"horizontal": ..., "other": ...}}
    5. numeric float/int: compared against `threshold` (score >= threshold -> True)
    6. tuple/list of bools: any element is True -> True
    """
    if isinstance(pred, bool):
        return pred

    if isinstance(pred, (int, float)):
        return float(pred) >= threshold

    if isinstance(pred, (tuple, list)):
        return any(bool(item) for item in pred)

    if isinstance(pred, dict):
        base = os.path.basename(filename)
        if base in pred:
            return parse_prediction(pred[base], filename, threshold)
        if filename in pred:
            return parse_prediction(pred[filename], filename, threshold)

        if "anomaly" in pred:
            val = pred["anomaly"]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val) >= threshold
            return bool(val)

        if "is_anomaly" in pred:
            val = pred["is_anomaly"]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val) >= threshold
            return bool(val)

        if "score" in pred:
            return float(pred["score"]) >= threshold

        if "horizontal" in pred or "other" in pred:
            return bool(pred.get("horizontal", False)) or bool(pred.get("other", False))

    return bool(pred)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division avoiding ZeroDivisionError."""
    return float(numerator) / float(denominator) if denominator != 0 else default


# -----------------------------------------------------------------------------
# Core Evaluation Function
# -----------------------------------------------------------------------------
def evaluate_anomaly_detection(
    method: Callable[[str], Any] = lambda x: {"horizontal": False, "other": False},
    annotations_path: str = ANNOTATIONS_FILE,
    patches_dir: str = FILE_DIR,
    threshold: float = 0.5,
    max_samples: Optional[int] = None,
    show: bool = False,
    num_show_examples: int = 4,
    save_plot_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate if a method can identify anomalies (horizontal and/or other classes) in spectrogram patches.

    Parameters
    ----------
    method : Callable[[str], Any]
        Function taking a patch file path and returning a prediction.
    annotations_path : str
        Path to annotations.json.
    patches_dir : str
        Path to directory containing test .npy patches.
    threshold : float
        Decision threshold if method outputs continuous anomaly scores.
    max_samples : int, optional
        Limit evaluation to first N samples (useful for quick benchmarking).
    show : bool
        If True, displays matplotlib visual inspections of false alarms (FP) and missed anomalies (FN).
    num_show_examples : int
        Number of FP and FN examples to visualize if show=True.
    save_plot_path : str, optional
        If provided, saves a summary evaluation figure (confusion matrix & metrics) to disk.
    verbose : bool
        If True, prints formatted evaluation tables to stdout.

    Returns
    -------
    dict
        Detailed performance report containing confusion matrix, metrics, subgroup recall,
        and lists of false positive and false negative file paths.
    """
    annotations = load_annotations(annotations_path)
    items = list(annotations.items())
    if max_samples is not None and max_samples > 0:
        items = items[:max_samples]

    n_samples = len(items)

    # Confusion matrix counters
    tp = 0  # Anomaly correctly predicted
    tn = 0  # Normal correctly predicted
    fp = 0  # Normal wrongly flagged as anomaly (False alarm)
    fn = 0  # Anomaly missed

    fp_paths: List[Dict[str, Any]] = []
    fn_paths: List[Dict[str, Any]] = []

    # Subgroup breakdown
    # Types: "horizontal_only", "other_only", "horizontal_and_other", "normal"
    subgroups = {
        "horizontal_only": {"total": 0, "detected": 0},
        "other_only": {"total": 0, "detected": 0},
        "horizontal_and_other": {"total": 0, "detected": 0},
        "normal": {"total": 0, "correct_rejected": 0},
    }

    iterator = tqdm(items, desc="Evaluating anomaly detection") if verbose else items

    for file, annot in iterator:
        patch_path = os.path.join(patches_dir, file)
        is_h = bool(annot.get("horizontal", False))
        is_o = bool(annot.get("other", False))
        true_anomaly = is_h or is_o

        if is_h and not is_o:
            subgroup_key = "horizontal_only"
        elif not is_h and is_o:
            subgroup_key = "other_only"
        elif is_h and is_o:
            subgroup_key = "horizontal_and_other"
        else:
            subgroup_key = "normal"

        subgroups[subgroup_key]["total"] += 1

        raw_pred = method(patch_path)
        pred_anomaly = parse_prediction(raw_pred, file, threshold=threshold)

        if pred_anomaly:
            if subgroup_key != "normal":
                subgroups[subgroup_key]["detected"] += 1
        else:
            if subgroup_key == "normal":
                subgroups[subgroup_key]["correct_rejected"] += 1

        if true_anomaly and pred_anomaly:
            tp += 1
        elif not true_anomaly and not pred_anomaly:
            tn += 1
        elif not true_anomaly and pred_anomaly:
            fp += 1
            fp_paths.append({
                "path": patch_path,
                "file": file,
                "annot": annot,
                "pred": raw_pred,
            })
        else:  # true_anomaly and not pred_anomaly
            fn += 1
            fn_paths.append({
                "path": patch_path,
                "file": file,
                "annot": annot,
                "subgroup": subgroup_key,
                "pred": raw_pred,
            })

    # Metric calculations
    total_anomalies = tp + fn
    total_normals = tn + fp

    accuracy = safe_div(tp + tn, n_samples)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)  # Sensitivity / Hit rate
    specificity = safe_div(tn, tn + fp)  # True Negative Rate
    f1 = safe_div(2 * precision * recall, precision + recall)
    balanced_acc = (recall + specificity) / 2.0
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)

    # Matthews Correlation Coefficient
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div(mcc_num, mcc_den)

    # Subgroup recall rates
    subgroup_results = {}
    for key, data in subgroups.items():
        tot = data["total"]
        if key == "normal":
            corr = data["correct_rejected"]
            rate = safe_div(corr, tot)
            subgroup_results[key] = {
                "total": tot,
                "correct": corr,
                "specificity": rate,
            }
        else:
            det = data["detected"]
            rate = safe_div(det, tot)
            subgroup_results[key] = {
                "total": tot,
                "detected": det,
                "recall": rate,
            }

    results = {
        "n_samples": n_samples,
        "total_anomalies": total_anomalies,
        "total_normals": total_normals,
        "confusion_matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        },
        "metrics": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "fpr": fpr,
            "fnr": fnr,
            "mcc": mcc,
        },
        "subgroups": subgroup_results,
        "false_positives": fp_paths,
        "false_negatives": fn_paths,
    }

    if verbose:
        print_evaluation_report(results)

    if save_plot_path or show:
        plot_evaluation_summary(
            results=results,
            patches_dir=patches_dir,
            show=show,
            save_path=save_plot_path,
            num_examples=num_show_examples,
        )

    return results


# -----------------------------------------------------------------------------
# Reporting & Pretty Printing
# -----------------------------------------------------------------------------
def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Print a clean, structured evaluation report to console."""
    cm = results["confusion_matrix"]
    m = results["metrics"]
    sg = results["subgroups"]
    n = results["n_samples"]
    n_ano = results["total_anomalies"]
    n_norm = results["total_normals"]

    print("\n" + "=" * 80)
    print("           ANOMALY DETECTION EVALUATION (Horizontal + Other vs Normal)")
    print("=" * 80)
    print(f"Total evaluated patches : {n}")
    print(f"  - Normal / Noise only : {n_norm} ({safe_div(n_norm, n)*100:.1f}%)")
    print(f"  - Anomaly patches     : {n_ano} ({safe_div(n_ano, n)*100:.1f}%)")
    print("-" * 80)

    print("CONFUSION MATRIX:")
    print(f"  True Positives  (TP) : {cm['TP']:>4}  (Correctly detected anomalies)")
    print(f"  True Negatives  (TN) : {cm['TN']:>4}  (Correctly identified normal noise)")
    print(f"  False Positives (FP) : {cm['FP']:>4}  (False alarms: normal flagged as anomaly)")
    print(f"  False Negatives (FN) : {cm['FN']:>4}  (Missed anomalies)")
    print("-" * 80)

    print("PERFORMANCE METRICS:")
    print(f"{'Metric':<26}{'Value':>10}    {'Description / Formula':<35}")
    print("-" * 80)
    print(f"{'Accuracy':<26}{m['accuracy']*100:>9.2f}%    (TP + TN) / Total")
    print(f"{'Balanced Accuracy':<26}{m['balanced_accuracy']*100:>9.2f}%    (Sensitivity + Specificity) / 2")
    print(f"{'Precision (PPV)':<26}{m['precision']*100:>9.2f}%    TP / (TP + FP)")
    print(f"{'Recall (Sensitivity)':<26}{m['recall']*100:>9.2f}%    TP / (TP + FN)  [Detection Rate]")
    print(f"{'Specificity (TNR)':<26}{m['specificity']*100:>9.2f}%    TN / (TN + FP)  [Noise Rejection]")
    print(f"{'F1-Score':<26}{m['f1_score']*100:>9.2f}%    Harmonic mean of Prec & Rec")
    print(f"{'False Positive Rate (FPR)':<26}{m['fpr']*100:>9.2f}%    FP / (FP + TN)  [False Alarm Rate]")
    print(f"{'False Negative Rate (FNR)':<26}{m['fnr']*100:>9.2f}%    FN / (FN + TP)  [Miss Rate]")
    print(f"{'Matthews Corr (MCC)':<26}{m['mcc']:>10.4f}    [-1.0 (worst) to +1.0 (perfect)]")
    print("-" * 80)

    print("SUBGROUP BREAKDOWN (Detection ability per anomaly class):")
    print(f"{'Subgroup':<26}{'Detected / Total':>18}    {'Rate':>10}")
    print("-" * 80)
    h_tot = sg["horizontal_only"]["total"]
    h_det = sg["horizontal_only"]["detected"]
    h_rate = sg["horizontal_only"]["recall"] * 100
    print(f"{'Horizontal only':<26}{f'{h_det} / {h_tot}':>18}    {h_rate:>9.2f}% Recall")

    o_tot = sg["other_only"]["total"]
    o_det = sg["other_only"]["detected"]
    o_rate = sg["other_only"]["recall"] * 100
    print(f"{'Other only':<26}{f'{o_det} / {o_tot}':>18}    {o_rate:>9.2f}% Recall")

    ho_tot = sg["horizontal_and_other"]["total"]
    ho_det = sg["horizontal_and_other"]["detected"]
    ho_rate = sg["horizontal_and_other"]["recall"] * 100
    print(f"{'Horizontal + Other':<26}{f'{ho_det} / {ho_tot}':>18}    {ho_rate:>9.2f}% Recall")
    print("-" * 80)
    tot_det = h_det + o_det + ho_det
    print(f"{'All Anomalies Combined':<26}{f'{tot_det} / {n_ano}':>18}    {m['recall']*100:>9.2f}% Overall Recall")

    n_tot = sg["normal"]["total"]
    n_corr = sg["normal"]["correct"]
    n_rate = sg["normal"]["specificity"] * 100
    print(f"{'Normal (Noise Background)':<26}{f'{n_corr} / {n_tot}':>18}    {n_rate:>9.2f}% Specificity")
    print("=" * 80 + "\n")


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def plot_evaluation_summary(
    results: Dict[str, Any],
    patches_dir: str,
    show: bool = False,
    save_path: Optional[str] = None,
    num_examples: int = 4,
) -> None:
    """
    Plot a visual diagnostic figure:
    1. Confusion Matrix bar / heatmap
    2. Subgroup Detection Rates bar chart
    3. Representative False Alarms (FP) and Missed Anomalies (FN)
    """
    cm = results["confusion_matrix"]
    m = results["metrics"]
    sg = results["subgroups"]
    fps = results["false_positives"]
    fns = results["false_negatives"]

    # Determine layout
    n_fp_to_show = min(num_examples // 2, len(fps))
    n_fn_to_show = min(num_examples // 2, len(fns))
    n_examples = n_fp_to_show + n_fn_to_show

    fig = plt.figure(figsize=(14, 8 if n_examples > 0 else 5))
    gs = fig.add_gridspec(2 if n_examples > 0 else 1, 3, height_ratios=[1, 1.2] if n_examples > 0 else [1])

    # 1. Confusion Matrix visual
    ax_cm = fig.add_subplot(gs[0, 0])
    cm_matrix = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    im = ax_cm.imshow(cm_matrix, cmap="Blues", interpolation="nearest")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Pred: Normal", "Pred: Anomaly"])
    ax_cm.set_yticklabels(["True: Normal", "True: Anomaly"])
    for i in range(2):
        for j in range(2):
            val = cm_matrix[i, j]
            color = "white" if val > cm_matrix.max() / 2 else "black"
            ax_cm.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14, weight="bold")
    ax_cm.set_title("Confusion Matrix")

    # 2. Key Metrics Bar Chart
    ax_met = fig.add_subplot(gs[0, 1])
    metric_labels = ["Accuracy", "Bal. Acc", "Precision", "Recall", "Specificity", "F1"]
    metric_values = [
        m["accuracy"] * 100,
        m["balanced_accuracy"] * 100,
        m["precision"] * 100,
        m["recall"] * 100,
        m["specificity"] * 100,
        m["f1_score"] * 100,
    ]
    bars = ax_met.bar(metric_labels, metric_values, color="#3470a3")
    ax_met.set_ylim(0, 105)
    ax_met.set_ylabel("Score (%)")
    ax_met.set_title("Key Performance Metrics")
    for bar in bars:
        h = bar.get_height()
        ax_met.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    ax_met.set_xticks(range(len(metric_labels)))
    ax_met.set_xticklabels(metric_labels, rotation=30, ha="right", fontsize=9)

    # 3. Subgroup Detection Rates Bar Chart
    ax_sg = fig.add_subplot(gs[0, 2])
    sg_labels = ["Horizontal", "Other", "Both", "Normal (spec)"]
    sg_rates = [
        sg["horizontal_only"]["recall"] * 100,
        sg["other_only"]["recall"] * 100,
        sg["horizontal_and_other"]["recall"] * 100,
        sg["normal"]["specificity"] * 100,
    ]
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#17becf"]
    bars_sg = ax_sg.bar(sg_labels, sg_rates, color=colors)
    ax_sg.set_ylim(0, 105)
    ax_sg.set_ylabel("Detection / Rejection (%)")
    ax_sg.set_title("Subgroup Breakdown")
    for bar in bars_sg:
        h = bar.get_height()
        ax_sg.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    ax_sg.set_xticks(range(len(sg_labels)))
    ax_sg.set_xticklabels(sg_labels, rotation=25, ha="right", fontsize=9)

    # 4. Error inspection panels (FP & FN)
    if n_examples > 0:
        total_cols = max(4, n_examples)
        gs_err = gs[1, :].subgridspec(1, total_cols)
        col_idx = 0

        # Show false alarms (Normal patches predicted as anomaly)
        for i in range(n_fp_to_show):
            sample = fps[i]
            ax = fig.add_subplot(gs_err[0, col_idx])
            col_idx += 1
            patch = load_patch(sample["path"])
            ax.imshow(patch, origin="lower", cmap="viridis")
            ax.set_title(f"FP (False Alarm)\n{sample['file']}", fontsize=8, color="red")
            ax.axis("off")

        # Show missed anomalies (True anomalies predicted as normal)
        for i in range(n_fn_to_show):
            sample = fns[i]
            ax = fig.add_subplot(gs_err[0, col_idx])
            col_idx += 1
            patch = load_patch(sample["path"])
            subg = sample["subgroup"].replace("_", " ")
            ax.imshow(patch, origin="lower", cmap="viridis")
            ax.set_title(f"FN (Missed)\n{sample['file']}\n[{subg}]", fontsize=8, color="crimson")
            ax.axis("off")

        # Fill remaining blank slots if any
        while col_idx < total_cols:
            ax = fig.add_subplot(gs_err[0, col_idx])
            ax.axis("off")
            col_idx += 1

    plt.suptitle("Anomaly Detection Evaluation Summary", fontsize=14, weight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Summary plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# -----------------------------------------------------------------------------
# Drop-in Compatibility Alias
# -----------------------------------------------------------------------------
def compare_method_to_annotations(
    method: Callable[[str], Any] = lambda x: {"horizontal": False, "other": False},
    show: bool = False,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Drop-in replacement for existing scripts' `compare_method_to_annotations`,
    focused specifically on binary anomaly detection (horizontal+other vs normal).
    """
    return evaluate_anomaly_detection(
        method=method,
        show=show,
        threshold=threshold,
    )


# -----------------------------------------------------------------------------
# Built-in Baseline Methods (for testing / demonstration)
# -----------------------------------------------------------------------------
def baseline_all_normal(path: str) -> Dict[str, bool]:
    """Baseline predicting no anomaly for all patches."""
    return {"horizontal": False, "other": False}


def baseline_all_anomaly(path: str) -> Dict[str, bool]:
    """Baseline predicting anomaly for all patches."""
    return {"horizontal": True, "other": True}


def baseline_random(path: str, p_anomaly: float = 0.275) -> Dict[str, bool]:
    """Random baseline based on the dataset class distribution."""
    is_anomaly = bool(np.random.rand() < p_anomaly)
    return {"horizontal": is_anomaly, "other": False}


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate anomaly detection performance (horizontal+other vs normal)."
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="all_normal",
        choices=["all_normal", "all_anomaly", "random", "lsd"],
        help="Baseline method to evaluate if no custom method is supplied.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of patches to evaluate (for quick test).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for probability / anomaly score outputs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display visual error inspection plots.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save evaluation plot figure.",
    )

    args = parser.parse_args()

    # Select method
    if args.baseline == "all_normal":
        method = baseline_all_normal
        print("[INFO] Evaluating baseline: all_normal (predicts normal everywhere)")
    elif args.baseline == "all_anomaly":
        method = baseline_all_anomaly
        print("[INFO] Evaluating baseline: all_anomaly (predicts anomaly everywhere)")
    elif args.baseline == "random":
        method = baseline_random
        print("[INFO] Evaluating baseline: random")
    elif args.baseline == "lsd":
        try:
            from LSD_anomaly_detection import method_lsd
            method = method_lsd
            print("[INFO] Evaluating baseline: LSD anomaly detection method")
        except ImportError as e:
            print(f"[ERROR] Could not import method_lsd from LSD_anomaly_detection: {e}")
            sys.exit(1)

    evaluate_anomaly_detection(
        method=method,
        max_samples=args.limit,
        threshold=args.threshold,
        show=args.show,
        save_plot_path=args.save_plot,
    )


if __name__ == "__main__":
    main()
