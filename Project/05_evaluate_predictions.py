#!/usr/bin/env python3
"""
Step 05: Evaluate nnU-Net predictions and residual center-wise variability.

Input:
    local_nnunet/nnUNet_raw/DatasetXXX_clean_raw_LOCO_CENTER/
        imagesTs/
        labelsTs/
        case_mapping.csv

    local_nnunet/predictions/DatasetXXX_clean_raw_LOCO_CENTER/
        case_XXXX.nii.gz

Output:
    local_nnunet/results/per_case_metrics.csv
    local_nnunet/results/final_summary.csv
    local_nnunet/results/statistical_tests.txt
    local_nnunet/results/statistical_tests.json
    local_nnunet/results/dice_by_hospital.png
    local_nnunet/results/mean_dice_by_hospital.png
    local_nnunet/results/overlays/

Run from repo root:
    python Project/05_evaluate_predictions.py \
        --nnunet_raw local_nnunet/nnUNet_raw \
        --pred_root local_nnunet/predictions \
        --out local_nnunet/results
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import stats


# ============================================================
# GENERAL HELPERS
# ============================================================

def json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    return value


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(json_safe(data), f, indent=2)


def strip_nii_gz(path: Path) -> str:
    name = path.name

    if name.endswith(".nii.gz"):
        return name[:-7]

    return path.stem


def parse_dataset_id(dataset_name: str) -> int:
    match = re.match(r"^Dataset(\d{3})_.+", dataset_name)

    if not match:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    return int(match.group(1))


def parse_test_center(dataset_name: str) -> str:
    """
    Extract held-out center from DatasetXXX_clean_raw_LOCO_CENTER.
    """
    if "LOCO_" in dataset_name:
        return dataset_name.split("LOCO_")[-1]

    return "UNKNOWN"


def parse_dataset_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None

    items = {x.strip() for x in value.split(",") if x.strip()}

    return items if items else None


def dataset_matches_filter(dataset_name: str, filter_items: set[str] | None) -> bool:
    if filter_items is None:
        return True

    dataset_id = f"{parse_dataset_id(dataset_name):03d}"

    for item in filter_items:
        item_upper = item.upper()

        if item == dataset_name:
            return True

        if item == dataset_id:
            return True

        if item_upper in dataset_name.upper():
            return True

    return False


def parse_labels(value: str) -> list[int]:
    labels = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        labels.append(int(item))

    return labels


def find_dataset_dirs(nnunet_raw: Path, pred_root: Path, dataset_filter: set[str] | None) -> list[Path]:
    if not nnunet_raw.exists():
        raise FileNotFoundError(f"nnUNet_raw folder not found: {nnunet_raw}")

    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction root not found: {pred_root}")

    dataset_dirs = []

    for path in sorted(nnunet_raw.iterdir()):
        if not path.is_dir():
            continue

        if not re.match(r"^Dataset\d{3}_.+", path.name):
            continue

        if not (path / "labelsTs").exists():
            continue

        if not (path / "imagesTs").exists():
            continue

        if not (pred_root / path.name).exists():
            continue

        if not dataset_matches_filter(path.name, dataset_filter):
            continue

        dataset_dirs.append(path)

    if not dataset_dirs:
        raise RuntimeError(
            "No matching datasets found. Check nnUNet_raw, prediction folders, and --datasets filter."
        )

    return dataset_dirs


# ============================================================
# IMAGE HELPERS
# ============================================================

def read_image(path: Path) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return sitk.ReadImage(str(path))


def image_to_array(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


def same_geometry(image_a: sitk.Image, image_b: sitk.Image, atol: float = 1e-4) -> bool:
    same_size = image_a.GetSize() == image_b.GetSize()
    same_spacing = np.allclose(image_a.GetSpacing(), image_b.GetSpacing(), atol=atol)
    same_origin = np.allclose(image_a.GetOrigin(), image_b.GetOrigin(), atol=atol)
    same_direction = np.allclose(image_a.GetDirection(), image_b.GetDirection(), atol=atol)

    return bool(same_size and same_spacing and same_origin and same_direction)


def resample_label_to_reference(label_img: sitk.Image, reference_img: sitk.Image) -> sitk.Image:
    """
    Resample prediction mask to ground-truth geometry if needed.
    """
    return sitk.Resample(
        label_img,
        reference_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def load_gt_pred(
    gt_path: Path,
    pred_path: Path,
    resample_pred_to_gt: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    gt_img = read_image(gt_path)
    pred_img = read_image(pred_path)

    was_resampled = False

    if not same_geometry(gt_img, pred_img):
        if not resample_pred_to_gt:
            gt_arr = image_to_array(gt_img)
            pred_arr = image_to_array(pred_img)
            raise ValueError(
                f"Prediction and GT geometry mismatch. "
                f"GT shape={gt_arr.shape}, prediction shape={pred_arr.shape}. "
                f"Use default resampling or remove --no_resample_pred_to_gt."
            )

        pred_img = resample_label_to_reference(pred_img, gt_img)
        was_resampled = True

    gt_arr = np.rint(image_to_array(gt_img)).astype(np.uint8)
    pred_arr = np.rint(image_to_array(pred_img)).astype(np.uint8)

    return gt_arr, pred_arr, was_resampled


def load_ct_for_overlay(ct_path: Path, reference_shape: tuple[int, int, int]) -> np.ndarray | None:
    if not ct_path.exists():
        return None

    try:
        ct_img = read_image(ct_path)
        ct_arr = image_to_array(ct_img).astype(np.float32)

        if ct_arr.shape != reference_shape:
            return None

        return ct_arr

    except Exception:
        return None


# ============================================================
# METRIC HELPERS
# ============================================================

def safe_divide(num: float, den: float) -> float:
    if den == 0:
        return float("nan")

    return float(num / den)


def binary_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """
    Compute binary segmentation metrics.

    gt and pred must be boolean arrays.
    """
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())

    gt_voxels = int(gt.sum())
    pred_voxels = int(pred.sum())
    total_voxels = int(gt.size)

    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn

    if dice_den == 0:
        dice = 1.0
    else:
        dice = (2.0 * tp) / dice_den

    if iou_den == 0:
        iou = 1.0
    else:
        iou = tp / iou_den

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    precision = safe_divide(tp, tp + fp)
    accuracy = safe_divide(tp + tn, total_voxels)

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Accuracy": accuracy,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "GT_Voxels": gt_voxels,
        "Pred_Voxels": pred_voxels,
        "Total_Voxels": total_voxels,
    }


def compute_case_metrics(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    label_list: list[int],
) -> dict[str, Any]:
    """
    Combined foreground metrics use all labels > 0.
    Label-wise metrics are also computed for requested labels.
    """
    combined = binary_metrics(gt_arr > 0, pred_arr > 0)

    metrics = dict(combined)

    for label in label_list:
        label_metrics = binary_metrics(gt_arr == label, pred_arr == label)

        for key, value in label_metrics.items():
            metrics[f"{key}_label_{label}"] = value

    return metrics


# ============================================================
# OVERLAY HELPERS
# ============================================================

def normalize_for_display(arr: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    lo, hi = np.percentile(finite, [p_low, p_high])

    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)

    return arr


def pick_best_slice(gt_arr: np.ndarray, pred_arr: np.ndarray) -> int:
    """
    Pick axial slice with largest union of GT and prediction.
    If prediction is empty, use largest GT slice.
    """
    union = np.logical_or(gt_arr > 0, pred_arr > 0)
    areas = union.sum(axis=(1, 2))

    if areas.max() > 0:
        return int(np.argmax(areas))

    gt_areas = (gt_arr > 0).sum(axis=(1, 2))

    if gt_areas.max() > 0:
        return int(np.argmax(gt_areas))

    return gt_arr.shape[0] // 2


def add_contour(ax, mask_2d: np.ndarray, color: str, label: str, linestyle: str = "solid") -> None:
    if mask_2d is None:
        return

    mask_2d = mask_2d.astype(bool)

    if not np.any(mask_2d):
        return

    ax.contour(
        mask_2d.astype(float),
        levels=[0.5],
        colors=color,
        linewidths=2,
        linestyles=linestyle,
    )

    ax.plot([], [], color=color, linewidth=2, linestyle=linestyle, label=label)


def make_error_map(gt_slice: np.ndarray, pred_slice: np.ndarray) -> np.ndarray:
    """
    Error map:
        0 = background / true negative
        1 = true positive
        2 = false positive
        3 = false negative
    """
    gt = gt_slice > 0
    pred = pred_slice > 0

    error = np.zeros(gt.shape, dtype=np.uint8)
    error[np.logical_and(gt, pred)] = 1
    error[np.logical_and(~gt, pred)] = 2
    error[np.logical_and(gt, ~pred)] = 3

    return error


def save_overlay_figure(
    ct_arr: np.ndarray,
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    dice_value: float,
    out_path: Path,
    title: str,
) -> None:
    """
    Save qualitative CT overlay:
        Panel A: CT + ground truth
        Panel B: CT + prediction
        Panel C: CT + both contours
        Panel D: error map
    """
    if ct_arr is None:
        return

    if ct_arr.shape != gt_arr.shape or pred_arr.shape != gt_arr.shape:
        return

    z = pick_best_slice(gt_arr, pred_arr)

    ct_slice = normalize_for_display(ct_arr[z])
    gt_slice = gt_arr[z] > 0
    pred_slice = pred_arr[z] > 0
    error_slice = make_error_map(gt_slice, pred_slice)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(ct_slice, cmap="gray")
    add_contour(axes[0], gt_slice, "lime", "Ground truth")
    axes[0].set_title("CT + Ground Truth")
    axes[0].axis("off")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].imshow(ct_slice, cmap="gray")
    add_contour(axes[1], pred_slice, "red", "Prediction")
    axes[1].set_title("CT + Prediction")
    axes[1].axis("off")
    axes[1].legend(loc="lower right", fontsize=8)

    axes[2].imshow(ct_slice, cmap="gray")
    add_contour(axes[2], gt_slice, "lime", "Ground truth")
    add_contour(axes[2], pred_slice, "red", "Prediction", linestyle="dashed")
    axes[2].set_title(f"CT + Both\nDice = {dice_value:.4f}")
    axes[2].axis("off")
    axes[2].legend(loc="lower right", fontsize=8)

    axes[3].imshow(ct_slice, cmap="gray")
    masked_error = np.ma.masked_where(error_slice == 0, error_slice)
    axes[3].imshow(masked_error, alpha=0.65)
    axes[3].set_title("Error Map\n1=TP, 2=FP, 3=FN")
    axes[3].axis("off")

    fig.suptitle(f"{title} | Slice {z}", fontsize=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# DATASET EVALUATION
# ============================================================

def load_case_mapping(dataset_dir: Path) -> pd.DataFrame:
    mapping_path = dataset_dir / "case_mapping.csv"

    if not mapping_path.exists():
        return pd.DataFrame()

    mapping_df = pd.read_csv(mapping_path)

    if "nnunet_case_id" not in mapping_df.columns:
        return pd.DataFrame()

    mapping_df["nnunet_case_id"] = mapping_df["nnunet_case_id"].astype(str)

    return mapping_df


def mapping_lookup(mapping_df: pd.DataFrame, case_id: str, fallback_center: str) -> dict[str, Any]:
    if mapping_df.empty:
        return {
            "PatientID": "",
            "Hospital": fallback_center,
            "Original_CT_Path": "",
            "Original_Mask_Path": "",
        }

    row = mapping_df[mapping_df["nnunet_case_id"] == case_id]

    if row.empty:
        return {
            "PatientID": "",
            "Hospital": fallback_center,
            "Original_CT_Path": "",
            "Original_Mask_Path": "",
        }

    row = row.iloc[0]

    patient_id = row.get("patient_id", "")
    center = row.get("center", fallback_center)
    ct_path = row.get("ct_path", "")
    mask_path = row.get("mask_path", "")

    return {
        "PatientID": patient_id,
        "Hospital": center,
        "Original_CT_Path": ct_path,
        "Original_Mask_Path": mask_path,
    }


def evaluate_one_dataset(
    dataset_dir: Path,
    pred_root: Path,
    out_dir: Path,
    label_list: list[int],
    overlay_samples: int,
    resample_pred_to_gt: bool,
) -> list[dict[str, Any]]:
    dataset_name = dataset_dir.name
    dataset_id = parse_dataset_id(dataset_name)
    fallback_center = parse_test_center(dataset_name)

    pred_dir = pred_root / dataset_name
    labels_ts = dataset_dir / "labelsTs"
    images_ts = dataset_dir / "imagesTs"

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction folder not found: {pred_dir}")

    if not labels_ts.exists():
        raise FileNotFoundError(f"labelsTs folder not found: {labels_ts}")

    if not images_ts.exists():
        raise FileNotFoundError(f"imagesTs folder not found: {images_ts}")

    mapping_df = load_case_mapping(dataset_dir)

    pred_files = sorted(pred_dir.glob("*.nii.gz"))

    rows = []
    overlays_written = 0

    print(f"\nEvaluating {dataset_name}")
    print("=" * 80)
    print(f"Prediction files: {len(pred_files)}")

    for pred_path in pred_files:
        case_id = strip_nii_gz(pred_path)

        gt_path = labels_ts / f"{case_id}.nii.gz"
        ct_path = images_ts / f"{case_id}_0000.nii.gz"

        if not gt_path.exists():
            print(f"Missing GT for {case_id}: {gt_path}")
            continue

        if not ct_path.exists():
            print(f"Missing CT image for {case_id}: {ct_path}")
            continue

        lookup = mapping_lookup(mapping_df, case_id, fallback_center)

        try:
            gt_arr, pred_arr, pred_resampled = load_gt_pred(
                gt_path=gt_path,
                pred_path=pred_path,
                resample_pred_to_gt=resample_pred_to_gt,
            )

            metrics = compute_case_metrics(
                gt_arr=gt_arr,
                pred_arr=pred_arr,
                label_list=label_list,
            )

            row = {
                "Dataset": dataset_name,
                "Dataset_ID": dataset_id,
                "Hospital": lookup["Hospital"],
                "Case": case_id,
                "PatientID": lookup["PatientID"],
                "Prediction_Path": str(pred_path),
                "GT_Path": str(gt_path),
                "Image_Path": str(ct_path),
                "Original_CT_Path": lookup["Original_CT_Path"],
                "Original_Mask_Path": lookup["Original_Mask_Path"],
                "Prediction_Resampled_To_GT": pred_resampled,
                "Shape_ZYX": tuple(int(x) for x in gt_arr.shape),
            }

            row.update(metrics)
            rows.append(row)

            if overlays_written < overlay_samples:
                ct_arr = load_ct_for_overlay(ct_path, reference_shape=gt_arr.shape)

                if ct_arr is not None:
                    overlay_path = (
                        out_dir
                        / "overlays"
                        / dataset_name
                        / f"{lookup['Hospital']}_{case_id}_overlay.png"
                    )

                    save_overlay_figure(
                        ct_arr=ct_arr,
                        gt_arr=gt_arr,
                        pred_arr=pred_arr,
                        dice_value=metrics["Dice"],
                        out_path=overlay_path,
                        title=f"{lookup['Hospital']} | {case_id}",
                    )

                    overlays_written += 1

        except Exception as exc:
            print(f"Failed case {case_id}: {exc}")
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Dataset_ID": dataset_id,
                    "Hospital": lookup["Hospital"],
                    "Case": case_id,
                    "PatientID": lookup["PatientID"],
                    "Prediction_Path": str(pred_path),
                    "GT_Path": str(gt_path),
                    "Image_Path": str(ct_path),
                    "Error": str(exc),
                }
            )

    print(f"Cases evaluated: {len(rows)}")
    print(f"Overlays written: {overlays_written}")

    return rows


# ============================================================
# SUMMARY AND STATISTICS
# ============================================================

def make_summary(per_case_df: pd.DataFrame) -> pd.DataFrame:
    valid = per_case_df.dropna(subset=["Dice"]).copy()

    if valid.empty:
        return pd.DataFrame()

    summary = (
        valid.groupby("Hospital")
        .agg(
            Cases=("Dice", "count"),
            Mean_Dice=("Dice", "mean"),
            Std_Dice=("Dice", "std"),
            Median_Dice=("Dice", "median"),
            Min_Dice=("Dice", "min"),
            Max_Dice=("Dice", "max"),
            Mean_IoU=("IoU", "mean"),
            Mean_Sensitivity=("Sensitivity", "mean"),
            Mean_Specificity=("Specificity", "mean"),
            Mean_Precision=("Precision", "mean"),
            Mean_Accuracy=("Accuracy", "mean"),
        )
        .reset_index()
        .sort_values("Hospital")
    )

    return summary


def run_statistical_tests(per_case_df: pd.DataFrame, alpha: float) -> dict[str, Any]:
    valid = per_case_df.dropna(subset=["Dice"]).copy()

    if valid.empty:
        return {
            "status": "failed",
            "reason": "No valid Dice scores available.",
        }

    hospitals = sorted(valid["Hospital"].dropna().unique().tolist())

    groups = []
    group_sizes = {}

    for hospital in hospitals:
        values = valid.loc[valid["Hospital"] == hospital, "Dice"].dropna().values
        group_sizes[hospital] = int(len(values))

        if len(values) > 0:
            groups.append(values)

    if len(groups) < 2:
        return {
            "status": "failed",
            "reason": "Fewer than two hospitals with valid Dice scores.",
            "group_sizes": group_sizes,
        }

    result = {
        "status": "completed",
        "alpha": alpha,
        "group_sizes": group_sizes,
    }

    try:
        levene_stat, levene_p = stats.levene(*groups, center="median")
        result["levene"] = {
            "statistic": float(levene_stat),
            "p_value": float(levene_p),
            "reject_null": bool(levene_p < alpha),
            "interpretation": (
                "Dice variance differs significantly across hospitals."
                if levene_p < alpha
                else "No statistically significant Dice variance difference across hospitals."
            ),
        }
    except Exception as exc:
        result["levene"] = {
            "error": str(exc),
        }

    try:
        anova_stat, anova_p = stats.f_oneway(*groups)
        result["anova"] = {
            "statistic": float(anova_stat),
            "p_value": float(anova_p),
            "reject_null": bool(anova_p < alpha),
            "interpretation": (
                "Mean Dice differs significantly across hospitals."
                if anova_p < alpha
                else "No statistically significant mean Dice difference across hospitals."
            ),
        }
    except Exception as exc:
        result["anova"] = {
            "error": str(exc),
        }

    try:
        kruskal_stat, kruskal_p = stats.kruskal(*groups)
        result["kruskal_wallis"] = {
            "statistic": float(kruskal_stat),
            "p_value": float(kruskal_p),
            "reject_null": bool(kruskal_p < alpha),
            "interpretation": (
                "Dice distributions differ significantly across hospitals."
                if kruskal_p < alpha
                else "No statistically significant Dice distribution difference across hospitals."
            ),
        }
    except Exception as exc:
        result["kruskal_wallis"] = {
            "error": str(exc),
        }

    return result


def write_statistical_tests_txt(path: Path, tests: dict[str, Any]) -> None:
    lines = []

    lines.append("Statistical Testing of Dice Scores Across Hospitals")
    lines.append("=" * 70)

    if tests.get("status") != "completed":
        lines.append(f"Status: {tests.get('status')}")
        lines.append(f"Reason: {tests.get('reason')}")
    else:
        lines.append(f"Alpha: {tests.get('alpha')}")
        lines.append("")
        lines.append("Group sizes:")
        for hospital, count in tests.get("group_sizes", {}).items():
            lines.append(f"  {hospital}: {count}")

        lines.append("")

        for test_name in ["anova", "levene", "kruskal_wallis"]:
            result = tests.get(test_name, {})
            lines.append(test_name.replace("_", " ").title())
            lines.append("-" * 70)

            if "error" in result:
                lines.append(f"Error: {result['error']}")
            else:
                lines.append(f"Statistic: {result.get('statistic'):.6f}")
                lines.append(f"p-value:   {result.get('p_value'):.6f}")
                lines.append(f"Reject H0: {result.get('reject_null')}")
                lines.append(f"Interpretation: {result.get('interpretation')}")

            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ============================================================
# PLOTS
# ============================================================

def save_dice_boxplot(per_case_df: pd.DataFrame, output_png: Path) -> None:
    valid = per_case_df.dropna(subset=["Dice"]).copy()

    if valid.empty:
        return

    hospitals = sorted(valid["Hospital"].unique().tolist())

    data = [
        valid.loc[valid["Hospital"] == hospital, "Dice"].values
        for hospital in hospitals
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(data, labels=hospitals)

    ax.set_title("Dice Score Distribution by Hospital", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hospital")
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_mean_dice_barplot(summary_df: pd.DataFrame, output_png: Path) -> None:
    if summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    hospitals = summary_df["Hospital"].tolist()
    means = summary_df["Mean_Dice"].tolist()
    stds = summary_df["Std_Dice"].fillna(0).tolist()

    bars = ax.bar(hospitals, means, yerr=stds, capsize=5)

    ax.set_title("Mean Dice Score by Hospital", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hospital")
    ax.set_ylabel("Mean Dice Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_individual_dice_scatter(per_case_df: pd.DataFrame, output_png: Path) -> None:
    valid = per_case_df.dropna(subset=["Dice"]).copy()

    if valid.empty:
        return

    hospitals = sorted(valid["Hospital"].unique().tolist())

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, hospital in enumerate(hospitals):
        values = valid.loc[valid["Hospital"] == hospital, "Dice"].values
        x = np.full(len(values), idx, dtype=float)

        if len(values) > 1:
            jitter = np.linspace(-0.12, 0.12, len(values))
        else:
            jitter = np.array([0.0])

        ax.scatter(x + jitter, values, alpha=0.7)

    ax.set_xticks(range(len(hospitals)))
    ax.set_xticklabels(hospitals)
    ax.set_title("Individual Dice Scores by Hospital", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hospital")
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    output_png.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def evaluate_all(
    nnunet_raw: Path,
    pred_root: Path,
    out_dir: Path,
    dataset_filter: set[str] | None,
    label_list: list[int],
    overlay_samples: int,
    resample_pred_to_gt: bool,
    alpha: float,
    continue_on_error: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = find_dataset_dirs(
        nnunet_raw=nnunet_raw,
        pred_root=pred_root,
        dataset_filter=dataset_filter,
    )

    print("\nEvaluation Configuration")
    print("=" * 80)
    print(f"nnUNet_raw:      {nnunet_raw}")
    print(f"Prediction root: {pred_root}")
    print(f"Output folder:   {out_dir}")
    print(f"Datasets found:  {len(dataset_dirs)}")
    print(f"Labels:          {label_list}")
    print(f"Overlay samples per dataset: {overlay_samples}")

    all_rows = []

    for dataset_dir in dataset_dirs:
        try:
            rows = evaluate_one_dataset(
                dataset_dir=dataset_dir,
                pred_root=pred_root,
                out_dir=out_dir,
                label_list=label_list,
                overlay_samples=overlay_samples,
                resample_pred_to_gt=resample_pred_to_gt,
            )

            all_rows.extend(rows)

        except Exception as exc:
            print(f"\nERROR while evaluating {dataset_dir.name}")
            print("=" * 80)
            print(str(exc))

            if not continue_on_error:
                raise

    per_case_df = pd.DataFrame(all_rows)

    if per_case_df.empty:
        raise RuntimeError("No prediction/ground-truth pairs were evaluated.")

    per_case_csv = out_dir / "per_case_metrics.csv"
    per_case_df.to_csv(per_case_csv, index=False)

    summary_df = make_summary(per_case_df)

    summary_csv = out_dir / "final_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    tests = run_statistical_tests(per_case_df, alpha=alpha)

    tests_json = out_dir / "statistical_tests.json"
    tests_txt = out_dir / "statistical_tests.txt"

    write_json(tests_json, tests)
    write_statistical_tests_txt(tests_txt, tests)

    save_dice_boxplot(
        per_case_df=per_case_df,
        output_png=out_dir / "dice_by_hospital.png",
    )

    save_mean_dice_barplot(
        summary_df=summary_df,
        output_png=out_dir / "mean_dice_by_hospital.png",
    )

    save_individual_dice_scatter(
        per_case_df=per_case_df,
        output_png=out_dir / "individual_dice_by_hospital.png",
    )

    print("\nStep 05 Complete")
    print("=" * 80)
    print(f"Per-case metrics:       {per_case_csv}")
    print(f"Center-wise summary:    {summary_csv}")
    print(f"Statistical tests TXT:  {tests_txt}")
    print(f"Statistical tests JSON: {tests_json}")
    print(f"Dice boxplot:           {out_dir / 'dice_by_hospital.png'}")
    print(f"Mean Dice bar plot:     {out_dir / 'mean_dice_by_hospital.png'}")
    print(f"Individual Dice plot:   {out_dir / 'individual_dice_by_hospital.png'}")
    print(f"Overlays folder:        {out_dir / 'overlays'}")

    print("\nCenter-wise Summary")
    print("-" * 80)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))

    print("\nStatistical Tests")
    print("-" * 80)
    if tests.get("status") == "completed":
        if "anova" in tests and "p_value" in tests["anova"]:
            print(
                f"ANOVA: F={tests['anova']['statistic']:.4f}, "
                f"p={tests['anova']['p_value']:.6f}"
            )

        if "levene" in tests and "p_value" in tests["levene"]:
            print(
                f"Levene: statistic={tests['levene']['statistic']:.4f}, "
                f"p={tests['levene']['p_value']:.6f}"
            )
    else:
        print(tests.get("reason"))

    print("\nNext step:")
    print(
        "python Project/06_visualize_pet_ct_gt_pred.py "
        "--ct path/to/PatientID__CT.nii.gz "
        "--pet path/to/PatientID__PT.nii.gz "
        "--gt path/to/PatientID.nii.gz "
        "--pred path/to/prediction.nii.gz "
        "--out figures/pet_ct_gt_prediction_overlay.png"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate nnU-Net predictions against labelsTs and analyze center-wise Dice variability."
    )

    parser.add_argument(
        "--nnunet_raw",
        required=True,
        type=Path,
        help="Path to local_nnunet/nnUNet_raw.",
    )

    parser.add_argument(
        "--pred_root",
        required=True,
        type=Path,
        help="Path to local_nnunet/predictions.",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output folder for metrics, statistics, plots, and overlays.",
    )

    parser.add_argument(
        "--datasets",
        default=None,
        type=str,
        help=(
            "Optional comma-separated dataset filter. "
            "Examples: CHUM,MDA or 001,002 or Dataset001_clean_raw_LOCO_CHUM."
        ),
    )

    parser.add_argument(
        "--labels",
        default="1",
        type=str,
        help=(
            "Comma-separated labels for label-wise metrics. "
            "Default is 1 for binary tumor segmentation."
        ),
    )

    parser.add_argument(
        "--overlay_samples",
        default=3,
        type=int,
        help="Number of qualitative CT overlays to save per dataset. Default: 3.",
    )

    parser.add_argument(
        "--no_resample_pred_to_gt",
        action="store_true",
        help="Disable prediction-to-GT resampling when geometry mismatch occurs.",
    )

    parser.add_argument(
        "--alpha",
        default=0.05,
        type=float,
        help="Significance level for ANOVA and Levene tests. Default: 0.05.",
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue if one dataset fails.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_filter = parse_dataset_filter(args.datasets)
    label_list = parse_labels(args.labels)

    evaluate_all(
        nnunet_raw=args.nnunet_raw,
        pred_root=args.pred_root,
        out_dir=args.out,
        dataset_filter=dataset_filter,
        label_list=label_list,
        overlay_samples=args.overlay_samples,
        resample_pred_to_gt=not args.no_resample_pred_to_gt,
        alpha=args.alpha,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    main()
