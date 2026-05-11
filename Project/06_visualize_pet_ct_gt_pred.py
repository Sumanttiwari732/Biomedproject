#!/usr/bin/env python3
"""
Step 06: Generate PET/CT visualization with ground truth and nnU-Net prediction.

This script overlays:
    1. CT only
    2. PET/CT fusion
    3. PET/CT + ground-truth contour + nnU-Net prediction contour
    4. Zoomed PET/CT segmentation error map

It supports predictions created from cropped/resampled nnU-Net datasets as long as
the prediction NIfTI file preserves physical image geometry.

Input example:
    Original CT:
        data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258__CT.nii.gz

    Original PET:
        data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258__PT.nii.gz

    Original ground truth:
        data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258.nii.gz

    nnU-Net prediction:
        local_nnunet/predictions/Dataset006_clean_raw_LOCO_MDA/case_XXXX.nii.gz

Output:
    figures/MDA-258_pet_ct_gt_prediction_overlay.png
    figures/MDA-258_pet_ct_gt_prediction_overlay_metrics.json
    figures/MDA-258_pet_ct_gt_prediction_overlay_metrics.txt

Run from repo root:
    python Project/06_visualize_pet_ct_gt_pred.py \
        --ct "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258__CT.nii.gz" \
        --pet "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258__PT.nii.gz" \
        --gt "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258.nii.gz" \
        --pred "local_nnunet/predictions/Dataset006_clean_raw_LOCO_MDA/case_XXXX.nii.gz" \
        --out figures/MDA-258_pet_ct_gt_prediction_overlay.png \
        --patient_id MDA-258
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def infer_patient_id(ct_path: Path) -> str:
    name = ct_path.name

    if name.endswith("__CT.nii.gz"):
        return name.replace("__CT.nii.gz", "")

    if name.endswith(".nii.gz"):
        return name.replace(".nii.gz", "")

    return ct_path.stem


# ============================================================
# IMAGE HELPERS
# ============================================================

def read_image(path: Path) -> sitk.Image:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return sitk.ReadImage(str(path))


def image_to_array(image: sitk.Image) -> np.ndarray:
    """
    SimpleITK returns arrays in z, y, x order.
    """
    return sitk.GetArrayFromImage(image)


def same_geometry(image_a: sitk.Image, image_b: sitk.Image, atol: float = 1e-4) -> bool:
    same_size = image_a.GetSize() == image_b.GetSize()
    same_spacing = np.allclose(image_a.GetSpacing(), image_b.GetSpacing(), atol=atol)
    same_origin = np.allclose(image_a.GetOrigin(), image_b.GetOrigin(), atol=atol)
    same_direction = np.allclose(image_a.GetDirection(), image_b.GetDirection(), atol=atol)

    return bool(same_size and same_spacing and same_origin and same_direction)


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    is_label: bool,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample image to the physical grid of reference.

    PET uses linear interpolation.
    GT and prediction masks use nearest-neighbor interpolation.
    """
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    pixel_type = sitk.sitkUInt8 if is_label else sitk.sitkFloat32

    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        interpolator,
        default_value,
        pixel_type,
    )


def load_and_align_images(
    ct_path: Path,
    pet_path: Path,
    gt_path: Path,
    pred_path: Path,
) -> tuple[sitk.Image, sitk.Image, sitk.Image, sitk.Image, dict[str, Any]]:
    """
    Load CT, PET, ground truth, and prediction.

    PET, GT, and prediction are resampled to original CT geometry.
    This makes original PET/CT overlays possible even when nnU-Net prediction
    was created from a cropped/resampled dataset.
    """
    ct_img = read_image(ct_path)
    pet_img = read_image(pet_path)
    gt_img = read_image(gt_path)
    pred_img = read_image(pred_path)

    resampling_info = {
        "pet_resampled_to_ct": not same_geometry(pet_img, ct_img),
        "gt_resampled_to_ct": not same_geometry(gt_img, ct_img),
        "pred_resampled_to_ct": not same_geometry(pred_img, ct_img),
        "ct_size_xyz": tuple(int(x) for x in ct_img.GetSize()),
        "ct_spacing_xyz": tuple(float(x) for x in ct_img.GetSpacing()),
        "pet_original_size_xyz": tuple(int(x) for x in pet_img.GetSize()),
        "gt_original_size_xyz": tuple(int(x) for x in gt_img.GetSize()),
        "pred_original_size_xyz": tuple(int(x) for x in pred_img.GetSize()),
        "pet_original_spacing_xyz": tuple(float(x) for x in pet_img.GetSpacing()),
        "gt_original_spacing_xyz": tuple(float(x) for x in gt_img.GetSpacing()),
        "pred_original_spacing_xyz": tuple(float(x) for x in pred_img.GetSpacing()),
    }

    pet_on_ct = resample_to_reference(
        image=pet_img,
        reference=ct_img,
        is_label=False,
        default_value=0.0,
    )

    gt_on_ct = resample_to_reference(
        image=gt_img,
        reference=ct_img,
        is_label=True,
        default_value=0,
    )

    pred_on_ct = resample_to_reference(
        image=pred_img,
        reference=ct_img,
        is_label=True,
        default_value=0,
    )

    return ct_img, pet_on_ct, gt_on_ct, pred_on_ct, resampling_info


# ============================================================
# PREPROCESSING FOR VISUALIZATION
# ============================================================

def ct_window(ct_slice: np.ndarray, window_min: float, window_max: float) -> np.ndarray:
    """
    Apply soft-tissue CT window and normalize to 0-1.
    """
    ct_slice = np.asarray(ct_slice, dtype=np.float32)
    ct_slice = np.clip(ct_slice, window_min, window_max)
    ct_slice = (ct_slice - window_min) / (window_max - window_min)

    return ct_slice


def normalize_pet_volume(
    pet_arr: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> np.ndarray:
    """
    Normalize PET volume using percentile scaling.
    """
    pet_arr = np.asarray(pet_arr, dtype=np.float32)

    valid = pet_arr[np.isfinite(pet_arr)]
    valid = valid[valid > 0]

    if valid.size == 0:
        return np.zeros_like(pet_arr, dtype=np.float32)

    low = np.percentile(valid, lower_percentile)
    high = np.percentile(valid, upper_percentile)

    if high <= low:
        return np.zeros_like(pet_arr, dtype=np.float32)

    pet_norm = np.clip(pet_arr, low, high)
    pet_norm = (pet_norm - low) / (high - low)

    return pet_norm


def clean_label_array(arr: np.ndarray) -> np.ndarray:
    """
    Convert label image to uint8 and keep non-negative integer labels.
    """
    arr = np.rint(arr).astype(np.int16)
    arr[arr < 0] = 0

    return arr.astype(np.uint8)


# ============================================================
# METRICS
# ============================================================

def safe_divide(num: float, den: float) -> float:
    if den == 0:
        return float("nan")

    return float(num / den)


def binary_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
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

    dice = 1.0 if dice_den == 0 else (2.0 * tp) / dice_den
    iou = 1.0 if iou_den == 0 else tp / iou_den

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Sensitivity": safe_divide(tp, tp + fn),
        "Specificity": safe_divide(tn, tn + fp),
        "Precision": safe_divide(tp, tp + fp),
        "Accuracy": safe_divide(tp + tn, total_voxels),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "GT_Voxels": gt_voxels,
        "Pred_Voxels": pred_voxels,
        "Total_Voxels": total_voxels,
    }


def compute_metrics(gt_arr: np.ndarray, pred_arr: np.ndarray) -> dict[str, Any]:
    """
    Compute combined foreground metrics and label-wise metrics for labels 1 and 2.
    """
    metrics = {
        "combined_labels_1_and_2": binary_metrics(gt_arr > 0, pred_arr > 0),
        "label_1_gtvp": binary_metrics(gt_arr == 1, pred_arr == 1),
        "label_2_gtvn": binary_metrics(gt_arr == 2, pred_arr == 2),
        "labels_present_gt": sorted(int(x) for x in np.unique(gt_arr)),
        "labels_present_prediction": sorted(int(x) for x in np.unique(pred_arr)),
    }

    return metrics


def format_metrics_text(patient_id: str, metrics: dict[str, Any], resampling_info: dict[str, Any]) -> str:
    lines = []

    lines.append(f"PET/CT Ground Truth vs nnU-Net Prediction Metrics: {patient_id}")
    lines.append("=" * 80)
    lines.append("")

    for section_name in ["combined_labels_1_and_2", "label_1_gtvp", "label_2_gtvn"]:
        section = metrics[section_name]

        lines.append(section_name)
        lines.append("-" * len(section_name))

        for key in ["Dice", "IoU", "Sensitivity", "Specificity", "Precision", "Accuracy"]:
            value = section.get(key)

            if value is None or np.isnan(value):
                lines.append(f"{key}: N/A")
            else:
                lines.append(f"{key}: {value:.4f}")

        for key in ["TP", "FP", "FN", "TN", "GT_Voxels", "Pred_Voxels", "Total_Voxels"]:
            lines.append(f"{key}: {section.get(key)}")

        lines.append("")

    lines.append("Labels")
    lines.append("-" * 80)
    lines.append(f"Ground truth labels: {metrics['labels_present_gt']}")
    lines.append(f"Prediction labels:   {metrics['labels_present_prediction']}")
    lines.append("")

    lines.append("Resampling Information")
    lines.append("-" * 80)

    for key, value in resampling_info.items():
        lines.append(f"{key}: {value}")

    return "\n".join(lines)


# ============================================================
# SLICE AND ROI SELECTION
# ============================================================

def select_slice(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    pet_arr: np.ndarray,
    mode: str,
) -> int:
    """
    Select axial slice for visualization.

    mode:
        union = largest union of GT and prediction
        gt = largest GT area
        pred = largest prediction area
        pet = largest PET uptake inside GT/pred union if possible
        middle = middle slice
    """
    if mode == "middle":
        return gt_arr.shape[0] // 2

    if mode == "gt":
        areas = (gt_arr > 0).sum(axis=(1, 2))

        if areas.max() > 0:
            return int(np.argmax(areas))

    if mode == "pred":
        areas = (pred_arr > 0).sum(axis=(1, 2))

        if areas.max() > 0:
            return int(np.argmax(areas))

    if mode == "pet":
        union = np.logical_or(gt_arr > 0, pred_arr > 0)

        if union.sum() > 0:
            scores = []

            for z in range(union.shape[0]):
                if union[z].sum() == 0:
                    scores.append(0.0)
                else:
                    scores.append(float(pet_arr[z][union[z]].sum()))

            if max(scores) > 0:
                return int(np.argmax(scores))

    # Default: largest union.
    union = np.logical_or(gt_arr > 0, pred_arr > 0)
    areas = union.sum(axis=(1, 2))

    if areas.max() > 0:
        return int(np.argmax(areas))

    gt_areas = (gt_arr > 0).sum(axis=(1, 2))

    if gt_areas.max() > 0:
        return int(np.argmax(gt_areas))

    pred_areas = (pred_arr > 0).sum(axis=(1, 2))

    if pred_areas.max() > 0:
        return int(np.argmax(pred_areas))

    return gt_arr.shape[0] // 2


def get_roi_crop(gt_slice: np.ndarray, pred_slice: np.ndarray, margin: int) -> tuple[slice, slice]:
    """
    Crop around union of GT and prediction on one axial slice.
    """
    combined = np.logical_or(gt_slice > 0, pred_slice > 0)

    coords = np.argwhere(combined)

    height, width = combined.shape

    if coords.size == 0:
        return slice(0, height), slice(0, width)

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min = max(0, int(y_min) - margin)
    y_max = min(height, int(y_max) + margin + 1)

    x_min = max(0, int(x_min) - margin)
    x_max = min(width, int(x_max) + margin + 1)

    return slice(y_min, y_max), slice(x_min, x_max)


# ============================================================
# PLOTTING
# ============================================================

def show_ct(ax, ct_slice: np.ndarray, title: str, aspect: float) -> None:
    ax.imshow(ct_slice, cmap="gray", aspect=aspect)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def show_pet_ct_fusion(
    ax,
    ct_slice: np.ndarray,
    pet_slice: np.ndarray,
    title: str,
    aspect: float,
    pet_alpha: float,
    pet_threshold: float,
) -> None:
    ax.imshow(ct_slice, cmap="gray", aspect=aspect)

    pet_overlay = np.ma.masked_where(pet_slice < pet_threshold, pet_slice)

    ax.imshow(
        pet_overlay,
        cmap="hot",
        alpha=pet_alpha,
        aspect=aspect,
    )

    ax.set_title(title, fontsize=11)
    ax.axis("off")


def add_gt_pred_contours(
    ax,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    linewidth: float = 2.4,
) -> None:
    """
    Cyan solid = ground truth
    Magenta dashed = nnU-Net prediction
    """
    gt_binary = gt_slice > 0
    pred_binary = pred_slice > 0

    if np.any(gt_binary):
        ax.contour(
            gt_binary.astype(float),
            levels=[0.5],
            colors="cyan",
            linewidths=linewidth,
            linestyles="solid",
        )

    if np.any(pred_binary):
        ax.contour(
            pred_binary.astype(float),
            levels=[0.5],
            colors="magenta",
            linewidths=linewidth,
            linestyles="dashed",
        )


def make_error_map(gt_slice: np.ndarray, pred_slice: np.ndarray) -> np.ndarray:
    """
    Error map:
        0 = transparent background
        1 = true positive overlap
        2 = false positive prediction
        3 = false negative missed tumor
    """
    gt = gt_slice > 0
    pred = pred_slice > 0

    error = np.zeros(gt.shape, dtype=np.uint8)
    error[np.logical_and(gt, pred)] = 1
    error[np.logical_and(~gt, pred)] = 2
    error[np.logical_and(gt, ~pred)] = 3

    return error


def show_error_map(
    ax,
    ct_slice: np.ndarray,
    pet_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    title: str,
    aspect: float,
    pet_alpha: float,
    pet_threshold: float,
) -> None:
    show_pet_ct_fusion(
        ax=ax,
        ct_slice=ct_slice,
        pet_slice=pet_slice,
        title=title,
        aspect=aspect,
        pet_alpha=pet_alpha,
        pet_threshold=pet_threshold,
    )

    error = make_error_map(gt_slice, pred_slice)

    cmap = ListedColormap(
        [
            (0, 0, 0, 0.0),      # 0 transparent
            (0, 1, 0, 0.55),    # 1 true positive
            (1, 1, 0, 0.65),    # 2 false positive
            (0, 0.4, 1, 0.65),  # 3 false negative
        ]
    )

    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    ax.imshow(error, cmap=cmap, norm=norm, aspect=aspect)
    ax.axis("off")


def make_caption(patient_id: str) -> str:
    caption = (
        f"Figure X. Qualitative evaluation of nnU-Net tumor segmentation using PET/CT fusion, "
        f"ground-truth annotation, and model prediction for case {patient_id}. CT alone provides "
        f"anatomical context but does not clearly separate tumor from surrounding head-and-neck "
        f"soft tissue. PET/CT fusion shows metabolic uptake, while the cyan contour represents "
        f"the ground-truth tumor mask and the magenta dashed contour represents the nnU-Net "
        f"prediction. The zoomed error map summarizes agreement between ground truth and prediction, "
        f"where green indicates true-positive overlap, yellow indicates false-positive prediction, "
        f"and blue indicates false-negative missed tumor. Evaluation is based on the mask-defined "
        f"region of interest rather than CT appearance alone."
    )

    return caption


def make_visualization(
    ct_img: sitk.Image,
    pet_img: sitk.Image,
    gt_img: sitk.Image,
    pred_img: sitk.Image,
    patient_id: str,
    output_path: Path,
    metrics: dict[str, Any],
    slice_mode: str,
    ct_min: float,
    ct_max: float,
    pet_lower_percentile: float,
    pet_upper_percentile: float,
    pet_alpha: float,
    pet_threshold: float,
    roi_margin: int,
    include_caption: bool,
) -> int:
    ct_arr = image_to_array(ct_img).astype(np.float32)
    pet_arr = image_to_array(pet_img).astype(np.float32)
    gt_arr = clean_label_array(image_to_array(gt_img))
    pred_arr = clean_label_array(image_to_array(pred_img))

    pet_norm = normalize_pet_volume(
        pet_arr,
        lower_percentile=pet_lower_percentile,
        upper_percentile=pet_upper_percentile,
    )

    z = select_slice(
        gt_arr=gt_arr,
        pred_arr=pred_arr,
        pet_arr=pet_norm,
        mode=slice_mode,
    )

    ct_slice = ct_window(ct_arr[z], window_min=ct_min, window_max=ct_max)
    pet_slice = pet_norm[z]
    gt_slice = gt_arr[z]
    pred_slice = pred_arr[z]

    spacing = ct_img.GetSpacing()
    aspect = spacing[1] / spacing[0]

    y_crop, x_crop = get_roi_crop(
        gt_slice=gt_slice,
        pred_slice=pred_slice,
        margin=roi_margin,
    )

    ct_zoom = ct_slice[y_crop, x_crop]
    pet_zoom = pet_slice[y_crop, x_crop]
    gt_zoom = gt_slice[y_crop, x_crop]
    pred_zoom = pred_slice[y_crop, x_crop]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
    axes = axes.ravel()

    show_ct(
        ax=axes[0],
        ct_slice=ct_slice,
        title="A. CT only\nTumor boundary is unclear",
        aspect=aspect,
    )

    show_pet_ct_fusion(
        ax=axes[1],
        ct_slice=ct_slice,
        pet_slice=pet_slice,
        title="B. PET/CT fusion\nAnatomy + metabolic uptake",
        aspect=aspect,
        pet_alpha=pet_alpha,
        pet_threshold=pet_threshold,
    )

    show_pet_ct_fusion(
        ax=axes[2],
        ct_slice=ct_slice,
        pet_slice=pet_slice,
        title="C. PET/CT + contours\nGround truth vs prediction",
        aspect=aspect,
        pet_alpha=pet_alpha,
        pet_threshold=pet_threshold,
    )

    add_gt_pred_contours(
        ax=axes[2],
        gt_slice=gt_slice,
        pred_slice=pred_slice,
        linewidth=2.4,
    )

    show_error_map(
        ax=axes[3],
        ct_slice=ct_zoom,
        pet_slice=pet_zoom,
        gt_slice=gt_zoom,
        pred_slice=pred_zoom,
        title="D. Zoomed error map\nTP / FP / FN",
        aspect=aspect,
        pet_alpha=pet_alpha,
        pet_threshold=pet_threshold,
    )

    add_gt_pred_contours(
        ax=axes[3],
        gt_slice=gt_zoom,
        pred_slice=pred_zoom,
        linewidth=2.0,
    )

    combined_dice = metrics["combined_labels_1_and_2"]["Dice"]
    combined_iou = metrics["combined_labels_1_and_2"]["IoU"]

    fig.suptitle(
        f"PET/CT Ground Truth vs nnU-Net Prediction: {patient_id}\n"
        f"Selected axial slice z = {z}; Dice = {combined_dice:.3f}, IoU = {combined_iou:.3f}",
        fontsize=14,
        y=0.98,
    )

    legend_items = [
        Line2D([0], [0], color="cyan", lw=2.5, linestyle="solid", label="Ground truth"),
        Line2D([0], [0], color="magenta", lw=2.5, linestyle="dashed", label="nnU-Net prediction"),
        Line2D([0], [0], color="green", lw=6, label="True positive"),
        Line2D([0], [0], color="yellow", lw=6, label="False positive"),
        Line2D([0], [0], color="dodgerblue", lw=6, label="False negative"),
    ]

    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.11),
    )

    if include_caption:
        caption = textwrap.fill(make_caption(patient_id), width=135)

        fig.text(
            0.05,
            0.015,
            caption,
            ha="left",
            va="bottom",
            fontsize=8.5,
        )

        plt.tight_layout(rect=[0, 0.16, 1, 0.92])
    else:
        plt.tight_layout(rect=[0, 0.10, 1, 0.92])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return z


# ============================================================
# MAIN
# ============================================================

def run_visualization(args: argparse.Namespace) -> None:
    ct_path = args.ct
    pet_path = args.pet
    gt_path = args.gt
    pred_path = args.pred
    output_path = args.out

    patient_id = args.patient_id

    if patient_id is None:
        patient_id = infer_patient_id(ct_path)

    ct_img, pet_on_ct, gt_on_ct, pred_on_ct, resampling_info = load_and_align_images(
        ct_path=ct_path,
        pet_path=pet_path,
        gt_path=gt_path,
        pred_path=pred_path,
    )

    gt_arr = clean_label_array(image_to_array(gt_on_ct))
    pred_arr = clean_label_array(image_to_array(pred_on_ct))

    metrics = compute_metrics(gt_arr, pred_arr)

    selected_slice = make_visualization(
        ct_img=ct_img,
        pet_img=pet_on_ct,
        gt_img=gt_on_ct,
        pred_img=pred_on_ct,
        patient_id=patient_id,
        output_path=output_path,
        metrics=metrics,
        slice_mode=args.slice_mode,
        ct_min=args.ct_min,
        ct_max=args.ct_max,
        pet_lower_percentile=args.pet_lower_percentile,
        pet_upper_percentile=args.pet_upper_percentile,
        pet_alpha=args.pet_alpha,
        pet_threshold=args.pet_threshold,
        roi_margin=args.roi_margin,
        include_caption=not args.no_caption,
    )

    metrics["patient_id"] = patient_id
    metrics["ct_path"] = str(ct_path)
    metrics["pet_path"] = str(pet_path)
    metrics["gt_path"] = str(gt_path)
    metrics["pred_path"] = str(pred_path)
    metrics["output_figure"] = str(output_path)
    metrics["selected_slice_z"] = selected_slice
    metrics["slice_mode"] = args.slice_mode
    metrics["resampling_info"] = resampling_info

    metrics_json = output_path.with_name(output_path.stem + "_metrics.json")
    metrics_txt = output_path.with_name(output_path.stem + "_metrics.txt")

    write_json(metrics_json, metrics)

    metrics_text = format_metrics_text(
        patient_id=patient_id,
        metrics=metrics,
        resampling_info=resampling_info,
    )

    write_text(metrics_txt, metrics_text)

    print("\nStep 06 Complete")
    print("=" * 80)
    print(f"Patient ID:       {patient_id}")
    print(f"Selected slice z: {selected_slice}")
    print(f"Figure:           {output_path}")
    print(f"Metrics JSON:     {metrics_json}")
    print(f"Metrics TXT:      {metrics_txt}")

    print("\nCombined Tumor Metrics")
    print("-" * 80)

    combined = metrics["combined_labels_1_and_2"]

    for key in ["Dice", "IoU", "Sensitivity", "Specificity", "Precision", "Accuracy"]:
        value = combined[key]

        if value is None or np.isnan(value):
            print(f"{key}: N/A")
        else:
            print(f"{key}: {value:.4f}")

    print("\nLabels")
    print("-" * 80)
    print(f"GT labels:         {metrics['labels_present_gt']}")
    print(f"Prediction labels: {metrics['labels_present_prediction']}")

    print("\nCaption")
    print("-" * 80)
    print(make_caption(patient_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PET/CT visualization with ground truth and nnU-Net prediction overlay."
    )

    parser.add_argument(
        "--ct",
        required=True,
        type=Path,
        help="Path to original CT image, e.g. PatientID__CT.nii.gz.",
    )

    parser.add_argument(
        "--pet",
        required=True,
        type=Path,
        help="Path to original PET image, e.g. PatientID__PT.nii.gz.",
    )

    parser.add_argument(
        "--gt",
        required=True,
        type=Path,
        help="Path to original ground-truth mask, e.g. PatientID.nii.gz.",
    )

    parser.add_argument(
        "--pred",
        required=True,
        type=Path,
        help="Path to nnU-Net prediction mask.",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output figure path, e.g. figures/MDA-258_pet_ct_gt_prediction_overlay.png.",
    )

    parser.add_argument(
        "--patient_id",
        default=None,
        type=str,
        help="Optional patient ID for title and caption.",
    )

    parser.add_argument(
        "--slice_mode",
        default="union",
        choices=["union", "gt", "pred", "pet", "middle"],
        help="Slice selection method. Default: union.",
    )

    parser.add_argument(
        "--ct_min",
        default=-200,
        type=float,
        help="CT window minimum HU. Default: -200.",
    )

    parser.add_argument(
        "--ct_max",
        default=300,
        type=float,
        help="CT window maximum HU. Default: 300.",
    )

    parser.add_argument(
        "--pet_lower_percentile",
        default=1,
        type=float,
        help="Lower PET percentile for display normalization. Default: 1.",
    )

    parser.add_argument(
        "--pet_upper_percentile",
        default=99.5,
        type=float,
        help="Upper PET percentile for display normalization. Default: 99.5.",
    )

    parser.add_argument(
        "--pet_alpha",
        default=0.45,
        type=float,
        help="PET overlay opacity. Default: 0.45.",
    )

    parser.add_argument(
        "--pet_threshold",
        default=0.08,
        type=float,
        help="Hide PET values below this normalized threshold. Default: 0.08.",
    )

    parser.add_argument(
        "--roi_margin",
        default=35,
        type=int,
        help="Pixel margin around GT/prediction union for zoomed ROI. Default: 35.",
    )

    parser.add_argument(
        "--no_caption",
        action="store_true",
        help="Do not embed caption text inside the saved figure.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_visualization(args)


if __name__ == "__main__":
    main()
