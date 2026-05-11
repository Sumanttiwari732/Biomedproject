#!/usr/bin/env python3
"""
Experiment 09: Preprocessing, normalization, and augmentation visualizations.

This script creates appendix-ready figures for the report:
    - Center-wise case distribution
    - Global CT HU histograms by center
    - Tumor-mask CT HU histograms by center
    - Tumor volume by center
    - Tumor mean HU by center
    - Slice thickness by center
    - ROI cropping example
    - Normalization strategy comparison
    - Candidate training-only augmentation examples

Example:
    python Project/09_preprocessing_normalization_augmentation_visuals.py \
      --hecktor_root "data/extracted/HECKTOR 2025 Training Data Defaced ALL" \
      --out preprocessing_visuals \
      --centers CHUM,CHUP,CHUS,HGJ,HMR,MDA,USZ \
      --max_cases_per_center 25
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from tqdm import tqdm

HU_MIN = -1000.0
HU_MAX = 400.0


def get_center(patient_id: str) -> str:
    return patient_id.split("-", 1)[0].upper()


def looks_like_patient_folder(path: Path) -> bool:
    return path.is_dir() and "-" in path.name


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-4) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol)
    )


def resample_mask_to_ct(mask_img: sitk.Image, ct_img: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        mask_img,
        ct_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def read_ct_and_mask(patient_dir: Path) -> tuple[sitk.Image, np.ndarray, np.ndarray]:
    patient_id = patient_dir.name
    ct_path = patient_dir / f"{patient_id}__CT.nii.gz"
    mask_path = patient_dir / f"{patient_id}.nii.gz"

    if not ct_path.exists():
        raise FileNotFoundError(f"Missing CT: {ct_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    ct_img = sitk.ReadImage(str(ct_path))
    mask_img = sitk.ReadImage(str(mask_path))

    if not same_geometry(ct_img, mask_img):
        mask_img = resample_mask_to_ct(mask_img, ct_img)

    ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    mask_arr = np.rint(mask_arr).astype(np.int16)
    binary_mask = (mask_arr > 0).astype(np.uint8)

    return ct_img, ct_arr, binary_mask


def collect_patient_dirs(root: Path, centers: set[str] | None) -> list[Path]:
    patient_dirs = sorted([p for p in root.iterdir() if looks_like_patient_folder(p)])
    if centers is not None:
        patient_dirs = [p for p in patient_dirs if get_center(p.name) in centers]
    return patient_dirs


def sample_cases_by_center(patient_dirs: list[Path], max_cases_per_center: int | None) -> list[Path]:
    if max_cases_per_center is None:
        return patient_dirs

    by_center: dict[str, list[Path]] = {}
    for patient_dir in patient_dirs:
        by_center.setdefault(get_center(patient_dir.name), []).append(patient_dir)

    selected: list[Path] = []
    for _, folders in sorted(by_center.items()):
        selected.extend(folders[:max_cases_per_center])
    return selected


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    return int(zmin), int(zmax), int(ymin), int(ymax), int(xmin), int(xmax)


def compute_case_stats(
    patient_dir: Path,
    rng: np.random.Generator,
    max_voxels: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    patient_id = patient_dir.name
    center = get_center(patient_id)
    ct_img, ct_arr, mask = read_ct_and_mask(patient_dir)

    spacing = ct_img.GetSpacing()
    bbox = bbox_from_mask(mask)

    global_vals = np.clip(ct_arr[np.isfinite(ct_arr)], HU_MIN, HU_MAX)
    if global_vals.size > max_voxels:
        global_vals = rng.choice(global_vals, size=max_voxels, replace=False)

    tumor_vals = ct_arr[mask > 0]
    tumor_vals = tumor_vals[np.isfinite(tumor_vals)]
    tumor_vals = np.clip(tumor_vals, HU_MIN, HU_MAX)
    if tumor_vals.size > max_voxels:
        tumor_vals_sample = rng.choice(tumor_vals, size=max_voxels, replace=False)
    else:
        tumor_vals_sample = tumor_vals

    voxel_volume_mm3 = float(spacing[0] * spacing[1] * spacing[2])
    tumor_voxels = int(mask.sum())

    row: dict[str, Any] = {
        "PatientID": patient_id,
        "Center": center,
        "SpacingX": spacing[0],
        "SpacingY": spacing[1],
        "SpacingZ": spacing[2],
        "SliceThickness": spacing[2],
        "ShapeZ": ct_arr.shape[0],
        "ShapeY": ct_arr.shape[1],
        "ShapeX": ct_arr.shape[2],
        "TumorVoxels": tumor_voxels,
        "TumorVolume_mm3": tumor_voxels * voxel_volume_mm3,
        "TumorVolume_cm3": tumor_voxels * voxel_volume_mm3 / 1000.0,
        "GlobalMeanHU": float(global_vals.mean()) if global_vals.size else np.nan,
        "GlobalStdHU": float(global_vals.std()) if global_vals.size else np.nan,
        "TumorMeanHU": float(tumor_vals.mean()) if tumor_vals.size else np.nan,
        "TumorStdHU": float(tumor_vals.std()) if tumor_vals.size else np.nan,
    }

    if bbox is not None:
        zmin, zmax, ymin, ymax, xmin, xmax = bbox
        row.update({"BBoxZ": zmax - zmin + 1, "BBoxY": ymax - ymin + 1, "BBoxX": xmax - xmin + 1})
    else:
        row.update({"BBoxZ": np.nan, "BBoxY": np.nan, "BBoxX": np.nan})

    return row, global_vals, tumor_vals_sample


def boxplot_by_center(df: pd.DataFrame, value_col: str, out_path: Path, title: str, ylabel: str) -> None:
    centers = sorted(df["Center"].dropna().unique())
    data = [pd.to_numeric(df.loc[df["Center"] == c, value_col], errors="coerce").dropna().values for c in centers]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, labels=centers, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel("Center")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def bar_case_distribution(patient_dirs: list[Path], out_path: Path) -> None:
    centers = [get_center(p.name) for p in patient_dirs]
    counts = pd.Series(centers).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Center-wise Case Distribution")
    ax.set_xlabel("Center")
    ax.set_ylabel("Number of Patient Folders")
    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(counts.values):
        ax.text(i, value, str(value), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def histogram_by_center(samples: dict[str, list[np.ndarray]], out_path: Path, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(HU_MIN, HU_MAX, 100)

    for center, arrays in sorted(samples.items()):
        vals = np.concatenate([a for a in arrays if a.size > 0]) if arrays else np.array([])
        if vals.size == 0:
            continue
        ax.hist(vals, bins=bins, density=True, histtype="step", linewidth=1.5, label=center)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def normalize_variants(ct_slice: np.ndarray) -> dict[str, np.ndarray]:
    x = ct_slice.astype(np.float32)
    clipped = np.clip(x, HU_MIN, HU_MAX)

    zscore = (clipped - clipped.mean()) / (clipped.std() + 1e-8)

    p25, p75 = np.percentile(clipped, [25, 75])
    iqr = p75 - p25
    if iqr < 1e-8:
        iqr = 1.0
    robust = (clipped - np.median(clipped)) / iqr

    minmax = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)

    return {
        "Raw": x,
        "HU clipped": clipped,
        "Z-score": zscore,
        "Robust": robust,
        "Min-max": minmax,
    }


def ct_window(x: np.ndarray, lo: float = -200.0, hi: float = 300.0) -> np.ndarray:
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)


def select_largest_tumor_slice(mask: np.ndarray) -> int:
    areas = mask.sum(axis=(1, 2))
    if areas.max() == 0:
        return mask.shape[0] // 2
    return int(np.argmax(areas))


def crop_around_mask(slice_img: np.ndarray, slice_mask: np.ndarray, margin: int = 40) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(slice_mask > 0)
    if coords.size == 0:
        return slice_img, slice_mask

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h, w = slice_img.shape

    y_min = max(0, int(y_min) - margin)
    y_max = min(h, int(y_max) + margin)
    x_min = max(0, int(x_min) - margin)
    x_max = min(w, int(x_max) + margin)

    return slice_img[y_min:y_max, x_min:x_max], slice_mask[y_min:y_max, x_min:x_max]


def save_roi_crop_example(patient_dir: Path, out_path: Path) -> None:
    _, ct_arr, mask = read_ct_and_mask(patient_dir)
    z = select_largest_tumor_slice(mask)
    ct_slice = ct_arr[z]
    mask_slice = mask[z]
    crop_img, crop_mask = crop_around_mask(ct_slice, mask_slice)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(ct_window(ct_slice), cmap="gray")
    axes[0].set_title("Full CT slice")
    axes[0].axis("off")

    axes[1].imshow(ct_window(ct_slice), cmap="gray")
    if mask_slice.any():
        axes[1].contour(mask_slice, levels=[0.5], linewidths=1.5)
    axes[1].set_title("CT + tumor mask")
    axes[1].axis("off")

    axes[2].imshow(ct_window(crop_img), cmap="gray")
    if crop_mask.any():
        axes[2].contour(crop_mask, levels=[0.5], linewidths=1.5)
    axes[2].set_title("Tumor ROI crop")
    axes[2].axis("off")

    fig.suptitle(f"ROI Cropping Example: {patient_dir.name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_normalization_comparison(patient_dir: Path, out_path: Path) -> None:
    _, ct_arr, mask = read_ct_and_mask(patient_dir)
    z = select_largest_tumor_slice(mask)
    ct_slice = ct_arr[z]
    mask_slice = mask[z]
    crop_img, _ = crop_around_mask(ct_slice, mask_slice)

    variants = normalize_variants(crop_img)
    fig, axes = plt.subplots(1, len(variants), figsize=(15, 4))

    for ax, (name, img) in zip(axes, variants.items()):
        if name in ["Raw", "HU clipped"]:
            ax.imshow(ct_window(img), cmap="gray")
        else:
            lo, hi = np.percentile(img, [1, 99])
            ax.imshow(np.clip((img - lo) / (hi - lo + 1e-8), 0, 1), cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(f"Normalization Comparison: {patient_dir.name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_augmentation_examples(patient_dir: Path, out_path: Path) -> None:
    _, ct_arr, mask = read_ct_and_mask(patient_dir)
    z = select_largest_tumor_slice(mask)
    ct_slice = ct_arr[z]
    mask_slice = mask[z].astype(bool)
    crop_img, crop_mask_uint8 = crop_around_mask(ct_slice, mask_slice.astype(np.uint8))
    crop_mask = crop_mask_uint8.astype(bool)

    eroded = ndimage.binary_erosion(crop_mask, iterations=1)
    dilated = ndimage.binary_dilation(crop_mask, iterations=1)
    blurred = ndimage.gaussian_filter(crop_mask.astype(float), sigma=1.0)
    perturbed = blurred > 0.35
    rotated = ndimage.rotate(crop_mask.astype(float), angle=8, reshape=False, order=0, mode="nearest") > 0.5

    variants = {
        "Original": crop_mask,
        "Erosion": eroded,
        "Dilation": dilated,
        "Contour perturbation": perturbed,
        "Rotation": rotated,
    }

    fig, axes = plt.subplots(1, len(variants), figsize=(15, 4))
    for ax, (name, mask_variant) in zip(axes, variants.items()):
        ax.imshow(ct_window(crop_img), cmap="gray")
        if mask_variant.any():
            ax.contour(mask_variant, levels=[0.5], linewidths=1.5)
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(f"Training-only Augmentation Examples: {patient_dir.name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate preprocessing, normalization, and augmentation visualizations.")
    parser.add_argument("--hecktor_root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--centers", default=None, type=str)
    parser.add_argument("--max_cases_per_center", default=25, type=int)
    parser.add_argument("--max_voxels_per_case", default=5000, type=int)
    parser.add_argument("--random_seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.hecktor_root.exists():
        raise FileNotFoundError(f"HECKTOR root not found: {args.hecktor_root}")

    args.out.mkdir(parents=True, exist_ok=True)

    centers = None
    if args.centers is not None:
        centers = {x.strip().upper() for x in args.centers.split(",") if x.strip()}

    all_patient_dirs = collect_patient_dirs(args.hecktor_root, centers)
    sampled_patient_dirs = sample_cases_by_center(all_patient_dirs, args.max_cases_per_center)
    rng = np.random.default_rng(args.random_seed)

    print("\nExperiment 9: Preprocessing Visualization")
    print("=" * 80)
    print(f"HECKTOR root: {args.hecktor_root}")
    print(f"All patient folders: {len(all_patient_dirs)}")
    print(f"Sampled folders for intensity statistics: {len(sampled_patient_dirs)}")
    print(f"Output: {args.out}")

    rows = []
    global_samples: dict[str, list[np.ndarray]] = {}
    tumor_samples: dict[str, list[np.ndarray]] = {}
    first_valid_case: Path | None = None

    for patient_dir in tqdm(sampled_patient_dirs, desc="Processing cases"):
        center = get_center(patient_dir.name)
        try:
            row, global_vals, tumor_vals = compute_case_stats(patient_dir, rng, args.max_voxels_per_case)
            rows.append(row)
            global_samples.setdefault(center, []).append(global_vals)
            tumor_samples.setdefault(center, []).append(tumor_vals)
            if first_valid_case is None and row.get("TumorVoxels", 0) > 0:
                first_valid_case = patient_dir
        except Exception as exc:
            rows.append({"PatientID": patient_dir.name, "Center": center, "Error": str(exc)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "preprocessing_case_statistics.csv", index=False)

    summary = (
        df.groupby("Center")
        .agg(
            Cases=("PatientID", "count"),
            MeanTumorVolume_cm3=("TumorVolume_cm3", "mean"),
            MeanTumorHU=("TumorMeanHU", "mean"),
            MeanSliceThickness=("SliceThickness", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.out / "preprocessing_center_summary.csv", index=False)

    bar_case_distribution(all_patient_dirs, args.out / "fig_center_case_distribution.png")
    histogram_by_center(global_samples, args.out / "fig_global_ct_hu_hist_by_center.png", "Global CT HU Distribution by Center", "HU")
    histogram_by_center(tumor_samples, args.out / "fig_tumor_ct_hu_hist_by_center.png", "Tumor-mask CT HU Distribution by Center", "HU")

    boxplot_by_center(df, "TumorVolume_cm3", args.out / "fig_tumor_volume_by_center.png", "Tumor Volume by Center", "Tumor volume (cm3)")
    boxplot_by_center(df, "TumorMeanHU", args.out / "fig_tumor_mean_hu_by_center.png", "Tumor Mean HU by Center", "Tumor mean HU")
    boxplot_by_center(df, "SliceThickness", args.out / "fig_slice_thickness_by_center.png", "Slice Thickness by Center", "Slice thickness (mm)")

    if first_valid_case is not None:
        save_roi_crop_example(first_valid_case, args.out / "fig_roi_cropping_example.png")
        save_normalization_comparison(first_valid_case, args.out / "fig_normalization_comparison.png")
        save_augmentation_examples(first_valid_case, args.out / "fig_augmentation_examples.png")

    print("\nExperiment 9 Complete")
    print("=" * 80)
    print(f"Statistics CSV: {args.out / 'preprocessing_case_statistics.csv'}")
    print(f"Center summary: {args.out / 'preprocessing_center_summary.csv'}")
    print(f"Figures saved in: {args.out}")


if __name__ == "__main__":
    main()
