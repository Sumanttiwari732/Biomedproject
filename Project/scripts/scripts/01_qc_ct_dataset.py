#!/usr/bin/env python3
"""
Step 01: Quality control and center-wise summary for HECKTOR CT/PET/mask data.

This script starts from the extracted HECKTOR 2025 patient-folder dataset and creates:

1. patient_inventory.csv
2. ct_dataset_raw.csv
3. ct_dataset_normalized.csv
4. abnormal_cases.csv
5. abnormal_by_center.csv
6. center_case_counts.csv
7. center_wise_ct_case_distribution.png
8. summary_raw.json
9. summary_normalized.json

Expected patient folder format:
    CENTER-XXX/
        CENTER-XXX__CT.nii.gz
        CENTER-XXX__PT.nii.gz
        CENTER-XXX.nii.gz

Example:
    python scripts/01_qc_ct_dataset.py \
        --root "data/extracted/HECKTOR 2025 Training Data Defaced ALL" \
        --out data/processed/step1_qc
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# -------------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------------
VALID_EXTENSIONS = [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"]

DEFAULT_HU_MIN = -1000
DEFAULT_HU_MAX = 400

DEFAULT_ABNORMAL_LOW = -2000
DEFAULT_ABNORMAL_HIGH = 5000

GEOMETRY_TOLERANCE = 1e-4


# -------------------------------------------------------------------------
# Basic helpers
# -------------------------------------------------------------------------
def has_valid_extension(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in VALID_EXTENSIONS)


def looks_like_patient_folder(folder: Path) -> bool:
    if not folder.is_dir():
        return False

    if "-" not in folder.name:
        return False

    center, case_number = folder.name.split("-", 1)

    return center.isalpha() and len(case_number) > 0


def extract_center(patient_id: str) -> str:
    return patient_id.split("-", 1)[0]


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    except Exception:
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
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


def find_dataset_root(root: Path) -> Path:
    """
    Accept either:
    1. The folder that directly contains patient folders.
    2. A parent folder that contains the dataset folder.
    """
    root = root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    direct_patient_dirs = [p for p in root.iterdir() if looks_like_patient_folder(p)]

    if direct_patient_dirs:
        return root

    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue

        patient_dirs = [p for p in candidate.iterdir() if looks_like_patient_folder(p)]

        if patient_dirs:
            return candidate

    for candidate in root.rglob("*"):
        if not candidate.is_dir():
            continue

        try:
            patient_dirs = [p for p in candidate.iterdir() if looks_like_patient_folder(p)]
        except PermissionError:
            continue

        if patient_dirs:
            return candidate

    raise RuntimeError(
        "Could not find HECKTOR patient folders. "
        "Expected folder names like CHUM-001, MDA-258, or USZ-001."
    )


def find_case_files(patient_folder: Path) -> dict[str, Path | None]:
    """
    Find CT, PET, mask, and optional RTDOSE files in one patient folder.
    """
    patient_id = patient_folder.name

    exact_ct = patient_folder / f"{patient_id}__CT.nii.gz"
    exact_pet = patient_folder / f"{patient_id}__PT.nii.gz"
    exact_mask = patient_folder / f"{patient_id}.nii.gz"
    exact_rtdose = patient_folder / f"{patient_id}__RTDOSE.nii.gz"

    files = [p for p in patient_folder.iterdir() if p.is_file() and has_valid_extension(p)]

    ct_candidates = []
    pet_candidates = []
    mask_candidates = []
    rtdose_candidates = []

    for file_path in files:
        name = file_path.name.lower()

        if "__ct" in name:
            ct_candidates.append(file_path)
        elif "__pt" in name:
            pet_candidates.append(file_path)
        elif "__rtdose" in name:
            rtdose_candidates.append(file_path)
        else:
            mask_candidates.append(file_path)

    ct_path = exact_ct if exact_ct.exists() else (sorted(ct_candidates)[0] if ct_candidates else None)
    pet_path = exact_pet if exact_pet.exists() else (sorted(pet_candidates)[0] if pet_candidates else None)
    mask_path = exact_mask if exact_mask.exists() else (sorted(mask_candidates)[0] if mask_candidates else None)
    rtdose_path = exact_rtdose if exact_rtdose.exists() else (sorted(rtdose_candidates)[0] if rtdose_candidates else None)

    return {
        "ct_path": ct_path,
        "pet_path": pet_path,
        "mask_path": mask_path,
        "rtdose_path": rtdose_path,
    }


# -------------------------------------------------------------------------
# Image helpers
# -------------------------------------------------------------------------
def read_image(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def image_to_array(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


def compute_stats(arr: np.ndarray) -> dict[str, float | None]:
    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "p01": None,
            "p99": None,
        }

    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p99": float(np.percentile(finite, 99)),
    }


def normalize_ct(arr: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, hu_min, hu_max)

    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return arr

    mean = float(np.mean(finite))
    std = float(np.std(finite))

    if std < 1e-6:
        std = 1.0

    return (arr - mean) / std


def compare_geometry(ct_img: sitk.Image, mask_img: sitk.Image) -> dict[str, Any]:
    ct_size = ct_img.GetSize()
    mask_size = mask_img.GetSize()

    ct_spacing = ct_img.GetSpacing()
    mask_spacing = mask_img.GetSpacing()

    ct_origin = ct_img.GetOrigin()
    mask_origin = mask_img.GetOrigin()

    ct_direction = ct_img.GetDirection()
    mask_direction = mask_img.GetDirection()

    same_size = ct_size == mask_size
    same_spacing = np.allclose(ct_spacing, mask_spacing, atol=GEOMETRY_TOLERANCE)
    same_origin = np.allclose(ct_origin, mask_origin, atol=GEOMETRY_TOLERANCE)
    same_direction = np.allclose(ct_direction, mask_direction, atol=GEOMETRY_TOLERANCE)

    geometry_match = same_size and same_spacing and same_origin and same_direction

    return {
        "geometry_match": bool(geometry_match),
        "same_size": bool(same_size),
        "same_spacing": bool(same_spacing),
        "same_origin": bool(same_origin),
        "same_direction": bool(same_direction),
        "ct_size_xyz": tuple(ct_size),
        "mask_size_xyz": tuple(mask_size),
        "ct_spacing_xyz": tuple(float(x) for x in ct_spacing),
        "mask_spacing_xyz": tuple(float(x) for x in mask_spacing),
        "ct_origin_xyz": tuple(float(x) for x in ct_origin),
        "mask_origin_xyz": tuple(float(x) for x in mask_origin),
    }


def mask_summary(mask_arr: np.ndarray) -> dict[str, Any]:
    rounded = np.rint(mask_arr).astype(np.int16)

    labels = sorted(int(x) for x in np.unique(rounded))
    label_set = set(labels)

    tumor_voxels = int(np.sum(rounded > 0))
    gtvp_voxels = int(np.sum(rounded == 1))
    gtvn_voxels = int(np.sum(rounded == 2))

    unexpected_labels = sorted(label_set.difference({0, 1, 2}))

    return {
        "mask_labels": labels,
        "unexpected_mask_labels": unexpected_labels,
        "mask_has_unexpected_labels": len(unexpected_labels) > 0,
        "mask_empty": tumor_voxels == 0,
        "tumor_voxels": tumor_voxels,
        "gtvp_label1_voxels": gtvp_voxels,
        "gtvn_label2_voxels": gtvn_voxels,
    }


def detect_abnormal_case(
    *,
    has_ct: bool,
    has_pet: bool,
    has_mask: bool,
    ct_read_error: str,
    pet_read_error: str,
    mask_read_error: str,
    ct_raw_stats: dict[str, float | None],
    mask_info: dict[str, Any],
    geometry_info: dict[str, Any],
    abnormal_low: float,
    abnormal_high: float,
) -> tuple[bool, str]:
    reasons = []

    if not has_ct:
        reasons.append("missing_ct")

    if not has_pet:
        reasons.append("missing_pet")

    if not has_mask:
        reasons.append("missing_mask")

    if ct_read_error:
        reasons.append("ct_read_error")

    if pet_read_error:
        reasons.append("pet_read_error")

    if mask_read_error:
        reasons.append("mask_read_error")

    ct_min = ct_raw_stats.get("min")
    ct_max = ct_raw_stats.get("max")
    ct_std = ct_raw_stats.get("std")

    if ct_min is None or ct_max is None:
        reasons.append("ct_no_finite_values")
    else:
        if ct_min < abnormal_low:
            reasons.append("low_intensity_outlier")
        if ct_max > abnormal_high:
            reasons.append("high_intensity_outlier")

    if ct_std is not None and ct_std < 1e-6:
        reasons.append("near_zero_ct_std")

    if mask_info.get("mask_empty", False):
        reasons.append("empty_mask")

    if mask_info.get("mask_has_unexpected_labels", False):
        reasons.append("unexpected_mask_labels")

    if geometry_info and not geometry_info.get("geometry_match", True):
        reasons.append("ct_mask_geometry_mismatch")

    is_abnormal = len(reasons) > 0
    reason = "|".join(reasons)

    return is_abnormal, reason


# -------------------------------------------------------------------------
# Processing
# -------------------------------------------------------------------------
def process_case(
    patient_folder: Path,
    hu_min: float,
    hu_max: float,
    abnormal_low: float,
    abnormal_high: float,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    patient_id = patient_folder.name
    center = extract_center(patient_id)

    files = find_case_files(patient_folder)

    ct_path = files["ct_path"]
    pet_path = files["pet_path"]
    mask_path = files["mask_path"]
    rtdose_path = files["rtdose_path"]

    has_ct = ct_path is not None and ct_path.exists()
    has_pet = pet_path is not None and pet_path.exists()
    has_mask = mask_path is not None and mask_path.exists()
    has_rtdose = rtdose_path is not None and rtdose_path.exists()

    ct_img = None
    mask_img = None

    ct_arr = None
    mask_arr = None

    ct_read_error = ""
    pet_read_error = ""
    mask_read_error = ""

    ct_raw_stats = compute_stats(np.array([], dtype=np.float32))
    ct_norm_stats = compute_stats(np.array([], dtype=np.float32))

    geometry_info: dict[str, Any] = {}
    mask_info: dict[str, Any] = {
        "mask_labels": [],
        "unexpected_mask_labels": [],
        "mask_has_unexpected_labels": False,
        "mask_empty": False,
        "tumor_voxels": None,
        "gtvp_label1_voxels": None,
        "gtvn_label2_voxels": None,
    }

    ct_spacing_xyz = None
    ct_size_xyz = None
    ct_shape_zyx = None
    slice_thickness_mm = None

    if has_ct:
        try:
            ct_img = read_image(ct_path)
            ct_arr = image_to_array(ct_img).astype(np.float32)

            ct_raw_stats = compute_stats(ct_arr)

            ct_norm_arr = normalize_ct(ct_arr, hu_min=hu_min, hu_max=hu_max)
            ct_norm_stats = compute_stats(ct_norm_arr)

            ct_spacing_xyz = tuple(float(x) for x in ct_img.GetSpacing())
            ct_size_xyz = tuple(int(x) for x in ct_img.GetSize())
            ct_shape_zyx = tuple(int(x) for x in ct_arr.shape)
            slice_thickness_mm = float(ct_img.GetSpacing()[2])

        except Exception as exc:
            ct_read_error = str(exc)

    if has_pet:
        try:
            _ = read_image(pet_path)
        except Exception as exc:
            pet_read_error = str(exc)

    if has_mask:
        try:
            mask_img = read_image(mask_path)
            mask_arr = image_to_array(mask_img)
            mask_info = mask_summary(mask_arr)

        except Exception as exc:
            mask_read_error = str(exc)

    if ct_img is not None and mask_img is not None:
        geometry_info = compare_geometry(ct_img, mask_img)

    is_abnormal, abnormal_reason = detect_abnormal_case(
        has_ct=has_ct,
        has_pet=has_pet,
        has_mask=has_mask,
        ct_read_error=ct_read_error,
        pet_read_error=pet_read_error,
        mask_read_error=mask_read_error,
        ct_raw_stats=ct_raw_stats,
        mask_info=mask_info,
        geometry_info=geometry_info,
        abnormal_low=abnormal_low,
        abnormal_high=abnormal_high,
    )

    complete_ct_pet_mask = has_ct and has_pet and has_mask

    usable_for_ct_training = (
        has_ct
        and has_mask
        and ct_read_error == ""
        and mask_read_error == ""
        and not mask_info.get("mask_empty", False)
    )

    base_record = {
        "patient_id": patient_id,
        "center": center,
        "patient_folder": str(patient_folder),
        "ct_path": str(ct_path) if has_ct else "",
        "pet_path": str(pet_path) if has_pet else "",
        "mask_path": str(mask_path) if has_mask else "",
        "rtdose_path": str(rtdose_path) if has_rtdose else "",
        "has_ct": has_ct,
        "has_pet": has_pet,
        "has_mask": has_mask,
        "has_rtdose": has_rtdose,
        "complete_ct_pet_mask": complete_ct_pet_mask,
        "usable_for_ct_training": usable_for_ct_training,
        "ct_read_error": ct_read_error,
        "pet_read_error": pet_read_error,
        "mask_read_error": mask_read_error,
        "ct_spacing_xyz": ct_spacing_xyz,
        "ct_size_xyz": ct_size_xyz,
        "ct_shape_zyx": ct_shape_zyx,
        "slice_thickness_mm": slice_thickness_mm,
        "is_abnormal": is_abnormal,
        "abnormal_reason": abnormal_reason,
    }

    base_record.update(mask_info)

    for key, value in geometry_info.items():
        base_record[key] = value

    raw_record = None
    norm_record = None

    if usable_for_ct_training:
        raw_record = {
            **base_record,
            "min": ct_raw_stats["min"],
            "max": ct_raw_stats["max"],
            "mean": ct_raw_stats["mean"],
            "std": ct_raw_stats["std"],
            "median": ct_raw_stats["median"],
            "p01": ct_raw_stats["p01"],
            "p99": ct_raw_stats["p99"],
        }

        norm_record = {
            **base_record,
            "min": ct_norm_stats["min"],
            "max": ct_norm_stats["max"],
            "mean": ct_norm_stats["mean"],
            "std": ct_norm_stats["std"],
            "median": ct_norm_stats["median"],
            "p01": ct_norm_stats["p01"],
            "p99": ct_norm_stats["p99"],
        }

    abnormal_record = None

    if is_abnormal:
        abnormal_record = {
            "patient_id": patient_id,
            "center": center,
            "patient_folder": str(patient_folder),
            "ct_path": str(ct_path) if has_ct else "",
            "pet_path": str(pet_path) if has_pet else "",
            "mask_path": str(mask_path) if has_mask else "",
            "abnormal_reason": abnormal_reason,
            "min_raw": ct_raw_stats["min"],
            "max_raw": ct_raw_stats["max"],
            "mean_raw": ct_raw_stats["mean"],
            "std_raw": ct_raw_stats["std"],
            "mask_empty": mask_info.get("mask_empty", False),
            "tumor_voxels": mask_info.get("tumor_voxels", None),
            "geometry_match": geometry_info.get("geometry_match", None),
        }

    return base_record, raw_record, norm_record, abnormal_record


def process_dataset(
    root: Path,
    out_dir: Path,
    hu_min: float,
    hu_max: float,
    abnormal_low: float,
    abnormal_high: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_root = find_dataset_root(root)

    patient_folders = sorted(
        [p for p in dataset_root.iterdir() if looks_like_patient_folder(p)]
    )

    if not patient_folders:
        raise RuntimeError(f"No patient folders found in: {dataset_root}")

    inventory_records = []
    raw_records = []
    norm_records = []
    abnormal_records = []

    print(f"Dataset root: {dataset_root}")
    print(f"Found {len(patient_folders)} patient folders")

    for patient_folder in tqdm(patient_folders, desc="Processing HECKTOR cases"):
        inventory, raw_record, norm_record, abnormal_record = process_case(
            patient_folder=patient_folder,
            hu_min=hu_min,
            hu_max=hu_max,
            abnormal_low=abnormal_low,
            abnormal_high=abnormal_high,
        )

        inventory_records.append(inventory)

        if raw_record is not None:
            raw_records.append(raw_record)

        if norm_record is not None:
            norm_records.append(norm_record)

        if abnormal_record is not None:
            abnormal_records.append(abnormal_record)

    inventory_df = pd.DataFrame(inventory_records)
    raw_df = pd.DataFrame(raw_records)
    norm_df = pd.DataFrame(norm_records)
    abnormal_df = pd.DataFrame(abnormal_records)

    return inventory_df, raw_df, norm_df, abnormal_df


# -------------------------------------------------------------------------
# Analysis and outputs
# -------------------------------------------------------------------------
def make_center_counts(inventory_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        inventory_df.groupby("center")
        .agg(
            patient_folders=("patient_id", "count"),
            complete_ct_pet_mask=("complete_ct_pet_mask", "sum"),
            usable_for_ct_training=("usable_for_ct_training", "sum"),
            missing_ct=("has_ct", lambda x: int((~x).sum())),
            missing_pet=("has_pet", lambda x: int((~x).sum())),
            missing_mask=("has_mask", lambda x: int((~x).sum())),
            abnormal_cases=("is_abnormal", "sum"),
        )
        .reset_index()
        .sort_values("center")
    )

    grouped["fraction_abnormal"] = (
        grouped["abnormal_cases"] / grouped["patient_folders"]
    ).round(4)

    return grouped


def make_abnormal_by_center(raw_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    if inventory_df.empty:
        return pd.DataFrame()

    result = (
        inventory_df.groupby("center")
        .agg(
            total_cases=("patient_id", "count"),
            abnormal_cases=("is_abnormal", "sum"),
        )
        .reset_index()
    )

    result["fraction_abnormal"] = (
        result["abnormal_cases"] / result["total_cases"]
    ).round(4)

    return result.sort_values("center")


def write_summary_json(
    df: pd.DataFrame,
    output_json: Path,
    stats_columns: list[str],
) -> None:
    if df.empty:
        summary = {"total_cases": 0}
    else:
        center_summary = {}

        for center, group in df.groupby("center"):
            center_summary[center] = {
                "cases": int(len(group)),
            }

            for col in stats_columns:
                if col not in group.columns:
                    continue

                values = pd.to_numeric(group[col], errors="coerce").dropna()

                if values.empty:
                    center_summary[center][col] = None
                else:
                    center_summary[center][col] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "median": float(values.median()),
                    }

        summary = {
            "total_cases": int(len(df)),
            "centers": center_summary,
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w") as f:
        json.dump(json_safe(summary), f, indent=2)


def save_center_barplot(center_counts: pd.DataFrame, output_png: Path) -> None:
    if center_counts.empty:
        return

    centers = center_counts["center"].tolist()
    counts = center_counts["patient_folders"].astype(int).tolist()
    total = int(sum(counts))

    fig, ax = plt.subplots(figsize=(13, 8))

    bars = ax.bar(centers, counts)

    ax.set_title(
        f"Center-wise CT Case Distribution",
        fontsize=22,
        fontweight="bold",
        pad=18,
    )
    ax.set_xlabel("Center", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Cases", fontsize=14, fontweight="bold")

    ymax = max(counts) if counts else 1
    ylim_top = int(np.ceil((ymax + 50) / 50) * 50)

    ax.set_ylim(0, ylim_top)
    ax.set_yticks(range(0, ylim_top + 1, 50))
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(5, ylim_top * 0.015),
            str(count),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.text(
        0.99,
        0.96,
        f"Total cases: {total}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "alpha": 0.12},
    )

    table_data = [
        centers,
        [str(x) for x in counts],
    ]

    row_labels = ["Center", "Number of Cases"]

    table = plt.table(
        cellText=table_data,
        rowLabels=row_labels,
        cellLoc="center",
        rowLoc="center",
        loc="bottom",
        bbox=[0.02, -0.34, 0.96, 0.20],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for cell in table.get_celld().values():
        cell.set_text_props(fontweight="bold")

    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.31)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_intensity_boxplot(raw_df: pd.DataFrame, output_png: Path) -> None:
    if raw_df.empty or "center" not in raw_df.columns or "mean" not in raw_df.columns:
        return

    centers = sorted(raw_df["center"].dropna().unique())

    data = [
        raw_df.loc[raw_df["center"] == center, "mean"].dropna().values
        for center in centers
    ]

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=centers)

    ax.set_title("Center-wise CT Mean Intensity Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Center")
    ax.set_ylabel("Mean CT Intensity")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_abnormal_barplot(abnormal_by_center: pd.DataFrame, output_png: Path) -> None:
    if abnormal_by_center.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    centers = abnormal_by_center["center"].tolist()
    counts = abnormal_by_center["abnormal_cases"].astype(int).tolist()

    bars = ax.bar(centers, counts)

    ax.set_title("Abnormal Case Count by Center", fontsize=14, fontweight="bold")
    ax.set_xlabel("Center")
    ax.set_ylabel("Number of Abnormal Cases")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ymax = max(counts) if counts else 1
    ax.set_ylim(0, ymax + max(2, ymax * 0.15))

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    out_dir: Path,
    inventory_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    norm_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    center_counts: pd.DataFrame,
    abnormal_by_center: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory_df.to_csv(out_dir / "patient_inventory.csv", index=False)
    raw_df.to_csv(out_dir / "ct_dataset_raw.csv", index=False)
    norm_df.to_csv(out_dir / "ct_dataset_normalized.csv", index=False)
    abnormal_df.to_csv(out_dir / "abnormal_cases.csv", index=False)
    center_counts.to_csv(out_dir / "center_case_counts.csv", index=False)
    abnormal_by_center.to_csv(out_dir / "abnormal_by_center.csv", index=False)

    write_summary_json(
        raw_df,
        out_dir / "summary_raw.json",
        stats_columns=["min", "max", "mean", "std", "median", "p01", "p99", "slice_thickness_mm"],
    )

    write_summary_json(
        norm_df,
        out_dir / "summary_normalized.json",
        stats_columns=["min", "max", "mean", "std", "median", "p01", "p99"],
    )

    save_center_barplot(
        center_counts,
        out_dir / "center_wise_ct_case_distribution.png",
    )

    save_intensity_boxplot(
        raw_df,
        out_dir / "ct_mean_intensity_by_center.png",
    )

    save_abnormal_barplot(
        abnormal_by_center,
        out_dir / "abnormal_case_count_by_center.png",
    )


def print_summary(
    inventory_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    center_counts: pd.DataFrame,
    out_dir: Path,
) -> None:
    total_folders = len(inventory_df)
    total_complete = int(inventory_df["complete_ct_pet_mask"].sum()) if not inventory_df.empty else 0
    total_usable = len(raw_df)
    total_abnormal = len(abnormal_df)

    print("\nQC summary")
    print("=" * 70)
    print(f"Total patient folders: {total_folders}")
    print(f"Complete CT/PET/mask cases: {total_complete}")
    print(f"Usable CT/mask training cases: {total_usable}")
    print(f"Abnormal or flagged cases: {total_abnormal}")

    print("\nCenter-wise summary")
    print("-" * 70)
    if not center_counts.empty:
        print(center_counts.to_string(index=False))

    print("\nSaved outputs")
    print("-" * 70)
    print(f"Output folder: {out_dir}")
    print(f"Patient inventory: {out_dir / 'patient_inventory.csv'}")
    print(f"Raw CT dataset CSV: {out_dir / 'ct_dataset_raw.csv'}")
    print(f"Normalized CT dataset CSV: {out_dir / 'ct_dataset_normalized.csv'}")
    print(f"Abnormal cases CSV: {out_dir / 'abnormal_cases.csv'}")
    print(f"Center counts CSV: {out_dir / 'center_case_counts.csv'}")
    print(f"Center bar plot: {out_dir / 'center_wise_ct_case_distribution.png'}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QC and center-wise summary for HECKTOR CT/PET/mask data."
    )

    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Path to extracted HECKTOR dataset folder.",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output folder for QC CSVs, JSON files, and plots.",
    )

    parser.add_argument(
        "--hu_min",
        default=DEFAULT_HU_MIN,
        type=float,
        help="Minimum HU value used for CT clipping.",
    )

    parser.add_argument(
        "--hu_max",
        default=DEFAULT_HU_MAX,
        type=float,
        help="Maximum HU value used for CT clipping.",
    )

    parser.add_argument(
        "--abnormal_low",
        default=DEFAULT_ABNORMAL_LOW,
        type=float,
        help="Raw CT minimum below this value is flagged as abnormal.",
    )

    parser.add_argument(
        "--abnormal_high",
        default=DEFAULT_ABNORMAL_HIGH,
        type=float,
        help="Raw CT maximum above this value is flagged as abnormal.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inventory_df, raw_df, norm_df, abnormal_df = process_dataset(
        root=args.root,
        out_dir=args.out,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        abnormal_low=args.abnormal_low,
        abnormal_high=args.abnormal_high,
    )

    center_counts = make_center_counts(inventory_df)
    abnormal_by_center = make_abnormal_by_center(raw_df, inventory_df)

    save_outputs(
        out_dir=args.out,
        inventory_df=inventory_df,
        raw_df=raw_df,
        norm_df=norm_df,
        abnormal_df=abnormal_df,
        center_counts=center_counts,
        abnormal_by_center=abnormal_by_center,
    )

    print_summary(
        inventory_df=inventory_df,
        raw_df=raw_df,
        abnormal_df=abnormal_df,
        center_counts=center_counts,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
