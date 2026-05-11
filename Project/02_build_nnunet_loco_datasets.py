#!/usr/bin/env python3
"""
Step 02: Build Leave-One-Center-Out nnU-Net datasets from HECKTOR QC CSV.

Input:
    data/processed/step1_qc/ct_dataset_raw.csv

Output:
    local_nnunet/nnUNet_raw/Dataset001_clean_raw_LOCO_CHUM/
    local_nnunet/nnUNet_raw/Dataset002_clean_raw_LOCO_CHUP/
    ...

Each dataset contains:
    imagesTr/
    labelsTr/
    imagesTs/
    labelsTs/
    dataset.json
    splits_final.json
    case_mapping.csv
    build_summary.json

Default output names match the trained model archive naming style:
    Dataset001_clean_raw_LOCO_CHUM
    Dataset002_clean_raw_LOCO_CHUP
    Dataset003_clean_raw_LOCO_CHUS
    Dataset004_clean_raw_LOCO_HGJ
    Dataset005_clean_raw_LOCO_HMR
    Dataset006_clean_raw_LOCO_MDA

Run from repo root:
    python Project/02_build_nnunet_loco_datasets.py \
        --qc_csv data/processed/step1_qc/ct_dataset_raw.csv \
        --out local_nnunet/nnUNet_raw \
        --label_mode binary \
        --crop_mode mask \
        --centers CHUM,CHUP,CHUS,HGJ,HMR,MDA
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_TARGET_SPACING = (1.0, 1.0, 1.0)

DEFAULT_HU_MIN = -1000.0
DEFAULT_HU_MAX = 400.0

DEFAULT_MARGIN = 20
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_RANDOM_SEED = 42


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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    value_str = str(value).strip().lower()

    return value_str in {"true", "1", "yes", "y", "t"}


def parse_centers(value: str | None) -> list[str] | None:
    if value is None:
        return None

    centers = [x.strip().upper() for x in value.split(",") if x.strip()]

    return centers if centers else None


def parse_spacing(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("--target_spacing requires exactly three values: x y z")

    return float(values[0]), float(values[1]), float(values[2])


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            "QC CSV is missing required columns: "
            + ", ".join(missing)
        )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "patient_id",
        "center",
        "ct_path",
        "mask_path",
    ]

    require_columns(df, required)

    df = df.copy()

    df["center"] = df["center"].astype(str).str.upper().str.strip()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["ct_path"] = df["ct_path"].astype(str).str.strip()
    df["mask_path"] = df["mask_path"].astype(str).str.strip()

    if "usable_for_ct_training" in df.columns:
        df["usable_for_ct_training"] = df["usable_for_ct_training"].apply(parse_bool)
    else:
        df["usable_for_ct_training"] = True

    if "is_abnormal" in df.columns:
        df["is_abnormal"] = df["is_abnormal"].apply(parse_bool)
    else:
        df["is_abnormal"] = False

    df = df[df["usable_for_ct_training"]].copy()

    df = df[df["ct_path"].str.len() > 0].copy()
    df = df[df["mask_path"].str.len() > 0].copy()

    df = df.sort_values(["center", "patient_id"]).reset_index(drop=True)

    return df


# ============================================================
# IMAGE PROCESSING HELPERS
# ============================================================

def read_image(path: Path) -> sitk.Image:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return sitk.ReadImage(str(path))


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    is_mask: bool,
    default_value: float = 0.0,
) -> sitk.Image:
    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    pixel_type = sitk.sitkUInt8 if is_mask else sitk.sitkFloat32

    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        interpolator,
        default_value,
        pixel_type,
    )


def make_reference_with_spacing(
    image: sitk.Image,
    target_spacing_xyz: tuple[float, float, float],
) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        max(1, int(round(original_size[i] * (original_spacing[i] / target_spacing_xyz[i]))))
        for i in range(3)
    ]

    reference = sitk.Image(new_size, sitk.sitkFloat32)
    reference.SetSpacing(target_spacing_xyz)
    reference.SetOrigin(image.GetOrigin())
    reference.SetDirection(image.GetDirection())

    return reference


def resample_to_spacing(
    image: sitk.Image,
    target_spacing_xyz: tuple[float, float, float],
    is_mask: bool,
    default_value: float = 0.0,
) -> sitk.Image:
    reference = make_reference_with_spacing(image, target_spacing_xyz)

    return resample_to_reference(
        image=image,
        reference=reference,
        is_mask=is_mask,
        default_value=default_value,
    )


def preprocess_ct_array(
    arr: np.ndarray,
    intensity_mode: str,
    hu_min: float,
    hu_max: float,
) -> np.ndarray:
    """
    intensity_mode:
        raw:
            Clip HU and divide by HU_MAX. This matches the earlier project workflow.
        normalized:
            Clip HU and apply z-score normalization.
        clipped:
            Clip HU only.
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, hu_min, hu_max)

    if intensity_mode == "raw":
        scale = hu_max if abs(hu_max) > 1e-6 else 1.0
        return arr / scale

    if intensity_mode == "normalized":
        finite = arr[np.isfinite(arr)]

        if finite.size == 0:
            return arr

        mean = float(np.mean(finite))
        std = float(np.std(finite))

        if std < 1e-6:
            std = 1.0

        return (arr - mean) / std

    if intensity_mode == "clipped":
        return arr

    raise ValueError(f"Unsupported intensity mode: {intensity_mode}")


def remap_mask_labels(mask_arr: np.ndarray, label_mode: str) -> np.ndarray:
    """
    HECKTOR labels:
        0 = background
        1 = GTVp
        2 = GTVn

    label_mode:
        binary:
            0 stays background.
            labels 1 and 2 become tumor label 1.
        multiclass:
            labels 0, 1, 2 are preserved.
            unexpected labels are set to 0.
    """
    mask_arr = np.rint(mask_arr).astype(np.uint8)

    if label_mode == "binary":
        return (mask_arr > 0).astype(np.uint8)

    if label_mode == "multiclass":
        return np.where(np.isin(mask_arr, [0, 1, 2]), mask_arr, 0).astype(np.uint8)

    raise ValueError(f"Unsupported label mode: {label_mode}")


def get_bbox_from_mask(mask_arr: np.ndarray, margin: int) -> tuple[int, int, int, int, int, int] | None:
    """
    Get bounding box from foreground mask.

    Array shape is z, y, x.

    Returns:
        z_min, z_max_exclusive, y_min, y_max_exclusive, x_min, x_max_exclusive
    """
    coords = np.argwhere(mask_arr > 0)

    if coords.size == 0:
        return None

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    z_min = max(0, int(z_min) - margin)
    y_min = max(0, int(y_min) - margin)
    x_min = max(0, int(x_min) - margin)

    z_max = min(mask_arr.shape[0], int(z_max) + margin + 1)
    y_max = min(mask_arr.shape[1], int(y_max) + margin + 1)
    x_max = min(mask_arr.shape[2], int(x_max) + margin + 1)

    return z_min, z_max, y_min, y_max, x_min, x_max


def crop_array(
    arr: np.ndarray,
    bbox: tuple[int, int, int, int, int, int],
) -> np.ndarray:
    z_min, z_max, y_min, y_max, x_min, x_max = bbox
    return arr[z_min:z_max, y_min:y_max, x_min:x_max]


def array_to_image_like(
    arr: np.ndarray,
    reference: sitk.Image,
    is_mask: bool,
    crop_bbox: tuple[int, int, int, int, int, int] | None = None,
) -> sitk.Image:
    """
    Convert z,y,x NumPy array to SimpleITK image and preserve geometry.

    If crop_bbox is provided, update origin to match the crop start index.
    """
    arr = np.asarray(arr)

    if is_mask:
        image = sitk.GetImageFromArray(arr.astype(np.uint8))
    else:
        image = sitk.GetImageFromArray(arr.astype(np.float32))

    image.SetSpacing(reference.GetSpacing())
    image.SetDirection(reference.GetDirection())

    if crop_bbox is None:
        image.SetOrigin(reference.GetOrigin())
    else:
        z_min, _, y_min, _, x_min, _ = crop_bbox
        new_origin = reference.TransformIndexToPhysicalPoint((int(x_min), int(y_min), int(z_min)))
        image.SetOrigin(new_origin)

    return image


def process_one_case(
    row: pd.Series,
    out_image_path: Path,
    out_label_path: Path,
    target_spacing_xyz: tuple[float, float, float],
    intensity_mode: str,
    label_mode: str,
    crop_mode: str,
    margin: int,
    hu_min: float,
    hu_max: float,
) -> dict[str, Any]:
    patient_id = row["patient_id"]
    ct_path = Path(row["ct_path"])
    mask_path = Path(row["mask_path"])

    ct_img = read_image(ct_path)
    mask_img = read_image(mask_path)

    # First align mask to CT grid.
    mask_on_ct = resample_to_reference(
        image=mask_img,
        reference=ct_img,
        is_mask=True,
        default_value=0,
    )

    # Resample CT to target spacing.
    ct_resampled = resample_to_spacing(
        image=ct_img,
        target_spacing_xyz=target_spacing_xyz,
        is_mask=False,
        default_value=float(hu_min),
    )

    # Resample mask to the resampled CT grid.
    mask_resampled = resample_to_reference(
        image=mask_on_ct,
        reference=ct_resampled,
        is_mask=True,
        default_value=0,
    )

    ct_arr = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
    mask_arr = sitk.GetArrayFromImage(mask_resampled)
    mask_arr = remap_mask_labels(mask_arr, label_mode=label_mode)

    if np.sum(mask_arr > 0) == 0:
        return {
            "patient_id": patient_id,
            "status": "skipped",
            "reason": "empty_mask_after_resampling",
        }

    crop_bbox = None

    if crop_mode == "mask":
        crop_bbox = get_bbox_from_mask(mask_arr, margin=margin)

        if crop_bbox is None:
            return {
                "patient_id": patient_id,
                "status": "skipped",
                "reason": "empty_mask_for_crop",
            }

        ct_arr = crop_array(ct_arr, crop_bbox)
        mask_arr = crop_array(mask_arr, crop_bbox)

    elif crop_mode == "none":
        crop_bbox = None

    else:
        raise ValueError(f"Unsupported crop mode: {crop_mode}")

    ct_arr = preprocess_ct_array(
        arr=ct_arr,
        intensity_mode=intensity_mode,
        hu_min=hu_min,
        hu_max=hu_max,
    )

    ct_out = array_to_image_like(
        arr=ct_arr,
        reference=ct_resampled,
        is_mask=False,
        crop_bbox=crop_bbox,
    )

    mask_out = array_to_image_like(
        arr=mask_arr,
        reference=ct_resampled,
        is_mask=True,
        crop_bbox=crop_bbox,
    )

    out_image_path.parent.mkdir(parents=True, exist_ok=True)
    out_label_path.parent.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(ct_out, str(out_image_path))
    sitk.WriteImage(mask_out, str(out_label_path))

    labels_present = sorted(int(x) for x in np.unique(mask_arr))

    return {
        "patient_id": patient_id,
        "status": "written",
        "reason": "",
        "input_ct_path": str(ct_path),
        "input_mask_path": str(mask_path),
        "output_image_path": str(out_image_path),
        "output_label_path": str(out_label_path),
        "output_shape_zyx": tuple(int(x) for x in ct_arr.shape),
        "labels_present": labels_present,
        "foreground_voxels": int(np.sum(mask_arr > 0)),
        "spacing_xyz": tuple(float(x) for x in ct_out.GetSpacing()),
        "crop_bbox_zyx": crop_bbox,
    }


# ============================================================
# SPLIT HELPERS
# ============================================================

def make_train_val_split(
    mapping_df: pd.DataFrame,
    val_fraction: float,
    random_seed: int,
) -> dict[str, list[str]]:
    """
    Create internal training/validation split from training centers only.

    This is not the held-out test center.
    The test center remains in imagesTs/labelsTs.
    """
    train_rows = mapping_df[mapping_df["split"] == "train"].copy()

    if train_rows.empty:
        return {"train": [], "val": []}

    rng = random.Random(random_seed)

    train_ids = []
    val_ids = []

    for center, center_group in train_rows.groupby("center"):
        case_ids = center_group["nnunet_case_id"].tolist()
        rng.shuffle(case_ids)

        if len(case_ids) <= 1:
            train_ids.extend(case_ids)
            continue

        n_val = int(round(len(case_ids) * val_fraction))
        n_val = max(1, n_val)
        n_val = min(n_val, len(case_ids) - 1)

        val_ids.extend(case_ids[:n_val])
        train_ids.extend(case_ids[n_val:])

    train_ids = sorted(train_ids)
    val_ids = sorted(val_ids)

    return {
        "train": train_ids,
        "val": val_ids,
    }


def write_dataset_json(
    dataset_dir: Path,
    label_mode: str,
    num_training: int,
) -> None:
    if label_mode == "binary":
        labels = {
            "background": 0,
            "tumor": 1,
        }
    elif label_mode == "multiclass":
        labels = {
            "background": 0,
            "GTVp": 1,
            "GTVn": 2,
        }
    else:
        raise ValueError(f"Unsupported label mode: {label_mode}")

    dataset_json = {
        "channel_names": {
            "0": "CT",
        },
        "labels": labels,
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }

    with (dataset_dir / "dataset.json").open("w") as f:
        json.dump(dataset_json, f, indent=2)


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(json_safe(data), f, indent=2)


# ============================================================
# DATASET BUILDING
# ============================================================

def build_one_loco_dataset(
    df: pd.DataFrame,
    test_center: str,
    dataset_id: int,
    out_root: Path,
    variant: str,
    intensity_mode: str,
    label_mode: str,
    crop_mode: str,
    target_spacing_xyz: tuple[float, float, float],
    margin: int,
    hu_min: float,
    hu_max: float,
    val_fraction: float,
    random_seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    dataset_name = f"Dataset{dataset_id:03d}_{variant}_{intensity_mode}_LOCO_{test_center}"
    dataset_dir = out_root / dataset_name

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    if dataset_dir.exists() and overwrite:
        shutil.rmtree(dataset_dir)

    for folder in [images_tr, labels_tr, images_ts, labels_ts]:
        folder.mkdir(parents=True, exist_ok=True)

    train_df = df[df["center"] != test_center].copy()
    test_df = df[df["center"] == test_center].copy()

    if train_df.empty:
        raise RuntimeError(f"No training cases for LOCO center {test_center}")

    if test_df.empty:
        raise RuntimeError(f"No test cases for LOCO center {test_center}")

    mapping_rows = []
    skipped_rows = []

    case_counter = 0

    print(f"\nBuilding {dataset_name}")
    print("=" * 80)
    print(f"Train cases before processing: {len(train_df)}")
    print(f"Test cases before processing:  {len(test_df)}")

    # ----------------------------
    # Training cases
    # ----------------------------
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"{dataset_name} train"):
        nnunet_case_id = f"case_{case_counter:04d}"
        case_counter += 1

        image_path = images_tr / f"{nnunet_case_id}_0000.nii.gz"
        label_path = labels_tr / f"{nnunet_case_id}.nii.gz"

        result = process_one_case(
            row=row,
            out_image_path=image_path,
            out_label_path=label_path,
            target_spacing_xyz=target_spacing_xyz,
            intensity_mode=intensity_mode,
            label_mode=label_mode,
            crop_mode=crop_mode,
            margin=margin,
            hu_min=hu_min,
            hu_max=hu_max,
        )

        if result["status"] == "written":
            mapping_rows.append(
                {
                    "nnunet_case_id": nnunet_case_id,
                    "split": "train",
                    "patient_id": row["patient_id"],
                    "center": row["center"],
                    "ct_path": row["ct_path"],
                    "mask_path": row["mask_path"],
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                }
            )
        else:
            skipped_rows.append(
                {
                    "nnunet_case_id": nnunet_case_id,
                    "split": "train",
                    "patient_id": row["patient_id"],
                    "center": row["center"],
                    "reason": result["reason"],
                }
            )

    # ----------------------------
    # Held-out test center cases
    # ----------------------------
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{dataset_name} test"):
        nnunet_case_id = f"case_{case_counter:04d}"
        case_counter += 1

        image_path = images_ts / f"{nnunet_case_id}_0000.nii.gz"
        label_path = labels_ts / f"{nnunet_case_id}.nii.gz"

        result = process_one_case(
            row=row,
            out_image_path=image_path,
            out_label_path=label_path,
            target_spacing_xyz=target_spacing_xyz,
            intensity_mode=intensity_mode,
            label_mode=label_mode,
            crop_mode=crop_mode,
            margin=margin,
            hu_min=hu_min,
            hu_max=hu_max,
        )

        if result["status"] == "written":
            mapping_rows.append(
                {
                    "nnunet_case_id": nnunet_case_id,
                    "split": "test",
                    "patient_id": row["patient_id"],
                    "center": row["center"],
                    "ct_path": row["ct_path"],
                    "mask_path": row["mask_path"],
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                }
            )
        else:
            skipped_rows.append(
                {
                    "nnunet_case_id": nnunet_case_id,
                    "split": "test",
                    "patient_id": row["patient_id"],
                    "center": row["center"],
                    "reason": result["reason"],
                }
            )

    mapping_df = pd.DataFrame(mapping_rows)
    skipped_df = pd.DataFrame(skipped_rows)

    train_written = mapping_df[mapping_df["split"] == "train"]
    test_written = mapping_df[mapping_df["split"] == "test"]

    write_dataset_json(
        dataset_dir=dataset_dir,
        label_mode=label_mode,
        num_training=len(train_written),
    )

    split_dict = make_train_val_split(
        mapping_df=mapping_df,
        val_fraction=val_fraction,
        random_seed=random_seed,
    )

    write_json(dataset_dir / "splits_final.json", [split_dict])

    mapping_df.to_csv(dataset_dir / "case_mapping.csv", index=False)

    if skipped_df.empty:
        skipped_df = pd.DataFrame(columns=["nnunet_case_id", "split", "patient_id", "center", "reason"])

    skipped_df.to_csv(dataset_dir / "skipped_cases.csv", index=False)

    build_summary = {
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "dataset_dir": str(dataset_dir),
        "variant": variant,
        "intensity_mode": intensity_mode,
        "label_mode": label_mode,
        "crop_mode": crop_mode,
        "target_spacing_xyz": target_spacing_xyz,
        "margin": margin,
        "hu_min": hu_min,
        "hu_max": hu_max,
        "test_center": test_center,
        "num_train_written": int(len(train_written)),
        "num_test_written": int(len(test_written)),
        "num_skipped": int(len(skipped_df)),
        "internal_val_fraction": val_fraction,
        "internal_train_cases": int(len(split_dict["train"])),
        "internal_val_cases": int(len(split_dict["val"])),
        "note": (
            "labelsTs is stored for local evaluation only. "
            "nnU-Net prediction uses imagesTs. "
            "If using splits_final.json for fold 0, copy it to the matching "
            "nnUNet_preprocessed dataset folder after planning/preprocessing."
        ),
    }

    write_json(dataset_dir / "build_summary.json", build_summary)

    print(f"Finished {dataset_name}")
    print(f"  Training cases written: {len(train_written)}")
    print(f"  Test cases written:     {len(test_written)}")
    print(f"  Skipped cases:          {len(skipped_df)}")
    print(f"  Dataset folder:         {dataset_dir}")

    return build_summary


def select_variant_dataframe(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "full":
        return df.copy()

    if variant == "clean":
        return df[df["is_abnormal"] == False].copy()

    raise ValueError(f"Unsupported variant: {variant}")


def build_all_datasets(
    qc_csv: Path,
    out_root: Path,
    dataset_id_start: int,
    centers: list[str] | None,
    variants: list[str],
    intensity_modes: list[str],
    label_mode: str,
    crop_mode: str,
    target_spacing_xyz: tuple[float, float, float],
    margin: int,
    hu_min: float,
    hu_max: float,
    val_fraction: float,
    random_seed: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    if not qc_csv.exists():
        raise FileNotFoundError(f"QC CSV not found: {qc_csv}")

    df = pd.read_csv(qc_csv)
    df = clean_dataframe(df)

    if df.empty:
        raise RuntimeError("No usable rows found in QC CSV.")

    if centers is None:
        centers = sorted(df["center"].unique().tolist())
    else:
        centers = [center.upper() for center in centers]

    available_centers = set(df["center"].unique().tolist())
    missing_centers = [center for center in centers if center not in available_centers]

    if missing_centers:
        raise ValueError(
            "Requested centers not found in QC CSV: "
            + ", ".join(missing_centers)
        )

    out_root.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    dataset_id = dataset_id_start

    print("\nLOCO Dataset Build Configuration")
    print("=" * 80)
    print(f"QC CSV: {qc_csv}")
    print(f"Output nnU-Net raw folder: {out_root}")
    print(f"Centers: {', '.join(centers)}")
    print(f"Variants: {', '.join(variants)}")
    print(f"Intensity modes: {', '.join(intensity_modes)}")
    print(f"Label mode: {label_mode}")
    print(f"Crop mode: {crop_mode}")
    print(f"Target spacing xyz: {target_spacing_xyz}")
    print(f"Start dataset ID: {dataset_id_start}")

    if crop_mode == "mask":
        print("\nWARNING")
        print("-" * 80)
        print(
            "crop_mode='mask' uses the ground-truth mask to crop both train and test cases. "
            "This reproduces the earlier cropped workflow but should not be used for strict "
            "unknown test-set deployment because it uses the target mask location."
        )

    for variant in variants:
        df_variant = select_variant_dataframe(df, variant=variant)

        if df_variant.empty:
            print(f"Skipping variant {variant}: no cases available.")
            continue

        for intensity_mode in intensity_modes:
            for center in centers:
                if center not in set(df_variant["center"].unique()):
                    print(f"Skipping {center} for {variant}/{intensity_mode}: no cases available.")
                    continue

                summary = build_one_loco_dataset(
                    df=df_variant,
                    test_center=center,
                    dataset_id=dataset_id,
                    out_root=out_root,
                    variant=variant,
                    intensity_mode=intensity_mode,
                    label_mode=label_mode,
                    crop_mode=crop_mode,
                    target_spacing_xyz=target_spacing_xyz,
                    margin=margin,
                    hu_min=hu_min,
                    hu_max=hu_max,
                    val_fraction=val_fraction,
                    random_seed=random_seed,
                    overwrite=overwrite,
                )

                all_summaries.append(summary)
                dataset_id += 1

    master_summary = {
        "qc_csv": str(qc_csv),
        "out_root": str(out_root),
        "dataset_id_start": dataset_id_start,
        "num_datasets_built": len(all_summaries),
        "datasets": all_summaries,
    }

    write_json(out_root / "loco_build_master_summary.json", master_summary)

    summary_df = pd.DataFrame(all_summaries)

    if not summary_df.empty:
        summary_df.to_csv(out_root / "loco_build_master_summary.csv", index=False)

    return all_summaries


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Leave-One-Center-Out nnU-Net datasets from HECKTOR QC CSV."
    )

    parser.add_argument(
        "--qc_csv",
        required=True,
        type=Path,
        help="Path to data/processed/step1_qc/ct_dataset_raw.csv.",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output nnU-Net raw folder, usually local_nnunet/nnUNet_raw.",
    )

    parser.add_argument(
        "--dataset_id_start",
        default=1,
        type=int,
        help="First nnU-Net dataset ID.",
    )

    parser.add_argument(
        "--centers",
        default=None,
        type=str,
        help="Optional comma-separated centers, e.g. CHUM,CHUP,CHUS,HGJ,HMR,MDA.",
    )

    parser.add_argument(
        "--variant",
        default="clean",
        choices=["clean", "full", "both"],
        help="clean excludes flagged abnormal cases; full includes all usable cases.",
    )

    parser.add_argument(
        "--intensity_mode",
        default="raw",
        choices=["raw", "normalized", "clipped", "both"],
        help="raw=clip/divide by HU_MAX, normalized=clip+zscore, clipped=clip only.",
    )

    parser.add_argument(
        "--label_mode",
        default="binary",
        choices=["binary", "multiclass"],
        help="binary combines labels 1 and 2; multiclass preserves labels 1 and 2.",
    )

    parser.add_argument(
        "--crop_mode",
        default="mask",
        choices=["mask", "none"],
        help="mask crops around GT mask; none keeps full image.",
    )

    parser.add_argument(
        "--target_spacing",
        nargs=3,
        type=float,
        default=list(DEFAULT_TARGET_SPACING),
        help="Target spacing x y z. Default: 1.0 1.0 1.0",
    )

    parser.add_argument(
        "--margin",
        default=DEFAULT_MARGIN,
        type=int,
        help="Voxel margin around tumor when crop_mode=mask.",
    )

    parser.add_argument(
        "--hu_min",
        default=DEFAULT_HU_MIN,
        type=float,
        help="Minimum HU for CT clipping.",
    )

    parser.add_argument(
        "--hu_max",
        default=DEFAULT_HU_MAX,
        type=float,
        help="Maximum HU for CT clipping.",
    )

    parser.add_argument(
        "--val_fraction",
        default=DEFAULT_VAL_FRACTION,
        type=float,
        help="Internal validation fraction from training centers.",
    )

    parser.add_argument(
        "--random_seed",
        default=DEFAULT_RANDOM_SEED,
        type=int,
        help="Random seed for internal validation split.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing DatasetXXX folders before rebuilding.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    centers = parse_centers(args.centers)

    if args.variant == "both":
        variants = ["clean", "full"]
    else:
        variants = [args.variant]

    if args.intensity_mode == "both":
        intensity_modes = ["raw", "normalized"]
    else:
        intensity_modes = [args.intensity_mode]

    target_spacing_xyz = parse_spacing(args.target_spacing)

    summaries = build_all_datasets(
        qc_csv=args.qc_csv,
        out_root=args.out,
        dataset_id_start=args.dataset_id_start,
        centers=centers,
        variants=variants,
        intensity_modes=intensity_modes,
        label_mode=args.label_mode,
        crop_mode=args.crop_mode,
        target_spacing_xyz=target_spacing_xyz,
        margin=args.margin,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        val_fraction=args.val_fraction,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )

    print("\nStep 02 Complete")
    print("=" * 80)
    print(f"Datasets built: {len(summaries)}")
    print(f"Output root: {args.out}")
    print(f"Master summary JSON: {args.out / 'loco_build_master_summary.json'}")
    print(f"Master summary CSV:  {args.out / 'loco_build_master_summary.csv'}")

    print("\nNext step:")
    print(
        "python Project/03_make_quartz_slurm.py "
        "--nnunet_raw local_nnunet/nnUNet_raw "
        "--out quartz/slurm_jobs "
        "--configuration 3d_fullres "
        "--fold 0"
    )


if __name__ == "__main__":
    main()
