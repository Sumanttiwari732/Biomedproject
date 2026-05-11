#!/usr/bin/env python3
"""
Experiment 07: Extract radiomic features from nnU-Net test images.

This script extracts PyRadiomics features from nnU-Net-preprocessed CT images
using two possible mask sources:

1. Ground-truth masks from labelsTs
2. nnU-Net prediction masks from predictions_latest

All nonzero labels are binarized:
    0 = background
    >0 = tumor foreground

Example:
    python Project/07_extract_radiomics_features.py \
      --nnunet_raw local_nnunet/nnunet_datasets_final \
      --pred_root local_nnunet/predictions_latest \
      --out radiomics_outputs_nnunet \
      --mask_source both \
      --metrics_csv local_nnunet/results/per_case_dice.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

try:
    from radiomics import featureextractor
except ImportError as exc:
    raise ImportError(
        "PyRadiomics is required for this script. Install with: pip install pyradiomics==3.0.1"
    ) from exc


def parse_dataset_id(dataset_name: str) -> int:
    match = re.match(r"^Dataset(\d{3})_.+", dataset_name)
    if not match:
        raise ValueError(f"Invalid nnU-Net dataset name: {dataset_name}")
    return int(match.group(1))


def parse_test_center(dataset_name: str) -> str:
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
    dataset_upper = dataset_name.upper()

    for item in filter_items:
        item_upper = item.upper()
        if item == dataset_name:
            return True
        if item == dataset_id:
            return True
        if item_upper in dataset_upper:
            return True

    return False


def find_dataset_dirs(nnunet_raw: Path, dataset_filter: set[str] | None) -> list[Path]:
    if not nnunet_raw.exists():
        raise FileNotFoundError(f"nnU-Net raw folder not found: {nnunet_raw}")

    dataset_dirs: list[Path] = []

    for path in sorted(nnunet_raw.iterdir()):
        if not path.is_dir():
            continue
        if not re.match(r"^Dataset\d{3}_.+", path.name):
            continue
        if not (path / "imagesTs").exists():
            continue
        if not (path / "labelsTs").exists():
            continue
        if not dataset_matches_filter(path.name, dataset_filter):
            continue
        dataset_dirs.append(path)

    if not dataset_dirs:
        raise RuntimeError("No matching nnU-Net LOCO dataset folders found.")

    return dataset_dirs


def load_case_mapping(dataset_dir: Path) -> pd.DataFrame:
    mapping_path = dataset_dir / "case_mapping.csv"
    if not mapping_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(mapping_path)

    possible_case_cols = ["nnunet_case_id", "case_id", "Case", "case"]
    case_col = next((c for c in possible_case_cols if c in df.columns), None)
    if case_col is None:
        return pd.DataFrame()

    df = df.rename(columns={case_col: "nnunet_case_id"})
    df["nnunet_case_id"] = df["nnunet_case_id"].astype(str)
    return df


def get_mapping_row(mapping_df: pd.DataFrame, case_id: str, fallback_center: str) -> dict[str, Any]:
    if mapping_df.empty:
        return {
            "PatientID": "",
            "Center": fallback_center,
            "Original_CT_Path": "",
            "Original_Mask_Path": "",
        }

    row = mapping_df[mapping_df["nnunet_case_id"] == case_id]
    if row.empty:
        return {
            "PatientID": "",
            "Center": fallback_center,
            "Original_CT_Path": "",
            "Original_Mask_Path": "",
        }

    row = row.iloc[0]
    return {
        "PatientID": row.get("patient_id", row.get("PatientID", "")),
        "Center": row.get("center", row.get("Center", fallback_center)),
        "Original_CT_Path": row.get("ct_path", row.get("CT_Path", "")),
        "Original_Mask_Path": row.get("mask_path", row.get("Mask_Path", "")),
    }


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-4) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol)
    )


def resample_mask_to_reference(mask_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        mask_img,
        ref_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )


def make_binary_mask_file(image_path: Path, mask_path: Path, out_path: Path) -> tuple[bool, int]:
    """Convert any nonzero mask label to tumor foreground label 1."""
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))

    if not same_geometry(image, mask):
        mask = resample_mask_to_reference(mask, image)

    arr = sitk.GetArrayFromImage(mask)
    arr = np.rint(arr).astype(np.int16)
    binary = (arr > 0).astype(np.uint8)

    foreground_voxels = int(binary.sum())
    if foreground_voxels == 0:
        return False, 0

    binary_img = sitk.GetImageFromArray(binary)
    binary_img.CopyInformation(image)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(binary_img, str(out_path))

    return True, foreground_voxels


def make_extractor(bin_width: float, correct_mask: bool) -> featureextractor.RadiomicsFeatureExtractor:
    settings = {
        "binWidth": bin_width,
        "label": 1,
        "correctMask": correct_mask,
        "force2D": False,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")

    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("firstorder")
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("glcm")
    extractor.enableFeatureClassByName("glrlm")
    extractor.enableFeatureClassByName("glszm")
    extractor.enableFeatureClassByName("gldm")
    extractor.enableFeatureClassByName("ngtdm")

    return extractor


def clean_feature_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def extract_one_feature_row(
    extractor: featureextractor.RadiomicsFeatureExtractor,
    image_path: Path,
    mask_path: Path,
    keep_diagnostics: bool,
) -> dict[str, Any]:
    features = extractor.execute(str(image_path), str(mask_path), label=1)

    cleaned = {}
    for key, value in features.items():
        key = str(key)
        if key.startswith("diagnostics_") and not keep_diagnostics:
            continue
        cleaned[key] = clean_feature_value(value)

    return cleaned


def extract_features_for_dataset(
    dataset_dir: Path,
    pred_root: Path,
    out_dir: Path,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    mask_source: str,
    keep_diagnostics: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_name = dataset_dir.name
    dataset_id = parse_dataset_id(dataset_name)
    fallback_center = parse_test_center(dataset_name)

    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"
    pred_dir = pred_root / dataset_name

    mapping_df = load_case_mapping(dataset_dir)

    gt_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    image_files = sorted(images_ts.glob("*_0000.nii.gz"))

    print(f"\nExtracting radiomics for {dataset_name}")
    print("=" * 80)
    print(f"Test images: {len(image_files)}")

    for image_path in tqdm(image_files, desc=dataset_name):
        case_id = image_path.name.replace("_0000.nii.gz", "")
        gt_path = labels_ts / f"{case_id}.nii.gz"
        pred_path = pred_dir / f"{case_id}.nii.gz"

        mapping = get_mapping_row(mapping_df, case_id, fallback_center)

        base = {
            "Dataset": dataset_name,
            "Dataset_ID": dataset_id,
            "Case": case_id,
            "PatientID": mapping["PatientID"],
            "Center": mapping["Center"],
            "Hospital": mapping["Center"],
            "Image_Path": str(image_path),
            "Original_CT_Path": mapping["Original_CT_Path"],
            "Original_Mask_Path": mapping["Original_Mask_Path"],
        }

        if mask_source in ["gt", "both"] and gt_path.exists():
            binary_gt_path = out_dir / "_binary_masks" / dataset_name / f"{case_id}_gt_binary.nii.gz"
            try:
                has_fg, fg_voxels = make_binary_mask_file(image_path, gt_path, binary_gt_path)
                if has_fg:
                    row = extract_one_feature_row(extractor, image_path, binary_gt_path, keep_diagnostics)
                    row.update(base)
                    row["MaskSource"] = "ground_truth"
                    row["Mask_Path"] = str(gt_path)
                    row["Binary_Mask_Path"] = str(binary_gt_path)
                    row["ForegroundVoxels"] = fg_voxels
                    row["Error"] = ""
                    gt_rows.append(row)
            except Exception as exc:
                error_row = dict(base)
                error_row["MaskSource"] = "ground_truth"
                error_row["Mask_Path"] = str(gt_path)
                error_row["Binary_Mask_Path"] = str(binary_gt_path)
                error_row["ForegroundVoxels"] = ""
                error_row["Error"] = str(exc)
                gt_rows.append(error_row)

        if mask_source in ["pred", "both"] and pred_path.exists():
            binary_pred_path = out_dir / "_binary_masks" / dataset_name / f"{case_id}_pred_binary.nii.gz"
            try:
                has_fg, fg_voxels = make_binary_mask_file(image_path, pred_path, binary_pred_path)
                if has_fg:
                    row = extract_one_feature_row(extractor, image_path, binary_pred_path, keep_diagnostics)
                    row.update(base)
                    row["MaskSource"] = "prediction"
                    row["Mask_Path"] = str(pred_path)
                    row["Binary_Mask_Path"] = str(binary_pred_path)
                    row["ForegroundVoxels"] = fg_voxels
                    row["Error"] = ""
                    pred_rows.append(row)
            except Exception as exc:
                error_row = dict(base)
                error_row["MaskSource"] = "prediction"
                error_row["Mask_Path"] = str(pred_path)
                error_row["Binary_Mask_Path"] = str(binary_pred_path)
                error_row["ForegroundVoxels"] = ""
                error_row["Error"] = str(exc)
                pred_rows.append(error_row)

    return gt_rows, pred_rows


def maybe_merge_metrics(features_df: pd.DataFrame, metrics_csv: Path | None) -> pd.DataFrame:
    if metrics_csv is None or not metrics_csv.exists() or features_df.empty:
        return features_df

    metrics = pd.read_csv(metrics_csv)
    if "Case" not in metrics.columns:
        print("Metrics CSV does not contain Case column. Skipping metrics merge.")
        return features_df

    keep_cols = [
        c
        for c in [
            "Dataset",
            "Case",
            "Center",
            "Hospital",
            "Dice",
            "IoU",
            "Sensitivity",
            "Specificity",
            "Precision",
            "Accuracy",
        ]
        if c in metrics.columns
    ]
    metrics = metrics[keep_cols].copy()

    if "Dataset" in metrics.columns:
        return features_df.merge(metrics, on=["Dataset", "Case"], how="left", suffixes=("", "_metric"))

    if "Hospital" in metrics.columns:
        return features_df.merge(
            metrics,
            left_on=["Center", "Case"],
            right_on=["Hospital", "Case"],
            how="left",
            suffixes=("", "_metric"),
        )

    if "Center" in metrics.columns:
        return features_df.merge(metrics, on=["Center", "Case"], how="left", suffixes=("", "_metric"))

    return features_df.merge(metrics, on="Case", how="left", suffixes=("", "_metric"))


def write_summary(out_dir: Path, gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> None:
    rows = []

    for name, df in [("ground_truth", gt_df), ("prediction", pred_df)]:
        if df.empty:
            rows.append(
                {
                    "MaskSource": name,
                    "Rows": 0,
                    "Centers": "",
                    "FeatureColumns": 0,
                    "ErrorRows": 0,
                }
            )
            continue

        feature_cols = [c for c in df.columns if c.startswith("original_")]
        error_rows = int((df["Error"].astype(str).str.len() > 0).sum()) if "Error" in df.columns else 0

        rows.append(
            {
                "MaskSource": name,
                "Rows": len(df),
                "Centers": ",".join(sorted(df["Center"].dropna().astype(str).unique())),
                "FeatureColumns": len(feature_cols),
                "ErrorRows": error_rows,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "radiomics_extraction_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PyRadiomics features from nnU-Net test images.")

    parser.add_argument("--nnunet_raw", required=True, type=Path)
    parser.add_argument("--pred_root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--datasets", default=None, type=str, help="Optional dataset filter, e.g. CHUM,MDA or 001,006.")
    parser.add_argument("--mask_source", default="both", choices=["gt", "pred", "both"])
    parser.add_argument("--bin_width", default=25.0, type=float)
    parser.add_argument("--correct_mask", action="store_true")
    parser.add_argument("--keep_diagnostics", action="store_true")
    parser.add_argument("--metrics_csv", default=None, type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dataset_filter = parse_dataset_filter(args.datasets)
    dataset_dirs = find_dataset_dirs(args.nnunet_raw, dataset_filter)

    extractor = make_extractor(bin_width=args.bin_width, correct_mask=args.correct_mask)

    all_gt_rows: list[dict[str, Any]] = []
    all_pred_rows: list[dict[str, Any]] = []

    for dataset_dir in dataset_dirs:
        gt_rows, pred_rows = extract_features_for_dataset(
            dataset_dir=dataset_dir,
            pred_root=args.pred_root,
            out_dir=args.out,
            extractor=extractor,
            mask_source=args.mask_source,
            keep_diagnostics=args.keep_diagnostics,
        )
        all_gt_rows.extend(gt_rows)
        all_pred_rows.extend(pred_rows)

    gt_df = pd.DataFrame(all_gt_rows)
    pred_df = pd.DataFrame(all_pred_rows)

    gt_df = maybe_merge_metrics(gt_df, args.metrics_csv)
    pred_df = maybe_merge_metrics(pred_df, args.metrics_csv)

    gt_path = args.out / "features_gt_masks.csv"
    pred_path = args.out / "features_pred_masks.csv"

    if not gt_df.empty:
        gt_df.to_csv(gt_path, index=False)
    if not pred_df.empty:
        pred_df.to_csv(pred_path, index=False)

    write_summary(args.out, gt_df, pred_df)

    print("\nExperiment 7 Complete")
    print("=" * 80)
    print(f"Datasets processed: {len(dataset_dirs)}")
    print(f"GT feature rows:    {len(gt_df)}")
    print(f"Pred feature rows:  {len(pred_df)}")
    print(f"GT output:          {gt_path}")
    print(f"Pred output:        {pred_path}")
    print(f"Summary:            {args.out / 'radiomics_extraction_summary.csv'}")


if __name__ == "__main__":
    main()
