from pathlib import Path
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = Path("/Users/sumantratiwari/PycharmProjects/Biomedproject/HECKTOR 2025 Training Data Defaced ALL")
PAIRING_CSV = ROOT_DIR / "hecktor_ct_patient_pairs_usable.csv"

OUT_ROOT = ROOT_DIR / "ct_step4_alignment_qc"
QC_DIR = OUT_ROOT / "before_after_qc"
CT_CROP_DIR = OUT_ROOT / "ct_crops"
MASK_CROP_DIR = OUT_ROOT / "mask_crops"
MASK_ALIGNED_DIR = OUT_ROOT / "mask_aligned_full"

ALIGNMENT_REPORT_CSV = OUT_ROOT / "alignment_report.csv"
SUMMARY_JSON = OUT_ROOT / "alignment_summary.json"

NUM_CASES_TO_SHOW = None   # set to an integer to limit QC figures, or None for all usable cases

MARGIN_X = 24
MARGIN_Y = 24
MARGIN_Z = 12

HU_CLIP_MIN = -1000.0
HU_CLIP_MAX = 400.0


# ============================================================
# HELPERS
# ============================================================
def ensure_dirs():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    CT_CROP_DIR.mkdir(parents=True, exist_ok=True)
    MASK_CROP_DIR.mkdir(parents=True, exist_ok=True)
    MASK_ALIGNED_DIR.mkdir(parents=True, exist_ok=True)


def read_image(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    return img, arr


def geometry_matches(ct_img, mask_img):
    same_size = ct_img.GetSize() == mask_img.GetSize()
    same_spacing = np.allclose(ct_img.GetSpacing(), mask_img.GetSpacing())
    same_origin = np.allclose(ct_img.GetOrigin(), mask_img.GetOrigin())
    same_direction = np.allclose(ct_img.GetDirection(), mask_img.GetDirection())
    return {
        "same_size": bool(same_size),
        "same_spacing": bool(same_spacing),
        "same_origin": bool(same_origin),
        "same_direction": bool(same_direction),
        "same_geometry_all": bool(same_size and same_spacing and same_origin and same_direction),
    }


def resample_mask_to_ct(mask_img, ct_img):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask_img)


def bbox_from_binary_mask(mask_arr):
    coords = np.argwhere(mask_arr > 0)
    if coords.size == 0:
        return None

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    return {
        "x_min": int(x_min),
        "y_min": int(y_min),
        "z_min": int(z_min),
        "x_max": int(x_max),
        "y_max": int(y_max),
        "z_max": int(z_max),
        "x_size": int(x_max - x_min + 1),
        "y_size": int(y_max - y_min + 1),
        "z_size": int(z_max - z_min + 1),
    }


def expand_bbox(bbox, image_shape_zyx, margin_x=24, margin_y=24, margin_z=12):
    z_size, y_size, x_size = image_shape_zyx

    x0 = max(0, bbox["x_min"] - margin_x)
    y0 = max(0, bbox["y_min"] - margin_y)
    z0 = max(0, bbox["z_min"] - margin_z)

    x1 = min(x_size - 1, bbox["x_max"] + margin_x)
    y1 = min(y_size - 1, bbox["y_max"] + margin_y)
    z1 = min(z_size - 1, bbox["z_max"] + margin_z)

    return {
        "x0": int(x0),
        "y0": int(y0),
        "z0": int(z0),
        "x1": int(x1),
        "y1": int(y1),
        "z1": int(z1),
        "size_x": int(x1 - x0 + 1),
        "size_y": int(y1 - y0 + 1),
        "size_z": int(z1 - z0 + 1),
    }


def crop_sitk(img, bbox_expanded):
    return sitk.RegionOfInterest(
        img,
        size=[bbox_expanded["size_x"], bbox_expanded["size_y"], bbox_expanded["size_z"]],
        index=[bbox_expanded["x0"], bbox_expanded["y0"], bbox_expanded["z0"]],
    )


def clip_and_zscore_ct(ct_arr):
    ct_arr = ct_arr.astype(np.float32)
    ct_arr = np.clip(ct_arr, HU_CLIP_MIN, HU_CLIP_MAX)

    mean = float(ct_arr.mean())
    std = float(ct_arr.std())
    if std < 1e-6:
        std = 1.0

    return ((ct_arr - mean) / std).astype(np.float32), mean, std


def window_for_display(ct_slice, low=-1000, high=400):
    ct_slice = ct_slice.astype(np.float32)
    ct_slice = np.clip(ct_slice, low, high)
    return (ct_slice - low) / (high - low)


def contour_2d(mask2d):
    return (mask2d > 0).astype(np.uint8)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def save_qc_figure(patient_id, ct_arr, mask_arr_aligned, ct_crop_norm, mask_crop_arr, bbox_expanded, out_path):
    """
    Left: original CT + aligned mask contour
    Right: cropped + normalized CT + cropped mask contour
    """
    # same anatomical level in both panels
    orig_z = clamp((bbox_expanded["z0"] + bbox_expanded["z1"]) // 2, 0, ct_arr.shape[0] - 1)
    roi_z = clamp(orig_z - bbox_expanded["z0"], 0, ct_crop_norm.shape[0] - 1)

    ct_before = window_for_display(ct_arr[orig_z])
    ct_after = np.clip(ct_crop_norm[roi_z], -3, 3)

    mask_before = contour_2d(mask_arr_aligned[orig_z])
    mask_after = contour_2d(mask_crop_arr[roi_z])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(ct_before, cmap="gray")
    if mask_before.sum() > 0:
        axes[0].contour(mask_before, levels=[0.5], colors="red", linewidths=1.5)
    axes[0].set_title(f"{patient_id} - Original CT + Mask Contour")
    axes[0].axis("off")

    axes[1].imshow(ct_after, cmap="gray")
    if mask_after.sum() > 0:
        axes[1].contour(mask_after, levels=[0.5], colors="red", linewidths=1.5)
    axes[1].set_title(f"{patient_id} - Cropped + Normalized CT + Mask Contour")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_sitk_from_array(arr_zyx, reference_img, out_path, pixel_type):
    img = sitk.GetImageFromArray(arr_zyx)
    img = sitk.Cast(img, pixel_type)
    img.CopyInformation(reference_img)
    sitk.WriteImage(img, str(out_path))


# ============================================================
# MAIN PROCESSING
# ============================================================
def process_case(row):
    patient_id = row["patient_id"]
    center = row["center"]
    ct_path = Path(row["ct_path"])
    mask_path = Path(row["mask_path"])

    record = {
        "patient_id": patient_id,
        "center": center,
        "ct_path": str(ct_path),
        "mask_path": str(mask_path),
        "status": "ok",
        "reason": None,
    }

    try:
        # Read original images
        ct_img, ct_arr = read_image(ct_path)
        mask_img, mask_arr = read_image(mask_path)

        # Geometry check before alignment
        geom = geometry_matches(ct_img, mask_img)
        record.update({f"pre_{k}": v for k, v in geom.items()})

        # Resample mask to CT grid
        mask_aligned_img = resample_mask_to_ct(mask_img, ct_img)
        mask_aligned_arr = sitk.GetArrayFromImage(mask_aligned_img).astype(np.uint8)

        record["aligned_mask_foreground_voxels"] = int(mask_aligned_arr.sum())
        record["aligned_mask_foreground_fraction"] = float(mask_aligned_arr.mean())

        if mask_aligned_arr.sum() == 0:
            record["status"] = "skipped_empty_mask_after_resample"
            record["reason"] = "Mask became empty after resampling to CT grid"
            return record

        # Save aligned full mask for reference
        aligned_mask_out = MASK_ALIGNED_DIR / f"{patient_id}_mask_aligned.nii.gz"
        sitk.WriteImage(mask_aligned_img, str(aligned_mask_out))
        record["aligned_mask_path"] = str(aligned_mask_out)

        # Bounding box in aligned mask space
        bbox = bbox_from_binary_mask(mask_aligned_arr)
        if bbox is None:
            record["status"] = "skipped_no_bbox"
            record["reason"] = "Could not compute foreground bounding box"
            return record

        bbox_expanded = expand_bbox(
            bbox,
            ct_arr.shape,
            margin_x=MARGIN_X,
            margin_y=MARGIN_Y,
            margin_z=MARGIN_Z,
        )
        record.update({f"bbox_{k}": v for k, v in bbox_expanded.items()})

        # Crop CT and mask using aligned mask bbox
        ct_crop_img = crop_sitk(ct_img, bbox_expanded)
        mask_crop_img = crop_sitk(mask_aligned_img, bbox_expanded)

        ct_crop_arr = sitk.GetArrayFromImage(ct_crop_img).astype(np.float32)
        mask_crop_arr = sitk.GetArrayFromImage(mask_crop_img).astype(np.uint8)

        # Normalize cropped CT
        ct_crop_norm_arr, ct_mean, ct_std = clip_and_zscore_ct(ct_crop_arr)
        record["ct_crop_mean_before_norm"] = ct_mean
        record["ct_crop_std_before_norm"] = ct_std
        record["ct_crop_shape_zyx"] = tuple(ct_crop_arr.shape)
        record["mask_crop_shape_zyx"] = tuple(mask_crop_arr.shape)

        # Save cropped mask and normalized CT
        ct_out = CT_CROP_DIR / f"{patient_id}_ct_crop_norm.nii.gz"
        mask_out = MASK_CROP_DIR / f"{patient_id}_mask_crop.nii.gz"

        ct_norm_img = sitk.GetImageFromArray(ct_crop_norm_arr)
        ct_norm_img.CopyInformation(ct_crop_img)
        ct_norm_img = sitk.Cast(ct_norm_img, sitk.sitkFloat32)

        mask_out_img = sitk.GetImageFromArray(mask_crop_arr.astype(np.uint8))
        mask_out_img.CopyInformation(mask_crop_img)
        mask_out_img = sitk.Cast(mask_out_img, sitk.sitkUInt8)

        sitk.WriteImage(ct_norm_img, str(ct_out))
        sitk.WriteImage(mask_out_img, str(mask_out))

        record["ct_crop_path"] = str(ct_out)
        record["mask_crop_path"] = str(mask_out)

        # Save QC figure
        qc_out = QC_DIR / f"{patient_id}_before_after.png"
        save_qc_figure(
            patient_id=patient_id,
            ct_arr=ct_arr,
            mask_arr_aligned=mask_aligned_arr,
            ct_crop_norm=ct_crop_norm_arr,
            mask_crop_arr=mask_crop_arr,
            bbox_expanded=bbox_expanded,
            out_path=qc_out,
        )
        record["qc_path"] = str(qc_out)

        return record

    except Exception as e:
        record["status"] = "error"
        record["reason"] = str(e)
        return record


def main():
    ensure_dirs()

    if not PAIRING_CSV.exists():
        raise FileNotFoundError(f"Missing pairing CSV: {PAIRING_CSV}")

    df = pd.read_csv(PAIRING_CSV)
    usable_df = df.dropna(subset=["ct_path", "mask_path"]).copy()

    if len(usable_df) == 0:
        raise RuntimeError("No usable CT + mask rows found in the pairing CSV.")

    if NUM_CASES_TO_SHOW is not None:
        usable_df = usable_df.head(NUM_CASES_TO_SHOW)

    print(f"Processing {len(usable_df)} cases...")
    records = []
    for _, row in tqdm(usable_df.iterrows(), total=len(usable_df), desc="Alignment + QC"):
        records.append(process_case(row))

    out_df = pd.DataFrame(records)
    out_df.to_csv(ALIGNMENT_REPORT_CSV, index=False)

    summary = {
        "total_cases": int(len(out_df)),
        "ok": int((out_df["status"] == "ok").sum()),
        "skipped_empty_mask_after_resample": int((out_df["status"] == "skipped_empty_mask_after_resample").sum()),
        "skipped_no_bbox": int((out_df["status"] == "skipped_no_bbox").sum()),
        "errors": int((out_df["status"] == "error").sum()),
        "pre_geometry_match_all": int(out_df.get("pre_same_geometry_all", pd.Series(dtype=bool)).sum()) if "pre_same_geometry_all" in out_df.columns else None,
        "pre_size_match": int(out_df.get("pre_same_size", pd.Series(dtype=bool)).sum()) if "pre_same_size" in out_df.columns else None,
        "pre_spacing_match": int(out_df.get("pre_same_spacing", pd.Series(dtype=bool)).sum()) if "pre_same_spacing" in out_df.columns else None,
        "pre_origin_match": int(out_df.get("pre_same_origin", pd.Series(dtype=bool)).sum()) if "pre_same_origin" in out_df.columns else None,
        "pre_direction_match": int(out_df.get("pre_same_direction", pd.Series(dtype=bool)).sum()) if "pre_same_direction" in out_df.columns else None,
    }

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== ALIGNMENT SUMMARY =====")
    print(json.dumps(summary, indent=2))

    print(f"\nSaved alignment report to: {ALIGNMENT_REPORT_CSV}")
    print(f"Saved QC images to: {QC_DIR}")
    print(f"Saved cropped CTs to: {CT_CROP_DIR}")
    print(f"Saved cropped masks to: {MASK_CROP_DIR}")
    print(f"Saved aligned full masks to: {MASK_ALIGNED_DIR}")

    if "status" in out_df.columns:
        print("\nStatus counts:")
        print(out_df["status"].value_counts(dropna=False).to_string())

    print("\nFirst few rows of alignment report:")
    cols = [
        "patient_id", "center", "status",
        "pre_same_geometry_all", "pre_same_size", "pre_same_spacing", "pre_same_origin", "pre_same_direction",
        "ct_crop_shape_zyx", "mask_crop_shape_zyx"
    ]
    cols = [c for c in cols if c in out_df.columns]
    print(out_df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()