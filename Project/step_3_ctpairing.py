from pathlib import Path
import json
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = Path("/Users/sumantratiwari/PycharmProjects/Biomedproject/HECKTOR 2025 Training Data Defaced ALL")

OUTPUT_CSV = ROOT_DIR / "hecktor_ct_patient_pairs.csv"
OUTPUT_USABLE_CSV = ROOT_DIR / "hecktor_ct_patient_pairs_usable.csv"
OUTPUT_JSON = ROOT_DIR / "hecktor_ct_patient_pairs_summary.json"

VALID_EXTS = [".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"]


# ============================================================
# HELPERS
# ============================================================
def has_valid_extension(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(ext) for ext in VALID_EXTS)


def strip_extension(filename: str) -> str:
    name = filename
    for ext in VALID_EXTS:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def read_image_meta(path: Path):
    """
    Read metadata only, not the full array.
    Returns:
        shape_zyx, spacing_xyz, origin, direction
    """
    img = sitk.ReadImage(str(path))
    size_xyz = img.GetSize()       # x, y, z
    spacing_xyz = img.GetSpacing() # x, y, z
    origin = img.GetOrigin()
    direction = img.GetDirection()
    shape_zyx = (size_xyz[2], size_xyz[1], size_xyz[0])
    return shape_zyx, spacing_xyz, origin, direction


def list_patient_folders(root_dir: Path):
    """
    In your dataset, each patient is in its own folder:
      ROOT_DIR / CHUM-001 / ...
    """
    folders = [p for p in root_dir.iterdir() if p.is_dir()]
    folders.sort()
    return folders


def extract_center(folder: Path, root_dir: Path) -> str:
    """
    For your data, the first folder level is the site/patient folder name.
    """
    rel = folder.relative_to(root_dir)
    return rel.parts[0] if len(rel.parts) > 0 else "unknown"


def find_ct_and_mask_files(folder: Path):
    """
    HECKTOR folder pattern:
      patient__CT.nii.gz   -> CT
      patient.nii.gz       -> segmentation mask
      patient__PT.nii.gz   -> PET (ignored)
      patient__RTDOSE.nii.gz -> dose (ignored)

    Returns:
        ct_files, mask_files, ignored_files
    """
    files = [p for p in folder.iterdir() if p.is_file() and has_valid_extension(p)]

    ct_files = []
    mask_files = []
    ignored_files = []

    folder_name = folder.name.lower()

    for f in files:
        name = f.name.lower()
        stem = strip_extension(f.name).lower()

        # CT
        if "__ct" in name or name.endswith("_ct.nii.gz") or name.endswith("_ct.nii"):
            ct_files.append(f)

        # Mask / label file: usually the plain file without __CT / __PT / __RTDOSE
        elif stem == folder_name or "__" not in stem:
            mask_files.append(f)

        else:
            ignored_files.append(f)

    return ct_files, mask_files, ignored_files


def get_single_or_none(file_list):
    if len(file_list) == 0:
        return None
    return sorted(file_list)[0]


# ============================================================
# MAIN BUILD FUNCTION
# ============================================================
def build_ct_patient_table(root_dir: Path):
    if not root_dir.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {root_dir}")

    patient_folders = list_patient_folders(root_dir)
    print(f"Found {len(patient_folders)} patient folders.")

    records = []

    for folder in tqdm(patient_folders, desc="Pairing CT patients"):
        patient_id = folder.name
        center = extract_center(folder, root_dir)

        ct_files, mask_files, ignored_files = find_ct_and_mask_files(folder)

        ct_path = get_single_or_none(ct_files)
        mask_path = get_single_or_none(mask_files)

        record = {
            "patient_id": patient_id,
            "center": center,
            "folder": str(folder),
            "ct_path": str(ct_path) if ct_path else None,
            "mask_path": str(mask_path) if mask_path else None,
            "num_ct_files": len(ct_files),
            "num_mask_files": len(mask_files),
            "num_ignored_files": len(ignored_files),
            "has_ct": ct_path is not None,
            "has_mask": mask_path is not None,
            "status": "ok",
            "error": None,
        }

        # Read metadata if files exist
        try:
            if ct_path:
                ct_shape, ct_spacing, ct_origin, ct_direction = read_image_meta(ct_path)
                record["ct_shape_zyx"] = ct_shape
                record["ct_spacing_xyz"] = ct_spacing
                record["ct_origin"] = ct_origin
                record["ct_direction"] = ct_direction
            else:
                record["ct_shape_zyx"] = None
                record["ct_spacing_xyz"] = None
                record["ct_origin"] = None
                record["ct_direction"] = None

            if mask_path:
                mask_shape, mask_spacing, mask_origin, mask_direction = read_image_meta(mask_path)
                record["mask_shape_zyx"] = mask_shape
                record["mask_spacing_xyz"] = mask_spacing
                record["mask_origin"] = mask_origin
                record["mask_direction"] = mask_direction
            else:
                record["mask_shape_zyx"] = None
                record["mask_spacing_xyz"] = None
                record["mask_origin"] = None
                record["mask_direction"] = None

        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)

        records.append(record)

    df = pd.DataFrame(records)
    return df


def summarize_table(df: pd.DataFrame):
    print("\n===== CT PAIRING SUMMARY =====")
    print(f"Total patient folders: {len(df)}")

    print("\nAvailability counts:")
    print(df[["has_ct", "has_mask"]].sum())

    print("\nStatus counts:")
    print(df["status"].value_counts(dropna=False))

    usable = df[(df["has_ct"]) & (df["has_mask"]) & (df["status"] == "ok")].copy()
    print(f"\nUsable CT segmentation cases (CT + MASK): {len(usable)}")

    missing = df[~((df["has_ct"]) & (df["has_mask"]) & (df["status"] == "ok"))].copy()
    if len(missing) > 0:
        print("\nExamples of incomplete cases:")
        cols = [
            "patient_id",
            "center",
            "has_ct",
            "has_mask",
            "num_ct_files",
            "num_mask_files",
            "num_ignored_files",
        ]
        print(missing[cols].head(10).to_string(index=False))


def save_summary_json(df: pd.DataFrame, out_path: Path):
    usable = df[(df["has_ct"]) & (df["has_mask"]) & (df["status"] == "ok")]
    summary = {
        "total_patient_folders": int(len(df)),
        "usable_ct_segmentation_cases": int(len(usable)),
        "has_ct": int(df["has_ct"].sum()),
        "has_mask": int(df["has_mask"].sum()),
        "status_counts": df["status"].value_counts(dropna=False).to_dict(),
        "centers": df["center"].value_counts(dropna=False).to_dict(),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    df = build_ct_patient_table(ROOT_DIR)

    # Save full table
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved CT patient pairing table to: {OUTPUT_CSV}")

    # Save only usable CT + mask cases
    usable_df = df[(df["has_ct"]) & (df["has_mask"]) & (df["status"] == "ok")].copy()
    usable_df.to_csv(OUTPUT_USABLE_CSV, index=False)
    print(f"Saved usable CT cases to: {OUTPUT_USABLE_CSV}")

    # Save JSON summary
    summary = save_summary_json(df, OUTPUT_JSON)
    print(f"Saved summary to: {OUTPUT_JSON}")

    # Print summary
    summarize_table(df)

    print("\nDone.")


if __name__ == "__main__":
    main()