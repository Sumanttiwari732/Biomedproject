from pathlib import Path
import re
import json
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = Path("/Users/sumantratiwari/PycharmProjects/Biomedproject/HECKTOR 2025 Training Data Defaced ALL")
OUTPUT_CSV = ROOT_DIR / "hecktor_patient_pairs.csv"
OUTPUT_JSON = ROOT_DIR / "hecktor_patient_pairs_summary.json"

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
    Read only metadata, not pixel array.
    Returns: shape_zyx, spacing_xyz, origin, direction
    """
    img = sitk.ReadImage(str(path))
    size_xyz = img.GetSize()        # x, y, z
    spacing_xyz = img.GetSpacing()  # x, y, z
    origin = img.GetOrigin()
    direction = img.GetDirection()
    shape_zyx = (size_xyz[2], size_xyz[1], size_xyz[0])
    return shape_zyx, spacing_xyz, origin, direction


def extract_patient_id(path: Path) -> str:
    """
    Examples:
      CHUM-001__CT.nii.gz -> CHUM-001
      CHUM-001.nii.gz     -> CHUM-001
    """
    stem = strip_extension(path.name)
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem


def infer_center_from_folder(path: Path, root_dir: Path) -> str:
    rel = path.relative_to(root_dir)
    parts = rel.parts
    if len(parts) == 0:
        return "unknown"
    return parts[0]


def list_patient_folders(root_dir: Path):
    """
    HECKTOR data in your setup is organized as:
      ROOT_DIR / patient_folder / files...
    """
    folders = [p for p in root_dir.iterdir() if p.is_dir()]
    folders.sort()
    return folders


def find_files_for_patient(folder: Path):
    """
    Find CT, PET/PT, mask, and other files inside one patient folder.
    """
    files = [p for p in folder.iterdir() if p.is_file() and has_valid_extension(p)]

    ct_files = []
    pet_files = []
    mask_files = []
    rtdose_files = []
    other_files = []

    for f in files:
        name = f.name.lower()
        stem = strip_extension(f.name).lower()

        # HECKTOR naming in your dataset:
        #   patient__CT.nii.gz
        #   patient__PT.nii.gz
        #   patient.nii.gz  (mask)
        if "__ct" in name or name.endswith("_ct.nii.gz") or name.endswith("_ct.nii"):
            ct_files.append(f)
        elif "__pt" in name or "_pt" in name or "__pet" in name or "pet" in name:
            pet_files.append(f)
        elif "__rtdose" in name:
            rtdose_files.append(f)
        elif stem == folder.name.lower() or "__" not in stem:
            # In your folder tree, the unnamed .nii.gz is the segmentation mask.
            mask_files.append(f)
        else:
            other_files.append(f)

    return ct_files, pet_files, mask_files, rtdose_files, other_files


def get_single_or_none(file_list):
    if len(file_list) == 0:
        return None
    return sorted(file_list)[0]


# ============================================================
# MAIN BUILD FUNCTION
# ============================================================
def build_patient_table(root_dir: Path):
    if not root_dir.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {root_dir}")

    patient_folders = list_patient_folders(root_dir)
    print(f"Found {len(patient_folders)} patient folders.")

    records = []

    for folder in tqdm(patient_folders, desc="Pairing patients"):
        patient_id = folder.name
        center = infer_center_from_folder(folder, root_dir)

        ct_files, pet_files, mask_files, rtdose_files, other_files = find_files_for_patient(folder)

        ct_path = get_single_or_none(ct_files)
        pet_path = get_single_or_none(pet_files)
        mask_path = get_single_or_none(mask_files)

        record = {
            "patient_id": patient_id,
            "center": center,
            "folder": str(folder),
            "ct_path": str(ct_path) if ct_path else None,
            "pet_path": str(pet_path) if pet_path else None,
            "mask_path": str(mask_path) if mask_path else None,
            "rtdose_path": str(get_single_or_none(rtdose_files)) if len(rtdose_files) > 0 else None,
            "num_ct_files": len(ct_files),
            "num_pet_files": len(pet_files),
            "num_mask_files": len(mask_files),
            "num_rtdose_files": len(rtdose_files),
            "num_other_files": len(other_files),
            "has_ct": ct_path is not None,
            "has_pet": pet_path is not None,
            "has_mask": mask_path is not None,
            "status": "ok",
            "error": None,
        }

        # Metadata only for files that exist
        try:
            if ct_path:
                ct_shape, ct_spacing, ct_origin, ct_dir = read_image_meta(ct_path)
                record["ct_shape_zyx"] = ct_shape
                record["ct_spacing_xyz"] = ct_spacing
                record["ct_origin"] = ct_origin
            else:
                record["ct_shape_zyx"] = None
                record["ct_spacing_xyz"] = None
                record["ct_origin"] = None

            if pet_path:
                pet_shape, pet_spacing, pet_origin, pet_dir = read_image_meta(pet_path)
                record["pet_shape_zyx"] = pet_shape
                record["pet_spacing_xyz"] = pet_spacing
                record["pet_origin"] = pet_origin
            else:
                record["pet_shape_zyx"] = None
                record["pet_spacing_xyz"] = None
                record["pet_origin"] = None

            if mask_path:
                mask_shape, mask_spacing, mask_origin, mask_dir = read_image_meta(mask_path)
                record["mask_shape_zyx"] = mask_shape
                record["mask_spacing_xyz"] = mask_spacing
                record["mask_origin"] = mask_origin
            else:
                record["mask_shape_zyx"] = None
                record["mask_spacing_xyz"] = None
                record["mask_origin"] = None

        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)

        records.append(record)

    df = pd.DataFrame(records)

    # Make sure important columns are present even if empty
    for col in [
        "ct_shape_zyx", "ct_spacing_xyz", "ct_origin",
        "pet_shape_zyx", "pet_spacing_xyz", "pet_origin",
        "mask_shape_zyx", "mask_spacing_xyz", "mask_origin",
    ]:
        if col not in df.columns:
            df[col] = None

    return df


def summarize_table(df: pd.DataFrame):
    print("\n===== PAIRING SUMMARY =====")
    print(f"Total patient folders: {len(df)}")
    print("\nAvailability counts:")
    print(df[["has_ct", "has_pet", "has_mask"]].sum())

    print("\nStatus counts:")
    print(df["status"].value_counts(dropna=False))

    usable = df[(df["has_ct"]) & (df["has_pet"]) & (df["has_mask"]) & (df["status"] == "ok")]
    print(f"\nUsable segmentation cases (CT + PET + MASK): {len(usable)}")

    missing = df[~((df["has_ct"]) & (df["has_pet"]) & (df["has_mask"]) & (df["status"] == "ok"))]
    if len(missing) > 0:
        print("\nExamples of incomplete cases:")
        cols = ["patient_id", "center", "has_ct", "has_pet", "has_mask", "num_ct_files", "num_pet_files", "num_mask_files", "num_rtdose_files"]
        print(missing[cols].head(10).to_string(index=False))


def save_summary_json(df: pd.DataFrame, out_path: Path):
    summary = {
        "total_patient_folders": int(len(df)),
        "usable_segmentation_cases": int(((df["has_ct"]) & (df["has_pet"]) & (df["has_mask"]) & (df["status"] == "ok")).sum()),
        "has_ct": int(df["has_ct"].sum()),
        "has_pet": int(df["has_pet"].sum()),
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
    df = build_patient_table(ROOT_DIR)

    # Save full table
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved patient pairing table to: {OUTPUT_CSV}")

    # Save JSON summary
    summary = save_summary_json(df, OUTPUT_JSON)
    print(f"Saved summary to: {OUTPUT_JSON}")

    # Print summary
    summarize_table(df)

    # Save only usable segmentation cases
    usable_df = df[(df["has_ct"]) & (df["has_pet"]) & (df["has_mask"]) & (df["status"] == "ok")].copy()
    usable_csv = ROOT_DIR / "hecktor_patient_pairs_usable_segmentation.csv"
    usable_df.to_csv(usable_csv, index=False)
    print(f"\nSaved usable segmentation cases to: {usable_csv}")

    print("\nDone.")


if __name__ == "__main__":
    main()