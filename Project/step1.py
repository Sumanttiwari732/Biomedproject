from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================
ROOT_DIR = Path("/Users/sumantratiwari/PycharmProjects/Biomedproject/HECKTOR 2025 Training Data Defaced ALL")

VALID_EXTS = [".nii.gz", ".nii", ".nrrd", ".mha", ".mhd"]


# ============================================================
# FILE UTILITIES
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


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def read_image(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    spacing = img.GetSpacing()         # x, y, z
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return img, arr, spacing, origin, direction


def image_stats(arr: np.ndarray):
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def sample_unique_count(arr: np.ndarray, max_samples: int = 50000) -> int:
    flat = np.asarray(arr).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0
    step = max(1, flat.size // max_samples)
    sample = flat[::step]
    if sample.size == 0:
        return 0
    return int(np.unique(sample).size)


def is_integer_like(arr: np.ndarray, sample_size: int = 100000) -> bool:
    flat = np.asarray(arr).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return False
    step = max(1, flat.size // sample_size)
    sample = flat[::step]
    if sample.size == 0:
        return False
    return float(np.mean(np.isclose(sample, np.round(sample)))) > 0.98


# ============================================================
# PATIENT / CENTER IDENTIFICATION
# ============================================================
def extract_patient_id(path: Path) -> str:
    """
    HECKTOR filenames often look like:
      MDA-258__CT.nii.gz
      MDA-258__PT.nii.gz
      MDA-258.nii.gz
    """
    stem = strip_extension(path.name)
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem


def extract_center(path: Path, root_dir: Path) -> str:
    """
    Try to infer the center from the first folder under root.
    Example: root/CHUM-001/CHUM-001__CT.nii.gz -> CHUM-001
    """
    rel = path.relative_to(root_dir)
    parts = rel.parts[:-1]
    if not parts:
        return "unknown"
    return parts[0]


# ============================================================
# MODALITY DETECTION
# ============================================================
def infer_modality_from_name(path: Path):
    """
    Detect from filename:
      CT, PET, MASK, RTDOSE, RTSTRUCT, UNKNOWN
    """
    name = path.name.lower()

    if "rtdose" in name:
        return "RTDOSE"
    if "rtstruct" in name:
        return "RTSTRUCT"

    if "__ct" in name or name.endswith("_ct.nii.gz") or name.endswith("_ct.nii"):
        return "CT"
    if "__pt" in name or "_pt" in name or "__pet" in name or "pet" in name:
        return "PET"

    if any(tok in name for tok in ["mask", "seg", "label", "gtv", "tumor", "node"]):
        return "MASK"

    return "UNKNOWN"


def infer_modality_from_content(path: Path):
    """
    Fallback classification when filename is not enough.
    """
    try:
        _, arr, _, _, _ = read_image(path)
        stats = image_stats(arr)
        uniq = sample_unique_count(arr)

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return "UNKNOWN"

        # Masks: few labels, integer-like, usually non-negative
        if is_integer_like(arr) and uniq <= 20 and stats["max"] <= 20:
            return "MASK"

        # CT: often has negative HU values
        if stats["min"] < -100:
            return "CT"

        # PET: usually non-negative floating point
        if stats["min"] >= 0:
            return "PET"

    except Exception:
        pass

    return "UNKNOWN"


def classify_file(path: Path):
    by_name = infer_modality_from_name(path)
    if by_name != "UNKNOWN":
        return by_name, "name"

    by_content = infer_modality_from_content(path)
    if by_content != "UNKNOWN":
        return by_content, "content"

    return "UNKNOWN", "unknown"


# ============================================================
# DISCOVERY
# ============================================================
def discover_all_image_files(root_dir: Path):
    return [p for p in root_dir.rglob("*") if p.is_file() and has_valid_extension(p)]


def group_by_patient(files):
    grouped = {}
    for p in files:
        pid = extract_patient_id(p)
        if pid not in grouped:
            grouped[pid] = []
        grouped[pid].append(p)
    return grouped


# ============================================================
# INSPECTION
# ============================================================
def inspect_file(path: Path, root_dir: Path):
    row = {
        "patient_id": extract_patient_id(path),
        "center": extract_center(path, root_dir),
        "filename": path.name,
        "full_path": str(path),
        "detected_modality": None,
        "detect_method": None,
        "shape_zyx": None,
        "spacing_xyz": None,
        "dtype": None,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "unique_count_sampled": None,
        "foreground_voxels": None,
        "foreground_fraction": None,
        "status": "ok",
        "error": None,
    }

    try:
        modality, method = classify_file(path)
        row["detected_modality"] = modality
        row["detect_method"] = method

        _, arr, spacing, _, _ = read_image(path)

        row["shape_zyx"] = tuple(arr.shape)
        row["spacing_xyz"] = tuple(float(x) for x in spacing)
        row["dtype"] = str(arr.dtype)

        stats = image_stats(arr)
        row["min"] = stats["min"]
        row["max"] = stats["max"]
        row["mean"] = stats["mean"]
        row["std"] = stats["std"]
        row["unique_count_sampled"] = sample_unique_count(arr)

        if modality == "MASK":
            fg = arr > 0
            row["foreground_voxels"] = int(fg.sum())
            row["foreground_fraction"] = float(fg.mean())

    except Exception as e:
        row["status"] = "error"
        row["error"] = str(e)

    return row


def summarize(df: pd.DataFrame):
    print("\n===== FILE-LEVEL SUMMARY =====")
    print(f"Total files: {len(df)}")
    print("\nDetected modalities:")
    print(df["detected_modality"].value_counts(dropna=False))

    print("\nCenters found:")
    print(df["center"].value_counts(dropna=False).head(20))

    print("\nExample files per modality:")
    for mod in ["CT", "PET", "MASK", "RTDOSE", "RTSTRUCT", "UNKNOWN"]:
        sub = df[df["detected_modality"] == mod]
        if len(sub) > 0:
            print(f"\n{mod}:")
            print(sub[["patient_id", "center", "filename"]].head(5).to_string(index=False))

    ct_df = df[(df["status"] == "ok") & (df["detected_modality"] == "CT")]
    pet_df = df[(df["status"] == "ok") & (df["detected_modality"] == "PET")]
    mask_df = df[(df["status"] == "ok") & (df["detected_modality"] == "MASK")]

    if len(ct_df) > 0:
        print("\nCT intensity summary:")
        print(ct_df[["min", "max", "mean", "std"]].describe())

    if len(pet_df) > 0:
        print("\nPET intensity summary:")
        print(pet_df[["min", "max", "mean", "std"]].describe())

    if len(mask_df) > 0:
        print("\nMASK foreground summary:")
        print(mask_df[["foreground_voxels", "foreground_fraction"]].describe())


def save_patient_inventory(df: pd.DataFrame, out_path: Path):
    rows = []
    for pid, grp in df.groupby("patient_id"):
        items = {"patient_id": pid}
        for mod in ["CT", "PET", "MASK", "RTDOSE", "RTSTRUCT", "UNKNOWN"]:
            files = grp[grp["detected_modality"] == mod]["filename"].tolist()
            items[f"{mod.lower()}_count"] = len(files)
            items[f"{mod.lower()}_files"] = " | ".join(files[:10]) if files else ""
        centers = grp["center"].mode()
        items["center"] = centers.iloc[0] if len(centers) else "unknown"
        rows.append(items)

    patient_df = pd.DataFrame(rows).sort_values("patient_id")
    patient_df.to_csv(out_path, index=False)
    return patient_df


def preview_folders(root_dir: Path, max_lines: int = 120):
    print("\n===== FOLDER PREVIEW =====")
    count = 0
    for p in sorted(root_dir.rglob("*")):
        if count >= max_lines:
            print("... (truncated)")
            break
        rel = p.relative_to(root_dir)
        if p.is_dir():
            print(f"[DIR]  {rel}")
        else:
            print(f"[FILE] {rel}")
        count += 1


# ============================================================
# PREVIEW IMAGES
# ============================================================
def safe_mid_slice(arr: np.ndarray) -> int:
    if arr is None or arr.ndim == 0:
        return 0
    return max(0, arr.shape[0] // 2)


def save_quick_previews(df: pd.DataFrame, out_dir: Path, n: int = 3):
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    patient_groups = df.groupby("patient_id")
    count = 0

    for pid, grp in patient_groups:
        if count >= n:
            break

        ct_rows = grp[grp["detected_modality"] == "CT"]
        pet_rows = grp[grp["detected_modality"] == "PET"]
        mask_rows = grp[grp["detected_modality"] == "MASK"]

        if len(ct_rows) == 0 or len(pet_rows) == 0:
            continue

        ct_path = Path(ct_rows.iloc[0]["full_path"])
        pet_path = Path(pet_rows.iloc[0]["full_path"])

        _, ct_arr, _, _, _ = read_image(ct_path)
        _, pet_arr, _, _, _ = read_image(pet_path)

        mask_arr = None
        if len(mask_rows) > 0:
            mask_path = Path(mask_rows.iloc[0]["full_path"])
            _, mask_arr, _, _, _ = read_image(mask_path)

        ct_z = safe_mid_slice(ct_arr)
        pet_z = safe_mid_slice(pet_arr)
        mask_z = safe_mid_slice(mask_arr) if mask_arr is not None else 0

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(ct_arr[ct_z], cmap="gray")
        axes[0].set_title(f"{pid} - CT (slice {ct_z})")
        axes[0].axis("off")

        axes[1].imshow(pet_arr[pet_z], cmap="gray")
        axes[1].set_title(f"{pid} - PET (slice {pet_z})")
        axes[1].axis("off")

        if mask_arr is not None:
            axes[2].imshow(mask_arr[mask_z], cmap="gray")
            axes[2].set_title(f"{pid} - MASK (slice {mask_z})")
        else:
            axes[2].text(0.5, 0.5, "No mask found", ha="center", va="center")
            axes[2].set_title(f"{pid} - MASK")
        axes[2].axis("off")

        plt.tight_layout()
        out_file = out_dir / f"{pid}_preview.png"
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        count += 1


# ============================================================
# MAIN
# ============================================================
def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {ROOT_DIR}")

    print(f"Scanning dataset under: {ROOT_DIR}")

    preview_folders(ROOT_DIR, max_lines=80)

    files = discover_all_image_files(ROOT_DIR)
    print(f"\nFound {len(files)} image files in total.")

    if not files:
        raise RuntimeError("No image files found. Check the folder path and extensions.")

    records = []
    for p in tqdm(files, desc="Inspecting files"):
        records.append(inspect_file(p, ROOT_DIR))

    df = pd.DataFrame(records)

    file_report = ROOT_DIR / "hecktor_file_inventory.csv"
    df.to_csv(file_report, index=False)
    print(f"\nSaved file inventory to: {file_report}")

    patient_report = ROOT_DIR / "hecktor_patient_inventory.csv"
    save_patient_inventory(df, patient_report)
    print(f"Saved patient inventory to: {patient_report}")

    summary = {
        "total_files": int(len(df)),
        "total_patients": int(df["patient_id"].nunique()),
        "modalities": df["detected_modality"].value_counts(dropna=False).to_dict(),
        "centers": df["center"].value_counts(dropna=False).to_dict(),
    }

    summary_path = ROOT_DIR / "hecktor_inspection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))

    summarize(df)

    preview_dir = ROOT_DIR / "qc_previews"
    save_quick_previews(df, preview_dir, n=3)
    print(f"\nSaved preview images to: {preview_dir}")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()