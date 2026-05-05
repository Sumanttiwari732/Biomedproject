from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import json


# CONFIG

ROOT_DIR = Path("/Users/sumantratiwari/PycharmProjects/Biomedproject/HECKTOR 2025 Training Data Defaced ALL")

OUT_DIR = ROOT_DIR / "step1_3_flagged"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUT_DIR / "ct_dataset_raw.csv"
NORM_CSV = OUT_DIR / "ct_dataset_normalized.csv"
RAW_JSON = OUT_DIR / "summary_raw.json"
NORM_JSON = OUT_DIR / "summary_normalized.json"
ABNORMAL_CSV = OUT_DIR / "abnormal_cases.csv"
ABNORMAL_ANALYSIS_CSV = OUT_DIR / "abnormal_by_center.csv"

VALID_EXTS = [".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"]

# HU thresholds
HU_MIN = -1000
HU_MAX = 400
ABNORMAL_LOW = -2000
ABNORMAL_HIGH = 5000



# HELPERS

def has_valid_extension(path: Path):
    return any(path.name.lower().endswith(ext) for ext in VALID_EXTS)


def extract_center(pid: str):
    return pid.split("-")[0]


def find_ct_and_mask(folder: Path):
    files = [p for p in folder.iterdir() if p.is_file() and has_valid_extension(p)]

    ct_candidates = []
    mask_candidates = []

    for f in files:
        name = f.name.lower()

        if "__ct" in name:
            ct_candidates.append(f)
        elif "__pt" in name or "__rtdose" in name:
            continue
        else:
            mask_candidates.append(f)

    ct = sorted(ct_candidates)[0] if ct_candidates else None
    mask = sorted(mask_candidates)[0] if mask_candidates else None

    return ct, mask


def load_ct(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return img, arr


def compute_stats(arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None

    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def detect_abnormal(arr):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False, "no_finite_values"

    reasons = []

    if finite.max() > ABNORMAL_HIGH:
        reasons.append("high_intensity_outlier")

    if finite.min() < ABNORMAL_LOW:
        reasons.append("low_intensity_outlier")

    return len(reasons) > 0, "|".join(reasons)



# CORE PROCESSING

def process_dataset():
    raw_records = []
    norm_records = []
    abnormal_records = []

    folders = [p for p in ROOT_DIR.iterdir() if p.is_dir()]
    print(f"Found {len(folders)} patient folders")

    for folder in tqdm(folders, desc="Processing patients"):
        pid = folder.name
        center = extract_center(pid)

        ct_path, mask_path = find_ct_and_mask(folder)

        if ct_path is None or mask_path is None:
            continue

        try:
            img, arr = load_ct(ct_path)

            is_abnormal, reason = detect_abnormal(arr)

            if is_abnormal:
                abnormal_records.append({
                    "patient_id": pid,
                    "center": center,
                    "ct_path": str(ct_path),
                    "reason": reason,
                    "min_raw": float(np.nanmin(arr)),
                    "max_raw": float(np.nanmax(arr)),
                })

            # Clip HU
            arr_clipped = np.clip(arr, HU_MIN, HU_MAX)

            raw_stats = compute_stats(arr_clipped)
            if raw_stats is None:
                continue

            # Normalize
            finite = arr_clipped[np.isfinite(arr_clipped)]
            mean = finite.mean()
            std = finite.std()
            if std < 1e-6:
                std = 1.0

            arr_norm = (arr_clipped - mean) / std
            norm_stats = compute_stats(arr_norm)

            base = {
                "patient_id": pid,
                "center": center,
                "ct_path": str(ct_path),
                "mask_path": str(mask_path),
                "spacing_xyz": img.GetSpacing(),
                "shape_zyx": tuple(arr.shape),
                "is_abnormal": is_abnormal,
                "abnormal_reason": reason if is_abnormal else ""
            }

            raw_records.append({**base, **raw_stats})
            norm_records.append({**base, **norm_stats})

        except Exception as e:
            print(f"Error: {pid} → {e}")
            continue

    return (
        pd.DataFrame(raw_records),
        pd.DataFrame(norm_records),
        pd.DataFrame(abnormal_records),
    )



# ANALYSIS 

def analyze_abnormal(df):
    print("\n===== ABNORMAL DISTRIBUTION =====")

    counts = df.groupby("center")["is_abnormal"].sum()
    ratio = df.groupby("center")["is_abnormal"].mean()

    result = pd.DataFrame({
        "num_abnormal": counts,
        "fraction_abnormal": ratio
    })

    print(result)
    result.to_csv(ABNORMAL_ANALYSIS_CSV)

    return result



# MAIN

def main():
    df_raw, df_norm, df_abnormal = process_dataset()

    print(f"\nTotal usable patients: {len(df_raw)}")
    print(f"Abnormal cases: {len(df_abnormal)}")

    df_raw.to_csv(RAW_CSV, index=False)
    df_norm.to_csv(NORM_CSV, index=False)
    df_abnormal.to_csv(ABNORMAL_CSV, index=False)

    print(f"Saved RAW → {RAW_CSV}")
    print(f"Saved NORMALIZED → {NORM_CSV}")
    print(f"Saved ABNORMAL → {ABNORMAL_CSV}")

   
    analyze_abnormal(df_raw)

    print("\nDone: Step 1–3 complete with analysis.")


if __name__ == "__main__":
    main()
