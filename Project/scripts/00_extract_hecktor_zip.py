#!/usr/bin/env python3
"""
Step 00: Extract the HECKTOR 2025 training dataset zip file.

This script:
1. Extracts the HECKTOR zip file into a target folder.
2. Prevents unsafe zip path traversal.
3. Detects the extracted dataset root.
4. Creates a patient-level inventory.
5. Counts cases by center.
6. Writes summary files for reproducibility.

Expected patient folder format:
    CENTER-XXX/
        CENTER-XXX__CT.nii.gz
        CENTER-XXX__PT.nii.gz
        CENTER-XXX.nii.gz

Example:
    python scripts/00_extract_hecktor_zip.py \
        --zip data/raw/HECKTOR_2025_Training_Data_Defaced_ALL.zip \
        --out data/extracted
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path


def is_safe_zip_member(output_dir: Path, member_name: str) -> bool:
    """
    Check that a zip member will be extracted inside output_dir.
    This prevents path traversal such as ../../file.
    """
    output_dir = output_dir.resolve()
    target_path = (output_dir / member_name).resolve()

    try:
        target_path.relative_to(output_dir)
        return True
    except ValueError:
        return False


def safe_extract_zip(zip_path: Path, output_dir: Path, overwrite: bool = False) -> None:
    """
    Safely extract a zip file.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"File is not a valid zip archive: {zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for item in output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()

        for member in members:
            name = member.filename

            # Skip macOS metadata files.
            if name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
                continue

            if not is_safe_zip_member(output_dir, name):
                raise RuntimeError(f"Unsafe zip member blocked: {name}")

            zf.extract(member, output_dir)


def looks_like_patient_folder(folder: Path) -> bool:
    """
    A patient folder should have a name like CHUM-001 or MDA-258.
    """
    if not folder.is_dir():
        return False

    name = folder.name

    if "-" not in name:
        return False

    center, case_id = name.split("-", 1)

    if len(center) < 2:
        return False

    if not center.isalpha():
        return False

    if len(case_id) == 0:
        return False

    return True


def find_dataset_root(output_dir: Path) -> Path:
    """
    Find the folder that contains patient folders.

    Handles both cases:
    1. Zip extracts directly into output_dir.
    2. Zip extracts into one top-level folder inside output_dir.
    """
    output_dir = output_dir.resolve()

    direct_patient_dirs = [
        p for p in output_dir.iterdir()
        if looks_like_patient_folder(p)
    ]

    if direct_patient_dirs:
        return output_dir

    candidate_dirs = [
        p for p in output_dir.iterdir()
        if p.is_dir() and not p.name.startswith("__MACOSX")
    ]

    for candidate in candidate_dirs:
        patient_dirs = [
            p for p in candidate.iterdir()
            if looks_like_patient_folder(p)
        ]

        if patient_dirs:
            return candidate

    # Fallback: recursive search.
    for candidate in output_dir.rglob("*"):
        if not candidate.is_dir():
            continue

        patient_dirs = [
            p for p in candidate.iterdir()
            if looks_like_patient_folder(p)
        ]

        if patient_dirs:
            return candidate

    raise RuntimeError(
        "Could not find a dataset root containing patient folders. "
        "Check whether the zip file extracted correctly."
    )


def build_patient_inventory(dataset_root: Path) -> list[dict]:
    """
    Build a patient-level inventory of CT, PET, mask, and optional RTDOSE files.
    """
    rows = []

    patient_dirs = sorted(
        [p for p in dataset_root.iterdir() if looks_like_patient_folder(p)]
    )

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        center = patient_id.split("-", 1)[0]

        ct_path = patient_dir / f"{patient_id}__CT.nii.gz"
        pet_path = patient_dir / f"{patient_id}__PT.nii.gz"
        mask_path = patient_dir / f"{patient_id}.nii.gz"
        rtdose_path = patient_dir / f"{patient_id}__RTDOSE.nii.gz"

        has_ct = ct_path.exists()
        has_pet = pet_path.exists()
        has_mask = mask_path.exists()
        has_rtdose = rtdose_path.exists()

        complete_ct_pet_mask = has_ct and has_pet and has_mask

        rows.append(
            {
                "patient_id": patient_id,
                "center": center,
                "patient_folder": str(patient_dir),
                "ct_path": str(ct_path) if has_ct else "",
                "pet_path": str(pet_path) if has_pet else "",
                "mask_path": str(mask_path) if has_mask else "",
                "rtdose_path": str(rtdose_path) if has_rtdose else "",
                "has_ct": has_ct,
                "has_pet": has_pet,
                "has_mask": has_mask,
                "has_rtdose": has_rtdose,
                "complete_ct_pet_mask": complete_ct_pet_mask,
            }
        )

    return rows


def write_csv(rows: list[dict], output_csv: Path) -> None:
    """
    Write patient inventory to CSV.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_csv.write_text("")
        return

    fieldnames = list(rows[0].keys())

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    zip_path: Path,
    output_dir: Path,
    dataset_root: Path,
    rows: list[dict],
    output_json: Path,
) -> None:
    """
    Write extraction summary JSON.
    """
    center_counts = Counter(row["center"] for row in rows)
    complete_counts = Counter(
        row["center"] for row in rows if row["complete_ct_pet_mask"]
    )

    summary = {
        "zip_path": str(zip_path),
        "extraction_output_dir": str(output_dir),
        "dataset_root": str(dataset_root),
        "total_patient_folders": len(rows),
        "total_complete_ct_pet_mask_cases": int(
            sum(row["complete_ct_pet_mask"] for row in rows)
        ),
        "center_counts": dict(sorted(center_counts.items())),
        "center_complete_ct_pet_mask_counts": dict(sorted(complete_counts.items())),
        "expected_files": {
            "ct": "PatientID__CT.nii.gz",
            "pet": "PatientID__PT.nii.gz",
            "mask": "PatientID.nii.gz",
            "optional_rtdose": "PatientID__RTDOSE.nii.gz",
        },
        "label_definitions": {
            "0": "background",
            "1": "GTVp_primary_gross_tumor_volume",
            "2": "GTVn_nodal_gross_tumor_volume",
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w") as f:
        json.dump(summary, f, indent=2)


def print_summary(rows: list[dict], dataset_root: Path) -> None:
    """
    Print center-wise summary to terminal.
    """
    center_counts = Counter(row["center"] for row in rows)
    complete_counts = Counter(
        row["center"] for row in rows if row["complete_ct_pet_mask"]
    )

    print("\nExtraction complete")
    print("=" * 60)
    print(f"Dataset root: {dataset_root}")
    print(f"Total patient folders: {len(rows)}")
    print(
        "Total complete CT/PET/mask cases: "
        f"{sum(row['complete_ct_pet_mask'] for row in rows)}"
    )

    print("\nCenter-wise counts")
    print("-" * 60)
    print(f"{'Center':<10}{'Patient folders':>18}{'Complete cases':>18}")

    for center in sorted(center_counts):
        print(
            f"{center:<10}"
            f"{center_counts[center]:>18}"
            f"{complete_counts.get(center, 0):>18}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract HECKTOR 2025 zip file and create a patient inventory."
    )

    parser.add_argument(
        "--zip",
        required=True,
        type=Path,
        help="Path to HECKTOR 2025 zip file.",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output folder where the dataset will be extracted.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output contents before extraction.",
    )

    parser.add_argument(
        "--inventory_csv",
        default=None,
        type=Path,
        help="Optional path for patient inventory CSV.",
    )

    parser.add_argument(
        "--summary_json",
        default=None,
        type=Path,
        help="Optional path for extraction summary JSON.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    zip_path = args.zip.resolve()
    output_dir = args.out.resolve()

    safe_extract_zip(
        zip_path=zip_path,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )

    dataset_root = find_dataset_root(output_dir)
    rows = build_patient_inventory(dataset_root)

    if not rows:
        raise RuntimeError(
            "No patient folders were found after extraction. "
            "Check the zip file and folder structure."
        )

    inventory_csv = (
        args.inventory_csv
        if args.inventory_csv is not None
        else output_dir / "extraction_patient_inventory.csv"
    )

    summary_json = (
        args.summary_json
        if args.summary_json is not None
        else output_dir / "extraction_summary.json"
    )

    write_csv(rows, inventory_csv)
    write_summary(
        zip_path=zip_path,
        output_dir=output_dir,
        dataset_root=dataset_root,
        rows=rows,
        output_json=summary_json,
    )

    print_summary(rows, dataset_root)

    print("\nSaved files")
    print("=" * 60)
    print(f"Inventory CSV: {inventory_csv}")
    print(f"Summary JSON:  {summary_json}")


if __name__ == "__main__":
    main()
