#!/usr/bin/env python3
"""
Step 04: Run local nnU-Net inference using trained LOCO model weights.

Input:
    local_nnunet/nnUNet_raw/DatasetXXX_clean_raw_LOCO_CENTER/imagesTs/
    local_nnunet/nnUNet_results/DatasetXXX_clean_raw_LOCO_CENTER/
        nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth

Output:
    local_nnunet/predictions/DatasetXXX_clean_raw_LOCO_CENTER/*.nii.gz
    local_nnunet/predictions/inference_summary.csv
    local_nnunet/predictions/inference_summary.json
    local_nnunet/predictions/qc_overlays/

Run from repo root:
    python Project/04_run_local_inference.py \
        --nnunet_raw local_nnunet/nnUNet_raw \
        --nnunet_results local_nnunet/nnUNet_results \
        --out local_nnunet/predictions \
        --configuration 3d_fullres \
        --fold 0 \
        --checkpoint checkpoint_best.pth
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch


try:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ImportError as exc:
    raise ImportError(
        "Could not import nnunetv2. Install it with: pip install nnunetv2"
    ) from exc


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


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(json_safe(data), f, indent=2)


def parse_dataset_id(dataset_name: str) -> int:
    match = re.match(r"^Dataset(\d{3})_.+", dataset_name)

    if not match:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    return int(match.group(1))


def parse_dataset_filter(value: str | None) -> set[str] | None:
    """
    Allows:
        --datasets Dataset001_clean_raw_LOCO_CHUM,Dataset002_clean_raw_LOCO_CHUP
        --datasets 001,002
        --datasets CHUM,MDA
    """
    if value is None:
        return None

    items = {x.strip() for x in value.split(",") if x.strip()}

    return items if items else None


def dataset_matches_filter(dataset_name: str, filter_items: set[str] | None) -> bool:
    if filter_items is None:
        return True

    dataset_id = f"{parse_dataset_id(dataset_name):03d}"

    for item in filter_items:
        item_upper = item.upper()

        if item == dataset_name:
            return True

        if item == dataset_id:
            return True

        if item_upper in dataset_name.upper():
            return True

    return False


def find_dataset_dirs(nnunet_raw: Path, dataset_filter: set[str] | None) -> list[Path]:
    if not nnunet_raw.exists():
        raise FileNotFoundError(f"nnUNet_raw folder not found: {nnunet_raw}")

    dataset_dirs = []

    for path in sorted(nnunet_raw.iterdir()):
        if not path.is_dir():
            continue

        if not re.match(r"^Dataset\d{3}_.+", path.name):
            continue

        if not (path / "dataset.json").exists():
            continue

        if not dataset_matches_filter(path.name, dataset_filter):
            continue

        dataset_dirs.append(path)

    if not dataset_dirs:
        raise RuntimeError(
            f"No matching DatasetXXX_* folders found in {nnunet_raw}"
        )

    return dataset_dirs


def parse_folds(value: str) -> tuple[int | str, ...]:
    """
    nnU-Net accepts folds such as:
        (0,)
        (0, 1, 2, 3, 4)
        ('all',)
    """
    value = str(value).strip().lower()

    if value == "all":
        return ("all",)

    folds = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        folds.append(int(item))

    if not folds:
        raise ValueError("No valid folds were provided.")

    return tuple(folds)


def select_device(device_arg: str, gpu_index: int) -> torch.device:
    device_arg = device_arg.lower()

    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", gpu_index)
        return torch.device("cpu")

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        return torch.device("cuda", gpu_index)

    if device_arg == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {device_arg}")


def count_input_images(images_ts: Path) -> int:
    return len(sorted(images_ts.glob("*_0000.nii.gz")))


def count_prediction_files(pred_dir: Path) -> int:
    return len(sorted(pred_dir.glob("*.nii.gz")))


# ============================================================
# MODEL FOLDER HELPERS
# ============================================================

def find_model_folder(
    nnunet_results: Path,
    dataset_name: str,
    trainer: str,
    plans: str,
    configuration: str,
) -> Path:
    """
    Expected:
        nnUNet_results/DatasetXXX_Name/nnUNetTrainer__nnUNetPlans__3d_fullres

    This function also handles cases where the model archive created one extra nesting level.
    """
    expected_relative = Path(dataset_name) / f"{trainer}__{plans}__{configuration}"
    direct = nnunet_results / expected_relative

    if direct.exists():
        return direct

    matches = []

    for candidate in nnunet_results.rglob(f"{trainer}__{plans}__{configuration}"):
        if not candidate.is_dir():
            continue

        if candidate.parent.name == dataset_name:
            matches.append(candidate)

    if matches:
        return sorted(matches, key=lambda p: len(str(p)))[0]

    raise FileNotFoundError(
        "Could not find model folder for dataset.\n"
        f"Dataset: {dataset_name}\n"
        f"Expected model folder: {direct}"
    )


def checkpoint_exists_for_folds(
    model_folder: Path,
    folds: tuple[int | str, ...],
    checkpoint_name: str,
) -> bool:
    for fold in folds:
        fold_folder = model_folder / f"fold_{fold}"

        if fold == "all":
            fold_folder = model_folder / "fold_all"

        if not (fold_folder / checkpoint_name).exists():
            return False

    return True


def choose_checkpoint(
    model_folder: Path,
    folds: tuple[int | str, ...],
    requested_checkpoint: str,
    fallback_checkpoints: list[str],
) -> str:
    """
    Use requested checkpoint if present. Otherwise try fallbacks.
    """
    if checkpoint_exists_for_folds(model_folder, folds, requested_checkpoint):
        return requested_checkpoint

    for fallback in fallback_checkpoints:
        if checkpoint_exists_for_folds(model_folder, folds, fallback):
            print(
                f"Requested checkpoint {requested_checkpoint} not found for all folds. "
                f"Using fallback checkpoint: {fallback}"
            )
            return fallback

    checked = [requested_checkpoint] + fallback_checkpoints

    raise FileNotFoundError(
        "No usable checkpoint found.\n"
        f"Model folder: {model_folder}\n"
        f"Folds: {folds}\n"
        f"Checked checkpoint names: {checked}"
    )


# ============================================================
# QC OVERLAY HELPERS
# ============================================================

def read_nifti_array(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image)


def normalize_for_display(arr: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)

    finite = arr[np.isfinite(arr)]

    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    lo, hi = np.percentile(finite, [p_low, p_high])

    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)

    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)

    return arr


def pick_best_slice(mask_arr: np.ndarray) -> int:
    """
    Select the axial slice with largest predicted foreground.
    """
    foreground = mask_arr > 0
    areas = foreground.sum(axis=(1, 2))

    if areas.max() > 0:
        return int(np.argmax(areas))

    return mask_arr.shape[0] // 2


def save_prediction_overlay(
    ct_path: Path,
    pred_path: Path,
    out_png: Path,
    title: str,
) -> None:
    ct_arr = read_nifti_array(ct_path)
    pred_arr = read_nifti_array(pred_path)

    if ct_arr.shape != pred_arr.shape:
        print(
            f"Skipping QC overlay due to shape mismatch: "
            f"CT {ct_arr.shape}, prediction {pred_arr.shape}"
        )
        return

    z = pick_best_slice(pred_arr)

    ct_slice = normalize_for_display(ct_arr[z])
    pred_slice = pred_arr[z] > 0

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(ct_slice, cmap="gray")

    if np.any(pred_slice):
        ax.contour(
            pred_slice.astype(float),
            levels=[0.5],
            colors="red",
            linewidths=2,
        )

    ax.set_title(f"{title}\nSlice {z}", fontsize=11)
    ax.axis("off")

    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_qc_overlays(
    images_ts: Path,
    pred_dir: Path,
    qc_dir: Path,
    dataset_name: str,
    max_cases: int,
) -> int:
    if max_cases <= 0:
        return 0

    image_files = sorted(images_ts.glob("*_0000.nii.gz"))

    if not image_files:
        return 0

    selected = image_files[: min(max_cases, len(image_files))]
    saved = 0

    for image_file in selected:
        case_id = image_file.name.replace("_0000.nii.gz", "")
        pred_file = pred_dir / f"{case_id}.nii.gz"

        if not pred_file.exists():
            continue

        out_png = qc_dir / dataset_name / f"{case_id}_prediction_overlay.png"

        save_prediction_overlay(
            ct_path=image_file,
            pred_path=pred_file,
            out_png=out_png,
            title=f"{dataset_name} - {case_id}",
        )

        saved += 1

    return saved


# ============================================================
# INFERENCE
# ============================================================

def run_one_dataset(
    dataset_dir: Path,
    nnunet_results: Path,
    out_root: Path,
    qc_root: Path,
    folds: tuple[int | str, ...],
    configuration: str,
    trainer: str,
    plans: str,
    checkpoint: str,
    fallback_checkpoints: list[str],
    device: torch.device,
    tile_step_size: float,
    use_mirroring: bool,
    overwrite: bool,
    save_probabilities: bool,
    num_processes_preprocessing: int,
    num_processes_segmentation_export: int,
    qc_samples: int,
) -> dict[str, Any]:
    dataset_name = dataset_dir.name
    dataset_id = parse_dataset_id(dataset_name)

    images_ts = dataset_dir / "imagesTs"

    if not images_ts.exists():
        raise FileNotFoundError(f"imagesTs folder not found: {images_ts}")

    input_count = count_input_images(images_ts)

    if input_count == 0:
        raise RuntimeError(f"No test images found in: {images_ts}")

    pred_dir = out_root / dataset_name
    pred_dir.mkdir(parents=True, exist_ok=True)

    existing_predictions = count_prediction_files(pred_dir)

    if existing_predictions >= input_count and not overwrite:
        print(f"\nSkipping {dataset_name}: predictions already exist.")
        print(f"Prediction folder: {pred_dir}")

        qc_saved = make_qc_overlays(
            images_ts=images_ts,
            pred_dir=pred_dir,
            qc_dir=qc_root,
            dataset_name=dataset_name,
            max_cases=qc_samples,
        )

        return {
            "dataset_name": dataset_name,
            "dataset_id": dataset_id,
            "status": "skipped_existing",
            "input_cases": input_count,
            "prediction_cases": existing_predictions,
            "qc_overlays": qc_saved,
            "model_folder": "",
            "checkpoint": "",
            "runtime_seconds": 0.0,
        }

    model_folder = find_model_folder(
        nnunet_results=nnunet_results,
        dataset_name=dataset_name,
        trainer=trainer,
        plans=plans,
        configuration=configuration,
    )

    checkpoint_to_use = choose_checkpoint(
        model_folder=model_folder,
        folds=folds,
        requested_checkpoint=checkpoint,
        fallback_checkpoints=fallback_checkpoints,
    )

    print("\nRunning inference")
    print("=" * 80)
    print(f"Dataset:       {dataset_name}")
    print(f"Dataset ID:    {dataset_id}")
    print(f"Input folder:  {images_ts}")
    print(f"Output folder: {pred_dir}")
    print(f"Model folder:  {model_folder}")
    print(f"Folds:         {folds}")
    print(f"Checkpoint:    {checkpoint_to_use}")
    print(f"Device:        {device}")

    start_time = time.time()

    predictor = nnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=(device.type == "cuda"),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_folder),
        use_folds=folds,
        checkpoint_name=checkpoint_to_use,
    )

    predictor.predict_from_files(
        list_of_lists_or_source_folder=str(images_ts),
        output_folder_or_list_of_truncated_output_files=str(pred_dir),
        save_probabilities=save_probabilities,
        overwrite=overwrite,
        num_processes_preprocessing=num_processes_preprocessing,
        num_processes_segmentation_export=num_processes_segmentation_export,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    runtime_seconds = time.time() - start_time

    prediction_count = count_prediction_files(pred_dir)

    qc_saved = make_qc_overlays(
        images_ts=images_ts,
        pred_dir=pred_dir,
        qc_dir=qc_root,
        dataset_name=dataset_name,
        max_cases=qc_samples,
    )

    print(f"Finished inference for {dataset_name}")
    print(f"Predictions written: {prediction_count}")
    print(f"QC overlays written: {qc_saved}")
    print(f"Runtime seconds: {runtime_seconds:.2f}")

    return {
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "status": "completed",
        "input_cases": input_count,
        "prediction_cases": prediction_count,
        "qc_overlays": qc_saved,
        "model_folder": str(model_folder),
        "checkpoint": checkpoint_to_use,
        "runtime_seconds": float(runtime_seconds),
    }


def set_nnunet_environment(
    nnunet_raw: Path,
    nnunet_results: Path,
    nnunet_preprocessed: Path | None,
) -> None:
    """
    Set nnU-Net environment variables for local inference.
    """
    if nnunet_preprocessed is None:
        nnunet_preprocessed = nnunet_raw.parent / "nnUNet_preprocessed"

    os.environ["nnUNet_raw"] = str(nnunet_raw.resolve())
    os.environ["nnUNet_preprocessed"] = str(nnunet_preprocessed.resolve())
    os.environ["nnUNet_results"] = str(nnunet_results.resolve())

    print("\nnnU-Net environment")
    print("=" * 80)
    print(f"nnUNet_raw:          {os.environ['nnUNet_raw']}")
    print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
    print(f"nnUNet_results:      {os.environ['nnUNet_results']}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local nnU-Net inference using trained LOCO model weights."
    )

    parser.add_argument(
        "--nnunet_raw",
        required=True,
        type=Path,
        help="Path to local_nnunet/nnUNet_raw.",
    )

    parser.add_argument(
        "--nnunet_results",
        required=True,
        type=Path,
        help="Path to local_nnunet/nnUNet_results containing trained models.",
    )

    parser.add_argument(
        "--nnunet_preprocessed",
        default=None,
        type=Path,
        help="Optional path to local_nnunet/nnUNet_preprocessed.",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output prediction root, usually local_nnunet/predictions.",
    )

    parser.add_argument(
        "--datasets",
        default=None,
        type=str,
        help=(
            "Optional comma-separated dataset filter. "
            "Examples: 001,002 or CHUM,MDA or Dataset001_clean_raw_LOCO_CHUM"
        ),
    )

    parser.add_argument(
        "--configuration",
        default="3d_fullres",
        help="nnU-Net configuration. Default: 3d_fullres.",
    )

    parser.add_argument(
        "--fold",
        default="0",
        help="Fold to use. Examples: 0, 0,1,2,3,4, or all.",
    )

    parser.add_argument(
        "--trainer",
        default="nnUNetTrainer",
        help="Trainer name. Default: nnUNetTrainer.",
    )

    parser.add_argument(
        "--plans",
        default="nnUNetPlans",
        help="Plans name. Default: nnUNetPlans.",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoint_best.pth",
        help="Checkpoint for inference. Default: checkpoint_best.pth.",
    )

    parser.add_argument(
        "--fallback_checkpoints",
        default="checkpoint_final.pth,checkpoint_latest.pth",
        help="Comma-separated fallback checkpoints if requested checkpoint is missing.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device. Default: auto.",
    )

    parser.add_argument(
        "--gpu_index",
        default=0,
        type=int,
        help="CUDA GPU index if using CUDA. Default: 0.",
    )

    parser.add_argument(
        "--tile_step_size",
        default=0.5,
        type=float,
        help="nnU-Net tile step size. Default: 0.5.",
    )

    parser.add_argument(
        "--disable_mirroring",
        action="store_true",
        help="Disable test-time mirroring.",
    )

    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        help="Save softmax probabilities in addition to segmentation.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions.",
    )

    parser.add_argument(
        "--num_processes_preprocessing",
        default=2,
        type=int,
        help="Number of preprocessing processes. Default: 2.",
    )

    parser.add_argument(
        "--num_processes_segmentation_export",
        default=2,
        type=int,
        help="Number of export processes. Default: 2.",
    )

    parser.add_argument(
        "--qc_samples",
        default=3,
        type=int,
        help="Number of quick QC overlays per dataset. Default: 3.",
    )

    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue with other datasets if one dataset fails.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_nnunet_environment(
        nnunet_raw=args.nnunet_raw,
        nnunet_results=args.nnunet_results,
        nnunet_preprocessed=args.nnunet_preprocessed,
    )

    dataset_filter = parse_dataset_filter(args.datasets)
    dataset_dirs = find_dataset_dirs(args.nnunet_raw, dataset_filter)

    folds = parse_folds(args.fold)

    fallback_checkpoints = [
        x.strip()
        for x in args.fallback_checkpoints.split(",")
        if x.strip()
    ]

    device = select_device(args.device, gpu_index=args.gpu_index)

    args.out.mkdir(parents=True, exist_ok=True)

    qc_root = args.out / "qc_overlays"
    qc_root.mkdir(parents=True, exist_ok=True)

    print("\nInference Configuration")
    print("=" * 80)
    print(f"Datasets found: {len(dataset_dirs)}")
    print(f"Output root:    {args.out}")
    print(f"QC root:        {qc_root}")
    print(f"Configuration:  {args.configuration}")
    print(f"Fold:           {folds}")
    print(f"Checkpoint:     {args.checkpoint}")
    print(f"Device:         {device}")

    summary_rows = []

    for dataset_dir in dataset_dirs:
        try:
            row = run_one_dataset(
                dataset_dir=dataset_dir,
                nnunet_results=args.nnunet_results,
                out_root=args.out,
                qc_root=qc_root,
                folds=folds,
                configuration=args.configuration,
                trainer=args.trainer,
                plans=args.plans,
                checkpoint=args.checkpoint,
                fallback_checkpoints=fallback_checkpoints,
                device=device,
                tile_step_size=args.tile_step_size,
                use_mirroring=not args.disable_mirroring,
                overwrite=args.overwrite,
                save_probabilities=args.save_probabilities,
                num_processes_preprocessing=args.num_processes_preprocessing,
                num_processes_segmentation_export=args.num_processes_segmentation_export,
                qc_samples=args.qc_samples,
            )

            summary_rows.append(row)

        except Exception as exc:
            error_row = {
                "dataset_name": dataset_dir.name,
                "dataset_id": parse_dataset_id(dataset_dir.name),
                "status": "failed",
                "input_cases": None,
                "prediction_cases": None,
                "qc_overlays": None,
                "model_folder": "",
                "checkpoint": "",
                "runtime_seconds": None,
                "error": str(exc),
            }

            summary_rows.append(error_row)

            print(f"\nERROR while processing {dataset_dir.name}")
            print("=" * 80)
            print(str(exc))

            if not args.continue_on_error:
                raise

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = args.out / "inference_summary.csv"
    summary_json = args.out / "inference_summary.json"

    summary_df.to_csv(summary_csv, index=False)
    write_json(summary_json, summary_rows)

    print("\nStep 04 Complete")
    print("=" * 80)
    print(f"Inference summary CSV:  {summary_csv}")
    print(f"Inference summary JSON: {summary_json}")
    print(f"Prediction root:        {args.out}")
    print(f"QC overlays:            {qc_root}")

    print("\nNext step:")
    print(
        "python Project/05_evaluate_predictions.py "
        "--nnunet_raw local_nnunet/nnUNet_raw "
        "--pred_root local_nnunet/predictions "
        "--out local_nnunet/results"
    )


if __name__ == "__main__":
    main()
