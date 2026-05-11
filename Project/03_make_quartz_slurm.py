#!/usr/bin/env python3
"""
Step 03: Generate Quartz Slurm scripts for nnU-Net LOCO training.

Input:
    local_nnunet/nnUNet_raw/Dataset001_clean_raw_LOCO_CHUM/
    local_nnunet/nnUNet_raw/Dataset002_clean_raw_LOCO_CHUP/
    ...

Output:
    quartz/slurm_jobs/train_Dataset001_clean_raw_LOCO_CHUM.sh
    quartz/slurm_jobs/train_Dataset002_clean_raw_LOCO_CHUP.sh
    ...
    quartz/slurm_jobs/submit_all.sh

Run from repo root:
    python Project/03_make_quartz_slurm.py \
        --nnunet_raw local_nnunet/nnUNet_raw \
        --out quartz/slurm_jobs \
        --quartz_nnunet_raw /N/project/YOUR_PROJECT/local_nnunet/nnUNet_raw \
        --quartz_nnunet_preprocessed /N/project/YOUR_PROJECT/local_nnunet/nnUNet_preprocessed \
        --quartz_nnunet_results /N/project/YOUR_PROJECT/local_nnunet/nnUNet_results \
        --configuration 3d_fullres \
        --fold 0
"""

from __future__ import annotations

import argparse
import re
import stat
from pathlib import Path


# ============================================================
# HELPERS
# ============================================================

def find_dataset_dirs(nnunet_raw: Path) -> list[Path]:
    """
    Find nnU-Net raw dataset folders named DatasetXXX_Name.
    """
    if not nnunet_raw.exists():
        raise FileNotFoundError(f"nnU-Net raw folder not found: {nnunet_raw}")

    dataset_dirs = []

    for path in sorted(nnunet_raw.iterdir()):
        if path.is_dir() and re.match(r"^Dataset\d{3}_.+", path.name):
            dataset_json = path / "dataset.json"

            if dataset_json.exists():
                dataset_dirs.append(path)

    if not dataset_dirs:
        raise RuntimeError(
            f"No DatasetXXX_* folders with dataset.json found in: {nnunet_raw}"
        )

    return dataset_dirs


def parse_dataset_id(dataset_name: str) -> int:
    """
    Extract numeric dataset ID from DatasetXXX_Name.
    Example:
        Dataset001_clean_raw_LOCO_CHUM -> 1
    """
    match = re.match(r"^Dataset(\d{3})_.+", dataset_name)

    if not match:
        raise ValueError(f"Invalid nnU-Net dataset name: {dataset_name}")

    return int(match.group(1))


def make_executable(path: Path) -> None:
    """
    Make a shell script executable.
    """
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def shell_quote(value: str) -> str:
    """
    Simple shell quoting for paths and strings.
    """
    return "'" + value.replace("'", "'\"'\"'") + "'"


# ============================================================
# SLURM SCRIPT GENERATION
# ============================================================

def make_slurm_header(
    job_name: str,
    partition: str,
    account: str | None,
    time: str,
    cpus: int,
    mem: str,
    gpus: int,
    mail_user: str | None,
    mail_type: str | None,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --gres=gpu:{gpus}",
        "#SBATCH --output=logs/%x_%j.out",
        "#SBATCH --error=logs/%x_%j.err",
    ]

    if account:
        lines.append(f"#SBATCH --account={account}")

    if mail_user:
        lines.append(f"#SBATCH --mail-user={mail_user}")

    if mail_type:
        lines.append(f"#SBATCH --mail-type={mail_type}")

    return "\n".join(lines)


def make_environment_block(
    quartz_nnunet_raw: str,
    quartz_nnunet_preprocessed: str,
    quartz_nnunet_results: str,
    module_loads: list[str],
    conda_env: str | None,
    venv_activate: str | None,
) -> str:
    lines = [
        "",
        "set -euo pipefail",
        "",
        "mkdir -p logs",
        "",
        "echo \"Job started on: $(date)\"",
        "echo \"Running on node: $(hostname)\"",
        "echo \"Current directory: $(pwd)\"",
        "",
    ]

    for module in module_loads:
        lines.append(f"module load {module}")

    if module_loads:
        lines.append("")

    if conda_env:
        lines.extend(
            [
                "source ~/.bashrc || true",
                f"conda activate {conda_env}",
                "",
            ]
        )

    if venv_activate:
        lines.extend(
            [
                f"source {shell_quote(venv_activate)}",
                "",
            ]
        )

    lines.extend(
        [
            f"export nnUNet_raw={shell_quote(quartz_nnunet_raw)}",
            f"export nnUNet_preprocessed={shell_quote(quartz_nnunet_preprocessed)}",
            f"export nnUNet_results={shell_quote(quartz_nnunet_results)}",
            "",
            "echo \"nnUNet_raw=$nnUNet_raw\"",
            "echo \"nnUNet_preprocessed=$nnUNet_preprocessed\"",
            "echo \"nnUNet_results=$nnUNet_results\"",
            "",
            "python --version",
            "which python",
            "which nnUNetv2_train",
            "",
        ]
    )

    return "\n".join(lines)


def make_training_block(
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    fold: str,
    trainer: str,
    plans: str,
    checkpoint: str | None,
    skip_preprocess: bool,
    verify_dataset_integrity: bool,
    copy_splits: bool,
    extra_train_args: str,
) -> str:
    lines = [
        f"DATASET_ID={dataset_id}",
        f"DATASET_NAME={shell_quote(dataset_name)}",
        f"CONFIGURATION={shell_quote(configuration)}",
        f"FOLD={shell_quote(fold)}",
        "",
        "RAW_DATASET_DIR=\"$nnUNet_raw/$DATASET_NAME\"",
        "PREPROCESSED_DATASET_DIR=\"$nnUNet_preprocessed/$DATASET_NAME\"",
        "",
        "echo \"Dataset ID: $DATASET_ID\"",
        "echo \"Dataset name: $DATASET_NAME\"",
        "echo \"Raw dataset dir: $RAW_DATASET_DIR\"",
        "echo \"Preprocessed dataset dir: $PREPROCESSED_DATASET_DIR\"",
        "",
        "if [ ! -d \"$RAW_DATASET_DIR\" ]; then",
        "    echo \"ERROR: Raw dataset folder not found: $RAW_DATASET_DIR\"",
        "    exit 1",
        "fi",
        "",
    ]

    if not skip_preprocess:
        preprocess_cmd = f"nnUNetv2_plan_and_preprocess -d {dataset_id}"

        if verify_dataset_integrity:
            preprocess_cmd += " --verify_dataset_integrity"

        lines.extend(
            [
                "echo \"Running nnU-Net planning and preprocessing...\"",
                preprocess_cmd,
                "",
            ]
        )
    else:
        lines.extend(
            [
                "echo \"Skipping planning and preprocessing because --skip_preprocess was used.\"",
                "",
            ]
        )

    if copy_splits:
        lines.extend(
            [
                "echo \"Checking for custom splits_final.json...\"",
                "if [ -f \"$RAW_DATASET_DIR/splits_final.json\" ]; then",
                "    mkdir -p \"$PREPROCESSED_DATASET_DIR\"",
                "    cp \"$RAW_DATASET_DIR/splits_final.json\" \"$PREPROCESSED_DATASET_DIR/splits_final.json\"",
                "    echo \"Copied splits_final.json to preprocessed dataset folder.\"",
                "else",
                "    echo \"No custom splits_final.json found in raw dataset folder.\"",
                "fi",
                "",
            ]
        )

    train_cmd = (
        f"nnUNetv2_train {dataset_id} {configuration} {fold} "
        f"-tr {trainer} -p {plans}"
    )

    if checkpoint:
        train_cmd += f" -pretrained_weights {shell_quote(checkpoint)}"

    if extra_train_args:
        train_cmd += f" {extra_train_args}"

    lines.extend(
        [
            "echo \"Starting nnU-Net training...\"",
            train_cmd,
            "",
            "echo \"Training finished on: $(date)\"",
        ]
    )

    return "\n".join(lines)


def write_one_slurm_script(
    dataset_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    dataset_name = dataset_dir.name
    dataset_id = parse_dataset_id(dataset_name)

    safe_job_name = f"nnunet_{dataset_name}"
    safe_job_name = safe_job_name[:128]

    header = make_slurm_header(
        job_name=safe_job_name,
        partition=args.partition,
        account=args.account,
        time=args.time,
        cpus=args.cpus_per_task,
        mem=args.mem,
        gpus=args.gpus,
        mail_user=args.mail_user,
        mail_type=args.mail_type,
    )

    environment_block = make_environment_block(
        quartz_nnunet_raw=args.quartz_nnunet_raw,
        quartz_nnunet_preprocessed=args.quartz_nnunet_preprocessed,
        quartz_nnunet_results=args.quartz_nnunet_results,
        module_loads=args.module_load,
        conda_env=args.conda_env,
        venv_activate=args.venv_activate,
    )

    training_block = make_training_block(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        configuration=args.configuration,
        fold=args.fold,
        trainer=args.trainer,
        plans=args.plans,
        checkpoint=args.pretrained_weights,
        skip_preprocess=args.skip_preprocess,
        verify_dataset_integrity=not args.no_verify_dataset_integrity,
        copy_splits=not args.no_copy_splits,
        extra_train_args=args.extra_train_args,
    )

    script_text = "\n".join(
        [
            header,
            environment_block,
            training_block,
            "",
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = output_dir / f"train_{dataset_name}.sh"
    script_path.write_text(script_text)
    make_executable(script_path)

    return script_path


def write_submit_all_script(slurm_scripts: list[Path], output_dir: Path) -> Path:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "echo \"Submitting all nnU-Net LOCO training jobs...\"",
        "",
    ]

    for script_path in slurm_scripts:
        lines.append(f"sbatch {script_path.name}")

    lines.extend(
        [
            "",
            "echo \"Submitted all jobs.\"",
        ]
    )

    submit_path = output_dir / "submit_all.sh"
    submit_path.write_text("\n".join(lines) + "\n")
    make_executable(submit_path)

    return submit_path


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Quartz Slurm scripts for nnU-Net LOCO training."
    )

    parser.add_argument(
        "--nnunet_raw",
        required=True,
        type=Path,
        help="Local nnUNet_raw folder used to discover DatasetXXX folders.",
    )

    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output folder for Slurm scripts.",
    )

    parser.add_argument(
        "--quartz_nnunet_raw",
        required=True,
        type=str,
        help="Quartz path to nnUNet_raw.",
    )

    parser.add_argument(
        "--quartz_nnunet_preprocessed",
        required=True,
        type=str,
        help="Quartz path to nnUNet_preprocessed.",
    )

    parser.add_argument(
        "--quartz_nnunet_results",
        required=True,
        type=str,
        help="Quartz path to nnUNet_results.",
    )

    parser.add_argument(
        "--configuration",
        default="3d_fullres",
        help="nnU-Net configuration. Default: 3d_fullres.",
    )

    parser.add_argument(
        "--fold",
        default="0",
        help="nnU-Net fold. Default: 0.",
    )

    parser.add_argument(
        "--trainer",
        default="nnUNetTrainer",
        help="nnU-Net trainer class. Default: nnUNetTrainer.",
    )

    parser.add_argument(
        "--plans",
        default="nnUNetPlans",
        help="nnU-Net plans identifier. Default: nnUNetPlans.",
    )

    parser.add_argument(
        "--partition",
        default="gpu",
        help="Quartz Slurm partition. Default: gpu.",
    )

    parser.add_argument(
        "--account",
        default=None,
        help="Optional Slurm account name.",
    )

    parser.add_argument(
        "--time",
        default="24:00:00",
        help="Slurm wall time. Default: 24:00:00.",
    )

    parser.add_argument(
        "--cpus_per_task",
        default=8,
        type=int,
        help="CPUs per task. Default: 8.",
    )

    parser.add_argument(
        "--mem",
        default="64G",
        help="Memory request. Default: 64G.",
    )

    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs. Default: 1.",
    )

    parser.add_argument(
        "--mail_user",
        default=None,
        help="Optional email for Slurm notifications.",
    )

    parser.add_argument(
        "--mail_type",
        default=None,
        help="Optional Slurm mail type, e.g. END,FAIL.",
    )

    parser.add_argument(
        "--module_load",
        action="append",
        default=[],
        help="Module to load on Quartz. Can be used multiple times.",
    )

    parser.add_argument(
        "--conda_env",
        default=None,
        help="Optional conda environment name to activate.",
    )

    parser.add_argument(
        "--venv_activate",
        default=None,
        help="Optional path to venv activate script on Quartz.",
    )

    parser.add_argument(
        "--pretrained_weights",
        default=None,
        help="Optional path to pretrained weights for nnUNetv2_train.",
    )

    parser.add_argument(
        "--extra_train_args",
        default="",
        help="Extra arguments appended to nnUNetv2_train.",
    )

    parser.add_argument(
        "--skip_preprocess",
        action="store_true",
        help="Skip nnUNetv2_plan_and_preprocess in generated scripts.",
    )

    parser.add_argument(
        "--no_verify_dataset_integrity",
        action="store_true",
        help="Do not use --verify_dataset_integrity during preprocessing.",
    )

    parser.add_argument(
        "--no_copy_splits",
        action="store_true",
        help="Do not copy splits_final.json to nnUNet_preprocessed dataset folder.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dirs = find_dataset_dirs(args.nnunet_raw)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(parents=True, exist_ok=True)

    slurm_scripts = []

    print("\nGenerating Quartz Slurm scripts")
    print("=" * 80)
    print(f"Local nnUNet_raw: {args.nnunet_raw}")
    print(f"Output folder:    {args.out}")
    print(f"Datasets found:   {len(dataset_dirs)}")

    for dataset_dir in dataset_dirs:
        script_path = write_one_slurm_script(
            dataset_dir=dataset_dir,
            output_dir=args.out,
            args=args,
        )

        slurm_scripts.append(script_path)
        print(f"Wrote: {script_path}")

    submit_all = write_submit_all_script(
        slurm_scripts=slurm_scripts,
        output_dir=args.out,
    )

    print("\nStep 03 Complete")
    print("=" * 80)
    print(f"Slurm scripts written: {len(slurm_scripts)}")
    print(f"Submit-all script:     {submit_all}")

    print("\nNext steps:")
    print("1. Sync local_nnunet and quartz/slurm_jobs to Quartz.")
    print("2. On Quartz, cd into the slurm job folder.")
    print("3. Submit one job with sbatch train_DatasetXXX_*.sh")
    print("4. Or submit all jobs with ./submit_all.sh")


if __name__ == "__main__":
    main()
