#!/usr/bin/env python3
"""
Experiment 10: Create final batch-effect summary tables and report.

Inputs:
    - Per-case Dice CSV
    - ComBat center-classifier result CSV for prediction masks
    - ComBat center-classifier result CSV for ground-truth masks

Outputs:
    - final_per_case_dice.csv
    - final_loco_dice_summary.csv
    - final_statistical_tests.csv
    - final_combat_summary.csv
    - final_mean_dice_by_center.png
    - final_batch_effect_summary.md

Example:
    python Project/10_make_final_batch_effect_summary.py \
      --dice_csv local_nnunet/results/per_case_dice.csv \
      --combat_pred radiomics_outputs_nnunet/combat_pred_masks/center_classifier_before_after.csv \
      --combat_gt radiomics_outputs_nnunet/combat_gt_masks/center_classifier_before_after.csv \
      --out final_project_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def detect_center_column(df: pd.DataFrame) -> str:
    for col in ["Center", "Hospital", "center", "hospital"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find center column in Dice CSV.")


def detect_dice_column(df: pd.DataFrame) -> str:
    for col in ["Dice", "dice", "DSC", "dsc"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find Dice column in Dice CSV.")


def dice_to_iou(dice: float) -> float:
    if not np.isfinite(dice):
        return np.nan
    return dice / (2.0 - dice) if dice < 2.0 else np.nan


def summarize_dice(dice_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(dice_csv)

    center_col = detect_center_column(df)
    dice_col = detect_dice_column(df)

    df[dice_col] = pd.to_numeric(df[dice_col], errors="coerce")
    df = df[df[dice_col].notna()].copy()

    summary = (
        df.groupby(center_col)[dice_col]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                center_col: "Center",
                "count": "n",
                "mean": "MeanDice",
                "std": "SDDice",
                "min": "MinDice",
                "max": "MaxDice",
            }
        )
    )
    summary["ApproxIoU"] = summary["MeanDice"].apply(dice_to_iou)

    groups = [df.loc[df[center_col] == center, dice_col].dropna().values for center in summary["Center"]]

    tests = []
    if len(groups) >= 2:
        anova_stat, anova_p = stats.f_oneway(*groups)
        levene_stat, levene_p = stats.levene(*groups, center="median")
        kruskal_stat, kruskal_p = stats.kruskal(*groups)

        tests.append(
            {
                "Test": "One-way ANOVA",
                "Statistic": anova_stat,
                "p_value": anova_p,
                "Interpretation": "Mean Dice differs across centers" if anova_p < 0.05 else "No significant mean Dice difference across centers",
            }
        )
        tests.append(
            {
                "Test": "Levene's test",
                "Statistic": levene_stat,
                "p_value": levene_p,
                "Interpretation": "Dice variance differs across centers" if levene_p < 0.05 else "No significant Dice variance difference across centers",
            }
        )
        tests.append(
            {
                "Test": "Kruskal-Wallis",
                "Statistic": kruskal_stat,
                "p_value": kruskal_p,
                "Interpretation": "Dice distributions differ across centers" if kruskal_p < 0.05 else "No significant Dice distribution difference across centers",
            }
        )

    tests_df = pd.DataFrame(tests)
    return df, summary, tests_df


def save_mean_dice_plot(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(summary))
    means = summary["MeanDice"].values
    errors = summary["SDDice"].fillna(0).values

    ax.bar(x, means, yerr=errors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["Center"].values)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Held-out center")
    ax.set_ylabel("Mean Dice")
    ax.set_title("Mean Dice Score by Held-out Center")
    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(means):
        ax.text(i, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_combat_table(path: Path | None, source_name: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.insert(0, "FeatureSource", source_name)
    return df


def make_combat_summary(combat_pred: Path | None, combat_gt: Path | None) -> pd.DataFrame:
    rows = []

    pred_df = load_combat_table(combat_pred, "nnU-Net prediction masks")
    if not pred_df.empty:
        rows.append(pred_df)

    gt_df = load_combat_table(combat_gt, "Ground-truth masks")
    if not gt_df.empty:
        rows.append(gt_df)

    if not rows:
        return pd.DataFrame()

    df = pd.concat(rows, ignore_index=True)

    keep_cols = [
        c
        for c in [
            "FeatureSource",
            "prefix",
            "n_samples",
            "n_features",
            "n_centers",
            "accuracy",
            "balanced_accuracy",
            "macro_auc_ovr",
        ]
        if c in df.columns
    ]

    out = df[keep_cols].copy()
    out["Condition"] = out["prefix"].replace({"before_combat": "Before ComBat", "after_combat": "After ComBat"})

    preferred_order = [
        "FeatureSource",
        "Condition",
        "n_samples",
        "n_features",
        "n_centers",
        "accuracy",
        "balanced_accuracy",
        "macro_auc_ovr",
    ]
    out = out[[c for c in preferred_order if c in out.columns]]

    return out


def markdown_table(df: pd.DataFrame, float_format: str = ".3f") -> str:
    if df.empty:
        return "_No data available._"

    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: f"{x:{float_format}}" if pd.notna(x) else "")

    try:
        return formatted.to_markdown(index=False)
    except ImportError:
        return formatted.to_csv(index=False)


def write_markdown_report(
    out_path: Path,
    dice_summary: pd.DataFrame,
    tests_df: pd.DataFrame,
    combat_summary: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# Final Batch-Effect Summary")
    lines.append("")
    lines.append("## 1. LOCO Segmentation Performance")
    lines.append("")
    lines.append(markdown_table(dice_summary))
    lines.append("")
    lines.append("## 2. Statistical Tests")
    lines.append("")
    lines.append(markdown_table(tests_df))
    lines.append("")
    lines.append("## 3. Radiomic Center Leakage and ComBat Harmonization")
    lines.append("")
    lines.append(markdown_table(combat_summary))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "LOCO nnU-Net evaluation measured segmentation-level center effects. "
        "Center-classifier performance before ComBat measured radiomic feature-level center leakage. "
        "Decreased classifier performance after ComBat indicates that feature-domain harmonization reduced center-specific radiomic signal. "
        "Because ComBat was applied after segmentation, it did not directly change nnU-Net predictions or Dice scores."
    )
    lines.append("")

    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create final batch-effect summary tables and report.")
    parser.add_argument("--dice_csv", required=True, type=Path)
    parser.add_argument("--combat_pred", default=None, type=Path)
    parser.add_argument("--combat_gt", default=None, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dice_csv.exists():
        raise FileNotFoundError(f"Dice CSV not found: {args.dice_csv}")

    args.out.mkdir(parents=True, exist_ok=True)

    dice_df, dice_summary, tests_df = summarize_dice(args.dice_csv)
    combat_summary = make_combat_summary(args.combat_pred, args.combat_gt)

    dice_df.to_csv(args.out / "final_per_case_dice.csv", index=False)
    dice_summary.to_csv(args.out / "final_loco_dice_summary.csv", index=False)
    tests_df.to_csv(args.out / "final_statistical_tests.csv", index=False)

    if not combat_summary.empty:
        combat_summary.to_csv(args.out / "final_combat_summary.csv", index=False)

    save_mean_dice_plot(dice_summary, args.out / "final_mean_dice_by_center.png")

    write_markdown_report(
        out_path=args.out / "final_batch_effect_summary.md",
        dice_summary=dice_summary,
        tests_df=tests_df,
        combat_summary=combat_summary,
    )

    print("\nExperiment 10 Complete")
    print("=" * 80)
    print(f"Per-case Dice:       {args.out / 'final_per_case_dice.csv'}")
    print(f"Dice summary:        {args.out / 'final_loco_dice_summary.csv'}")
    print(f"Statistical tests:   {args.out / 'final_statistical_tests.csv'}")
    print(f"ComBat summary:      {args.out / 'final_combat_summary.csv'}")
    print(f"Mean Dice plot:      {args.out / 'final_mean_dice_by_center.png'}")
    print(f"Markdown summary:    {args.out / 'final_batch_effect_summary.md'}")


if __name__ == "__main__":
    main()
