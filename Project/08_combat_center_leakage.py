#!/usr/bin/env python3
"""
Experiment 08: Center-classifier leakage test and ComBat harmonization.

This script tests whether radiomic features can predict hospital center before
and after ComBat harmonization.

Example:
    python Project/08_combat_center_leakage.py \
      --features radiomics_outputs_nnunet/features_pred_masks.csv \
      --out radiomics_outputs_nnunet/combat_pred_masks \
      --center_col Center
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from neuroCombat import neuroCombat
except ImportError:
    try:
        from neuroCombat.neuroCombat import neuroCombat
    except ImportError as exc:
        raise ImportError("neuroCombat is required. Install with: pip install neuroCombat==0.2.12") from exc


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


def find_center_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"Center column not found: {requested}")
        return requested

    for col in ["Center", "Hospital", "center", "hospital"]:
        if col in df.columns:
            return col

    raise ValueError("No center column found. Use --center_col.")


def get_feature_columns(df: pd.DataFrame, include_shape: bool) -> list[str]:
    cols = [c for c in df.columns if c.startswith("original_")]

    if not include_shape:
        cols = [c for c in cols if not c.startswith("original_shape")]

    numeric_cols: list[str] = []
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() > 0:
            numeric_cols.append(col)

    return numeric_cols


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    missing_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    missing_fraction = X.isna().mean()
    keep_cols = missing_fraction[missing_fraction <= missing_threshold].index.tolist()
    X = X[keep_cols].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )

    variances = X_imputed.var(axis=0)
    nonzero_cols = variances[variances > 1e-12].index.tolist()
    X_imputed = X_imputed[nonzero_cols].copy()

    info = {
        "initial_feature_count": len(feature_cols),
        "after_missing_filter": len(keep_cols),
        "after_variance_filter": len(nonzero_cols),
        "dropped_missing_count": len(feature_cols) - len(keep_cols),
        "dropped_zero_variance_count": len(keep_cols) - len(nonzero_cols),
    }

    return X_imputed, info


def make_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def run_center_classifier(
    X: pd.DataFrame,
    centers: pd.Series,
    out_dir: Path,
    prefix: str,
    random_seed: int,
) -> dict[str, Any]:
    encoder = LabelEncoder()
    y = encoder.fit_transform(centers.astype(str).values)

    counts = pd.Series(y).value_counts()
    min_class_count = int(counts.min())

    if min_class_count < 2:
        return {
            "prefix": prefix,
            "status": "failed",
            "reason": "At least one center has fewer than two samples.",
        }

    n_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    clf = make_classifier()

    y_pred = cross_val_predict(clf, X.values, y, cv=cv, method="predict")
    y_prob = cross_val_predict(clf, X.values, y, cv=cv, method="predict_proba")

    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)

    try:
        macro_auc = roc_auc_score(y, y_prob, multi_class="ovr", average="macro")
    except Exception:
        macro_auc = np.nan

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Center Classifier: {prefix}")
    plt.tight_layout()

    cm_path = out_dir / f"confusion_matrix_{prefix}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "prefix": prefix,
        "status": "completed",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_centers": int(len(encoder.classes_)),
        "centers": list(encoder.classes_),
        "cv_folds": int(n_splits),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_auc_ovr": None if np.isnan(macro_auc) else float(macro_auc),
        "confusion_matrix_png": str(cm_path),
    }


def save_pca_plot(
    X: pd.DataFrame,
    centers: pd.Series,
    out_path: Path,
    title: str,
) -> dict[str, Any]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Center": centers.astype(str).values})

    fig, ax = plt.subplots(figsize=(8, 6))
    for center, group in plot_df.groupby("Center"):
        ax.scatter(group["PC1"], group["PC2"], label=center, alpha=0.75)

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.grid(alpha=0.3)
    ax.legend(title="Center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "pca_png": str(out_path),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_variance": float(pca.explained_variance_ratio_[1]),
    }


def apply_combat(X: pd.DataFrame, centers: pd.Series) -> pd.DataFrame:
    covars = pd.DataFrame({"batch": centers.astype(str).values})
    combat_result = neuroCombat(dat=X.values.T, covars=covars, batch_col="batch")
    harmonized = combat_result["data"].T
    return pd.DataFrame(harmonized, columns=X.columns, index=X.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run center classifier before and after ComBat harmonization.")
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--center_col", default=None, type=str)
    parser.add_argument("--missing_threshold", default=0.20, type=float)
    parser.add_argument("--include_shape_features", action="store_true")
    parser.add_argument("--random_seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.features.exists():
        raise FileNotFoundError(f"Feature file not found: {args.features}")

    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.features)
    center_col = find_center_column(df, args.center_col)
    df = df[df[center_col].notna()].copy()
    df[center_col] = df[center_col].astype(str)

    feature_cols = get_feature_columns(df, include_shape=args.include_shape_features)
    if len(feature_cols) == 0:
        raise RuntimeError("No radiomic feature columns found.")

    X_before, prep_info = prepare_feature_matrix(
        df,
        feature_cols=feature_cols,
        missing_threshold=args.missing_threshold,
    )
    centers = df.loc[X_before.index, center_col].copy()

    print("\nExperiment 8: ComBat Center Leakage Analysis")
    print("=" * 80)
    print(f"Feature file: {args.features}")
    print(f"Rows: {len(df)}")
    print(f"Center column: {center_col}")
    print(f"Centers: {sorted(centers.unique())}")
    print(f"Features used: {X_before.shape[1]}")
    print(f"Shape features included: {args.include_shape_features}")

    before = run_center_classifier(X_before, centers, args.out, "before_combat", args.random_seed)
    pca_before = save_pca_plot(X_before, centers, args.out / "pca_before_combat.png", "PCA Before ComBat")

    X_after = apply_combat(X_before, centers)

    after = run_center_classifier(X_after, centers, args.out, "after_combat", args.random_seed)
    pca_after = save_pca_plot(X_after, centers, args.out / "pca_after_combat.png", "PCA After ComBat")

    metadata_cols = [
        c
        for c in [
            "Dataset",
            "Dataset_ID",
            "Case",
            "PatientID",
            "Center",
            "Hospital",
            "MaskSource",
            "Dice",
            "IoU",
            "Sensitivity",
            "Specificity",
            "Precision",
            "Accuracy",
        ]
        if c in df.columns
    ]

    harmonized_df = pd.concat(
        [df.loc[X_after.index, metadata_cols].reset_index(drop=True), X_after.reset_index(drop=True)],
        axis=1,
    )

    harmonized_path = args.out / "features_combat_harmonized.csv"
    harmonized_df.to_csv(harmonized_path, index=False)

    comparison_df = pd.DataFrame([before, after])
    comparison_path = args.out / "center_classifier_before_after.csv"
    comparison_df.to_csv(comparison_path, index=False)

    summary = {
        "features": str(args.features),
        "out": str(args.out),
        "center_column": center_col,
        "centers": sorted(centers.unique().tolist()),
        "preprocessing": prep_info,
        "shape_features_included": args.include_shape_features,
        "before_classifier": before,
        "after_classifier": after,
        "pca_before": pca_before,
        "pca_after": pca_after,
        "harmonized_features_csv": str(harmonized_path),
        "classifier_comparison_csv": str(comparison_path),
    }
    write_json(args.out / "combat_summary.json", summary)

    print("\nExperiment 8 Complete")
    print("=" * 80)
    print(f"Harmonized features: {harmonized_path}")
    print(f"Classifier metrics:  {comparison_path}")
    print(f"PCA before:          {args.out / 'pca_before_combat.png'}")
    print(f"PCA after:           {args.out / 'pca_after_combat.png'}")
    print(f"Summary JSON:        {args.out / 'combat_summary.json'}")

    print("\nKey Results")
    print("-" * 80)
    for label, result in [("Before ComBat", before), ("After ComBat", after)]:
        print(label)
        print(f"  Status:            {result.get('status')}")
        print(f"  Accuracy:          {result.get('accuracy')}")
        print(f"  Balanced accuracy: {result.get('balanced_accuracy')}")
        print(f"  Macro AUC:         {result.get('macro_auc_ovr')}")


if __name__ == "__main__":
    main()
