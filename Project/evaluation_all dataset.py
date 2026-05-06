from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
ROOT = Path("/Users/sumantratiwari/PycharmProjects/Biomedproject/local_nnunet")

DATASETS = [
    ("Dataset001_clean_raw_LOCO_CHUM", "CHUM"),
    ("Dataset002_clean_raw_LOCO_CHUP", "CHUP"),
    ("Dataset003_clean_raw_LOCO_CHUS", "CHUS"),
    ("Dataset004_clean_raw_LOCO_HGJ", "HGJ"),
    ("Dataset005_clean_raw_LOCO_HMR", "HMR"),
    ("Dataset006_clean_raw_LOCO_MDA", "MDA"),
]

OUT_FILE = ROOT / "results/final_summary.csv"
BOXPLOT_FILE = ROOT / "results/dice_by_hospital.png"


# ----------------------------
# FUNCTIONS
# ----------------------------
def load_nii(path):
    return nib.load(str(path)).get_fdata()

def dice(pred, gt, eps=1e-6):
    pred = pred > 0
    gt = gt > 0
    intersection = np.logical_and(pred, gt).sum()
    return (2 * intersection + eps) / (pred.sum() + gt.sum() + eps)

def compute_dataset(pred_dir, gt_dir):
    scores = []

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    gt_map = {f.name: f for f in gt_files}

    for pf in pred_files:
        if pf.name not in gt_map:
            continue

        pred = load_nii(pf)
        gt = load_nii(gt_map[pf.name])

        if pred.shape != gt.shape:
            continue

        scores.append(dice(pred, gt))

    return np.array(scores)


# ----------------------------
# MAIN
# ----------------------------
results = []
all_scores = {}

for dataset, hospital in DATASETS:
    print(f"Processing {dataset}")

    pred_dir = ROOT / f"predictions_latest/{dataset}"
    gt_dir = ROOT / f"nnunet_datasets_final/{dataset}/labelsTs"

    scores = compute_dataset(pred_dir, gt_dir)
    all_scores[hospital] = scores

    if len(scores) == 0:
        print(f"No data found for {hospital}, skipping")
        continue

    results.append({
        "Dataset": dataset,
        "Hospital": hospital,
        "Cases": len(scores),
        "Mean Dice": round(scores.mean(), 4),
        "Std Dice": round(scores.std(ddof=1), 4),
        "Min Dice": round(scores.min(), 4),
        "Max Dice": round(scores.max(), 4),
    })

# Save summary table
df = pd.DataFrame(results)
OUT_FILE.parent.mkdir(exist_ok=True, parents=True)
df.to_csv(OUT_FILE, index=False)

print("\nPer-hospital Dice summary:")
print(df)

# ----------------------------
# HYPOTHESIS TESTING
# H0: No difference in Dice performance across hospitals
# H1: Dice performance differs across hospitals
# ----------------------------
groups = [scores for scores in all_scores.values() if len(scores) > 0]

if len(groups) < 2:
    raise ValueError("Need at least two hospitals with Dice scores for ANOVA.")

# Check variance differences across hospitals
levene_stat, levene_p = stats.levene(*groups, center='median')

# Compare mean Dice across hospitals
anova_stat, anova_p = stats.f_oneway(*groups)

print("\nHypothesis test across hospitals:")
print(f"Levene test (variance equality): statistic={levene_stat:.4f}, p-value={levene_p:.6f}")
print(f"One-way ANOVA (mean Dice equality): statistic={anova_stat:.4f}, p-value={anova_p:.6f}")

alpha = 0.05
if anova_p < alpha:
    print("\nDecision: Reject H0")
    print("Conclusion: Dice segmentation performance differs significantly across hospitals.")
    print("This supports the presence of residual site-specific variability / batch effects.")
else:
    print("\nDecision: Fail to reject H0")
    print("Conclusion: No statistically significant difference in Dice performance across hospitals.")
    print("This suggests preprocessing and normalization reduced inter-hospital variability.")

# Optional: per-hospital pairwise t-tests against pooled mean are NOT the main test
# for the hypothesis, so we avoid using a one-sample test against a fixed baseline.

# ----------------------------
# OPTIONAL VISUALIZATION
# ----------------------------
plot_df = pd.DataFrame(
    [(h, d) for h, scores in all_scores.items() for d in scores],
    columns=["Hospital", "Dice"]
)

plt.figure(figsize=(10, 6))
plot_df.boxplot(column="Dice", by="Hospital")
plt.title("Dice Score Distribution by Hospital")
plt.suptitle("")
plt.xlabel("Hospital")
plt.ylabel("Dice Score")
plt.tight_layout()
plt.savefig(BOXPLOT_FILE, dpi=300)
print(f"\nSaved boxplot to: {BOXPLOT_FILE}")
