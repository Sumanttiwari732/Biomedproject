The nnU-Net workflow below follows the standard nnU-Net v2 logic: datasets are placed under `nnUNet_raw`, each dataset uses a `DatasetXXX_Name` folder format, and nnU-Net performs planning, preprocessing, training, and inference through its command-line workflow. nnU-Net is a self-configuring biomedical segmentation framework that adapts preprocessing, architecture, and training settings to the dataset. ([GitHub][1])

````markdown
# Batch Effects in Multi-Center PET/CT Radiomics Using HECKTOR 2025 and nnU-Net

This repository contains a reproducible pipeline for studying batch effects in multi-center PET/CT imaging data for deep learning-based head-and-neck tumor segmentation.

The project uses the HECKTOR 2025 training dataset and evaluates how center-specific imaging differences affect nnU-Net tumor segmentation performance. The workflow includes dataset extraction, quality control, center-wise exploratory analysis, nnU-Net dataset preparation, Quartz HPC training, local inference, Dice evaluation, residual batch-effect analysis, and qualitative visualization of ground truth versus model prediction.

## Project Title

**Batch effects in multi-center data for Deep Learning Applications in Radiomics**

Authors:

- Jakob Morales
- Sumant Tiwari

Course:

- Biomedical Image Processing
- Professor Rakesh Shiradkar
- SP26-IN-BMEG-E511-32707

---

## 1. Project Overview

Multi-center medical imaging datasets often contain non-biological variability caused by differences in scanners, acquisition protocols, reconstruction settings, patient population, and institutional workflow. These sources of variation are commonly referred to as **batch effects**.

In radiomics and deep learning-based medical image analysis, batch effects can reduce model generalizability and bias segmentation performance across hospitals. This project investigates residual center-related variability in head-and-neck tumor segmentation using the HECKTOR 2025 PET/CT dataset.

The project focuses on the following goals:

1. Extract and organize the HECKTOR 2025 training dataset.
2. Perform CT quality control and center-wise exploratory data analysis.
3. Prepare nnU-Net-compatible datasets using Leave-One-Center-Out evaluation.
4. Train nnU-Net models on Indiana University Quartz HPC.
5. Run local inference using downloaded trained models.
6. Evaluate segmentation performance using Dice and related metrics.
7. Analyze residual batch effects across centers.
8. Generate qualitative overlays of CT, PET/CT, ground truth, and nnU-Net prediction.

---

## 2. Dataset

This project uses the **HECKTOR 2025 Training Data Defaced ALL** dataset.

The dataset is not included in this repository because it contains large medical imaging files and may require authorized access.

Expected downloaded file:

```text
data/raw/HECKTOR_2025_Training_Data_Defaced_ALL.zip
````

After extraction, the expected patient-folder structure is:

```text
data/extracted/HECKTOR 2025 Training Data Defaced ALL/
├── CHUM-001/
│   ├── CHUM-001__CT.nii.gz
│   ├── CHUM-001__PT.nii.gz
│   └── CHUM-001.nii.gz
├── MDA-258/
│   ├── MDA-258__CT.nii.gz
│   ├── MDA-258__PT.nii.gz
│   └── MDA-258.nii.gz
└── ...
```

Expected file types:

| File                       | Meaning                                                |
| -------------------------- | ------------------------------------------------------ |
| `PatientID__CT.nii.gz`     | CT image                                               |
| `PatientID__PT.nii.gz`     | PET image                                              |
| `PatientID.nii.gz`         | Ground-truth tumor segmentation mask                   |
| `PatientID__RTDOSE.nii.gz` | Optional radiation dose file; not used in this project |

Segmentation label definitions:

| Label | Meaning                          |
| ----: | -------------------------------- |
|     0 | Background                       |
|     1 | GTVp, primary gross tumor volume |
|     2 | GTVn, nodal gross tumor volume   |

For nnU-Net binary tumor segmentation, labels 1 and 2 are combined into a single tumor foreground label.

---

## 3. Repository Structure

```text
Biomedproject/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── example_config.json
├── scripts/
│   ├── 00_extract_hecktor_zip.py
│   ├── 01_qc_ct_dataset.py
│   ├── 02_build_nnunet_loco_datasets.py
│   ├── 03_make_quartz_slurm.py
│   ├── 04_run_local_inference.py
│   ├── 05_evaluate_predictions.py
│   └── 06_visualize_pet_ct_gt_pred.py
├── quartz/
│   └── README_QUARTZ.md
├── docs/
│   └── workflow.md
├── data/
│   ├── raw/
│   ├── extracted/
│   └── processed/
├── local_nnunet/
│   ├── nnUNet_raw/
│   ├── nnUNet_preprocessed/
│   ├── nnUNet_results/
│   ├── predictions/
│   └── results/
└── figures/
```

The `data/`, `local_nnunet/`, `figures/`, model checkpoints, predictions, and NIfTI files should not be committed to GitHub.

---

## 4. Software Requirements

Recommended environment:

```text
Python 3.10 or 3.11
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
tqdm
SimpleITK
nibabel
torch
nnunetv2
```

Optional packages:

```text
openpyxl
seaborn
```

---

## 5. Environment Setup

Clone the repository:

```bash
git clone https://github.com/Sumanttiwari732/Biomedproject.git
cd Biomedproject
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Create required local folders:

```bash
mkdir -p data/raw
mkdir -p data/extracted
mkdir -p data/processed
mkdir -p figures
mkdir -p local_nnunet/nnUNet_raw
mkdir -p local_nnunet/nnUNet_preprocessed
mkdir -p local_nnunet/nnUNet_results
mkdir -p local_nnunet/predictions
mkdir -p local_nnunet/results
```

Set nnU-Net environment variables:

```bash
export nnUNet_raw="$PWD/local_nnunet/nnUNet_raw"
export nnUNet_preprocessed="$PWD/local_nnunet/nnUNet_preprocessed"
export nnUNet_results="$PWD/local_nnunet/nnUNet_results"
```

To make these permanent, add them to your shell configuration file, such as `~/.zshrc` or `~/.bashrc`.

---

## 6. Reproducible Workflow

The pipeline is organized into six main steps.

---

## Step 0: Extract the HECKTOR 2025 Zip File

Place the downloaded HECKTOR zip file here:

```text
data/raw/HECKTOR_2025_Training_Data_Defaced_ALL.zip
```

Run:

```bash
python scripts/00_extract_hecktor_zip.py \
  --zip data/raw/HECKTOR_2025_Training_Data_Defaced_ALL.zip \
  --out data/extracted
```

Expected output:

```text
data/extracted/HECKTOR 2025 Training Data Defaced ALL/
```

---

## Step 1: CT Dataset Quality Control and Center-Wise Summary

Run:

```bash
python scripts/01_qc_ct_dataset.py \
  --root "data/extracted/HECKTOR 2025 Training Data Defaced ALL" \
  --out data/processed/step1_qc
```

This script performs:

* Patient-folder inventory
* CT file detection
* PET file detection
* Mask file detection
* Complete CT/PET/mask case detection
* CT intensity statistics
* CT spacing and slice-thickness extraction
* Abnormal intensity flagging
* Center-wise case counting

Expected outputs:

```text
data/processed/step1_qc/ct_dataset_raw.csv
data/processed/step1_qc/ct_dataset_normalized.csv
data/processed/step1_qc/abnormal_cases.csv
data/processed/step1_qc/abnormal_by_center.csv
data/processed/step1_qc/center_case_counts.csv
data/processed/step1_qc/center_wise_ct_case_distribution.png
```

The center-wise distribution should include centers such as:

```text
CHUM
CHUP
CHUS
HGJ
HMR
MDA
USZ
```

Example center-wise counts from the local dataset:

| Center | Cases |
| ------ | ----: |
| CHUM   |    56 |
| CHUP   |    72 |
| CHUS   |    72 |
| HGJ    |    55 |
| HMR    |    18 |
| MDA    |   442 |
| USZ    |    11 |

Total:

```text
726 cases
```

---

## Step 2: Build Leave-One-Center-Out nnU-Net Datasets

Run:

```bash
python scripts/02_build_nnunet_loco_datasets.py \
  --qc_csv data/processed/step1_qc/ct_dataset_raw.csv \
  --out local_nnunet/nnUNet_raw \
  --label_mode binary \
  --crop_mode none
```

Recommended default:

```text
--label_mode binary
--crop_mode none
```

Label modes:

| Mode         | Description                                  |
| ------------ | -------------------------------------------- |
| `binary`     | Combines labels 1 and 2 into one tumor label |
| `multiclass` | Preserves background 0, GTVp 1, and GTVn 2   |

Crop modes:

| Mode   | Description                                                                                                                                |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `none` | Uses full image after resampling/preprocessing                                                                                             |
| `mask` | Crops around ground-truth mask; useful for experiments but not recommended for strict test inference because it uses ground-truth location |

Important:

For strict Leave-One-Center-Out evaluation, the held-out center should not be used for training. The held-out center should be stored as test data and evaluated only after training.

Standard logic:

```text
Training set: all centers except held-out center
Validation set: internal split from training centers only
Test set: held-out center
```

Expected nnU-Net dataset examples:

```text
local_nnunet/nnUNet_raw/
├── Dataset001_HECKTOR_clean_raw_LOCO_CHUM/
├── Dataset002_HECKTOR_clean_raw_LOCO_CHUP/
├── Dataset003_HECKTOR_clean_raw_LOCO_CHUS/
├── Dataset004_HECKTOR_clean_raw_LOCO_HGJ/
├── Dataset005_HECKTOR_clean_raw_LOCO_HMR/
├── Dataset006_HECKTOR_clean_raw_LOCO_MDA/
└── Dataset007_HECKTOR_clean_raw_LOCO_USZ/
```

Each dataset should contain:

```text
DatasetXXX_HECKTOR_clean_raw_LOCO_CENTER/
├── dataset.json
├── imagesTr/
├── labelsTr/
├── imagesTs/
└── labelsTs/
```

Note:

`labelsTs/` is retained for local evaluation after prediction. nnU-Net itself uses `imagesTs/` for prediction.

---

## Step 3: Generate Quartz Slurm Training Scripts

Run:

```bash
python scripts/03_make_quartz_slurm.py \
  --nnunet_raw local_nnunet/nnUNet_raw \
  --out quartz/slurm_jobs \
  --configuration 3d_fullres \
  --fold 0
```

Expected output:

```text
quartz/slurm_jobs/train_Dataset001_HECKTOR_clean_raw_LOCO_CHUM.sh
quartz/slurm_jobs/train_Dataset002_HECKTOR_clean_raw_LOCO_CHUP.sh
...
```

Each Slurm script should run:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_train DATASET_ID 3d_fullres 0
```

Example:

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
nnUNetv2_train 1 3d_fullres 0
```

---

## Step 4: Sync Data to Quartz and Train nnU-Net

Example local-to-Quartz sync:

```bash
rsync -avz local_nnunet/ username@quartz.uits.iu.edu:/N/project/your_project/local_nnunet/
rsync -avz quartz/slurm_jobs/ username@quartz.uits.iu.edu:/N/project/your_project/slurm_jobs/
```

On Quartz:

```bash
cd /N/project/your_project
```

Set nnU-Net paths:

```bash
export nnUNet_raw="/N/project/your_project/local_nnunet/nnUNet_raw"
export nnUNet_preprocessed="/N/project/your_project/local_nnunet/nnUNet_preprocessed"
export nnUNet_results="/N/project/your_project/local_nnunet/nnUNet_results"
```

Submit training jobs:

```bash
sbatch slurm_jobs/train_Dataset001_HECKTOR_clean_raw_LOCO_CHUM.sh
sbatch slurm_jobs/train_Dataset002_HECKTOR_clean_raw_LOCO_CHUP.sh
sbatch slurm_jobs/train_Dataset003_HECKTOR_clean_raw_LOCO_CHUS.sh
sbatch slurm_jobs/train_Dataset004_HECKTOR_clean_raw_LOCO_HGJ.sh
sbatch slurm_jobs/train_Dataset005_HECKTOR_clean_raw_LOCO_HMR.sh
sbatch slurm_jobs/train_Dataset006_HECKTOR_clean_raw_LOCO_MDA.sh
sbatch slurm_jobs/train_Dataset007_HECKTOR_clean_raw_LOCO_USZ.sh
```

Monitor jobs:

```bash
squeue -u username
```

Expected trained-model output:

```text
local_nnunet/nnUNet_results/
└── DatasetXXX_HECKTOR_clean_raw_LOCO_CENTER/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
        └── fold_0/
            ├── checkpoint_best.pth
            ├── checkpoint_final.pth
            └── progress.png
```

---

## Step 5: Download Trained Models from Quartz

After training, sync results back to the local machine:

```bash
rsync -avz username@quartz.uits.iu.edu:/N/project/your_project/local_nnunet/nnUNet_results/ \
  local_nnunet/nnUNet_results/
```

Optional: sync preprocessed files if needed:

```bash
rsync -avz username@quartz.uits.iu.edu:/N/project/your_project/local_nnunet/nnUNet_preprocessed/ \
  local_nnunet/nnUNet_preprocessed/
```

---

## Step 6: Run Local Inference

Run inference on held-out center test images:

```bash
python scripts/04_run_local_inference.py \
  --nnunet_raw local_nnunet/nnUNet_raw \
  --nnunet_results local_nnunet/nnUNet_results \
  --out local_nnunet/predictions \
  --configuration 3d_fullres \
  --fold 0 \
  --checkpoint checkpoint_best.pth
```

Expected output:

```text
local_nnunet/predictions/
├── Dataset001_HECKTOR_clean_raw_LOCO_CHUM/
├── Dataset002_HECKTOR_clean_raw_LOCO_CHUP/
├── Dataset003_HECKTOR_clean_raw_LOCO_CHUS/
├── Dataset004_HECKTOR_clean_raw_LOCO_HGJ/
├── Dataset005_HECKTOR_clean_raw_LOCO_HMR/
├── Dataset006_HECKTOR_clean_raw_LOCO_MDA/
└── Dataset007_HECKTOR_clean_raw_LOCO_USZ/
```

---

## Step 7: Evaluate Dice and Residual Batch Effects

Run:

```bash
python scripts/05_evaluate_predictions.py \
  --nnunet_raw local_nnunet/nnUNet_raw \
  --pred_root local_nnunet/predictions \
  --out local_nnunet/results
```

This script computes:

* Dice similarity coefficient
* Intersection over Union
* Sensitivity / recall
* Specificity
* Precision
* Accuracy
* Per-case metrics
* Center-wise summary
* ANOVA across centers
* Levene’s test for variance differences
* Dice boxplots
* Mean Dice bar plots

Expected outputs:

```text
local_nnunet/results/per_case_metrics.csv
local_nnunet/results/final_summary.csv
local_nnunet/results/statistical_tests.txt
local_nnunet/results/dice_by_hospital.png
local_nnunet/results/mean_dice_by_hospital.png
local_nnunet/results/overlays/
```

---

## Step 8: Generate Qualitative PET/CT + Ground Truth + Prediction Overlay

Run:

```bash
python scripts/06_visualize_pet_ct_gt_pred.py \
  --ct "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258__CT.nii.gz" \
  --pet "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258__PT.nii.gz" \
  --gt "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258.nii.gz" \
  --pred "local_nnunet/predictions/Dataset006_HECKTOR_clean_raw_LOCO_MDA/case_XXXX.nii.gz" \
  --out figures/MDA-258_pet_ct_gt_prediction_overlay.png
```

The figure should show:

1. CT only
2. PET/CT fusion
3. PET/CT with ground-truth and prediction contours
4. Zoomed error map

Recommended contour colors:

| Region                | Color          |
| --------------------- | -------------- |
| Ground truth          | Cyan or green  |
| nnU-Net prediction    | Magenta or red |
| True positive overlap | Green          |
| False positive        | Yellow         |
| False negative        | Blue           |

Suggested figure caption:

```text
Figure X. Qualitative evaluation of nnU-Net tumor segmentation using PET/CT fusion, ground-truth annotation, and model prediction. CT alone provides anatomical context but does not clearly separate tumor from surrounding soft tissue. PET/CT fusion shows metabolic uptake, while the ground-truth contour represents the manual tumor annotation and the prediction contour represents the nnU-Net output. The zoomed error map summarizes agreement between ground truth and prediction, where true-positive overlap, false-positive prediction, and false-negative missed tumor regions are visualized separately.
```

---

## 9. Leave-One-Center-Out Evaluation Design

This project uses Leave-One-Center-Out evaluation to measure cross-center generalization.

For each experiment:

```text
Held-out center = test center
Training centers = all other centers
Validation data = internal split from training centers
Test data = held-out center only
```

Example for LOCO-MDA:

```text
Test center: MDA
Training centers: CHUM, CHUP, CHUS, HGJ, HMR, USZ
```

Example for LOCO-USZ:

```text
Test center: USZ
Training centers: CHUM, CHUP, CHUS, HGJ, HMR, MDA
```

This design prevents direct center overlap between training and testing and is appropriate for evaluating residual batch effects.

---

## 10. Evaluation Metrics

The main segmentation metric is the Dice Similarity Coefficient:

```text
Dice = 2|A ∩ B| / (|A| + |B|)
```

where:

```text
A = predicted tumor mask
B = ground-truth tumor mask
```

Additional metrics:

| Metric               | Meaning                                             |
| -------------------- | --------------------------------------------------- |
| Dice                 | Spatial overlap between prediction and ground truth |
| IoU                  | Intersection over union                             |
| Sensitivity / Recall | Fraction of ground-truth tumor detected             |
| Specificity          | Fraction of background correctly classified         |
| Precision            | Fraction of predicted tumor that is correct         |
| Accuracy             | Overall voxel-wise correctness                      |

Statistical tests:

| Test          | Purpose                                            |
| ------------- | -------------------------------------------------- |
| One-way ANOVA | Tests whether mean Dice differs across centers     |
| Levene’s test | Tests whether Dice variance differs across centers |

---

## 11. Expected Main Figures

Recommended report figures:

1. Center-wise CT case distribution bar graph
2. CT intensity distribution by center
3. Abnormal scan count by center
4. Dice score boxplot by hospital
5. Mean Dice score by hospital
6. Representative CT/PET/GT/prediction overlay
7. Example nnU-Net segmentation overlay with ground truth and prediction

---

## 12. Example Report Interpretation

Center-wise imbalance:

```text
The HECKTOR dataset demonstrated strong center imbalance. MDA contributed the largest proportion of cases, while USZ and HMR contributed the fewest. This imbalance is important because dominant centers may contribute more strongly to learned imaging patterns and may influence cross-center generalization.
```

Qualitative PET/CT visualization:

```text
CT alone may not clearly separate tumor from surrounding soft tissue in head-and-neck imaging. PET/CT fusion provides functional uptake information, while the ground-truth mask defines the tumor region used for training and evaluation. Therefore, qualitative assessment should compare model prediction directly with the annotated mask rather than relying on CT appearance alone.
```

Residual batch effects:

```text
Center-wise Dice score differences and statistical testing were used to assess whether residual batch effects remained after preprocessing. Significant differences in performance or variance across hospitals would suggest that scanner, protocol, or institutional factors continued to influence segmentation robustness.
```

---

## 13. Files Not Tracked by Git

The following should not be committed:

```text
data/
*.nii
*.nii.gz
*.mha
*.mhd
*.nrrd
*.zip
*.pth
*.pt
*.ckpt
local_nnunet/
nnUNet_raw/
nnUNet_preprocessed/
nnUNet_results/
predictions/
results/
figures/*.png
figures/*.pdf
```

Use `.gitignore` to exclude these files.

---

## 14. Reproducibility Checklist

Before running the full project, verify:

```text
[ ] HECKTOR zip file downloaded
[ ] Zip file placed in data/raw/
[ ] Python environment created
[ ] requirements.txt installed
[ ] nnUNet_raw variable set
[ ] nnUNet_preprocessed variable set
[ ] nnUNet_results variable set
[ ] Dataset extracted successfully
[ ] CT/PET/mask file inventory completed
[ ] Center-wise counts generated
[ ] nnU-Net LOCO datasets created
[ ] Data synced to Quartz
[ ] nnU-Net training completed
[ ] Best model downloaded locally
[ ] Local inference completed
[ ] Dice and statistical analysis completed
[ ] Qualitative overlays generated
```

---

## 15. Acknowledgments

The authors thank the HECKTOR Challenge organizers for providing the dataset and acknowledge the Indiana University Quartz HPC environment for computational support.

---

## 16. References

Recommended references for the project report:

1. Oreiller V, Andrearczyk V, Jreige M, et al. Head and neck tumor segmentation in PET/CT: The HECKTOR challenge. Medical Image Analysis. 2022;77:102336.

2. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods. 2021;18:203–211.

3. Leek JT, Scharpf RB, Bravo HC, et al. Tackling the widespread and critical impact of batch effects in high-throughput data. Nature Reviews Genetics. 2010;11:733–739.

4. Orlhac F, Eertink JJ, Cottereau AS, et al. A Guide to ComBat Harmonization of Imaging Biomarkers in Multicenter Studies. Journal of Nuclear Medicine. 2022;63(2):172–179.

5. Da-Ano R, Visvikis D, Hatt M. Harmonization strategies for multicenter radiomics investigations. Physics in Medicine & Biology. 2020;65(24):24TR02.

6. Wu H, Liu X, Peng L, et al. Optimal batch determination for improved harmonization and prognostication of multi-center PET/CT radiomics feature in head and neck cancer. Physics in Medicine & Biology. 2023;68(22):225014.

```
