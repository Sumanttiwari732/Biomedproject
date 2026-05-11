# Batch Effects in Multi-Center PET/CT Radiomics Using HECKTOR 2025 and nnU-Net

This repository contains a reproducible pipeline for studying batch effects in multi-center PET/CT imaging data for head-and-neck tumor segmentation.

The goal of this project is to investigate residual batch effects in multi-center CT-based head-and-neck tumor segmentation using a reproducible nnU-Net workflow. The pipeline includes HECKTOR dataset extraction, CT/PET/mask quality control, center-wise exploratory analysis, Leave-One-Center-Out dataset preparation, Quartz-based nnU-Net training, local inference, Dice score evaluation, residual batch-effect statistical analysis, and qualitative PET/CT visualization of ground truth versus model prediction.

## Project

**Batch effects in multi-center data for Deep Learning Applications in Radiomics**

Authors:

- Jakob Morales
- Sumant Tiwari

Course:

- Biomedical Image Processing
- Professor Rakesh Shiradkar
- SP26-IN-BMEG-E511-32707

---

## 1. Research Motivation

Multi-center medical imaging datasets contain non-biological variability caused by scanner vendor, acquisition protocol, reconstruction settings, voxel spacing, image quality, and institutional workflow. These sources of variability are known as batch effects.

In radiomics and deep learning-based medical image analysis, batch effects can reduce model generalization and cause segmentation performance to vary across hospitals. This project evaluates whether residual center-specific variability remains after preprocessing and nnU-Net segmentation.

---

## 2. Objectives

The project objectives are:

1. Organize and quality-control the HECKTOR 2025 PET/CT dataset.
2. Quantify center-wise differences in case distribution, CT intensity, scanner information, and slice thickness.
3. Prepare nnU-Net-compatible datasets using Leave-One-Center-Out splitting.
4. Train nnU-Net tumor segmentation models on Indiana University Quartz HPC.
5. Run local inference using trained nnU-Net model checkpoints.
6. Evaluate segmentation performance using Dice score and related metrics.
7. Test whether segmentation performance differs across hospitals.
8. Generate qualitative PET/CT overlays comparing ground truth and nnU-Net prediction.

---

## 3. Hypothesis

**Null hypothesis:** After preprocessing and normalization, segmentation performance does not differ significantly across hospitals.

**Alternative hypothesis:** Segmentation performance differs across hospitals because residual batch effects remain after preprocessing.

---

## 4. Repository Structure

```text
Biomedproject/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── example_config.json
├── Project/
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
├── model_zoo/
├── local_nnunet/
│   ├── nnUNet_raw/
│   ├── nnUNet_preprocessed/
│   ├── nnUNet_results/
│   ├── predictions/
│   └── results/
└── figures/
```

The folders `data/`, `model_zoo/`, `local_nnunet/`, `figures/`, model checkpoints, predictions, and NIfTI files are not tracked by Git.

---

## 5. Dataset

This project uses the **HECKTOR 2025 Training Data Defaced ALL** dataset.

The dataset is not included in this repository because it contains large medical imaging files.

Place the downloaded dataset zip file here:

```text
data/raw/HECKTOR_2025_Training_Data_Defaced_ALL.zip
```

After extraction, the expected folder structure is:

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

File meanings:

| File | Description |
|---|---|
| `PatientID__CT.nii.gz` | CT image |
| `PatientID__PT.nii.gz` | PET image |
| `PatientID.nii.gz` | Ground-truth tumor segmentation mask |
| `PatientID__RTDOSE.nii.gz` | Radiation dose file, not used in this project |

Segmentation labels:

| Label | Meaning |
|---:|---|
| 0 | Background |
| 1 | GTVp, primary gross tumor volume |
| 2 | GTVn, nodal gross tumor volume |

For binary nnU-Net tumor segmentation, labels 1 and 2 are combined into one foreground tumor label.

---

## 6. Center-Wise Dataset Summary

Example center-wise counts from the local HECKTOR folder:

| Center | Cases |
|---|---:|
| CHUM | 56 |
| CHUP | 72 |
| CHUS | 72 |
| HGJ | 55 |
| HMR | 18 |
| MDA | 442 |
| USZ | 11 |

Total:

```text
726 cases
```

USZ is included in dataset distribution plots. If no trained LOCO model is available for USZ, USZ is excluded from model-performance comparison.

---

## 7. Trained LOCO Model Weights

The trained Leave-One-Center-Out nnU-Net model weights are not stored in this GitHub repository.

Download the trained model archive here:

[Download trained HECKTOR LOCO models](https://indiana-my.sharepoint.com/my?id=%2Fpersonal%2Fsrtiwari%5Fiu%5Fedu%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files%2Fhecktor%5Floco%5Fmodels%2Etar%2Egz&parent=%2Fpersonal%2Fsrtiwari%5Fiu%5Fedu%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files&ct=1778459525253&or=Teams%2DHL&ga=1&LOF=1)

The SharePoint link may require Indiana University login.

Save the archive as:

```text
model_zoo/hecktor_loco_models.tar.gz
```

Create the model folders:

```bash
mkdir -p model_zoo
mkdir -p local_nnunet/nnUNet_results
```

Check the archive structure:

```bash
tar -tzf model_zoo/hecktor_loco_models.tar.gz | head
```

If the archive contains dataset folders such as `Dataset001_clean_raw_LOCO_CHUM/`, extract it into `nnUNet_results`:

```bash
tar -xzf model_zoo/hecktor_loco_models.tar.gz -C local_nnunet/nnUNet_results
```

If the archive contains a top-level `nnUNet_results/` folder, extract it into `local_nnunet`:

```bash
tar -xzf model_zoo/hecktor_loco_models.tar.gz -C local_nnunet
```

Expected model structure:

```text
local_nnunet/nnUNet_results/
├── Dataset001_clean_raw_LOCO_CHUM/
│   └── nnUNetTrainer__nnUNetPlans__3d_fullres/
│       └── fold_0/
│           ├── checkpoint_best.pth
│           └── checkpoint_final.pth
├── Dataset002_clean_raw_LOCO_CHUP/
├── Dataset003_clean_raw_LOCO_CHUS/
├── Dataset004_clean_raw_LOCO_HGJ/
├── Dataset005_clean_raw_LOCO_HMR/
└── Dataset006_clean_raw_LOCO_MDA/
```

---

## 8. Software Setup

Recommended Python version:

```text
Python 3.10 or 3.11
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Create local folders:

```bash
mkdir -p data/raw
mkdir -p data/extracted
mkdir -p data/processed
mkdir -p figures
mkdir -p model_zoo
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

To make the variables permanent, add the same lines to `~/.zshrc` or `~/.bashrc`.

---

## 9. Requirements

`requirements.txt`

```text
numpy>=1.24,<2.0
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
SimpleITK>=2.3
nibabel>=5.2
torch>=2.1
nnunetv2>=2.4
matplotlib>=3.7
tqdm>=4.66
openpyxl>=3.1
```

On Quartz, PyTorch may need to be installed with the CUDA version supported by the cluster.

---

## 10. Quick Start: Run Inference Using Trained Models

Use this workflow if you want to reproduce prediction, Dice score evaluation, and residual batch-effect analysis using the trained LOCO model weights.

### Step 1: Extract the HECKTOR dataset

```bash
python Project/00_extract_hecktor_zip.py \
  --zip data/raw/HECKTOR_2025_Training_Data_Defaced_ALL.zip \
  --out data/extracted
```

Expected output:

```text
data/extracted/HECKTOR 2025 Training Data Defaced ALL/
data/extracted/extraction_patient_inventory.csv
data/extracted/extraction_summary.json
```

### Step 2: Run CT/PET/mask quality control

```bash
python Project/01_qc_ct_dataset.py \
  --root "data/extracted/HECKTOR 2025 Training Data Defaced ALL" \
  --out data/processed/step1_qc
```

Expected outputs:

```text
data/processed/step1_qc/patient_inventory.csv
data/processed/step1_qc/ct_dataset_raw.csv
data/processed/step1_qc/ct_dataset_normalized.csv
data/processed/step1_qc/abnormal_cases.csv
data/processed/step1_qc/abnormal_by_center.csv
data/processed/step1_qc/center_case_counts.csv
data/processed/step1_qc/center_wise_ct_case_distribution.png
data/processed/step1_qc/ct_mean_intensity_by_center.png
data/processed/step1_qc/abnormal_case_count_by_center.png
```

This step checks:

- CT file availability
- PET file availability
- Mask file availability
- CT intensity statistics
- CT slice thickness
- CT-mask geometry consistency
- Mask labels
- Abnormal intensity values
- Empty masks
- Center-wise case counts

### Step 3: Build nnU-Net LOCO datasets

To reproduce the trained cropped LOCO workflow:

```bash
python Project/02_build_nnunet_loco_datasets.py \
  --qc_csv data/processed/step1_qc/ct_dataset_raw.csv \
  --out local_nnunet/nnUNet_raw \
  --label_mode binary \
  --crop_mode mask \
  --centers CHUM,CHUP,CHUS,HGJ,HMR,MDA
```

Use `--crop_mode mask` only when reproducing the cropped workflow used for the trained model weights.

For strict future test-set inference, use:

```bash
--crop_mode none
```

Expected output:

```text
local_nnunet/nnUNet_raw/
├── Dataset001_clean_raw_LOCO_CHUM/
├── Dataset002_clean_raw_LOCO_CHUP/
├── Dataset003_clean_raw_LOCO_CHUS/
├── Dataset004_clean_raw_LOCO_HGJ/
├── Dataset005_clean_raw_LOCO_HMR/
└── Dataset006_clean_raw_LOCO_MDA/
```

Each dataset contains:

```text
imagesTr/
labelsTr/
imagesTs/
labelsTs/
dataset.json
splits_final.json
case_mapping.csv
build_summary.json
```

### Step 4: Download and extract trained weights

Download the trained model archive from the SharePoint link in Section 7.

Save it as:

```text
model_zoo/hecktor_loco_models.tar.gz
```

Extract it:

```bash
tar -tzf model_zoo/hecktor_loco_models.tar.gz | head
tar -xzf model_zoo/hecktor_loco_models.tar.gz -C local_nnunet/nnUNet_results
```

The `DatasetXXX_clean_raw_LOCO_CENTER` folders should sit directly inside:

```text
local_nnunet/nnUNet_results/
```

### Step 5: Run local inference

```bash
python Project/04_run_local_inference.py \
  --nnunet_raw local_nnunet/nnUNet_raw \
  --nnunet_results local_nnunet/nnUNet_results \
  --out local_nnunet/predictions \
  --configuration 3d_fullres \
  --fold 0 \
  --checkpoint checkpoint_best.pth
```

This step creates nnU-Net predicted tumor masks for each held-out center.

Expected output:

```text
local_nnunet/predictions/
├── Dataset001_clean_raw_LOCO_CHUM/
├── Dataset002_clean_raw_LOCO_CHUP/
├── Dataset003_clean_raw_LOCO_CHUS/
├── Dataset004_clean_raw_LOCO_HGJ/
├── Dataset005_clean_raw_LOCO_HMR/
├── Dataset006_clean_raw_LOCO_MDA/
├── inference_summary.csv
├── inference_summary.json
└── qc_overlays/
```

### Step 6: Evaluate Dice score and residual batch effects

```bash
python Project/05_evaluate_predictions.py \
  --nnunet_raw local_nnunet/nnUNet_raw \
  --pred_root local_nnunet/predictions \
  --out local_nnunet/results
```

This step compares nnU-Net predictions with ground-truth masks.

The evaluation includes:

- Dice score
- Intersection over Union
- Sensitivity
- Specificity
- Precision
- Accuracy
- Center-wise summary
- One-way ANOVA
- Levene’s test
- Dice boxplots
- Mean Dice bar plots
- Qualitative CT overlays

Expected outputs:

```text
local_nnunet/results/per_case_metrics.csv
local_nnunet/results/final_summary.csv
local_nnunet/results/statistical_tests.txt
local_nnunet/results/statistical_tests.json
local_nnunet/results/dice_by_hospital.png
local_nnunet/results/mean_dice_by_hospital.png
local_nnunet/results/individual_dice_by_hospital.png
local_nnunet/results/overlays/
```

---

## 11. Full Workflow: Train Models from Scratch

Use this workflow if you want to train the LOCO models again on Quartz.

### Step 1: Extract dataset

```bash
python Project/00_extract_hecktor_zip.py \
  --zip data/raw/HECKTOR_2025_Training_Data_Defaced_ALL.zip \
  --out data/extracted
```

### Step 2: Run quality control

```bash
python Project/01_qc_ct_dataset.py \
  --root "data/extracted/HECKTOR 2025 Training Data Defaced ALL" \
  --out data/processed/step1_qc
```

### Step 3: Build LOCO nnU-Net datasets

```bash
python Project/02_build_nnunet_loco_datasets.py \
  --qc_csv data/processed/step1_qc/ct_dataset_raw.csv \
  --out local_nnunet/nnUNet_raw \
  --label_mode binary \
  --crop_mode mask \
  --centers CHUM,CHUP,CHUS,HGJ,HMR,MDA
```

### Step 4: Generate Quartz Slurm scripts

Replace `/N/project/YOUR_PROJECT` with your Quartz project path.

```bash
python Project/03_make_quartz_slurm.py \
  --nnunet_raw local_nnunet/nnUNet_raw \
  --out quartz/slurm_jobs \
  --quartz_nnunet_raw /N/project/YOUR_PROJECT/local_nnunet/nnUNet_raw \
  --quartz_nnunet_preprocessed /N/project/YOUR_PROJECT/local_nnunet/nnUNet_preprocessed \
  --quartz_nnunet_results /N/project/YOUR_PROJECT/local_nnunet/nnUNet_results \
  --configuration 3d_fullres \
  --fold 0
```

Expected output:

```text
quartz/slurm_jobs/
├── submit_all.sh
├── train_Dataset001_clean_raw_LOCO_CHUM.sh
├── train_Dataset002_clean_raw_LOCO_CHUP.sh
├── train_Dataset003_clean_raw_LOCO_CHUS.sh
├── train_Dataset004_clean_raw_LOCO_HGJ.sh
├── train_Dataset005_clean_raw_LOCO_HMR.sh
└── train_Dataset006_clean_raw_LOCO_MDA.sh
```

### Step 5: Sync data to Quartz

```bash
rsync -avz local_nnunet/ username@quartz.uits.iu.edu:/N/project/YOUR_PROJECT/local_nnunet/

rsync -avz quartz/slurm_jobs/ username@quartz.uits.iu.edu:/N/project/YOUR_PROJECT/slurm_jobs/
```

### Step 6: Train on Quartz

On Quartz:

```bash
cd /N/project/YOUR_PROJECT/slurm_jobs
```

Submit one training job:

```bash
sbatch train_Dataset001_clean_raw_LOCO_CHUM.sh
```

Or submit all jobs:

```bash
./submit_all.sh
```

Check job status:

```bash
squeue -u username
```

Expected model output:

```text
/N/project/YOUR_PROJECT/local_nnunet/nnUNet_results/
└── DatasetXXX_clean_raw_LOCO_CENTER/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
        └── fold_0/
            ├── checkpoint_best.pth
            ├── checkpoint_final.pth
            └── progress.png
```

### Step 7: Download trained models

```bash
rsync -avz username@quartz.uits.iu.edu:/N/project/YOUR_PROJECT/local_nnunet/nnUNet_results/ \
  local_nnunet/nnUNet_results/
```

### Step 8: Run inference and evaluation locally

```bash
python Project/04_run_local_inference.py \
  --nnunet_raw local_nnunet/nnUNet_raw \
  --nnunet_results local_nnunet/nnUNet_results \
  --out local_nnunet/predictions \
  --configuration 3d_fullres \
  --fold 0 \
  --checkpoint checkpoint_best.pth
```

```bash
python Project/05_evaluate_predictions.py \
  --nnunet_raw local_nnunet/nnUNet_raw \
  --pred_root local_nnunet/predictions \
  --out local_nnunet/results
```

---

## 12. Leave-One-Center-Out Design

This project uses Leave-One-Center-Out evaluation to test cross-center generalization.

For each experiment:

```text
Held-out center = test center
Training centers = all other centers
Validation data = internal split from training centers
Test data = held-out center only
```

Example:

```text
LOCO-CHUM:
Test center = CHUM
Training centers = CHUP, CHUS, HGJ, HMR, MDA
```

Example:

```text
LOCO-MDA:
Test center = MDA
Training centers = CHUM, CHUP, CHUS, HGJ, HMR
```

This design prevents center overlap between training and testing and evaluates whether the model generalizes to unseen hospitals.

---

## 13. Inference and Dice Score Evaluation

Inference is performed after nnU-Net training or after downloading trained model weights.

The inference script generates predicted tumor masks:

```text
Project/04_run_local_inference.py
```

The evaluation script compares predicted masks with ground-truth masks:

```text
Project/05_evaluate_predictions.py
```

Dice score is the main segmentation metric:

```text
Dice = 2|A ∩ B| / (|A| + |B|)
```

where:

```text
A = predicted tumor mask
B = ground-truth tumor mask
```

Other metrics:

| Metric | Meaning |
|---|---|
| IoU | Intersection over union |
| Sensitivity | Fraction of ground-truth tumor detected |
| Specificity | Fraction of background correctly classified |
| Precision | Fraction of predicted tumor that is correct |
| Accuracy | Overall voxel-wise correctness |

Statistical tests:

| Test | Purpose |
|---|---|
| One-way ANOVA | Tests whether mean Dice differs across centers |
| Levene’s test | Tests whether Dice variance differs across centers |
| Kruskal-Wallis | Non-parametric comparison of Dice distributions |

---

## 14. Residual Batch-Effect Analysis

Residual batch effects are evaluated using center-wise segmentation performance.

The analysis includes:

- Mean Dice by hospital
- Dice standard deviation by hospital
- Dice boxplot by hospital
- Individual Dice scatter plot
- One-way ANOVA across hospitals
- Levene’s test across hospitals

Significant differences in Dice score or Dice variance across hospitals suggest that residual center-specific variability remains after preprocessing.

---

## 15. Visualization

Generate a PET/CT overlay with ground truth and nnU-Net prediction:

```bash
python Project/06_visualize_pet_ct_gt_pred.py \
  --ct "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258__CT.nii.gz" \
  --pet "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258__PT.nii.gz" \
  --gt "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258.nii.gz" \
  --pred "local_nnunet/predictions/Dataset006_clean_raw_LOCO_MDA/case_XXXX.nii.gz" \
  --out figures/MDA-258_pet_ct_gt_prediction_overlay.png \
  --patient_id MDA-258
```

The visualization shows:

1. CT only
2. PET/CT fusion
3. PET/CT with ground-truth and prediction contours
4. Zoomed error map

Color convention:

| Region | Color |
|---|---|
| Ground truth | Cyan or green |
| nnU-Net prediction | Magenta or red |
| True positive overlap | Green |
| False positive | Yellow |
| False negative | Blue |

This visualization is important because head-and-neck tumors are not always clearly distinguishable on CT alone. The mask-defined region is used for training and evaluation.

---

## 16. Main Outputs

```text
data/processed/step1_qc/center_case_counts.csv
data/processed/step1_qc/center_wise_ct_case_distribution.png
data/processed/step1_qc/ct_mean_intensity_by_center.png
data/processed/step1_qc/abnormal_case_count_by_center.png

local_nnunet/predictions/
local_nnunet/results/per_case_metrics.csv
local_nnunet/results/final_summary.csv
local_nnunet/results/statistical_tests.txt
local_nnunet/results/statistical_tests.json
local_nnunet/results/dice_by_hospital.png
local_nnunet/results/mean_dice_by_hospital.png
local_nnunet/results/individual_dice_by_hospital.png
local_nnunet/results/overlays/

figures/
```

---

## 17. Files Not Tracked by Git

Do not commit:

```text
data/
model_zoo/
local_nnunet/
*.nii
*.nii.gz
*.mha
*.mhd
*.nrrd
*.zip
*.tar
*.tar.gz
*.pth
*.pt
*.ckpt
predictions/
results/
figures/*.png
figures/*.pdf
```

---

## 18. Reproducibility Checklist

```text
[ ] HECKTOR zip file placed in data/raw/
[ ] Python environment created
[ ] requirements.txt installed
[ ] nnUNet_raw set
[ ] nnUNet_preprocessed set
[ ] nnUNet_results set
[ ] Dataset extracted
[ ] QC completed
[ ] Center-wise counts generated
[ ] LOCO datasets created
[ ] Trained model weights downloaded or models trained on Quartz
[ ] Inference completed
[ ] Dice evaluation completed
[ ] Statistical tests completed
[ ] Qualitative overlays generated
```

---

## 19. Notes

The trained model archive reproduces the cropped LOCO workflow used in this project. The cropped workflow uses the ground-truth mask to crop the tumor region before training and testing. This is useful for controlled analysis but should not be treated as strict deployment inference.

For future strict external testing, use full-image inference with:

```bash
--crop_mode none
```

---

## 20. References

1. Oreiller V, Andrearczyk V, Jreige M, et al. Head and neck tumor segmentation in PET/CT: The HECKTOR challenge. Medical Image Analysis. 2022;77:102336.

2. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods. 2021;18:203–211.

3. Leek JT, Scharpf RB, Bravo HC, et al. Tackling the widespread and critical impact of batch effects in high-throughput data. Nature Reviews Genetics. 2010;11:733–739.

4. Orlhac F, Eertink JJ, Cottereau AS, et al. A Guide to ComBat Harmonization of Imaging Biomarkers in Multicenter Studies. Journal of Nuclear Medicine. 2022;63(2):172–179.

5. Da-Ano R, Visvikis D, Hatt M. Harmonization strategies for multicenter radiomics investigations. Physics in Medicine & Biology. 2020;65(24):24TR02.

6. Wu H, Liu X, Peng L, et al. Optimal batch determination for improved harmonization and prognostication of multi-center PET/CT radiomics feature in head and neck cancer. Physics in Medicine & Biology. 2023;68(22):225014.
