
---

# Batch Effects in Multi-Center PET/CT Radiomics Using HECKTOR 2025 and nnU-Net

This repository contains a reproducible pipeline for studying residual batch effects in multi-center PET/CT imaging data for head-and-neck tumor segmentation.

The project investigates whether center-specific variability remains after preprocessing and deep learning-based tumor segmentation. The workflow uses the HECKTOR 2025 dataset, Leave-One-Center-Out evaluation, nnU-Net segmentation, Dice-based performance analysis, post-segmentation radiomic feature extraction, center-classifier leakage testing, and ComBat feature harmonization.

## Project

**Batch effects in multi-center data for Deep Learning Applications in Radiomics**

Authors:

* Jakob Morales
* Sumant Tiwari

Course:

* Biomedical Image Processing
* Professor Rakesh Shiradkar
* SP26-IN-BMEG-E511-32707

---

## 1. Research Motivation

Multi-center medical imaging datasets contain non-biological variability caused by scanner vendor, acquisition protocol, reconstruction settings, voxel spacing, image quality, segmentation workflow, and institutional practice. These sources of variation are commonly known as **batch effects**.

In radiomics and deep learning-based medical image analysis, batch effects can reduce model generalization and cause segmentation performance to vary across hospitals. This project evaluates residual batch effects at two levels:

1. **Segmentation-performance level:** whether nnU-Net Dice scores differ across held-out centers.
2. **Radiomic feature level:** whether extracted tumor radiomic features can predict hospital identity.

ComBat harmonization is then used as a post-segmentation feature-domain method to reduce center-specific radiomic signal.

---

## 2. Objectives

The project objectives are:

1. Organize and quality-control the HECKTOR 2025 PET/CT dataset.
2. Quantify center-wise differences in case distribution, scanner information, CT intensity, and slice thickness.
3. Prepare nnU-Net-compatible datasets using Leave-One-Center-Out splitting.
4. Train nnU-Net segmentation models on Indiana University Quartz HPC.
5. Run local inference using trained nnU-Net checkpoints.
6. Evaluate segmentation performance using Dice score and related metrics.
7. Test whether segmentation performance differs across hospitals.
8. Generate qualitative PET/CT overlays comparing ground truth and nnU-Net prediction.
9. Extract radiomic features from ground-truth and nnU-Net-predicted tumor masks.
10. Use a center classifier to quantify radiomic feature-level batch leakage.
11. Apply ComBat harmonization to reduce center-specific radiomic feature signal.
12. Generate final summary tables, plots, and reproducibility outputs.

---

## 3. Hypothesis

**Null hypothesis:** After preprocessing and normalization, segmentation performance does not differ significantly across hospitals.

**Alternative hypothesis:** Segmentation performance differs across hospitals because residual batch effects remain after preprocessing.

For the radiomic feature analysis, the project also tests whether tumor radiomic features retain enough center-specific information to predict hospital identity, and whether ComBat reduces that center leakage.

---

## 4. Repository Structure

```text
Biomedproject/
├── README.md
├── requirements.txt
├── .gitignore
├── Project/
│   ├── 00_extract_hecktor_zip.py
│   ├── 01_qc_ct_dataset.py
│   ├── 02_build_nnunet_loco_datasets.py
│   ├── 03_make_quartz_slurm.py
│   ├── 04_run_local_inference.py
│   ├── 05_evaluate_predictions.py
│   ├── 06_visualize_pet_ct_gt_pred.py
│   ├── 07_extract_radiomics_features.py
│   ├── 08_combat_center_leakage.py
│   ├── 09_preprocessing_normalization_augmentation_visuals.py
│   └── 10_make_final_batch_effect_summary.py
├── configs/
│   └── example_config.json
├── quartz/
│   └── README_QUARTZ.md
├── docs/
│   └── workflow.md
├── data/
│   ├── raw/
│   ├── extracted/
│   └── processed/
├── local_nnunet/
│   ├── nnunet_datasets_final/
│   ├── nnunet_preprocessed/
│   ├── nnunet_results/
│   ├── predictions_latest/
│   └── results/
├── radiomics_outputs_nnunet/
├── preprocessing_visuals/
├── final_project_outputs/
└── figures/
```

Large medical images, model checkpoints, predictions, radiomic feature tables, and generated figures are not tracked by Git.

---

## 5. Dataset

This project uses the **HECKTOR 2025 Training Data Defaced ALL** dataset.

The dataset is not included in this repository because it contains large medical imaging files.

Expected downloaded file:

```text
data/raw/HECKTOR_2025_Training_Data_Defaced_ALL.zip
```

After extraction, the expected structure is:

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

| File                       | Description                                                    |
| -------------------------- | -------------------------------------------------------------- |
| `PatientID__CT.nii.gz`     | CT image                                                       |
| `PatientID__PT.nii.gz`     | PET image                                                      |
| `PatientID.nii.gz`         | Ground-truth tumor segmentation mask                           |
| `PatientID__RTDOSE.nii.gz` | Radiation dose file, not used for segmentation in this project |

Segmentation labels:

| Label | Meaning                          |
| ----: | -------------------------------- |
|     0 | Background                       |
|     1 | GTVp, primary gross tumor volume |
|     2 | GTVn, nodal gross tumor volume   |

For binary nnU-Net tumor segmentation, labels 1 and 2 are combined into one foreground tumor class.

---

## 6. Center-Wise Dataset Summary

The full descriptive HECKTOR cohort contained seven centers:

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
726 patient folders
```

USZ was included in descriptive exploratory analysis. The completed nnU-Net segmentation and ComBat analysis used the available six-center prediction subset:

```text
CHUM, CHUP, CHUS, HGJ, HMR, MDA
```

USZ was excluded from completed segmentation-performance statistics because a trained LOCO-USZ model and corresponding predictions were not available at the time of analysis. The scripts support adding USZ if a LOCO-USZ model is trained.

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

Create folders:

```bash
mkdir -p model_zoo
mkdir -p local_nnunet/nnunet_results
```

Check the archive structure:

```bash
tar -tzf model_zoo/hecktor_loco_models.tar.gz | head
```

If the archive contains dataset folders such as `Dataset001_clean_raw_LOCO_CHUM/`, extract it into `nnunet_results`:

```bash
tar -xzf model_zoo/hecktor_loco_models.tar.gz -C local_nnunet/nnunet_results
```

Expected six-center model structure:

```text
local_nnunet/nnunet_results/
├── Dataset001_clean_raw_LOCO_CHUM/
├── Dataset002_clean_raw_LOCO_CHUP/
├── Dataset003_clean_raw_LOCO_CHUS/
├── Dataset004_clean_raw_LOCO_HGJ/
├── Dataset005_clean_raw_LOCO_HMR/
└── Dataset006_clean_raw_LOCO_MDA/
```

Each dataset should contain:

```text
DatasetXXX_clean_raw_LOCO_CENTER/
└── nnUNetTrainer__nnUNetPlans__3d_fullres/
    └── fold_0/
        ├── checkpoint_best.pth
        └── checkpoint_final.pth
```

Check available models:

```bash
find local_nnunet/nnunet_results -maxdepth 1 -type d -name "Dataset*LOCO*"
```

If `Dataset007_clean_raw_LOCO_USZ` is not present, the USZ LOCO model must be trained before including USZ in segmentation-performance statistics.

---

## 8. Software Setup

Recommended Python version:

```text
Python 3.11
```

Create the main environment:

```bash
cd /Users/sumantratiwari/PycharmProjects/Biomedproject

python3.11 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

A separate radiomics environment is recommended for Experiments 7–10:

```bash
python3.11 -m venv .venv_radiomics
source .venv_radiomics/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Verify PyRadiomics and neuroCombat:

```bash
python -c "from radiomics import featureextractor; print(featureextractor.RadiomicsFeatureExtractor)"
python -c "from neuroCombat import neuroCombat; print('neuroCombat works')"
```

Set nnU-Net paths:

```bash
export nnUNet_raw="$PWD/local_nnunet/nnunet_datasets_final"
export nnUNet_preprocessed="$PWD/local_nnunet/nnunet_preprocessed"
export nnUNet_results="$PWD/local_nnunet/nnunet_results"
```

To make these permanent, add them to `~/.zshrc` or `~/.bashrc`.

---

## 9. Requirements

Recommended `requirements.txt`:

```text
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
tqdm>=4.66
SimpleITK>=2.3
nibabel>=5.2
openpyxl>=3.1
tabulate>=0.9.0
pyradiomics==3.0.1
neuroCombat==0.2.12
torch>=2.1
nnunetv2>=2.4
```

On Quartz, PyTorch may need to be installed with the CUDA version supported by the cluster.

---

## 10. Quick Start: Use Existing Trained Models

Use this workflow if trained LOCO model weights are already available.

### Step 1: Extract the HECKTOR dataset

```bash
python Project/00_extract_hecktor_zip.py \
  --zip data/raw/HECKTOR_2025_Training_Data_Defaced_ALL.zip \
  --out data/extracted
```

Expected output:

```text
data/extracted/HECKTOR 2025 Training Data Defaced ALL/
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
```

This step checks:

* CT file availability
* PET file availability
* Mask file availability
* CT intensity statistics
* CT slice thickness
* CT-mask geometry consistency
* Mask labels
* Abnormal intensity values
* Empty masks
* Center-wise case counts

### Step 3: Build LOCO nnU-Net datasets

For the completed six-center analysis:

```bash
python Project/02_build_nnunet_loco_datasets.py \
  --qc_csv data/processed/step1_qc/ct_dataset_raw.csv \
  --out local_nnunet/nnunet_datasets_final \
  --label_mode binary \
  --crop_mode mask \
  --centers CHUM,CHUP,CHUS,HGJ,HMR,MDA \
  --overwrite
```

For optional seven-center dataset construction including USZ:

```bash
python Project/02_build_nnunet_loco_datasets.py \
  --qc_csv data/processed/step1_qc/ct_dataset_raw.csv \
  --out local_nnunet/nnunet_datasets_final \
  --label_mode binary \
  --crop_mode mask \
  --centers CHUM,CHUP,CHUS,HGJ,HMR,MDA,USZ \
  --overwrite
```

Use `--crop_mode mask` only when reproducing the cropped workflow used in this project. For stricter future deployment-style inference, use:

```bash
--crop_mode none
```

Expected six-center output:

```text
local_nnunet/nnunet_datasets_final/
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

### Step 4: Extract trained weights

```bash
tar -tzf model_zoo/hecktor_loco_models.tar.gz | head

tar -xzf model_zoo/hecktor_loco_models.tar.gz \
  -C local_nnunet/nnunet_results
```

Check models:

```bash
find local_nnunet/nnunet_results -maxdepth 1 -type d -name "Dataset*LOCO*"
```

### Step 5: Run local inference

For available six-center models:

```bash
python Project/04_run_local_inference.py \
  --nnunet_raw local_nnunet/nnunet_datasets_final \
  --nnunet_results local_nnunet/nnunet_results \
  --out local_nnunet/predictions_latest \
  --datasets CHUM,CHUP,CHUS,HGJ,HMR,MDA \
  --configuration 3d_fullres \
  --fold 0 \
  --checkpoint checkpoint_best.pth
```

Expected output:

```text
local_nnunet/predictions_latest/
├── Dataset001_clean_raw_LOCO_CHUM/
├── Dataset002_clean_raw_LOCO_CHUP/
├── Dataset003_clean_raw_LOCO_CHUS/
├── Dataset004_clean_raw_LOCO_HGJ/
├── Dataset005_clean_raw_LOCO_HMR/
└── Dataset006_clean_raw_LOCO_MDA/
```

### Step 6: Evaluate Dice and residual batch effects

```bash
python Project/05_evaluate_predictions.py \
  --nnunet_raw local_nnunet/nnunet_datasets_final \
  --pred_root local_nnunet/predictions_latest \
  --out local_nnunet/results \
  --datasets CHUM,CHUP,CHUS,HGJ,HMR,MDA
```

Expected outputs:

```text
local_nnunet/results/per_case_dice.csv
local_nnunet/results/final_summary.csv
local_nnunet/results/statistical_tests.txt
local_nnunet/results/dice_by_hospital.png
local_nnunet/results/mean_dice_by_hospital.png
local_nnunet/results/individual_dice_by_hospital.png
local_nnunet/results/overlays/
```

---

## 11. Full Workflow: Train LOCO Models on Quartz

Use this workflow to train models from scratch.

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

### Step 3: Build LOCO datasets

Completed six-center setup:

```bash
python Project/02_build_nnunet_loco_datasets.py \
  --qc_csv data/processed/step1_qc/ct_dataset_raw.csv \
  --out local_nnunet/nnunet_datasets_final \
  --label_mode binary \
  --crop_mode mask \
  --centers CHUM,CHUP,CHUS,HGJ,HMR,MDA \
  --overwrite
```

Optional seven-center setup:

```bash
python Project/02_build_nnunet_loco_datasets.py \
  --qc_csv data/processed/step1_qc/ct_dataset_raw.csv \
  --out local_nnunet/nnunet_datasets_final \
  --label_mode binary \
  --crop_mode mask \
  --centers CHUM,CHUP,CHUS,HGJ,HMR,MDA,USZ \
  --overwrite
```

### Step 4: Generate Quartz Slurm scripts

Replace `/N/project/YOUR_PROJECT` with your Quartz project path.

```bash
python Project/03_make_quartz_slurm.py \
  --nnunet_raw local_nnunet/nnunet_datasets_final \
  --out quartz/slurm_jobs \
  --quartz_nnunet_raw /N/project/YOUR_PROJECT/local_nnunet/nnunet_datasets_final \
  --quartz_nnunet_preprocessed /N/project/YOUR_PROJECT/local_nnunet/nnunet_preprocessed \
  --quartz_nnunet_results /N/project/YOUR_PROJECT/local_nnunet/nnunet_results \
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

If USZ is included, an additional script should be generated:

```text
train_Dataset007_clean_raw_LOCO_USZ.sh
```

### Step 5: Sync to Quartz

```bash
rsync -avz local_nnunet/ username@quartz.uits.iu.edu:/N/project/YOUR_PROJECT/local_nnunet/

rsync -avz quartz/slurm_jobs/ username@quartz.uits.iu.edu:/N/project/YOUR_PROJECT/slurm_jobs/
```

### Step 6: Train on Quartz

On Quartz:

```bash
cd /N/project/YOUR_PROJECT/slurm_jobs
```

Submit one job:

```bash
sbatch train_Dataset001_clean_raw_LOCO_CHUM.sh
```

Or submit all jobs:

```bash
./submit_all.sh
```

Check status:

```bash
squeue -u username
```

Expected model output:

```text
/N/project/YOUR_PROJECT/local_nnunet/nnunet_results/
└── DatasetXXX_clean_raw_LOCO_CENTER/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
        └── fold_0/
            ├── checkpoint_best.pth
            ├── checkpoint_final.pth
            └── progress.png
```

### Step 7: Download trained models

```bash
rsync -avz username@quartz.uits.iu.edu:/N/project/YOUR_PROJECT/local_nnunet/nnunet_results/ \
  local_nnunet/nnunet_results/
```

### Step 8: Run inference and evaluation locally

```bash
python Project/04_run_local_inference.py \
  --nnunet_raw local_nnunet/nnunet_datasets_final \
  --nnunet_results local_nnunet/nnunet_results \
  --out local_nnunet/predictions_latest \
  --configuration 3d_fullres \
  --fold 0 \
  --checkpoint checkpoint_best.pth
```

```bash
python Project/05_evaluate_predictions.py \
  --nnunet_raw local_nnunet/nnunet_datasets_final \
  --pred_root local_nnunet/predictions_latest \
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

Examples:

```text
LOCO-CHUM:
Test center = CHUM
Training centers = CHUP, CHUS, HGJ, HMR, MDA
```

```text
LOCO-MDA:
Test center = MDA
Training centers = CHUM, CHUP, CHUS, HGJ, HMR
```

Optional USZ experiment:

```text
LOCO-USZ:
Test center = USZ
Training centers = CHUM, CHUP, CHUS, HGJ, HMR, MDA
```

LOCO evaluation prevents center overlap between training and testing and provides a stricter test of cross-center generalization than random splitting.

---

## 13. Inference and Dice Score Evaluation

Inference generates predicted tumor masks using trained nnU-Net models:

```text
Project/04_run_local_inference.py
```

Evaluation compares predictions with ground-truth masks:

```text
Project/05_evaluate_predictions.py
```

Main metric:

```text
Dice = 2|A ∩ B| / (|A| + |B|)
```

where:

```text
A = predicted tumor mask
B = ground-truth tumor mask
```

Other metrics:

| Metric      | Meaning                                     |
| ----------- | ------------------------------------------- |
| IoU         | Intersection over union                     |
| Sensitivity | Fraction of ground-truth tumor detected     |
| Specificity | Fraction of background correctly classified |
| Precision   | Fraction of predicted tumor that is correct |
| Accuracy    | Overall voxel-wise correctness              |

Statistical tests:

| Test           | Purpose                                            |
| -------------- | -------------------------------------------------- |
| One-way ANOVA  | Tests whether mean Dice differs across centers     |
| Levene’s test  | Tests whether Dice variance differs across centers |
| Kruskal-Wallis | Non-parametric comparison of Dice distributions    |

---

## 14. PET/CT Qualitative Visualization

Generate a PET/CT overlay with ground truth and nnU-Net prediction:

```bash
grep "MDA-258" local_nnunet/nnunet_datasets_final/Dataset006_clean_raw_LOCO_MDA/case_mapping.csv
```

Use the `case_XXXX` value from `case_mapping.csv` in the prediction path:

```bash
python Project/06_visualize_pet_ct_gt_pred.py \
  --ct "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258__CT.nii.gz" \
  --pet "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258__PT.nii.gz" \
  --gt "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258.nii.gz" \
  --pred "local_nnunet/predictions_latest/Dataset006_clean_raw_LOCO_MDA/case_XXXX.nii.gz" \
  --out figures/MDA-258_pet_ct_gt_prediction_overlay.png \
  --patient_id MDA-258
```

The visualization shows:

1. CT only
2. PET/CT fusion
3. PET/CT with ground-truth and prediction contours
4. Zoomed error map

This visualization is important because head-and-neck tumors are not always clearly distinguishable on CT alone.

---

## 15. Experiment 7: Radiomic Feature Extraction

After nnU-Net inference, radiomic features are extracted from nnU-Net-preprocessed CT images using two mask sources:

1. Ground-truth tumor masks
2. nnU-Net prediction masks

Masks are binarized:

```text
0 = background
>0 = tumor foreground
```

Run:

```bash
source .venv_radiomics/bin/activate

python Project/07_extract_radiomics_features.py \
  --nnunet_raw local_nnunet/nnunet_datasets_final \
  --pred_root local_nnunet/predictions_latest \
  --out radiomics_outputs_nnunet \
  --mask_source both \
  --metrics_csv local_nnunet/results/per_case_dice.csv
```

Expected outputs:

```text
radiomics_outputs_nnunet/features_gt_masks.csv
radiomics_outputs_nnunet/features_pred_masks.csv
radiomics_outputs_nnunet/radiomics_extraction_summary.csv
radiomics_outputs_nnunet/_binary_masks/
```

These feature tables are used to evaluate whether tumor radiomic features retain center-specific signal after preprocessing and segmentation.

---

## 16. Experiment 8: Center Leakage and ComBat Harmonization

A center classifier is trained to predict hospital identity from extracted radiomic features. Classification above chance indicates feature-level batch leakage.

For six centers, chance balanced accuracy is approximately:

```text
1 / 6 = 0.167
```

ComBat is then applied using center as the batch variable. The center classifier and PCA visualization are repeated after harmonization.

Run on nnU-Net prediction-mask features:

```bash
python Project/08_combat_center_leakage.py \
  --features radiomics_outputs_nnunet/features_pred_masks.csv \
  --out radiomics_outputs_nnunet/combat_pred_masks \
  --center_col Center
```

Run on ground-truth-mask features:

```bash
python Project/08_combat_center_leakage.py \
  --features radiomics_outputs_nnunet/features_gt_masks.csv \
  --out radiomics_outputs_nnunet/combat_gt_masks \
  --center_col Center
```

Expected outputs:

```text
radiomics_outputs_nnunet/combat_pred_masks/center_classifier_before_after.csv
radiomics_outputs_nnunet/combat_pred_masks/features_combat_harmonized.csv
radiomics_outputs_nnunet/combat_pred_masks/pca_before_combat.png
radiomics_outputs_nnunet/combat_pred_masks/pca_after_combat.png
radiomics_outputs_nnunet/combat_pred_masks/confusion_matrix_before_combat.png
radiomics_outputs_nnunet/combat_pred_masks/confusion_matrix_after_combat.png
radiomics_outputs_nnunet/combat_pred_masks/combat_summary.json
```

The same output structure is produced for:

```text
radiomics_outputs_nnunet/combat_gt_masks/
```

Completed six-center ComBat results:

| Feature source           |   n | Features | Condition     | Accuracy | Balanced accuracy | Macro AUC |
| ------------------------ | --: | -------: | ------------- | -------: | ----------------: | --------: |
| nnU-Net prediction masks | 323 |       93 | Before ComBat |    0.505 |             0.456 |     0.761 |
| nnU-Net prediction masks | 323 |       93 | After ComBat  |    0.173 |             0.148 |     0.397 |
| Ground-truth masks       | 319 |       93 | Before ComBat |    0.549 |             0.490 |     0.812 |
| Ground-truth masks       | 319 |       93 | After ComBat  |    0.232 |             0.223 |     0.529 |

Important interpretation:

```text
ComBat is applied after segmentation.
It reduces feature-level center leakage.
It does not directly modify nnU-Net predictions or improve Dice scores.
```

---

## 17. Experiment 9: Preprocessing, Normalization, and Augmentation Visualizations

This experiment generates appendix-ready figures for preprocessing and image-level variability.

It creates:

* Center-wise case distribution
* Global CT HU histograms by center
* Tumor-mask HU histograms by center
* Tumor volume by center
* Tumor mean HU by center
* Slice thickness by center
* ROI cropping example
* Normalization strategy comparison
* Candidate augmentation examples

Run:

```bash
python Project/09_preprocessing_normalization_augmentation_visuals.py \
  --hecktor_root "data/extracted/HECKTOR 2025 Training Data Defaced ALL" \
  --out preprocessing_visuals \
  --centers CHUM,CHUP,CHUS,HGJ,HMR,MDA,USZ \
  --max_cases_per_center 25
```

Expected outputs:

```text
preprocessing_visuals/preprocessing_case_statistics.csv
preprocessing_visuals/preprocessing_center_summary.csv
preprocessing_visuals/fig_center_case_distribution.png
preprocessing_visuals/fig_global_ct_hu_hist_by_center.png
preprocessing_visuals/fig_tumor_ct_hu_hist_by_center.png
preprocessing_visuals/fig_tumor_volume_by_center.png
preprocessing_visuals/fig_tumor_mean_hu_by_center.png
preprocessing_visuals/fig_slice_thickness_by_center.png
preprocessing_visuals/fig_roi_cropping_example.png
preprocessing_visuals/fig_normalization_comparison.png
preprocessing_visuals/fig_augmentation_examples.png
```

These figures support the preprocessing, normalization, EDA, and future augmentation sections of the final report.

---

## 18. Experiment 10: Final Batch-Effect Summary

This experiment creates final summary tables and figures for the report.

Inputs:

```text
local_nnunet/results/per_case_dice.csv
radiomics_outputs_nnunet/combat_pred_masks/center_classifier_before_after.csv
radiomics_outputs_nnunet/combat_gt_masks/center_classifier_before_after.csv
```

Run:

```bash
python Project/10_make_final_batch_effect_summary.py \
  --dice_csv local_nnunet/results/per_case_dice.csv \
  --combat_pred radiomics_outputs_nnunet/combat_pred_masks/center_classifier_before_after.csv \
  --combat_gt radiomics_outputs_nnunet/combat_gt_masks/center_classifier_before_after.csv \
  --out final_project_outputs
```

Expected outputs:

```text
final_project_outputs/final_per_case_dice.csv
final_project_outputs/final_loco_dice_summary.csv
final_project_outputs/final_statistical_tests.csv
final_project_outputs/final_combat_summary.csv
final_project_outputs/final_mean_dice_by_center.png
final_project_outputs/final_batch_effect_summary.md
```

These files summarize:

1. LOCO Dice performance by held-out center
2. ANOVA and Levene’s test results
3. Center-classifier performance before and after ComBat
4. Final interpretation of segmentation-level and feature-level batch effects

---

## 19. Completed Results Snapshot

Completed six-center LOCO segmentation analysis showed:

```text
Mean Dice range: 0.68 to 0.81
One-way ANOVA: F = 5.42, p = 0.0017
Levene’s test: statistic = 3.87, p = 0.006
```

Interpretation:

```text
Segmentation performance and variance differed across hospitals,
indicating residual center-specific segmentation effects.
```

Completed radiomic feature-level analysis showed:

```text
Prediction-mask features:
Balanced accuracy before ComBat = 0.456
Balanced accuracy after ComBat  = 0.148

Ground-truth-mask features:
Balanced accuracy before ComBat = 0.490
Balanced accuracy after ComBat  = 0.223
```

Interpretation:

```text
Radiomic features retained center-specific signal after preprocessing and segmentation.
ComBat reduced feature-level center leakage.
```

---

## 20. Main Outputs

```text
data/processed/step1_qc/center_case_counts.csv
data/processed/step1_qc/center_wise_ct_case_distribution.png

local_nnunet/nnunet_datasets_final/
local_nnunet/nnunet_results/
local_nnunet/predictions_latest/
local_nnunet/results/per_case_dice.csv
local_nnunet/results/final_summary.csv

radiomics_outputs_nnunet/features_gt_masks.csv
radiomics_outputs_nnunet/features_pred_masks.csv
radiomics_outputs_nnunet/combat_pred_masks/
radiomics_outputs_nnunet/combat_gt_masks/

preprocessing_visuals/
final_project_outputs/
figures/
```

---

## 21. Files Not Tracked by Git

Do not commit:

```text
data/
model_zoo/
local_nnunet/
radiomics_outputs_nnunet/
preprocessing_visuals/
final_project_outputs/
HECKTOR 2025 Training Data Defaced ALL/
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

## 22. Reproducibility Checklist

```text
[ ] HECKTOR zip file placed in data/raw/
[ ] Python 3.11 environment created
[ ] requirements.txt installed
[ ] nnUNet_raw set
[ ] nnUNet_preprocessed set
[ ] nnUNet_results set
[ ] Dataset extracted
[ ] QC completed
[ ] Center-wise counts generated
[ ] LOCO datasets created
[ ] Trained model weights downloaded or trained on Quartz
[ ] Inference completed
[ ] Dice evaluation completed
[ ] Statistical tests completed
[ ] PET/CT overlays generated
[ ] Radiomic features extracted
[ ] Center-classifier leakage test completed
[ ] ComBat harmonization completed
[ ] Preprocessing and augmentation visualizations generated
[ ] Final summary report generated
```

---

## 23. Notes

The trained model archive reproduces the cropped LOCO workflow used in this project. The cropped workflow uses ground-truth tumor location during dataset preparation and is useful for controlled analysis. It should not be treated as strict deployment inference.

For future strict external testing, use full-image inference with:

```bash
--crop_mode none
```

The scripts support adding USZ to the LOCO process. However, USZ should only be included in segmentation-performance results after a trained LOCO-USZ model and corresponding predictions are available.

---

## 24. References

1. Oreiller V, Andrearczyk V, Jreige M, et al. Head and neck tumor segmentation in PET/CT: The HECKTOR challenge. *Medical Image Analysis*. 2022;77:102336.

2. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*. 2021;18:203–211.

3. Leek JT, Scharpf RB, Bravo HC, et al. Tackling the widespread and critical impact of batch effects in high-throughput data. *Nature Reviews Genetics*. 2010;11:733–739.

4. Orlhac F, Eertink JJ, Cottereau AS, et al. A Guide to ComBat Harmonization of Imaging Biomarkers in Multicenter Studies. *Journal of Nuclear Medicine*. 2022;63(2):172–179.

5. Da-Ano R, Visvikis D, Hatt M. Harmonization strategies for multicenter radiomics investigations. *Physics in Medicine & Biology*. 2020;65(24):24TR02.

6. Wu H, Liu X, Peng L, Yang Y, Zhou Z, Du D, Xu H, Lv W, Lu L. Optimal batch determination for improved harmonization and prognostication of multi-center PET/CT radiomics feature in head and neck cancer. *Physics in Medicine & Biology*. 2023;68(22):225014.
