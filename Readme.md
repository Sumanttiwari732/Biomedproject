````markdown
# Batch Effects in Multi-Center PET/CT Radiomics Using HECKTOR 2025 and nnU-Net

This repository contains a reproducible pipeline for studying batch effects in multi-center PET/CT data for head-and-neck tumor segmentation.

The project uses the HECKTOR 2025 training dataset and nnU-Net. The workflow includes dataset extraction, CT quality control, center-wise exploratory analysis, Leave-One-Center-Out dataset preparation, nnU-Net training on Quartz, local inference, Dice evaluation, residual batch-effect analysis, and qualitative visualization of ground truth versus model prediction.

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

## 1. Repository Contents

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
````

The folders `data/`, `local_nnunet/`, `figures/`, predictions, model checkpoints, and NIfTI files are not tracked by Git.

---

## 2. Dataset

This project uses the **HECKTOR 2025 Training Data Defaced ALL** dataset.

The dataset is not included in this repository because it contains large medical imaging files.

Place the downloaded zip file here:

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

| File                       | Description                                   |
| -------------------------- | --------------------------------------------- |
| `PatientID__CT.nii.gz`     | CT image                                      |
| `PatientID__PT.nii.gz`     | PET image                                     |
| `PatientID.nii.gz`         | Ground-truth tumor mask                       |
| `PatientID__RTDOSE.nii.gz` | Radiation dose file, not used in this project |

Segmentation labels:

| Label | Meaning                          |
| ----: | -------------------------------- |
|     0 | Background                       |
|     1 | GTVp, primary gross tumor volume |
|     2 | GTVn, nodal gross tumor volume   |

For binary nnU-Net training, labels 1 and 2 are combined into one tumor foreground label.

---

## 3. Trained LOCO Model Weights

The trained Leave-One-Center-Out nnU-Net model weights are not stored in this GitHub repository.

Download the trained model archive here:

[Download trained HECKTOR LOCO models](https://indiana-my.sharepoint.com/my?id=%2Fpersonal%2Fsrtiwari%5Fiu%5Fedu%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files%2Fhecktor%5Floco%5Fmodels%2Etar%2Egz&parent=%2Fpersonal%2Fsrtiwari%5Fiu%5Fedu%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files&ct=1778459525253&or=Teams%2DHL&ga=1&LOF=1)

The SharePoint link may require Indiana University login.

Save the file as:

```text
model_zoo/hecktor_loco_models.tar.gz
```

Create the model folder:

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

If the archive already contains a top-level `nnUNet_results/` folder, extract it into `local_nnunet`:

```bash
tar -xzf model_zoo/hecktor_loco_models.tar.gz -C local_nnunet
```

After extraction, the expected model structure should look like this:

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

## 4. Software Setup

Recommended Python version:

```text
Python 3.10 or 3.11
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install packages:

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

## 5. Requirements

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

## 6. Quick Start: Use the Trained Models for Inference

Use this path if you only want to run prediction and evaluation using the trained LOCO models.

### Step 1: Extract the HECKTOR dataset

```bash
python scripts/00_extract_hecktor_zip.py \
  --zip data/raw/HECKTOR_2025_Training_Data_Defaced_ALL.zip \
  --out data/extracted
```

Expected output:

```text
data/extracted/HECKTOR 2025 Training Data Defaced ALL/
```

### Step 2: Run quality control

```bash
python scripts/01_qc_ct_dataset.py \
  --root "data/extracted/HECKTOR 2025 Training Data Defaced ALL" \
  --out data/processed/step1_qc
```

Expected outputs:

```text
data/processed/step1_qc/ct_dataset_raw.csv
data/processed/step1_qc/ct_dataset_normalized.csv
data/processed/step1_qc/abnormal_cases.csv
data/processed/step1_qc/abnormal_by_center.csv
data/processed/step1_qc/center_case_counts.csv
data/processed/step1_qc/center_wise_ct_case_distribution.png
```

### Step 3: Build nnU-Net LOCO datasets

```bash
python scripts/02_build_nnunet_loco_datasets.py \
  --qc_csv data/processed/step1_qc/ct_dataset_raw.csv \
  --out local_nnunet/nnUNet_raw \
  --label_mode binary \
  --crop_mode mask
```

Use `--crop_mode mask` only when reproducing the cropped workflow used for the trained LOCO models.

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
```

### Step 4: Download and extract trained weights

Download:

[Download trained HECKTOR LOCO models](https://indiana-my.sharepoint.com/my?id=%2Fpersonal%2Fsrtiwari%5Fiu%5Fedu%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files%2Fhecktor%5Floco%5Fmodels%2Etar%2Egz&parent=%2Fpersonal%2Fsrtiwari%5Fiu%5Fedu%2FDocuments%2FMicrosoft%20Teams%20Chat%20Files&ct=1778459525253&or=Teams%2DHL&ga=1&LOF=1)

Save as:

```text
model_zoo/hecktor_loco_models.tar.gz
```

Extract:

```bash
tar -tzf model_zoo/hecktor_loco_models.tar.gz | head
tar -xzf model_zoo/hecktor_loco_models.tar.gz -C local_nnunet/nnUNet_results
```

If the extracted folder is nested incorrectly, move the `DatasetXXX...` folders so they sit directly inside:

```text
local_nnunet/nnUNet_results/
```

### Step 5: Run local inference

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
├── Dataset001_clean_raw_LOCO_CHUM/
├── Dataset002_clean_raw_LOCO_CHUP/
├── Dataset003_clean_raw_LOCO_CHUS/
├── Dataset004_clean_raw_LOCO_HGJ/
├── Dataset005_clean_raw_LOCO_HMR/
└── Dataset006_clean_raw_LOCO_MDA/
```

### Step 6: Evaluate Dice and residual batch effects

```bash
python scripts/05_evaluate_predictions.py \
  --nnunet_raw local_nnunet/nnUNet_raw \
  --pred_root local_nnunet/predictions \
  --out local_nnunet/results
```

Expected outputs:

```text
local_nnunet/results/per_case_metrics.csv
local_nnunet/results/final_summary.csv
local_nnunet/results/statistical_tests.txt
local_nnunet/results/dice_by_hospital.png
local_nnunet/results/mean_dice_by_hospital.png
local_nnunet/results/overlays/
```

The evaluation script computes:

* Dice
* IoU
* Sensitivity
* Specificity
* Precision
* Accuracy
* One-way ANOVA across centers
* Levene’s test across centers

---

## 7. Full Workflow: Train Models from Scratch

Use this path if you want to train the LOCO models again.

### Step 1: Extract dataset

```bash
python scripts/00_extract_hecktor_zip.py \
  --zip data/raw/HECKTOR_2025_Training_Data_Defaced_ALL.zip \
  --out data/extracted
```

### Step 2: Run QC

```bash
python scripts/01_qc_ct_dataset.py \
  --root "data/extracted/HECKTOR 2025 Training Data Defaced ALL" \
  --out data/processed/step1_qc
```

### Step 3: Build LOCO nnU-Net datasets

```bash
python scripts/02_build_nnunet_loco_datasets.py \
  --qc_csv data/processed/step1_qc/ct_dataset_raw.csv \
  --out local_nnunet/nnUNet_raw \
  --label_mode binary \
  --crop_mode mask
```

### Step 4: Generate Quartz Slurm scripts

```bash
python scripts/03_make_quartz_slurm.py \
  --nnunet_raw local_nnunet/nnUNet_raw \
  --out quartz/slurm_jobs \
  --configuration 3d_fullres \
  --fold 0
```

Expected output:

```text
quartz/slurm_jobs/train_Dataset001_clean_raw_LOCO_CHUM.sh
quartz/slurm_jobs/train_Dataset002_clean_raw_LOCO_CHUP.sh
...
```

### Step 5: Sync data to Quartz

```bash
rsync -avz local_nnunet/ username@quartz.uits.iu.edu:/N/project/your_project/local_nnunet/
rsync -avz quartz/slurm_jobs/ username@quartz.uits.iu.edu:/N/project/your_project/slurm_jobs/
```

### Step 6: Train on Quartz

On Quartz:

```bash
cd /N/project/your_project
```

Set paths:

```bash
export nnUNet_raw="/N/project/your_project/local_nnunet/nnUNet_raw"
export nnUNet_preprocessed="/N/project/your_project/local_nnunet/nnUNet_preprocessed"
export nnUNet_results="/N/project/your_project/local_nnunet/nnUNet_results"
```

Submit jobs:

```bash
sbatch slurm_jobs/train_Dataset001_clean_raw_LOCO_CHUM.sh
sbatch slurm_jobs/train_Dataset002_clean_raw_LOCO_CHUP.sh
sbatch slurm_jobs/train_Dataset003_clean_raw_LOCO_CHUS.sh
sbatch slurm_jobs/train_Dataset004_clean_raw_LOCO_HGJ.sh
sbatch slurm_jobs/train_Dataset005_clean_raw_LOCO_HMR.sh
sbatch slurm_jobs/train_Dataset006_clean_raw_LOCO_MDA.sh
```

Check jobs:

```bash
squeue -u username
```

Expected model output:

```text
local_nnunet/nnUNet_results/
└── DatasetXXX_clean_raw_LOCO_CENTER/
    └── nnUNetTrainer__nnUNetPlans__3d_fullres/
        └── fold_0/
            ├── checkpoint_best.pth
            ├── checkpoint_final.pth
            └── progress.png
```

### Step 7: Download trained models

```bash
rsync -avz username@quartz.uits.iu.edu:/N/project/your_project/local_nnunet/nnUNet_results/ \
  local_nnunet/nnUNet_results/
```

### Step 8: Run inference and evaluation

```bash
python scripts/04_run_local_inference.py \
  --nnunet_raw local_nnunet/nnUNet_raw \
  --nnunet_results local_nnunet/nnUNet_results \
  --out local_nnunet/predictions \
  --configuration 3d_fullres \
  --fold 0 \
  --checkpoint checkpoint_best.pth
```

```bash
python scripts/05_evaluate_predictions.py \
  --nnunet_raw local_nnunet/nnUNet_raw \
  --pred_root local_nnunet/predictions \
  --out local_nnunet/results
```

---

## 8. Leave-One-Center-Out Design

This project uses Leave-One-Center-Out evaluation.

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

```text
LOCO-MDA:
Test center = MDA
Training centers = CHUM, CHUP, CHUS, HGJ, HMR
```

This design tests whether the model generalizes to unseen centers.

---

## 9. Center-Wise Dataset Summary

Example counts from the local dataset:

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

USZ is included in dataset distribution plots. If no trained LOCO model is available for USZ, USZ is excluded from model-performance comparisons.

---

## 10. Evaluation Metrics

The main metric is Dice:

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

| Test          | Purpose                                            |
| ------------- | -------------------------------------------------- |
| One-way ANOVA | Tests whether mean Dice differs across centers     |
| Levene’s test | Tests whether Dice variance differs across centers |

---

## 11. Visualization

Generate a PET/CT ground-truth and prediction overlay:

```bash
python scripts/06_visualize_pet_ct_gt_pred.py \
  --ct "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258__CT.nii.gz" \
  --pet "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258__PT.nii.gz" \
  --gt "data/extracted/HECKTOR 2025 Training Data Defaced ALL/MDA-258/MDA-258.nii.gz" \
  --pred "local_nnunet/predictions/Dataset006_clean_raw_LOCO_MDA/case_XXXX.nii.gz" \
  --out figures/MDA-258_pet_ct_gt_prediction_overlay.png
```

The visualization shows:

1. CT only
2. PET/CT fusion
3. PET/CT with ground truth and prediction contours
4. Zoomed error map

Color convention:

| Region                | Color          |
| --------------------- | -------------- |
| Ground truth          | Cyan or green  |
| nnU-Net prediction    | Magenta or red |
| True positive overlap | Green          |
| False positive        | Yellow         |
| False negative        | Blue           |

---

## 12. Outputs

Main outputs:

```text
data/processed/step1_qc/center_case_counts.csv
data/processed/step1_qc/center_wise_ct_case_distribution.png
local_nnunet/predictions/
local_nnunet/results/per_case_metrics.csv
local_nnunet/results/final_summary.csv
local_nnunet/results/statistical_tests.txt
local_nnunet/results/dice_by_hospital.png
local_nnunet/results/mean_dice_by_hospital.png
local_nnunet/results/overlays/
figures/
```

---

## 13. Files Not Tracked by Git

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

## 14. Reproducibility Checklist

```text
[ ] HECKTOR zip file placed in data/raw/
[ ] Python environment created
[ ] requirements.txt installed
[ ] nnUNet_raw set
[ ] nnUNet_preprocessed set
[ ] nnUNet_results set
[ ] Dataset extracted
[ ] QC completed
[ ] LOCO datasets created
[ ] Trained model weights downloaded or models trained on Quartz
[ ] Inference completed
[ ] Dice evaluation completed
[ ] Statistical tests completed
[ ] Qualitative overlays generated
```

---

## 15. References

1. Oreiller V, Andrearczyk V, Jreige M, et al. Head and neck tumor segmentation in PET/CT: The HECKTOR challenge. Medical Image Analysis. 2022;77:102336.

2. Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods. 2021;18:203–211.

3. Leek JT, Scharpf RB, Bravo HC, et al. Tackling the widespread and critical impact of batch effects in high-throughput data. Nature Reviews Genetics. 2010;11:733–739.

4. Orlhac F, Eertink JJ, Cottereau AS, et al. A Guide to ComBat Harmonization of Imaging Biomarkers in Multicenter Studies. Journal of Nuclear Medicine. 2022;63(2):172–179.

5. Da-Ano R, Visvikis D, Hatt M. Harmonization strategies for multicenter radiomics investigations. Physics in Medicine & Biology. 2020;65(24):24TR02.

6. Wu H, Liu X, Peng L, et al. Optimal batch determination for improved harmonization and prognostication of multi-center PET/CT radiomics feature in head and neck cancer. Physics in Medicine & Biology. 2023;68(22):225014.
