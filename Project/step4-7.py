from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
import json


# CONFIG

ROOT_DIR = Path("/Users/sumantratiwari/PycharmProjects/Biomedproject/HECKTOR 2025 Training Data Defaced ALL")
STEP1_CSV = ROOT_DIR / "step1_3_flagged/ct_dataset_raw.csv"

OUT_ROOT = ROOT_DIR / "nnunet_datasets_final"

TARGET_SPACING = (1.0, 1.0, 1.0)
HU_MIN = -1000
HU_MAX = 400
MARGIN = 20



# HELPERS

def read_image(path):
    return sitk.ReadImage(str(path))


def resample(img, spacing, is_mask=False):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(img)


def normalize_ct(arr, mode):
    arr = np.clip(arr, HU_MIN, HU_MAX)

    if mode == "normalized":
        mean = arr.mean()
        std = arr.std()
        if std < 1e-6:
            std = 1.0
        arr = (arr - mean) / std
    else:
        arr = arr / HU_MAX

    return arr


def get_bbox(mask_arr):
    coords = np.argwhere(mask_arr > 0)
    if coords.size == 0:
        return None

    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)

    return zmin, zmax, ymin, ymax, xmin, xmax


def crop(arr, bbox):
    zmin, zmax, ymin, ymax, xmin, xmax = bbox

    zmin = max(0, zmin - MARGIN)
    ymin = max(0, ymin - MARGIN)
    xmin = max(0, xmin - MARGIN)

    zmax += MARGIN
    ymax += MARGIN
    xmax += MARGIN

    return arr[zmin:zmax, ymin:ymax, xmin:xmax]


# PROCESS CASE

def process_case(row, case_id, img_dir, lbl_dir, mode):
    ct = read_image(row["ct_path"])
    mask = read_image(row["mask_path"])

    ct = resample(ct, TARGET_SPACING, False)
    mask = resample(mask, TARGET_SPACING, True)

    ct_arr = sitk.GetArrayFromImage(ct)
    mask_arr = sitk.GetArrayFromImage(mask)

    bbox = get_bbox(mask_arr)
    if bbox is None:
        return False

    ct_crop = crop(ct_arr, bbox)
    mask_crop = crop(mask_arr, bbox)

    ct_norm = normalize_ct(ct_crop, mode)

    ct_img = sitk.GetImageFromArray(ct_norm.astype(np.float32))
    mask_img = sitk.GetImageFromArray(mask_crop.astype(np.uint8))

    sitk.WriteImage(ct_img, str(img_dir / f"{case_id}_0000.nii.gz"))
    sitk.WriteImage(mask_img, str(lbl_dir / f"{case_id}.nii.gz"))

    return True


# BUILD DATASET (UPDATED)

def build_dataset(df_train, df_test, dataset_name, mode):
    print(f"\nProcessing {dataset_name}")

    out_dir = OUT_ROOT / dataset_name
    imagesTr = out_dir / "imagesTr"
    labelsTr = out_dir / "labelsTr"
    imagesTs = out_dir / "imagesTs"
    labelsTs = out_dir / "labelsTs"

    for d in [imagesTr, labelsTr, imagesTs, labelsTs]:
        d.mkdir(parents=True, exist_ok=True)

    train_ids = []
    test_ids = []

    # TRAIN
    for i, (_, row) in enumerate(tqdm(df_train.iterrows(), total=len(df_train))):
        case_id = f"case_{i:04d}"
        ok = process_case(row, case_id, imagesTr, labelsTr, mode)
        if ok:
            train_ids.append(case_id)

    # TEST
    for i, (_, row) in enumerate(tqdm(df_test.iterrows(), total=len(df_test))):
        case_id = f"case_{i+len(train_ids):04d}"
        ok = process_case(row, case_id, imagesTs, labelsTs, mode)
        if ok:
            test_ids.append(case_id)

    # dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz"
    }

    with open(out_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # IMPORTANT: nnU-Net controlled split
    splits = [{
        "train": train_ids,
        "val": test_ids
    }]

    with open(out_dir / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=2)



# MAIN (UPDATED WITH CENTER SPLIT)

def main():
    df = pd.read_csv(STEP1_CSV)

    centers = sorted(df["center"].unique())
    dataset_counter = 1

    print(f"Total patients: {len(df)}")

    for mode in ["raw", "normalized"]:
        for variant in ["clean", "full"]:

            if variant == "clean":
                df_variant = df[df["is_abnormal"] == False]
            else:
                df_variant = df.copy()

            for test_center in centers:

                train_df = df_variant[df_variant["center"] != test_center]
                test_df = df_variant[df_variant["center"] == test_center]

                if len(test_df) == 0:
                    continue

                dataset_name = f"Dataset{dataset_counter:03d}_{variant}_{mode}_LOCO_{test_center}"
                dataset_counter += 1

                build_dataset(train_df, test_df, dataset_name, mode)

    print("\n✅ Step 4–7 complete with center-wise splitting.")


if __name__ == "__main__":
    main()
